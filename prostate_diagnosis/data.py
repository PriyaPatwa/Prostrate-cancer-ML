from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pydicom

DICOM_SUFFIXES = {".dcm", ".dicom"}
FOLDER_LABEL_RE = re.compile(r"^ProstateDx-(\d+)-\d+$", re.IGNORECASE)


def find_dicom_files(data_dir: Path) -> list[Path]:
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    files = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DICOM_SUFFIXES
    ]
    return sorted(files)


def patient_id_from_path(path: Path, data_dir: Path) -> str:
    rel = path.resolve().relative_to(data_dir.resolve())
    if not rel.parts:
        raise ValueError(f"Could not extract patient id from path: {path}")
    return rel.parts[0]


def parse_binary_label(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"0", "false", "benign", "normal", "negative", "no", "non-cancer"}:
        return 0
    if text in {"1", "true", "malignant", "cancer", "positive", "yes", "tumor"}:
        return 1
    try:
        number = int(float(text))
    except ValueError as exc:
        raise ValueError(f"Unsupported label value: {value!r}") from exc
    if number not in (0, 1):
        raise ValueError(f"Only binary labels 0/1 are supported, got: {value!r}")
    return number


def load_labels_from_csv(labels_csv: Path, patient_col: str, label_col: str) -> dict[str, int]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV does not exist: {labels_csv}")
    with labels_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV has no header row: {labels_csv}")
        missing = {patient_col, label_col} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Labels CSV is missing columns {sorted(missing)}. "
                f"Available columns: {reader.fieldnames}"
            )
        labels: dict[str, int] = {}
        for row_number, row in enumerate(reader, start=2):
            patient_id = str(row[patient_col]).strip()
            if not patient_id:
                raise ValueError(f"Empty patient id at CSV row {row_number}")
            labels[patient_id] = parse_binary_label(row[label_col])
    if not labels:
        raise ValueError(f"Labels CSV contains no usable rows: {labels_csv}")
    return labels


def infer_labels_from_patient_folders(patient_ids: Iterable[str]) -> dict[str, int]:
    codes: dict[str, str] = {}
    for patient_id in patient_ids:
        match = FOLDER_LABEL_RE.match(patient_id)
        if match:
            codes[patient_id] = match.group(1)
    if not codes:
        raise ValueError(
            "Could not infer labels from patient folders. Provide --labels-csv with "
            "patient and label columns."
        )
    unique_codes = sorted(set(codes.values()))
    if len(unique_codes) != 2:
        raise ValueError(
            "Folder-label fallback expects exactly 2 class codes, but found "
            f"{unique_codes}. Provide --labels-csv for reliable labels."
        )
    code_to_label = {code: idx for idx, code in enumerate(unique_codes)}
    print(
        "WARNING: using folder-name label fallback: "
        + ", ".join(f"ProstateDx-{code}-* -> {label}" for code, label in code_to_label.items())
    )
    return {patient_id: code_to_label[code] for patient_id, code in codes.items()}


def build_manifest(
    data_dir: Path,
    labels_csv: Path | None = None,
    patient_col: str = "patient_id",
    label_col: str = "label",
    max_slices_per_patient: int | None = None,
    max_total_slices: int | None = None,
    seed: int = 42,
) -> list[tuple[str, str, int]]:
    files = find_dicom_files(data_dir)
    if not files:
        raise ValueError(f"No DICOM files found under: {data_dir.resolve()}")

    patient_to_files: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        patient_to_files[patient_id_from_path(path, data_dir)].append(path)

    if labels_csv is not None:
        labels = load_labels_from_csv(labels_csv, patient_col, label_col)
        label_source = str(labels_csv)
    else:
        labels = infer_labels_from_patient_folders(patient_to_files)
        label_source = "patient folder names"

    manifest: list[tuple[str, str, int]] = []
    rng = random.Random(seed)
    missing_label_patients: list[str] = []

    for patient_id, paths in sorted(patient_to_files.items()):
        if patient_id not in labels:
            missing_label_patients.append(patient_id)
            continue
        selected = sorted(paths)
        if max_slices_per_patient is not None and len(selected) > max_slices_per_patient:
            selected = sorted(rng.sample(selected, max_slices_per_patient))
        for path in selected:
            manifest.append((str(path), patient_id, labels[patient_id]))

    if missing_label_patients:
        preview = ", ".join(missing_label_patients[:10])
        print(
            f"WARNING: skipped {len(missing_label_patients)} patients without labels from "
            f"{label_source}. First skipped: {preview}"
        )

    if max_total_slices is not None and len(manifest) > max_total_slices:
        manifest = sorted(rng.sample(manifest, max_total_slices), key=lambda item: item[0])

    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: list[tuple[str, str, int]]) -> None:
    if not manifest:
        raise ValueError("Dataset manifest is empty after applying labels and slice limits.")
    labels = [label for _, _, label in manifest]
    label_counts = Counter(labels)
    if len(label_counts) < 2:
        raise ValueError(
            "Training needs both classes. Label counts after filtering: "
            f"{dict(label_counts)}"
        )
    bad_labels = sorted(label for label in label_counts if label not in (0, 1))
    if bad_labels:
        raise ValueError(f"Only binary labels 0/1 are supported. Found: {bad_labels}")


def split_manifest_by_patient(
    manifest: list[tuple[str, str, int]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")

    patient_to_items: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    patient_to_label: dict[str, int] = {}
    for item in manifest:
        _, patient_id, label = item
        patient_to_items[patient_id].append(item)
        patient_to_label[patient_id] = label

    label_to_patients: dict[int, list[str]] = defaultdict(list)
    for patient_id, label in patient_to_label.items():
        label_to_patients[label].append(patient_id)

    rng = random.Random(seed)
    train_patients: set[str] = set()
    val_patients: set[str] = set()

    for label, patients in sorted(label_to_patients.items()):
        rng.shuffle(patients)
        if len(patients) < 2:
            raise ValueError(
                f"Class {label} has only {len(patients)} patient(s); cannot make "
                "a patient-level train/validation split."
            )
        val_count = max(1, int(round(len(patients) * validation_fraction)))
        val_count = min(val_count, len(patients) - 1)
        val_patients.update(patients[:val_count])
        train_patients.update(patients[val_count:])

    train = [item for pid in sorted(train_patients) for item in patient_to_items[pid]]
    val = [item for pid in sorted(val_patients) for item in patient_to_items[pid]]
    validate_manifest(train)
    validate_manifest(val)
    return train, val


def _first_number(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, pydicom.multival.MultiValue):
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_dicom_image(path: str, image_size: int) -> np.ndarray:
    dataset = pydicom.dcmread(path, force=True)
    image = dataset.pixel_array.astype(np.float32)
    if image.size == 0:
        raise ValueError(f"Empty pixel array in DICOM file: {path}")

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    if getattr(dataset, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        image = image.max() - image

    center = _first_number(getattr(dataset, "WindowCenter", None))
    width = _first_number(getattr(dataset, "WindowWidth", None))
    if center is not None and width is not None and width > 1:
        low = center - width / 2.0
        high = center + width / 2.0
    else:
        low, high = np.percentile(image, [1.0, 99.0])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(image)), float(np.max(image))

    if high <= low:
        image = np.zeros_like(image, dtype=np.float32)
    else:
        image = np.clip(image, low, high)
        image = (image - low) / (high - low)

    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D DICOM slice after preprocessing, got {image.shape}")
    return np.expand_dims(image, axis=-1)


def make_numpy_generator(manifest: list[tuple[str, str, int]], image_size: int):
    def generator():
        for path, _, label in manifest:
            try:
                image = read_dicom_image(path, image_size)
            except Exception as exc:
                raise RuntimeError(f"Failed preprocessing DICOM file {path}: {exc}") from exc
            yield image, np.asarray(label, dtype=np.float32)

    return generator


def build_tf_dataset(
    manifest: list[tuple[str, str, int]],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    import tensorflow as tf

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        make_numpy_generator(manifest, image_size),
        output_signature=output_signature,
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(manifest), max(batch_size * 16, 128)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def summarize_manifest(name: str, manifest: list[tuple[str, str, int]]) -> None:
    labels = Counter(label for _, _, label in manifest)
    patients = defaultdict(set)
    for _, patient_id, label in manifest:
        patients[label].add(patient_id)
    print(f"{name}: {len(manifest)} slices")
    print(f"{name}: label counts by slice: {dict(sorted(labels.items()))}")
    print(
        f"{name}: patient counts by label: "
        f"{dict(sorted((label, len(ids)) for label, ids in patients.items()))}"
    )
    print(f"{name}: first sample path: {manifest[0][0]}")

