from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from prostate_diagnosis.config import TrainConfig
from prostate_diagnosis.data import (
    build_manifest,
    build_tf_dataset,
    split_manifest_by_patient,
    summarize_manifest,
)
from prostate_diagnosis.model import build_model


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train prostate cancer DICOM classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-slices-per-patient", type=int, default=64)
    parser.add_argument("--max-total-slices", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    return TrainConfig(**vars(args))


def compute_class_weight(manifest: list[tuple[str, str, int]]) -> dict[int, float]:
    counts = Counter(label for _, _, label in manifest)
    total = sum(counts.values())
    classes = sorted(counts)
    return {label: total / (len(classes) * counts[label]) for label in classes}


def main() -> None:
    import tensorflow as tf

    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset manifest...")
    manifest = build_manifest(
        data_dir=config.data_dir,
        labels_csv=config.labels_csv,
        patient_col=config.patient_col,
        label_col=config.label_col,
        max_slices_per_patient=config.max_slices_per_patient,
        max_total_slices=config.max_total_slices,
        seed=config.seed,
    )
    summarize_manifest("all", manifest)

    train_manifest, val_manifest = split_manifest_by_patient(
        manifest,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    summarize_manifest("train", train_manifest)
    summarize_manifest("validation", val_manifest)

    print("Creating streaming TensorFlow datasets...")
    train_ds = build_tf_dataset(
        train_manifest,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_ds = build_tf_dataset(
        val_manifest,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    model = build_model(config.image_size, config.learning_rate)
    model.summary()

    checkpoint_path = config.output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(filename=str(config.output_dir / "training_log.csv")),
    ]

    class_weight = compute_class_weight(train_manifest)
    print(f"class_weight={class_weight}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    final_path = config.output_dir / "final_model.keras"
    model.save(final_path)
    print(f"Saved final model: {final_path.resolve()}")
    print(f"Saved best model: {checkpoint_path.resolve()}")
    print(f"Training history keys: {list(history.history.keys())}")


if __name__ == "__main__":
    main()
