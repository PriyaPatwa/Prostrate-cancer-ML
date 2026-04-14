from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    labels_csv: Path | None = None
    patient_col: str = "patient_id"
    label_col: str = "label"
    image_size: int = 160
    batch_size: int = 16
    epochs: int = 10
    validation_fraction: float = 0.2
    seed: int = 42
    max_slices_per_patient: int | None = 64
    max_total_slices: int | None = None
    output_dir: Path = Path("outputs")
    learning_rate: float = 1e-3

