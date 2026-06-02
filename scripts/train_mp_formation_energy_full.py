from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import warnings

for warning_message in (
    r".*not find enough neighbors to build graph.*",
    r".*Issues encountered while parsing CIF.*",
    r".*No Pauling electronegativity.*",
):
    warnings.filterwarnings(
        "ignore",
        message=warning_message,
        category=UserWarning,
    )

os.environ["PYTHONWARNINGS"] = ",".join([
    "ignore:.*not find enough neighbors to build graph.*:UserWarning",
    "ignore:.*Issues encountered while parsing CIF.*:UserWarning",
    "ignore:.*No Pauling electronegativity.*:UserWarning",
])

from cgcnn.training import train_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "data" / "mp-all-formation-energy"
PRETRAINED = (
    Path.home()
    / "Downloads"
    / "training_runs"
    / "mp_all_formation_energy_subset_converged_cpu"
    / "model_best.pth.tar"
)
RUN_DIR = (
    Path.home()
    / "Downloads"
    / "training_runs"
    / "mp_all_formation_energy_full_from_subset_20260420"
)
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
METRICS_PATH = RUN_DIR / "metrics_history.json"
SUMMARY_PATH = RUN_DIR / "training_summary.json"


def regression_metrics(path: Path) -> dict[str, float | int]:
    y_true: list[float] = []
    y_pred: list[float] = []
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            y_true.append(float(row[1]))
            y_pred.append(float(row[2]))
    if not y_true:
        raise ValueError(f"No predictions found in {path}")
    errors = [pred - true for true, pred in zip(y_true, y_pred)]
    abs_errors = [abs(err) for err in errors]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(err * err for err in errors) / len(errors))
    mean_true = sum(y_true) / len(y_true)
    mean_pred = sum(y_pred) / len(y_pred)
    ss_res = sum(err * err for err in errors)
    ss_tot = sum((true - mean_true) ** 2 for true in y_true)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    cov = sum((true - mean_true) * (pred - mean_pred) for true, pred in zip(y_true, y_pred))
    var_true = sum((true - mean_true) ** 2 for true in y_true)
    var_pred = sum((pred - mean_pred) ** 2 for pred in y_pred)
    r = cov / math.sqrt(var_true * var_pred) if var_true and var_pred else float("nan")
    return {
        "count": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "r": r,
        "r2": r2,
        "min_abs_error": min(abs_errors),
        "max_abs_error": max(abs_errors),
    }


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    if not PRETRAINED.exists():
        raise FileNotFoundError(PRETRAINED)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(RUN_DIR)

    best_checkpoint = train_model(
        str(DATASET),
        task="regression",
        epochs=100,
        batch_size=256,
        workers=8,
        cuda=None,
        initialize_from=str(PRETRAINED),
        checkpoint_dir=str(CHECKPOINT_DIR),
        metrics_history_path=str(METRICS_PATH),
        print_freq=50,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        early_stopping_patience=8,
        early_stopping_min_delta=0.002,
    )

    history = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else []
    best_history = min(history, key=lambda row: float(row["val_metric"])) if history else None
    summary = {
        "dataset": str(DATASET),
        "source_properties_csv": str(Path.home() / "Downloads" / "cifs" / "mp_all_summary.csv"),
        "target": "formation_energy_per_atom",
        "training_rows": 154879,
        "split": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        "initialized_from": str(PRETRAINED),
        "best_checkpoint": str(best_checkpoint),
        "epochs_completed": max((int(row["epoch"]) for row in history), default=0),
        "best_epoch": int(best_history["epoch"]) if best_history else None,
        "best_validation_mae": float(best_history["val_metric"]) if best_history else None,
        "early_stopping": {"patience": 8, "min_delta": 0.002, "max_epochs": 100},
        "test_metrics": regression_metrics(RUN_DIR / "test_results.csv"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
