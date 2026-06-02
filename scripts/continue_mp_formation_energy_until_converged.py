from __future__ import annotations

import csv
import json
import math
import os
import shutil
import warnings
from pathlib import Path

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
SOURCE_RUN = REPO_ROOT / "runs" / "mp_all_formation_energy_subset_epoch1_cpu"
SOURCE_CHECKPOINT = SOURCE_RUN / "checkpoint.pth.tar"
SOURCE_HISTORY = SOURCE_RUN / "metrics_history.json"
OUT_ROOT = Path.home() / "Downloads" / "training_runs"
RUN_DIR = OUT_ROOT / "mp_all_formation_energy_subset_converged_cpu"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
HISTORY_PATH = RUN_DIR / "metrics_history.json"
SUMMARY_PATH = RUN_DIR / "convergence_summary.json"

TRAIN_SIZE = 1000
VAL_SIZE = 200
TEST_SIZE = 200
BATCH_SIZE = 64
WORKERS = 4
MAX_EPOCHS = 50
PATIENCE = 6
MIN_DELTA = 0.005


def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())
    if SOURCE_HISTORY.exists():
        history = json.loads(SOURCE_HISTORY.read_text())
        HISTORY_PATH.write_text(json.dumps(history, indent=2))
        return history
    return []


def best_epoch(history: list[dict]) -> tuple[int, float]:
    best = min(history, key=lambda row: float(row["val_metric"]))
    return int(best["epoch"]), float(best["val_metric"])


def epochs_without_improvement(history: list[dict], min_delta: float) -> int:
    best = math.inf
    stale = 0
    for row in history:
        val = float(row["val_metric"])
        if val < best - min_delta:
            best = val
            stale = 0
        else:
            stale += 1
    return stale


def regression_metrics(path: Path) -> dict:
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
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    checkpoint = RUN_DIR / "checkpoint.pth.tar"
    if not checkpoint.exists():
        shutil.copy2(SOURCE_CHECKPOINT, checkpoint)
    os.chdir(RUN_DIR)

    while True:
        history = load_history()
        current_epoch = max(int(row["epoch"]) for row in history) if history else 1
        stale = epochs_without_improvement(history, MIN_DELTA) if history else 0
        best_ep, best_val = best_epoch(history) if history else (current_epoch, math.inf)
        print(
            f"status epoch={current_epoch} best_epoch={best_ep} "
            f"best_val_mae={best_val:.6f} stale={stale}/{PATIENCE}",
            flush=True,
        )
        if current_epoch >= MAX_EPOCHS or stale >= PATIENCE:
            break
        next_epoch = current_epoch + 1
        train_model(
            str(DATASET),
            task="regression",
            epochs=next_epoch,
            batch_size=BATCH_SIZE,
            workers=WORKERS,
            cuda=False,
            checkpoint_dir=str(CHECKPOINT_DIR),
            metrics_history_path=str(HISTORY_PATH),
            print_freq=1,
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            resume=str(checkpoint),
        )

    metrics = regression_metrics(RUN_DIR / "test_results.csv")
    history = load_history()
    best_ep, best_val = best_epoch(history)
    summary = {
        "criterion": {
            "primary": "validation MAE",
            "patience_epochs": PATIENCE,
            "min_delta": MIN_DELTA,
            "max_epochs": MAX_EPOCHS,
        },
        "epochs_completed": max(int(row["epoch"]) for row in history),
        "best_epoch": best_ep,
        "best_validation_mae": best_val,
        "test_metrics": metrics,
        "run_dir": str(RUN_DIR),
        "resumed_from": str(SOURCE_CHECKPOINT),
        "dataset": str(DATASET),
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
