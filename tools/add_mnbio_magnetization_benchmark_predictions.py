from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_magnetization_multiclass_top_bin import run_inference

BENCHMARK_CSV = Path(
    "~/Library/CloudStorage/CloudMounter-StampedeWork/run/MnBiO/predict/magnetization/"
    "magnetization_volume_summary.csv"
).expanduser()
RUN_DIR = Path(
    "/Users/qz/.ghq/github.com/singularitti/cgcnn/tmp/prediction_runs/"
    "mnbio_magnetization_combined_20260529_131358"
)
OUTPUTS_DIR = RUN_DIR / "outputs"
ALL_DATASET_DIR = RUN_DIR / "classifier_dataset_all"
TOP_BIN_REGRESSOR_CHECKPOINT = RUN_DIR / "inputs" / "checkpoints" / "epoch_029.pth.tar"
SOURCE_TOP_BIN_REGRESSOR_CHECKPOINT = Path(
    "~/Library/CloudStorage/CloudMounter-StampedeWork/run/MnBiO/predict/magnetization/"
    "epoch_029.pth.tar"
).expanduser()
ALL_REGRESSOR_PREDICTIONS_CSV = OUTPUTS_DIR / "top_bin_regressor_predictions_all.csv"
MATCHED_CSV = OUTPUTS_DIR / "magnetization_volume_summary_with_cgcnn_all.csv"
PARITY_PNG = OUTPUTS_DIR / "magnetization_volume_parity_all.png"


def read_predictions(path: Path) -> dict[str, float]:
    predictions: dict[str, float] = {}
    with path.open(newline="") as handle:
        for material_id, _target, prediction in csv.reader(handle):
            predictions[material_id] = float(prediction)
    return predictions


def run_regressor_on_all_structures() -> Path:
    run_inference(
        dataset_dir=ALL_DATASET_DIR,
        model_path=(
            TOP_BIN_REGRESSOR_CHECKPOINT
            if TOP_BIN_REGRESSOR_CHECKPOINT.is_file()
            else SOURCE_TOP_BIN_REGRESSOR_CHECKPOINT
        ),
        task="regression",
        output_csv=ALL_REGRESSOR_PREDICTIONS_CSV,
        batch_size=256,
        workers=0,
    )
    return ALL_REGRESSOR_PREDICTIONS_CSV


def format_float(value: float) -> str:
    return f"{value:.16g}"


def update_benchmark_csv(
    benchmark_csv: Path,
    predictions: dict[str, float],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with benchmark_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for field in ["M", "CGCNN_M", "CGCNN_error"]:
        if field not in fieldnames:
            fieldnames.append(field)

    matched_rows: list[dict[str, str]] = []
    for row in rows:
        target = float(row["d_vol"]) + float(row["p_vol"]) + float(row["s_vol"])
        row["M"] = format_float(target)
        prediction = predictions.get(row["ID"])
        if prediction is None:
            row["CGCNN_M"] = ""
            row["CGCNN_error"] = ""
            continue
        row["CGCNN_M"] = format_float(prediction)
        row["CGCNN_error"] = format_float(prediction - target)
        matched_rows.append(row.copy())

    with benchmark_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows, matched_rows


def write_matched_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("No matched rows available for plotting.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parity_plot(path: Path, rows: list[dict[str, str]]) -> dict[str, float]:
    target = np.array([float(row["M"]) for row in rows], dtype=float)
    predicted = np.array([float(row["CGCNN_M"]) for row in rows], dtype=float)
    error = predicted - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    lo = float(min(target.min(), predicted.min()))
    hi = float(max(target.max(), predicted.max()))
    pad = (hi - lo) * 0.08 if hi > lo else 0.01

    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    ax.scatter(target, predicted, s=42, color="#2563eb", edgecolor="white", linewidth=0.7)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111827", linewidth=1.1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Target magnetization per volume")
    ax.set_ylabel("CGCNN predicted magnetization per volume")
    ax.set_title("MnBiO Magnetization Parity")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.04,
        0.96,
        f"n = {len(rows)}\nMAE = {mae:.4g}\nRMSE = {rmse:.4g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {"n": float(len(rows)), "mae": mae, "rmse": rmse}


if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    all_prediction_csv = run_regressor_on_all_structures()
    prediction_by_id = read_predictions(all_prediction_csv)
    all_rows, matched = update_benchmark_csv(BENCHMARK_CSV, prediction_by_id)
    write_matched_csv(MATCHED_CSV, matched)
    metrics = write_parity_plot(PARITY_PNG, matched)
    print(f"updated_csv={BENCHMARK_CSV}")
    print(f"all_regressor_predictions={all_prediction_csv}")
    print(f"matched_csv={MATCHED_CSV}")
    print(f"parity_png={PARITY_PNG}")
    print(f"rows={len(all_rows)} matched={len(matched)} missing={len(all_rows) - len(matched)}")
    print(f"mae={metrics['mae']:.16g} rmse={metrics['rmse']:.16g}")
