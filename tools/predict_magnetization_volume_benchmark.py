from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_magnetization_multiclass_top_bin import run_inference, write_atom_init


SYSTEM_NAME = os.environ.get("CGCNN_BENCHMARK_SYSTEM", "MnBiAl")
SOURCE_DIR = Path(
    os.environ.get(
        "CGCNN_BENCHMARK_SOURCE_DIR",
        "~/Library/CloudStorage/CloudMounter-StampedeWork/run/MnBiAl/cgcnn_predict",
    )
).expanduser()
BENCHMARK_CSV = SOURCE_DIR / "magnetization_volume_summary.csv"
TOP_BIN_REGRESSOR_CHECKPOINT = Path(
    os.environ.get(
        "CGCNN_TOP_BIN_REGRESSOR_CHECKPOINT",
        "~/Library/CloudStorage/CloudMounter-StampedeWork/run/MnBiO/predict/magnetization/"
        "epoch_029.pth.tar",
    )
).expanduser()
RUNS_ROOT = REPO_ROOT / "tmp" / "prediction_runs"
RUN_PREFIX = os.environ.get("CGCNN_BENCHMARK_RUN_PREFIX", "mnbial_magnetization_all_regressor")
BATCH_SIZE = 256
WORKERS = 0


def make_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"{RUN_PREFIX}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collect_cifs(source_dir: Path) -> list[Path]:
    cifs = sorted(source_dir.glob("*.cif"))
    if not cifs:
        raise RuntimeError(f"No CIF files found in {source_dir}")
    return cifs


def symlink_force(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


def write_dataset(dataset_dir: Path, cifs: list[Path]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=False)
    write_atom_init(dataset_dir / "atom_init.json")
    with (dataset_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for cif_path in cifs:
            symlink_force(cif_path, dataset_dir / f"{cif_path.stem}.cif")
            writer.writerow([cif_path.stem, "0.0"])


def read_predictions(path: Path) -> dict[str, float]:
    predictions: dict[str, float] = {}
    with path.open(newline="") as handle:
        for material_id, _target, prediction in csv.reader(handle):
            predictions[material_id] = float(prediction)
    return predictions


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

    required_fields = {"ID", "d_vol", "p_vol", "s_vol"}
    missing_fields = required_fields - set(fieldnames)
    if missing_fields:
        raise ValueError(f"Missing benchmark columns: {sorted(missing_fields)}")

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


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("No matched rows available.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
    ax.set_title(f"{SYSTEM_NAME} Magnetization Parity")
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
    return {"n": len(rows), "mae": mae, "rmse": rmse}


def write_readme(run_dir: Path, metadata: dict[str, object]) -> None:
    readme = run_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# {SYSTEM_NAME} Magnetization Volume Prediction",
                "",
                f"Created: {metadata['created_at_utc']}",
                "",
                "## Purpose",
                "",
                "Run the saved top-bin CGCNN magnetization regressor on every supplied CIF structure, then compare the predictions against the benchmark magnetization-per-volume target.",
                "",
                "## Inputs",
                "",
                f"- Source folder: `{metadata['source_dir']}`",
                f"- Benchmark CSV updated in place: `{metadata['benchmark_csv']}`",
                f"- Top-bin regressor checkpoint symlink: `{metadata['inputs']['top_bin_regressor_checkpoint']}`",
                f"- All-CIF symlink dataset: `{metadata['inputs']['dataset']}`",
                f"- Script snapshot: `{metadata['inputs']['script_snapshot']}`",
                f"- Batch size: `{metadata['parameters']['batch_size']}`",
                f"- Workers: `{metadata['parameters']['workers']}`",
                "",
                "All CIFs and the checkpoint are symlinked into this run folder; they are not copied.",
                "",
                "## Outputs",
                "",
                f"- Raw all-structure regressor predictions: `{metadata['outputs']['raw_predictions']}`",
                f"- Benchmark rows matched with CGCNN predictions: `{metadata['outputs']['matched_predictions']}`",
                f"- Parity plot: `{metadata['outputs']['parity_plot']}`",
                f"- Run metadata: `{metadata['outputs']['metadata']}`",
                "",
                "The benchmark CSV now includes `M = d_vol + p_vol + s_vol`, `CGCNN_M`, and `CGCNN_error = CGCNN_M - M`.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    if not BENCHMARK_CSV.is_file():
        raise FileNotFoundError(BENCHMARK_CSV)
    if not TOP_BIN_REGRESSOR_CHECKPOINT.is_file():
        raise FileNotFoundError(TOP_BIN_REGRESSOR_CHECKPOINT)

    source_cifs = collect_cifs(SOURCE_DIR)
    run_dir = make_run_dir()
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    linked_checkpoint = run_dir / "inputs" / "checkpoints" / TOP_BIN_REGRESSOR_CHECKPOINT.name
    symlink_force(TOP_BIN_REGRESSOR_CHECKPOINT, linked_checkpoint)
    dataset_dir = run_dir / "dataset_all"
    write_dataset(dataset_dir, source_cifs)

    raw_predictions = outputs_dir / "top_bin_regressor_predictions_all.csv"
    run_inference(
        dataset_dir=dataset_dir,
        model_path=linked_checkpoint,
        task="regression",
        output_csv=raw_predictions,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
    )

    prediction_by_id = read_predictions(raw_predictions)
    all_rows, matched_rows = update_benchmark_csv(BENCHMARK_CSV, prediction_by_id)
    matched_predictions = outputs_dir / "magnetization_volume_summary_with_cgcnn_all.csv"
    write_rows(matched_predictions, matched_rows)
    parity_plot = outputs_dir / "magnetization_volume_parity_all.png"
    metrics = write_parity_plot(parity_plot, matched_rows)

    script_snapshot = run_dir / "inputs" / "scripts" / Path(__file__).name
    script_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), script_snapshot)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(SOURCE_DIR),
        "benchmark_csv": str(BENCHMARK_CSV),
        "parameters": {"batch_size": BATCH_SIZE, "workers": WORKERS, "cuda": False},
        "counts": {
            "input_cifs": len(source_cifs),
            "benchmark_rows": len(all_rows),
            "matched_predictions": len(matched_rows),
            "missing_predictions": len(all_rows) - len(matched_rows),
        },
        "metrics": metrics,
        "inputs": {
            "top_bin_regressor_checkpoint": str(linked_checkpoint),
            "dataset": str(dataset_dir),
            "script_snapshot": str(script_snapshot),
        },
        "outputs": {
            "raw_predictions": str(raw_predictions),
            "matched_predictions": str(matched_predictions),
            "parity_plot": str(parity_plot),
            "metadata": str(run_dir / "run_metadata.json"),
        },
    }
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    write_readme(run_dir, metadata)

    print(f"run_dir={run_dir}")
    print(f"updated_csv={BENCHMARK_CSV}")
    print(f"raw_predictions={raw_predictions}")
    print(f"matched_predictions={matched_predictions}")
    print(f"parity_plot={parity_plot}")
    print(
        "rows={benchmark_rows} matched={matched_predictions} missing={missing_predictions}".format(
            **metadata["counts"]
        )
    )
    print(f"mae={metrics['mae']:.16g} rmse={metrics['rmse']:.16g}")
