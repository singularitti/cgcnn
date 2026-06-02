from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_magnetization_multiclass_top_bin import run_inference

DEFAULT_SOURCE_DIR = Path("/Users/Shared/Remote.noindex/Work/run/XMnBi_Kharel/cgcnn_predict")
DEFAULT_BENCHMARK_CSV = Path("/Users/Shared/Remote.noindex/Work/run/XMnBi_Kharel/formation_energy2.csv")
DEFAULT_NEW_MODEL = Path(
    "/Users/Shared/Remote.noindex/Work/run/MnBiO/predict/my new/my_formation_energy_mp_all.pth.tar"
)
DEFAULT_OLD_MODEL = Path(
    "/Users/Shared/Remote.noindex/Work/run/MnBiO/predict/xie old/xie-formation-energy-per-atom.pth.tar"
)
DEFAULT_ATOM_INIT = REPO_ROOT / "data" / "sample-regression" / "atom_init.json"
BENCHMARK_TARGET_COLUMN = "energy per atom (eV)"
ID_PATTERN = re.compile(r"/((?:mp-\d+)|(?:MnBi[^/]+))/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare new and old formation-energy CGCNN checkpoints on XMnBi Kharel CIFs."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--new-model", type=Path, default=DEFAULT_NEW_MODEL)
    parser.add_argument("--old-model", type=Path, default=DEFAULT_OLD_MODEL)
    parser.add_argument("--atom-init", type=Path, default=DEFAULT_ATOM_INIT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def make_run_dir(source_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = source_dir / f"xmnbi_kharel_formation_energy_compare_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def symlink_checked(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source)


def collect_cifs(source_dir: Path) -> list[Path]:
    cifs = sorted(path for path in source_dir.glob("*.cif") if path.is_file())
    if not cifs:
        raise RuntimeError(f"No CIF files found in {source_dir}")
    return cifs


def benchmark_id(abs_path: str) -> str:
    match = ID_PATTERN.search(abs_path)
    if match is None:
        raise ValueError(f"Could not extract mp-* or MnBi* ID from abs_path: {abs_path}")
    return match.group(1)


def read_benchmark_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    required = {"abs_path", BENCHMARK_TARGET_COLUMN}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")
    return rows, fieldnames


def write_dataset(dataset_dir: Path, cifs: list[Path], atom_init: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=False)
    symlink_checked(atom_init, dataset_dir / "atom_init.json")
    with (dataset_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for cif_path in cifs:
            material_id = cif_path.stem
            symlink_checked(cif_path, dataset_dir / f"{material_id}.cif")
            writer.writerow([material_id, "0.0"])


def load_regression_predictions(path: Path) -> dict[str, float]:
    predictions: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0] == "material_id":
                continue
            predictions[row[0]] = float(row[2])
    return predictions


def add_regression_header(path: Path, prediction_column: str) -> None:
    header = ["material_id", "dummy_target", prediction_column]
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if rows and rows[0] == header:
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run_model_inference(
    *,
    label: str,
    model_path: Path,
    run_dir: Path,
    source_cifs: list[Path],
    atom_init: Path,
    batch_size: int,
    workers: int,
) -> tuple[Path, dict[str, float]]:
    dataset_dir = run_dir / f"{label}_dataset_all"
    write_dataset(dataset_dir, source_cifs, atom_init)
    output_csv = run_dir / "outputs" / f"{label}_predictions.csv"
    run_inference(
        dataset_dir=dataset_dir,
        model_path=model_path,
        task="regression",
        output_csv=output_csv,
        batch_size=batch_size,
        workers=workers,
    )
    predictions = load_regression_predictions(output_csv)
    add_regression_header(output_csv, f"{label}_predicted_formation_energy_per_atom")
    return output_csv, predictions


def format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.16g}"


def metric_summary(targets: list[float], predictions: list[float]) -> dict[str, float | int | None]:
    if not targets:
        return {"n": 0, "mae": None, "rmse": None, "mean_error": None}
    target = np.array(targets, dtype=float)
    predicted = np.array(predictions, dtype=float)
    error = predicted - target
    return {
        "n": int(len(target)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_error": float(np.mean(error)),
    }


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parity_plot(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    targets = np.array([float(row["benchmark_energy_per_atom_eV"]) for row in rows], dtype=float)
    new_predictions = np.array(
        [float(row["new_predicted_formation_energy_per_atom"]) for row in rows],
        dtype=float,
    )
    old_predictions = np.array(
        [float(row["old_predicted_formation_energy_per_atom"]) for row in rows],
        dtype=float,
    )
    new_metrics = metric_summary(targets.tolist(), new_predictions.tolist())
    old_metrics = metric_summary(targets.tolist(), old_predictions.tolist())
    lo = float(min(targets.min(), new_predictions.min(), old_predictions.min()))
    hi = float(max(targets.max(), new_predictions.max(), old_predictions.max()))
    pad = (hi - lo) * 0.08 if hi > lo else 0.01

    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    ax.scatter(
        targets,
        new_predictions,
        s=44,
        color="#2563eb",
        edgecolor="white",
        linewidth=0.6,
        label=f"new model (MAE {new_metrics['mae']:.4g})",
    )
    ax.scatter(
        targets,
        old_predictions,
        s=44,
        color="#dc2626",
        edgecolor="white",
        linewidth=0.6,
        label=f"old Xie model (MAE {old_metrics['mae']:.4g})",
    )
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111827", linewidth=1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Benchmark energy per atom (eV)")
    ax.set_ylabel("CGCNN predicted value (eV/atom)")
    ax.set_title("XMnBi Kharel formation-energy checkpoint comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {"new_model": new_metrics, "old_model": old_metrics}


def summarize_checkpoint(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "best_validation_score": checkpoint.get("best_validation_score"),
        "args": args,
    }


def copy_script_snapshot(run_dir: Path) -> Path:
    snapshot = run_dir / "inputs" / "scripts" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), snapshot)
    return snapshot


def write_readme(run_dir: Path, metadata: dict[str, object]) -> None:
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                "# XMnBi Kharel Formation Energy Checkpoint Comparison",
                "",
                f"Created: {metadata['created_at_utc']}",
                "",
                "## Purpose",
                "",
                "Predict per-atom energy values for the XMnBi Kharel CIF set with two CGCNN formation-energy checkpoints, then compare both prediction sets against the benchmark `energy per atom (eV)` column in one parity plot.",
                "",
                "## Inputs",
                "",
                f"- Source CIF folder: `{metadata['source_dir']}`",
                f"- Benchmark CSV symlink: `{metadata['inputs']['benchmark_csv']}`",
                f"- New checkpoint symlink: `{metadata['inputs']['new_model']}`",
                f"- Old Xie checkpoint symlink: `{metadata['inputs']['old_model']}`",
                f"- Atom features symlink: `{metadata['inputs']['atom_init']}`",
                f"- New model dataset: `{metadata['inputs']['new_dataset']}`",
                f"- Old model dataset: `{metadata['inputs']['old_dataset']}`",
                f"- Script snapshot: `{metadata['inputs']['script_snapshot']}`",
                f"- Batch size: `{metadata['parameters']['batch_size']}`",
                f"- Workers: `{metadata['parameters']['workers']}`",
                "",
                "CIFs, checkpoints, and the benchmark CSV are symlinked into this run folder; they are not copied.",
                "",
                "## Outputs",
                "",
                f"- New model raw predictions: `{metadata['outputs']['new_predictions']}`",
                f"- Old model raw predictions: `{metadata['outputs']['old_predictions']}`",
                f"- Combined benchmark comparison table: `{metadata['outputs']['comparison_csv']}`",
                f"- Shared parity plot: `{metadata['outputs']['parity_plot']}`",
                f"- Run metadata: `{metadata['outputs']['metadata']}`",
                "",
                "The final table matches benchmark rows to CIFs by extracting `/mp-*/` or `/MnBi*/` from `abs_path`, ignoring surrounding path prefix and suffix.",
                "",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    benchmark_csv = args.benchmark_csv.expanduser().resolve()
    new_model = args.new_model.expanduser().resolve()
    old_model = args.old_model.expanduser().resolve()
    atom_init = args.atom_init.expanduser().resolve()
    for path in [source_dir, benchmark_csv, new_model, old_model, atom_init]:
        if not path.exists():
            raise FileNotFoundError(path)

    source_cifs = collect_cifs(source_dir)
    cif_by_id = {path.stem: path for path in source_cifs}
    benchmark_rows, benchmark_fieldnames = read_benchmark_rows(benchmark_csv)
    benchmark_ids = [benchmark_id(row["abs_path"]) for row in benchmark_rows]
    missing_cifs = sorted(set(benchmark_ids) - set(cif_by_id))
    extra_cifs = sorted(set(cif_by_id) - set(benchmark_ids))
    if missing_cifs:
        raise RuntimeError(f"Benchmark rows without matching CIFs: {missing_cifs}")

    run_dir = make_run_dir(source_dir)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    linked_benchmark = run_dir / "inputs" / "benchmark" / benchmark_csv.name
    linked_new_model = run_dir / "inputs" / "checkpoints" / new_model.name
    linked_old_model = run_dir / "inputs" / "checkpoints" / old_model.name
    symlink_checked(benchmark_csv, linked_benchmark)
    symlink_checked(new_model, linked_new_model)
    symlink_checked(old_model, linked_old_model)

    new_predictions_csv, new_predictions = run_model_inference(
        label="new",
        model_path=linked_new_model,
        run_dir=run_dir,
        source_cifs=source_cifs,
        atom_init=atom_init,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    old_predictions_csv, old_predictions = run_model_inference(
        label="old",
        model_path=linked_old_model,
        run_dir=run_dir,
        source_cifs=source_cifs,
        atom_init=atom_init,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    output_fields = [
        "material_id",
        "benchmark_abs_path",
        "source_cif",
        "benchmark_energy_per_atom_eV",
        "new_predicted_formation_energy_per_atom",
        "new_error_vs_benchmark_energy_per_atom",
        "old_predicted_formation_energy_per_atom",
        "old_error_vs_benchmark_energy_per_atom",
    ]
    comparison_rows: list[dict[str, object]] = []
    for benchmark_row, material_id in zip(benchmark_rows, benchmark_ids, strict=True):
        target = float(benchmark_row[BENCHMARK_TARGET_COLUMN])
        new_prediction = new_predictions[material_id]
        old_prediction = old_predictions[material_id]
        comparison_rows.append(
            {
                "material_id": material_id,
                "benchmark_abs_path": benchmark_row["abs_path"],
                "source_cif": str(cif_by_id[material_id]),
                "benchmark_energy_per_atom_eV": format_float(target),
                "new_predicted_formation_energy_per_atom": format_float(new_prediction),
                "new_error_vs_benchmark_energy_per_atom": format_float(new_prediction - target),
                "old_predicted_formation_energy_per_atom": format_float(old_prediction),
                "old_error_vs_benchmark_energy_per_atom": format_float(old_prediction - target),
            }
        )

    comparison_csv = outputs_dir / "formation_energy_model_comparison.csv"
    write_rows(comparison_csv, comparison_rows, output_fields)
    parity_plot = outputs_dir / "formation_energy_parity_comparison.png"
    metrics = write_parity_plot(parity_plot, comparison_rows)

    script_snapshot = copy_script_snapshot(run_dir)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "benchmark_source_csv": str(benchmark_csv),
        "benchmark_target_column": BENCHMARK_TARGET_COLUMN,
        "benchmark_id_rule": "Extract /mp-*/ or /MnBi*/ from abs_path and compare to CIF stem.",
        "benchmark_original_columns": benchmark_fieldnames,
        "parameters": {
            "batch_size": args.batch_size,
            "workers": args.workers,
            "cuda": False,
        },
        "counts": {
            "input_cifs": len(source_cifs),
            "benchmark_rows": len(benchmark_rows),
            "extra_cifs_without_benchmark_rows": extra_cifs,
        },
        "metrics": metrics,
        "inputs": {
            "benchmark_csv": str(linked_benchmark),
            "new_model": str(linked_new_model),
            "old_model": str(linked_old_model),
            "atom_init": str(atom_init),
            "new_dataset": str(run_dir / "new_dataset_all"),
            "old_dataset": str(run_dir / "old_dataset_all"),
            "script_snapshot": str(script_snapshot),
            "new_model_checkpoint_summary": summarize_checkpoint(linked_new_model),
            "old_model_checkpoint_summary": summarize_checkpoint(linked_old_model),
        },
        "outputs": {
            "new_predictions": str(new_predictions_csv),
            "old_predictions": str(old_predictions_csv),
            "comparison_csv": str(comparison_csv),
            "parity_plot": str(parity_plot),
            "metadata": str(run_dir / "run_metadata.json"),
        },
    }
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    write_readme(run_dir, metadata)
    print(run_dir)


if __name__ == "__main__":
    main()
