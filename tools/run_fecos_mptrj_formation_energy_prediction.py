from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cgcnn.data import CIFData
from cgcnn.inference import predict_model


POS_RE = re.compile(r"^pos_(\d+)$")
FINAL_RE = re.compile(r"^final_(\d+)\.cif$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MPTrj formation-energy CGCNN checkpoint on FeCoS CIFs."
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--benchmark-csv", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--atom-init", required=True, type=Path)
    parser.add_argument("--previous-run-dir", required=True, type=Path)
    parser.add_argument("--scratch-output-dir", required=True, type=Path)
    parser.add_argument("--work-output-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--expected-count", default=1314, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--initial-target-mode",
        choices=["benchmark-predicted", "constant"],
        default="benchmark-predicted",
        help="How to populate id_prop.csv target values.",
    )
    parser.add_argument("--constant-initial-value", default=0.0, type=float)
    parser.add_argument(
        "--mptrj-baseline-run-dir",
        type=Path,
        help="Optional prior MPTrj run folder for initial-value sensitivity comparison.",
    )
    return parser.parse_args()


def numeric_pos_key(structure_id: str) -> int:
    match = POS_RE.match(structure_id)
    if not match:
        raise ValueError(f"Structure ID is not a pos_N folder name: {structure_id}")
    return int(match.group(1))


def read_csv_dict(path: Path, key_column: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or key_column not in reader.fieldnames:
            raise ValueError(f"{path} must contain a {key_column!r} column")
        rows = {}
        for row in reader:
            key = row[key_column]
            if key in rows:
                raise ValueError(f"Duplicate {key_column}={key!r} in {path}")
            rows[key] = row
    return list(reader.fieldnames), rows


def discover_cifs(source_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for pos_dir in source_root.iterdir():
        if not pos_dir.is_dir() or not POS_RE.match(pos_dir.name):
            continue
        cifs = sorted(pos_dir.glob("final_*.cif"))
        if len(cifs) != 1:
            raise ValueError(f"Expected exactly one final_*.cif in {pos_dir}, found {len(cifs)}")
        final_match = FINAL_RE.match(cifs[0].name)
        pos_index = numeric_pos_key(pos_dir.name)
        if final_match is None or int(final_match.group(1)) != pos_index:
            raise ValueError(f"Folder/file mismatch: {pos_dir.name} contains {cifs[0].name}")
        records.append(
            {
                "structure_id": pos_dir.name,
                "material_index": str(pos_index),
                "source_pos_dir": str(pos_dir),
                "source_cif_path": str(cifs[0]),
                "source_cif_filename": cifs[0].name,
            }
        )
    return sorted(records, key=lambda row: int(row["material_index"]))


def force_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink():
        if os.readlink(link_path) == str(target_path):
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Refusing to overwrite non-symlink path: {link_path}")
    link_path.symlink_to(target_path)


def prepare_dataset(
    *,
    output_dir: Path,
    source_root: Path,
    benchmark_csv: Path,
    atom_init: Path,
    expected_count: int,
    initial_target_mode: str,
    constant_initial_value: float,
) -> dict[str, object]:
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    links_dir = output_dir / "links"
    links_dir.mkdir(parents=True, exist_ok=True)

    force_symlink(dataset_dir / "atom_init.json", atom_init)
    force_symlink(links_dir / "source_cif_root", source_root)
    force_symlink(links_dir / "benchmark_csv", benchmark_csv)
    force_symlink(links_dir / "atom_init.json", atom_init)

    benchmark_fields, benchmark_rows = read_csv_dict(benchmark_csv, "structure_id")
    required = {"formation_energy", "predicted_formation_energy_per_atom"}
    missing_columns = required.difference(benchmark_fields)
    if missing_columns:
        raise ValueError(f"Benchmark CSV is missing columns: {sorted(missing_columns)}")

    records = discover_cifs(source_root)
    if len(records) != expected_count:
        raise ValueError(f"Discovered {len(records)} CIFs, expected {expected_count}")

    missing_benchmark = [row["structure_id"] for row in records if row["structure_id"] not in benchmark_rows]
    if missing_benchmark:
        raise ValueError(f"{len(missing_benchmark)} CIF IDs are absent from benchmark CSV")

    id_prop_path = dataset_dir / "id_prop.csv"
    manifest_path = output_dir / "source_manifest.csv"
    with id_prop_path.open("w", newline="") as id_handle, manifest_path.open("w", newline="") as manifest_handle:
        id_writer = csv.writer(id_handle)
        manifest_fields = [
            "structure_id",
            "material_index",
            "source_pos_dir",
            "source_cif_path",
            "source_cif_filename",
            "linked_cif_path",
            "benchmark_formation_energy",
            "id_prop_initial_value",
            "id_prop_initial_value_source",
        ]
        manifest_writer = csv.DictWriter(manifest_handle, fieldnames=manifest_fields)
        manifest_writer.writeheader()
        for record in records:
            structure_id = record["structure_id"]
            benchmark = benchmark_rows[structure_id]
            if initial_target_mode == "benchmark-predicted":
                initial_value = benchmark["predicted_formation_energy_per_atom"]
                initial_source = "benchmark.predicted_formation_energy_per_atom"
            elif initial_target_mode == "constant":
                initial_value = f"{constant_initial_value:.17g}"
                initial_source = f"constant:{constant_initial_value:.17g}"
            else:
                raise ValueError(f"Unsupported initial target mode: {initial_target_mode}")
            link_path = dataset_dir / f"{structure_id}.cif"
            force_symlink(link_path, Path(record["source_cif_path"]))
            id_writer.writerow([structure_id, initial_value])
            manifest_writer.writerow(
                {
                    **record,
                    "linked_cif_path": str(link_path),
                    "benchmark_formation_energy": benchmark["formation_energy"],
                    "id_prop_initial_value": initial_value,
                    "id_prop_initial_value_source": initial_source,
                }
            )

    return {
        "dataset_dir": str(dataset_dir),
        "id_prop_csv": str(id_prop_path),
        "source_manifest_csv": str(manifest_path),
        "structure_count": len(records),
        "id_prop_initial_target_mode": initial_target_mode,
        "constant_initial_value": constant_initial_value if initial_target_mode == "constant" else None,
    }


def load_prediction_csv(path: Path, prediction_column: str) -> dict[str, float]:
    _, rows = read_csv_dict(path, "structure_id")
    return {structure_id: float(row[prediction_column]) for structure_id, row in rows.items()}


def load_raw_cgcnn_predictions(path: Path) -> dict[str, tuple[float, float]]:
    predictions: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(f"Expected three columns in raw CGCNN output row: {row}")
            predictions[row[0]] = (float(row[1]), float(row[2]))
    return predictions


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    diff = predicted - actual
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    pearson_r = float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 else math.nan
    return {
        "count": int(len(actual)),
        "pearson_r": pearson_r,
        "pearson_r_squared": float(pearson_r**2),
        "r2_score": float(1.0 - ss_res / ss_tot) if ss_tot else math.nan,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "target_mean": float(np.mean(actual)),
        "prediction_mean": float(np.mean(predicted)),
        "prediction_min": float(np.min(predicted)),
        "prediction_max": float(np.max(predicted)),
    }


def write_joined_outputs(
    *,
    output_dir: Path,
    benchmark_csv: Path,
    previous_run_dir: Path,
    raw_prediction_csv: Path,
) -> dict[str, object]:
    _, benchmark_rows = read_csv_dict(benchmark_csv, "structure_id")
    current_predictions = load_raw_cgcnn_predictions(raw_prediction_csv)
    previous_new = load_prediction_csv(
        previous_run_dir / "full_run" / "merged" / "predictions.csv",
        "predicted_formation_energy_per_atom",
    )
    previous_old = load_prediction_csv(
        previous_run_dir / "old_model_full_run" / "merged" / "predictions.csv",
        "predicted_formation_energy_per_atom",
    )

    structure_ids = sorted(current_predictions, key=numeric_pos_key)
    missing = {
        "benchmark": [sid for sid in structure_ids if sid not in benchmark_rows],
        "previous_20260421_model": [sid for sid in structure_ids if sid not in previous_new],
        "original_cgcnn_paper_model": [sid for sid in structure_ids if sid not in previous_old],
    }
    missing = {name: ids for name, ids in missing.items() if ids}
    if missing:
        raise ValueError(f"Missing comparison rows: {missing}")

    joined_csv = output_dir / "formation_energy_prediction_comparison.csv"
    fields = [
        "structure_id",
        "benchmark_formation_energy",
        "id_prop_initial_value",
        "benchmark_predicted_formation_energy_per_atom",
        "mptrj_20260705_predicted_formation_energy_per_atom",
        "previous_20260421_predicted_formation_energy_per_atom",
        "original_cgcnn_paper_predicted_formation_energy_per_atom",
        "mptrj_20260705_error",
        "previous_20260421_error",
        "original_cgcnn_paper_error",
        "raw_cgcnn_target_value",
    ]
    with joined_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for structure_id in structure_ids:
            benchmark = benchmark_rows[structure_id]
            actual = float(benchmark["formation_energy"])
            raw_target, current_pred = current_predictions[structure_id]
            prev_new_pred = previous_new[structure_id]
            prev_old_pred = previous_old[structure_id]
            writer.writerow(
                {
                    "structure_id": structure_id,
                    "benchmark_formation_energy": actual,
                    "id_prop_initial_value": raw_target,
                    "benchmark_predicted_formation_energy_per_atom": benchmark["predicted_formation_energy_per_atom"],
                    "mptrj_20260705_predicted_formation_energy_per_atom": current_pred,
                    "previous_20260421_predicted_formation_energy_per_atom": prev_new_pred,
                    "original_cgcnn_paper_predicted_formation_energy_per_atom": prev_old_pred,
                    "mptrj_20260705_error": current_pred - actual,
                    "previous_20260421_error": prev_new_pred - actual,
                    "original_cgcnn_paper_error": prev_old_pred - actual,
                    "raw_cgcnn_target_value": raw_target,
                }
            )

    actual = np.array([float(benchmark_rows[sid]["formation_energy"]) for sid in structure_ids])
    model_arrays = {
        "mptrj_20260705_best_validation": np.array([current_predictions[sid][1] for sid in structure_ids]),
        "previous_20260421_mp_all_model": np.array([previous_new[sid] for sid in structure_ids]),
        "original_cgcnn_paper_model": np.array([previous_old[sid] for sid in structure_ids]),
    }
    metrics = {
        "benchmark_column": "formation_energy",
        "models": {name: compute_metrics(actual, values) for name, values in model_arrays.items()},
        "joined_csv": str(joined_csv),
    }
    metrics_json = output_dir / "model_comparison_metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    metrics_csv = output_dir / "model_comparison_metrics.csv"
    with metrics_csv.open("w", newline="") as handle:
        fieldnames = ["model"] + list(next(iter(metrics["models"].values())).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, values in metrics["models"].items():
            writer.writerow({"model": model_name, **values})

    return {
        "joined_csv": str(joined_csv),
        "metrics_json": str(metrics_json),
        "metrics_csv": str(metrics_csv),
        "metrics": metrics,
    }


def write_initial_value_sensitivity(
    *,
    output_dir: Path,
    raw_prediction_csv: Path,
    baseline_run_dir: Path | None,
) -> dict[str, object] | None:
    if baseline_run_dir is None:
        return None
    baseline_csv = baseline_run_dir / "formation_energy_prediction_comparison.csv"
    if not baseline_csv.is_file():
        raise FileNotFoundError(f"Baseline comparison CSV does not exist: {baseline_csv}")

    current_predictions = load_raw_cgcnn_predictions(raw_prediction_csv)
    _, baseline_rows = read_csv_dict(baseline_csv, "structure_id")
    structure_ids = sorted(current_predictions, key=numeric_pos_key)
    missing = [structure_id for structure_id in structure_ids if structure_id not in baseline_rows]
    if missing:
        raise ValueError(f"Baseline run is missing {len(missing)} structure IDs")

    output_csv = output_dir / "initial_value_sensitivity_vs_previous_mptrj_run.csv"
    diffs = []
    with output_csv.open("w", newline="") as handle:
        fields = [
            "structure_id",
            "current_id_prop_initial_value",
            "current_mptrj_prediction",
            "baseline_mptrj_prediction",
            "prediction_difference_current_minus_baseline",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for structure_id in structure_ids:
            current_target, current_prediction = current_predictions[structure_id]
            baseline_prediction = float(
                baseline_rows[structure_id]["mptrj_20260705_predicted_formation_energy_per_atom"]
            )
            diff = current_prediction - baseline_prediction
            diffs.append(diff)
            writer.writerow(
                {
                    "structure_id": structure_id,
                    "current_id_prop_initial_value": current_target,
                    "current_mptrj_prediction": current_prediction,
                    "baseline_mptrj_prediction": baseline_prediction,
                    "prediction_difference_current_minus_baseline": diff,
                }
            )

    diff_array = np.array(diffs)
    summary = {
        "baseline_run_dir": str(baseline_run_dir),
        "baseline_csv": str(baseline_csv),
        "sensitivity_csv": str(output_csv),
        "count": int(len(diff_array)),
        "max_abs_prediction_difference": float(np.max(np.abs(diff_array))),
        "mean_abs_prediction_difference": float(np.mean(np.abs(diff_array))),
        "rmse_prediction_difference": float(np.sqrt(np.mean(diff_array**2))),
        "all_predictions_exactly_equal": bool(np.all(diff_array == 0.0)),
        "all_predictions_close_at_1e_12": bool(np.allclose(diff_array, 0.0, atol=1e-12, rtol=0.0)),
    }
    output_json = output_dir / "initial_value_sensitivity_summary.json"
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    summary["sensitivity_json"] = str(output_json)
    return summary


def plot_parity(output_dir: Path, joined_csv: Path, metrics: dict[str, object]) -> dict[str, str]:
    rows = []
    with joined_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    actual = np.array([float(row["benchmark_formation_energy"]) for row in rows])
    series = [
        (
            "MPTrj 20260705",
            np.array([float(row["mptrj_20260705_predicted_formation_energy_per_atom"]) for row in rows]),
            "#1167b1",
            "o",
        ),
        (
            "20260421 MP-all",
            np.array([float(row["previous_20260421_predicted_formation_energy_per_atom"]) for row in rows]),
            "#c44536",
            "^",
        ),
        (
            "Original CGCNN paper",
            np.array([float(row["original_cgcnn_paper_predicted_formation_energy_per_atom"]) for row in rows]),
            "#248f24",
            "s",
        ),
    ]

    all_values = np.concatenate([actual] + [values for _, values, _, _ in series])
    lo = float(np.floor((np.min(all_values) - 0.05) * 10) / 10)
    hi = float(np.ceil((np.max(all_values) + 0.05) * 10) / 10)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=180)
    for label, values, color, marker in series:
        metric_key = {
            "MPTrj 20260705": "mptrj_20260705_best_validation",
            "20260421 MP-all": "previous_20260421_mp_all_model",
            "Original CGCNN paper": "original_cgcnn_paper_model",
        }[label]
        model_metrics = metrics["models"][metric_key]
        legend_label = (
            f"{label}: MAE={model_metrics['mae']:.3f}, "
            f"r^2={model_metrics['pearson_r_squared']:.3f}"
        )
        ax.scatter(
            actual,
            values,
            s=18,
            alpha=0.68,
            linewidths=0.25,
            edgecolors="white",
            c=color,
            marker=marker,
            label=legend_label,
        )
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.1, linestyle="--", label="Parity")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Benchmark formation energy per atom (eV/atom)")
    ax.set_ylabel("CGCNN predicted formation energy per atom (eV/atom)")
    ax.set_title("FeCoS formation-energy parity")
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    png_path = figures_dir / "formation_energy_parity_mptrj_vs_previous.png"
    pdf_path = figures_dir / "formation_energy_parity_mptrj_vs_previous.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def git_revision(repo_dir: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def write_readme(output_dir: Path, summary: dict[str, object]) -> None:
    metrics = summary["comparison"]["metrics"]["models"]
    sensitivity = summary.get("initial_value_sensitivity")
    if sensitivity is None:
        sensitivity_text = "No prior MPTrj run was provided for an initial-value sensitivity comparison."
    else:
        sensitivity_text = (
            f"- Baseline MPTrj run: `{sensitivity['baseline_run_dir']}`\n"
            f"- Sensitivity CSV: `initial_value_sensitivity_vs_previous_mptrj_run.csv`\n"
            f"- Max absolute prediction difference: `{sensitivity['max_abs_prediction_difference']}`\n"
            f"- Mean absolute prediction difference: `{sensitivity['mean_abs_prediction_difference']}`\n"
            f"- All predictions exactly equal: `{sensitivity['all_predictions_exactly_equal']}`\n"
            f"- All predictions close at 1e-12: `{sensitivity['all_predictions_close_at_1e_12']}`"
        )
    readme = f"""# FeCoS CGCNN Formation-Energy Prediction With MPTrj Checkpoint

## Purpose

This run predicts formation energies for the 1314 FeCoS materials whose source CIFs are stored as `final_*.cif` files under `pos_*` folders. It uses the MPTrj formation-energy-per-atom CGCNN checkpoint requested by the user, then compares the new predictions against the benchmark `formation_energy` column and two earlier CGCNN model result sets from the April 2026 FeCoS run.

## Creation

- Created at: `{summary["created_at_utc"]}` UTC
- Host: `{summary["host"]}`
- Scratch run folder: `{summary["scratch_output_dir"]}`
- Work copy-back folder: `{summary["work_output_dir"]}`
- Slurm job ID: `{summary.get("slurm_job_id")}`
- Slurm partition: `{summary.get("slurm_partition")}`

## Inputs

- Source CIF root, read in place and symlinked: `{summary["inputs"]["source_root"]}`
- Benchmark CSV, read in place: `{summary["inputs"]["benchmark_csv"]}`
- Benchmark target column: `formation_energy`
- Initial `id_prop.csv` target mode: `{summary["preparation"]["id_prop_initial_target_mode"]}`
- Constant initial value: `{summary["preparation"]["constant_initial_value"]}`
- MPTrj checkpoint, read in place: `{summary["inputs"]["model_path"]}`
- Atom feature JSON, symlinked into the dataset: `{summary["inputs"]["atom_init"]}`
- Previous model comparison folder, read in place: `{summary["inputs"]["previous_run_dir"]}`
- CGCNN repository used in place: `{summary["inputs"]["repo_dir"]}`
- Git commit at run time: `{summary["inputs"]["git_revision"]}`
- Command-line options: `{summary["command_line"]}`

## Staged Dataset

- Dataset folder: `dataset/`
- CIF links: `dataset/pos_N.cif` -> source `pos_N/final_N.cif`
- Target table: `dataset/id_prop.csv`
- Manifest: `source_manifest.csv`

Each `id_prop.csv` row uses the folder/material ID from the benchmark CSV, for example `pos_1`, and the linked CIF file has the matching `pos_1.cif` name required by `CIFData`.

## Initial-Value Sensitivity

{sensitivity_text}

## Outputs

- Raw CGCNN output: `predictions/raw_mptrj_predictions.csv`
- Joined comparison table: `formation_energy_prediction_comparison.csv`
- Metrics JSON: `model_comparison_metrics.json`
- Metrics CSV: `model_comparison_metrics.csv`
- Initial-value sensitivity summary: `initial_value_sensitivity_summary.json`, when a baseline run is provided
- Parity plot PNG: `figures/formation_energy_parity_mptrj_vs_previous.png`
- Parity plot PDF: `figures/formation_energy_parity_mptrj_vs_previous.pdf`
- Run summary: `run_summary.json`
- Slurm logs: `logs/`

## Metrics

| Model | Count | MAE | RMSE | Bias | Pearson r^2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPTrj 20260705 best validation | {metrics["mptrj_20260705_best_validation"]["count"]} | {metrics["mptrj_20260705_best_validation"]["mae"]:.6f} | {metrics["mptrj_20260705_best_validation"]["rmse"]:.6f} | {metrics["mptrj_20260705_best_validation"]["bias"]:.6f} | {metrics["mptrj_20260705_best_validation"]["pearson_r_squared"]:.6f} |
| Previous 20260421 MP-all model | {metrics["previous_20260421_mp_all_model"]["count"]} | {metrics["previous_20260421_mp_all_model"]["mae"]:.6f} | {metrics["previous_20260421_mp_all_model"]["rmse"]:.6f} | {metrics["previous_20260421_mp_all_model"]["bias"]:.6f} | {metrics["previous_20260421_mp_all_model"]["pearson_r_squared"]:.6f} |
| Original CGCNN paper model | {metrics["original_cgcnn_paper_model"]["count"]} | {metrics["original_cgcnn_paper_model"]["mae"]:.6f} | {metrics["original_cgcnn_paper_model"]["rmse"]:.6f} | {metrics["original_cgcnn_paper_model"]["bias"]:.6f} | {metrics["original_cgcnn_paper_model"]["pearson_r_squared"]:.6f} |
"""
    (output_dir / "README.md").write_text(readme)


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = args.scratch_output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse existing scratch output dir: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    args.source_root = args.source_root.resolve()
    args.benchmark_csv = args.benchmark_csv.resolve()
    args.model_path = args.model_path.resolve()
    args.atom_init = args.atom_init.resolve()
    args.previous_run_dir = args.previous_run_dir.resolve()
    args.repo_dir = args.repo_dir.resolve()
    if args.mptrj_baseline_run_dir is not None:
        args.mptrj_baseline_run_dir = args.mptrj_baseline_run_dir.resolve()

    links_dir = output_dir / "links"
    links_dir.mkdir(exist_ok=True)
    force_symlink(links_dir / "model_best_validation.pth.tar", args.model_path)
    force_symlink(links_dir / "previous_fecos_run", args.previous_run_dir)
    force_symlink(links_dir / "cgcnn_repo", args.repo_dir)
    if args.mptrj_baseline_run_dir is not None:
        force_symlink(links_dir / "mptrj_baseline_run", args.mptrj_baseline_run_dir)

    preparation = prepare_dataset(
        output_dir=output_dir,
        source_root=args.source_root,
        benchmark_csv=args.benchmark_csv,
        atom_init=args.atom_init,
        expected_count=args.expected_count,
        initial_target_mode=args.initial_target_mode,
        constant_initial_value=args.constant_initial_value,
    )

    raw_prediction_csv = predictions_dir / "raw_mptrj_predictions.csv"
    dataset = CIFData(preparation["dataset_dir"], shuffle=False)
    predict_model(
        dataset=dataset,
        task="regression",
        modelpath=str(args.model_path),
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        print_freq=1,
        shuffle=False,
        output_csv=str(raw_prediction_csv),
    )

    comparison = write_joined_outputs(
        output_dir=output_dir,
        benchmark_csv=args.benchmark_csv,
        previous_run_dir=args.previous_run_dir,
        raw_prediction_csv=raw_prediction_csv,
    )
    figures = plot_parity(output_dir, Path(comparison["joined_csv"]), comparison["metrics"])
    initial_value_sensitivity = write_initial_value_sensitivity(
        output_dir=output_dir,
        raw_prediction_csv=raw_prediction_csv,
        baseline_run_dir=args.mptrj_baseline_run_dir,
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start,
        "host": platform.node(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "scratch_output_dir": str(output_dir),
        "work_output_dir": str(args.work_output_dir.resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "inputs": {
            "source_root": str(args.source_root),
            "benchmark_csv": str(args.benchmark_csv),
            "model_path": str(args.model_path),
            "atom_init": str(args.atom_init),
            "previous_run_dir": str(args.previous_run_dir),
            "mptrj_baseline_run_dir": (
                str(args.mptrj_baseline_run_dir) if args.mptrj_baseline_run_dir is not None else None
            ),
            "repo_dir": str(args.repo_dir),
            "git_revision": git_revision(args.repo_dir),
        },
        "command_line": " ".join(sys.argv),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "preparation": preparation,
        "raw_prediction_csv": str(raw_prediction_csv),
        "comparison": comparison,
        "figures": figures,
        "initial_value_sensitivity": initial_value_sensitivity,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(output_dir, summary)

    work_output_dir = args.work_output_dir.resolve()
    work_output_dir.parent.mkdir(parents=True, exist_ok=True)
    if work_output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing work output dir: {work_output_dir}")
    shutil.copytree(output_dir, work_output_dir, symlinks=True)
    print(json.dumps({"scratch_output_dir": str(output_dir), "work_output_dir": str(work_output_dir)}, indent=2))


if __name__ == "__main__":
    main()
