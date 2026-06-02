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

from train_magnetization_multiclass_top_bin import (
    CLASS_LARGE,
    CLASS_NAMES,
    NUM_CLASSES,
    load_classification_predictions,
    load_regression_predictions,
    run_inference,
    write_atom_init,
)

DEFAULT_SOURCE_DIR = Path("~/Downloads/Work/run/XMnBi_Kharel/cgcnn_predict").expanduser()
DEFAULT_BENCHMARK_CSV = Path(
    "~/Downloads/Work/run/XMnBi_Kharel/formation_energy2.csv"
).expanduser()
DEFAULT_CLASSIFIER_CHECKPOINT = Path(
    "~/Downloads/Work/run/MnBiO/predict/magnetization/classifier_model_best.pth.tar"
).expanduser()
DEFAULT_REGRESSOR_CHECKPOINT = Path(
    "~/Downloads/Work/run/MnBiO/predict/magnetization/epoch_029.pth.tar"
).expanduser()
DEFAULT_BATCH_SIZE = 256
DEFAULT_WORKERS = 0
BENCHMARK_TARGET_COLUMN = "magnetization M (mu_B / angstrom^3)"
ID_PATTERN = re.compile(r"/((?:mp-\d+)|(?:MnBi[^/]+))/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict magnetization per volume for the XMnBi Kharel CIF benchmark."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--regressor-checkpoint", type=Path, default=DEFAULT_REGRESSOR_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def symlink_force(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source)


def make_run_dir(source_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = source_dir / f"xmnbi_kharel_magnetization_combined_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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


def write_dataset(dataset_dir: Path, cifs: list[Path], target_value: str) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=False)
    write_atom_init(dataset_dir / "atom_init.json")
    with (dataset_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for cif_path in cifs:
            material_id = cif_path.stem
            symlink_force(cif_path, dataset_dir / f"{material_id}.cif")
            writer.writerow([material_id, target_value])


def format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.16g}"


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


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_classifier_header(path: Path) -> None:
    header = [
        "material_id",
        "target_class",
        "predicted_class",
        "class_0_zero_probability",
        "class_1_tiny_positive_probability",
        "class_2_small_positive_probability",
        "class_3_large_positive_probability",
    ]
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if rows and rows[0] == header:
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


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


def write_parity_plot(
    path: Path,
    rows: list[dict[str, object]],
    prediction_field: str,
    title: str,
) -> dict[str, float | int | None]:
    targets: list[float] = []
    predictions: list[float] = []
    for row in rows:
        prediction_text = str(row[prediction_field])
        if prediction_text:
            targets.append(float(row["benchmark_magnetization_per_volume"]))
            predictions.append(float(prediction_text))
    metrics = metric_summary(targets, predictions)
    if not targets:
        return metrics

    target = np.array(targets, dtype=float)
    predicted = np.array(predictions, dtype=float)
    lo = float(min(target.min(), predicted.min()))
    hi = float(max(target.max(), predicted.max()))
    pad = (hi - lo) * 0.08 if hi > lo else 0.01

    fig, ax = plt.subplots(figsize=(5.7, 5.1), constrained_layout=True)
    ax.scatter(target, predicted, s=38, color="#2563eb", edgecolor="white", linewidth=0.6)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111827", linewidth=1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Benchmark magnetization M (mu_B / angstrom^3)")
    ax.set_ylabel("CGCNN prediction (mu_B / angstrom^3)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.04,
        0.96,
        f"n = {metrics['n']}\nMAE = {metrics['mae']:.4g}\nRMSE = {metrics['rmse']:.4g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return metrics


def copy_script_snapshot(run_dir: Path) -> Path:
    snapshot = run_dir / "inputs" / "scripts" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), snapshot)
    return snapshot


def write_readme(run_dir: Path, metadata: dict[str, object]) -> None:
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                "# XMnBi Kharel Magnetization Prediction",
                "",
                f"Created: {metadata['created_at_utc']}",
                "",
                "## Purpose",
                "",
                "Predict magnetization per unit volume for the XMnBi Kharel CIF set using the saved MnBiO magnetization workflow: a four-class classifier routes structures predicted as class 3 / large_positive / M > 1e-3 to the top-bin regressor. The run also records the top-bin regressor applied to every CIF as a diagnostic, because that regressor was trained for class-3 structures only.",
                "",
                "## Inputs",
                "",
                f"- Source CIF folder: `{metadata['source_dir']}`",
                f"- Benchmark CSV symlink: `{metadata['inputs']['benchmark_csv']}`",
                f"- Classifier checkpoint symlink: `{metadata['inputs']['classifier_checkpoint']}`",
                f"- Top-bin regressor checkpoint symlink: `{metadata['inputs']['top_bin_regressor_checkpoint']}`",
                f"- All-CIF classifier dataset: `{metadata['inputs']['classifier_dataset']}`",
                f"- Selected class-3 regressor dataset: `{metadata['inputs']['selected_regressor_dataset']}`",
                f"- All-CIF diagnostic regressor dataset: `{metadata['inputs']['all_regressor_dataset']}`",
                f"- Script snapshot: `{metadata['inputs']['script_snapshot']}`",
                f"- Batch size: `{metadata['parameters']['batch_size']}`",
                f"- Workers: `{metadata['parameters']['workers']}`",
                "",
                "CIFs, checkpoints, and the benchmark CSV are symlinked into this run folder; they are not copied.",
                "",
                "## Outputs",
                "",
                f"- Classifier raw predictions: `{metadata['outputs']['classifier_predictions']}`",
                f"- Selected class-3 regressor raw predictions: `{metadata['outputs']['selected_regressor_predictions']}`",
                f"- All-CIF diagnostic regressor raw predictions: `{metadata['outputs']['all_regressor_predictions']}`",
                f"- Final benchmark comparison table: `{metadata['outputs']['benchmark_predictions']}`",
                f"- Routed parity plot: `{metadata['outputs']['routed_parity_plot']}`",
                f"- All-regressor diagnostic parity plot: `{metadata['outputs']['all_regressor_parity_plot']}`",
                f"- Run metadata: `{metadata['outputs']['metadata']}`",
                "",
                "The final table matches benchmark rows to CIFs by extracting `/mp-*/` or `/MnBi*/` from `abs_path`, ignoring the surrounding path prefix and suffix.",
                "",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    benchmark_csv = args.benchmark_csv.expanduser().resolve()
    classifier_checkpoint = args.classifier_checkpoint.expanduser().resolve()
    regressor_checkpoint = args.regressor_checkpoint.expanduser().resolve()

    for path in [source_dir, benchmark_csv, classifier_checkpoint, regressor_checkpoint]:
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
    inputs_dir = run_dir / "inputs"
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    linked_benchmark = inputs_dir / "benchmark" / benchmark_csv.name
    linked_classifier = inputs_dir / "checkpoints" / classifier_checkpoint.name
    linked_regressor = inputs_dir / "checkpoints" / regressor_checkpoint.name
    symlink_force(benchmark_csv, linked_benchmark)
    symlink_force(classifier_checkpoint, linked_classifier)
    symlink_force(regressor_checkpoint, linked_regressor)

    classifier_dataset = run_dir / "classifier_dataset_all"
    all_regressor_dataset = run_dir / "top_bin_regressor_dataset_all_diagnostic"
    write_dataset(classifier_dataset, source_cifs, target_value="0")
    write_dataset(all_regressor_dataset, source_cifs, target_value="0.0")

    classifier_predictions = outputs_dir / "classifier_predictions.csv"
    run_inference(
        dataset_dir=classifier_dataset,
        model_path=linked_classifier,
        task="classification",
        output_csv=classifier_predictions,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    classifier_rows = load_classification_predictions(classifier_predictions)
    add_classifier_header(classifier_predictions)

    selected_ids = sorted(
        material_id
        for material_id, row in classifier_rows.items()
        if int(row["predicted_class"]) == CLASS_LARGE
    )
    selected_cifs = [cif_by_id[material_id] for material_id in selected_ids]
    selected_regressor_dataset = run_dir / "top_bin_regressor_dataset_selected"
    selected_regressor_predictions = outputs_dir / "selected_top_bin_regressor_predictions.csv"
    selected_regressor_rows: dict[str, tuple[float, float]] = {}
    if selected_cifs:
        write_dataset(selected_regressor_dataset, selected_cifs, target_value="0.0")
        run_inference(
            dataset_dir=selected_regressor_dataset,
            model_path=linked_regressor,
            task="regression",
            output_csv=selected_regressor_predictions,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        selected_regressor_rows = load_regression_predictions(selected_regressor_predictions)
    else:
        selected_regressor_dataset.mkdir(parents=True, exist_ok=False)
        write_atom_init(selected_regressor_dataset / "atom_init.json")
        (selected_regressor_dataset / "id_prop.csv").write_text("")
        selected_regressor_predictions.write_text("")

    all_regressor_predictions = outputs_dir / "all_top_bin_regressor_predictions.csv"
    run_inference(
        dataset_dir=all_regressor_dataset,
        model_path=linked_regressor,
        task="regression",
        output_csv=all_regressor_predictions,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    all_regressor_rows = load_regression_predictions(all_regressor_predictions)

    class_probability_fields = [
        f"class_{label}_{CLASS_NAMES[label]}_probability" for label in range(NUM_CLASSES)
    ]
    output_fields = [
        "material_id",
        "benchmark_abs_path",
        "source_cif",
        "benchmark_magnetization_per_volume",
        "predicted_class",
        "predicted_class_name",
        "selected_for_top_bin_regressor",
        "routed_predicted_magnetization_per_volume",
        "routed_error",
        "all_regressor_predicted_magnetization_per_volume",
        "all_regressor_error",
    ] + class_probability_fields

    comparison_rows: list[dict[str, object]] = []
    for benchmark_row, material_id in zip(benchmark_rows, benchmark_ids, strict=True):
        target = float(benchmark_row[BENCHMARK_TARGET_COLUMN])
        classifier_row = classifier_rows[material_id]
        predicted_class = int(classifier_row["predicted_class"])
        probabilities = list(classifier_row["probabilities"])
        selected_prediction = selected_regressor_rows.get(material_id)
        routed_prediction = None if selected_prediction is None else selected_prediction[1]
        all_prediction = all_regressor_rows[material_id][1]
        row: dict[str, object] = {
            "material_id": material_id,
            "benchmark_abs_path": benchmark_row["abs_path"],
            "source_cif": str(cif_by_id[material_id]),
            "benchmark_magnetization_per_volume": format_float(target),
            "predicted_class": predicted_class,
            "predicted_class_name": CLASS_NAMES[predicted_class],
            "selected_for_top_bin_regressor": predicted_class == CLASS_LARGE,
            "routed_predicted_magnetization_per_volume": format_float(routed_prediction),
            "routed_error": format_float(
                None if routed_prediction is None else routed_prediction - target
            ),
            "all_regressor_predicted_magnetization_per_volume": format_float(all_prediction),
            "all_regressor_error": format_float(all_prediction - target),
        }
        for label, field in enumerate(class_probability_fields):
            row[field] = f"{probabilities[label]:.16g}"
        comparison_rows.append(row)

    benchmark_predictions = outputs_dir / "benchmark_predictions.csv"
    write_rows(benchmark_predictions, comparison_rows, output_fields)

    routed_parity_plot = outputs_dir / "routed_top_bin_parity.png"
    all_regressor_parity_plot = outputs_dir / "all_top_bin_regressor_diagnostic_parity.png"
    routed_metrics = write_parity_plot(
        routed_parity_plot,
        comparison_rows,
        "routed_predicted_magnetization_per_volume",
        "XMnBi Kharel routed top-bin predictions",
    )
    all_regressor_metrics = write_parity_plot(
        all_regressor_parity_plot,
        comparison_rows,
        "all_regressor_predicted_magnetization_per_volume",
        "XMnBi Kharel all-CIF top-bin regressor diagnostic",
    )

    script_snapshot = copy_script_snapshot(run_dir)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "benchmark_source_csv": str(benchmark_csv),
        "benchmark_target_column": BENCHMARK_TARGET_COLUMN,
        "benchmark_id_rule": "Extract /mp-*/ or /MnBi*/ from abs_path and compare to CIF stem.",
        "benchmark_original_columns": benchmark_fieldnames,
        "class_definitions": {
            "class_0": "M == 0",
            "class_1": "0 < M <= 1e-6",
            "class_2": "1e-6 < M <= 1e-3",
            "class_3": "M > 1e-3",
        },
        "parameters": {
            "batch_size": args.batch_size,
            "workers": args.workers,
            "cuda": False,
        },
        "counts": {
            "input_cifs": len(source_cifs),
            "benchmark_rows": len(benchmark_rows),
            "selected_for_top_bin_regressor": len(selected_ids),
            "extra_cifs_without_benchmark_rows": extra_cifs,
            "predicted_class_counts": {
                CLASS_NAMES[label]: sum(
                    1 for row in comparison_rows if int(row["predicted_class"]) == label
                )
                for label in range(NUM_CLASSES)
            },
        },
        "metrics": {
            "routed_top_bin_predictions": routed_metrics,
            "all_top_bin_regressor_diagnostic": all_regressor_metrics,
        },
        "inputs": {
            "benchmark_csv": str(linked_benchmark),
            "classifier_checkpoint": str(linked_classifier),
            "top_bin_regressor_checkpoint": str(linked_regressor),
            "classifier_dataset": str(classifier_dataset),
            "selected_regressor_dataset": str(selected_regressor_dataset),
            "all_regressor_dataset": str(all_regressor_dataset),
            "script_snapshot": str(script_snapshot),
            "classifier_checkpoint_summary": summarize_checkpoint(linked_classifier),
            "top_bin_regressor_checkpoint_summary": summarize_checkpoint(linked_regressor),
        },
        "outputs": {
            "classifier_predictions": str(classifier_predictions),
            "selected_regressor_predictions": str(selected_regressor_predictions),
            "all_regressor_predictions": str(all_regressor_predictions),
            "benchmark_predictions": str(benchmark_predictions),
            "routed_parity_plot": str(routed_parity_plot),
            "all_regressor_parity_plot": str(all_regressor_parity_plot),
            "metadata": str(run_dir / "run_metadata.json"),
        },
    }
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    write_readme(run_dir, metadata)
    print(run_dir)


if __name__ == "__main__":
    main()
