from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pymatgen.io.vasp import Poscar

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

DEFAULT_ROOTS = [
    Path("~/.mnt.noindex/Work/run/MnBiS").expanduser(),
    Path("~/.mnt.noindex/Work/run/MnBiSe").expanduser(),
    Path("~/.mnt.noindex/Work/run/MnBiSb").expanduser(),
    Path("~/.mnt.noindex/Work/run/MnBiTe").expanduser(),
]
DEFAULT_BENCHMARK_CSV = Path(
    "~/.mnt.noindex/Work/run/MnBiS/magnetization_summary_by_parent.csv"
).expanduser()
DEFAULT_CLASSIFIER_CHECKPOINT = Path(
    "~/.mnt.noindex/Work/run/MnBiO/predict/magnetization/classifier_model_best.pth.tar"
).expanduser()
DEFAULT_REGRESSOR_CHECKPOINT = Path(
    "~/.mnt.noindex/Work/run/MnBiO/predict/magnetization/epoch_029.pth.tar"
).expanduser()
DEFAULT_RUNS_ROOT = REPO_ROOT / "tmp" / "prediction_runs"
DEFAULT_BATCH_SIZE = 256
DEFAULT_WORKERS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict and benchmark MnBiX magnetization per volume."
    )
    parser.add_argument("--source-root", type=Path, action="append", default=None)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--regressor-checkpoint", type=Path, default=DEFAULT_REGRESSOR_CHECKPOINT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def symlink_force(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source)


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.runs_root.expanduser().resolve() / f"mnbix_magnetization_volume_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parent_from_root(root: Path) -> str:
    return root.expanduser().resolve().name


def collect_prediction_cifs(source_roots: list[Path]) -> dict[str, Path]:
    cifs_by_parent: dict[str, Path] = {}
    for root in source_roots:
        root = root.expanduser().resolve()
        parent = parent_from_root(root)
        predict_dir = root / "cgcnn_predict"
        cifs = sorted(
            path
            for path in predict_dir.glob("*.cif")
            if path.is_file() and not path.name.startswith("._")
        )
        if len(cifs) != 1:
            raise RuntimeError(f"Expected exactly one CIF in {predict_dir}, found {len(cifs)}")
        cifs_by_parent[parent] = cifs[0]
    return cifs_by_parent


def parse_benchmark(path: Path) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {"rows": 0, "files": set(), "total_moment_mu_b": 0.0}
    )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"parent", "file", "s", "p", "d"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing benchmark columns: {sorted(missing)}")
        for row in reader:
            parent = row["parent"]
            grouped[parent]["rows"] = int(grouped[parent]["rows"]) + 1
            grouped[parent]["files"].add(row["file"])
            grouped[parent]["total_moment_mu_b"] = float(grouped[parent]["total_moment_mu_b"]) + sum(
                float(row[orbital]) for orbital in ("s", "p", "d")
            )

    result: dict[str, dict[str, object]] = {}
    for parent, info in grouped.items():
        files = sorted(str(path) for path in info["files"])
        if len(files) != 1:
            raise RuntimeError(f"Expected one unique structure file for {parent}, found {files}")
        outcar = Path(files[0]).expanduser()
        volume_path = outcar
        volume_source = "listed_outcar"
        volume_error = ""
        try:
            volume = get_volume(volume_path)
        except Exception as exc:
            volume_error = f"{type(exc).__name__}: {exc}"
            volume_path = outcar.with_name("CONTCAR")
            volume_source = "sibling_contcar_fallback"
            volume = get_volume(volume_path)
        total_moment = float(info["total_moment_mu_b"])
        result[parent] = {
            "parent": parent,
            "benchmark_rows": int(info["rows"]),
            "listed_outcar": str(outcar),
            "volume_structure": str(volume_path),
            "volume_source": volume_source,
            "listed_outcar_volume_error": volume_error,
            "volume_angstrom3": volume,
            "total_moment_mu_b": total_moment,
            "benchmark_magnetization_per_volume": total_moment / volume,
        }
    return result


def get_volume(filename: Path) -> float:
    poscar = Poscar.from_file(filename)
    return float(poscar.structure.volume)


def write_dataset(dataset_dir: Path, parent_to_cif: dict[str, Path], target_value: str) -> dict[str, str]:
    dataset_dir.mkdir(parents=True, exist_ok=False)
    write_atom_init(dataset_dir / "atom_init.json")
    material_id_to_parent: dict[str, str] = {}
    with (dataset_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for parent, cif_path in sorted(parent_to_cif.items()):
            material_id = cif_path.stem
            if material_id in material_id_to_parent:
                raise RuntimeError(f"Duplicate material ID: {material_id}")
            material_id_to_parent[material_id] = parent
            symlink_force(cif_path, dataset_dir / f"{material_id}.cif")
            writer.writerow([material_id, target_value])
    return material_id_to_parent


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


def add_regression_header(path: Path) -> None:
    header = ["material_id", "target", "predicted_magnetization_per_volume"]
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if rows and rows[0] == header:
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.16g}"


def metric_summary(rows: list[dict[str, object]], prediction_field: str) -> dict[str, float | int | None]:
    targets: list[float] = []
    predictions: list[float] = []
    for row in rows:
        text = str(row[prediction_field])
        if not text:
            continue
        targets.append(float(row["benchmark_magnetization_per_volume"]))
        predictions.append(float(text))
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


def write_parity_plot(path: Path, rows: list[dict[str, object]]) -> dict[str, dict[str, float | int | None]]:
    series = [
        ("routed_predicted_magnetization_per_volume", "routed class-3 workflow", "#2563eb"),
        ("all_regressor_predicted_magnetization_per_volume", "all-CIF regressor diagnostic", "#dc2626"),
    ]
    metrics = {field: metric_summary(rows, field) for field, _, _ in series}

    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    plotted_targets: list[float] = []
    plotted_predictions: list[float] = []
    for field, label, color in series:
        targets: list[float] = []
        predictions: list[float] = []
        for row in rows:
            text = str(row[field])
            if not text:
                continue
            target = float(row["benchmark_magnetization_per_volume"])
            prediction = float(text)
            targets.append(target)
            predictions.append(prediction)
        if targets:
            ax.scatter(targets, predictions, s=48, color=color, edgecolor="white", linewidth=0.7, label=label)
            plotted_targets.extend(targets)
            plotted_predictions.extend(predictions)
    if plotted_targets:
        lo = float(min(min(plotted_targets), min(plotted_predictions)))
        hi = float(max(max(plotted_targets), max(plotted_predictions)))
        pad = (hi - lo) * 0.08 if hi > lo else 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111827", linewidth=1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Benchmark magnetization per volume (mu_B / angstrom^3)")
    ax.set_ylabel("CGCNN prediction per volume (mu_B / angstrom^3)")
    ax.set_title("MnBiX magnetization parity")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    text = "\n".join(
        f"{label}: n={metrics[field]['n']}, MAE={metrics[field]['mae']:.4g}"
        for field, label, _ in series
        if metrics[field]["n"]
    )
    if text:
        ax.text(
            0.04,
            0.96,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"},
        )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return metrics


def summarize_checkpoint(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    args = checkpoint.get("args", {})
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "best_validation_score": checkpoint.get("best_validation_score"),
        "args": args if isinstance(args, dict) else {},
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
                "# MnBiX Magnetization Per Volume Prediction",
                "",
                f"Created: {metadata['created_at_utc']}",
                "",
                "## Purpose",
                "",
                "Predict magnetization per unit volume for MnBiS, MnBiSe, MnBiSb, and MnBiTe CIFs using the saved MnBiO classifier plus top-bin regressor workflow, then compare against benchmark magnetization per volume computed from `magnetization_summary_by_parent.csv`.",
                "",
                "## Inputs",
                "",
                f"- Source roots: `{metadata['source_roots']}`",
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
                "CIFs, checkpoints, benchmark CSV, and volume structure files are symlinked into this run folder; they are not copied.",
                "",
                "## Benchmark Construction",
                "",
                "Rows are grouped by `parent`. For each parent, `s + p + d` is summed over all rows to get total magnetic moment. The benchmark magnetization is total magnetic moment divided by structure volume. The CSV lists unique `OUTCAR` files, but `pymatgen.io.vasp.Poscar.from_file(OUTCAR)` failed for these VASP output files, so volume was read from the sibling `CONTCAR`; this fallback and the original parse error are recorded in the comparison table and metadata.",
                "",
                "## Outputs",
                "",
                f"- Classifier raw predictions: `{metadata['outputs']['classifier_predictions']}`",
                f"- Selected class-3 regressor raw predictions: `{metadata['outputs']['selected_regressor_predictions']}`",
                f"- All-CIF diagnostic regressor raw predictions: `{metadata['outputs']['all_regressor_predictions']}`",
                f"- Final comparison table: `{metadata['outputs']['comparison_csv']}`",
                f"- Parity plot: `{metadata['outputs']['parity_plot']}`",
                f"- Run metadata: `{metadata['outputs']['metadata']}`",
                "",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    source_roots = [path.expanduser().resolve() for path in (args.source_root or DEFAULT_ROOTS)]
    benchmark_csv = args.benchmark_csv.expanduser().resolve()
    classifier_checkpoint = args.classifier_checkpoint.expanduser().resolve()
    regressor_checkpoint = args.regressor_checkpoint.expanduser().resolve()

    for path in [*source_roots, benchmark_csv, classifier_checkpoint, regressor_checkpoint]:
        if not path.exists():
            raise FileNotFoundError(path)

    parent_to_cif = collect_prediction_cifs(source_roots)
    benchmark = parse_benchmark(benchmark_csv)
    missing_benchmarks = sorted(set(parent_to_cif) - set(benchmark))
    if missing_benchmarks:
        raise RuntimeError(f"Missing benchmark rows for parents: {missing_benchmarks}")

    run_dir = make_run_dir(args)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = run_dir / "inputs"
    linked_benchmark = inputs_dir / "benchmark" / benchmark_csv.name
    linked_classifier = inputs_dir / "checkpoints" / classifier_checkpoint.name
    linked_regressor = inputs_dir / "checkpoints" / regressor_checkpoint.name
    symlink_force(benchmark_csv, linked_benchmark)
    symlink_force(classifier_checkpoint, linked_classifier)
    symlink_force(regressor_checkpoint, linked_regressor)
    linked_volume_files: dict[str, str] = {}
    for parent, row in sorted(benchmark.items()):
        source = Path(str(row["volume_structure"]))
        destination = inputs_dir / "volume_structures" / parent / source.name
        symlink_force(source, destination)
        linked_volume_files[parent] = str(destination)

    classifier_dataset = run_dir / "classifier_dataset_all"
    all_regressor_dataset = run_dir / "top_bin_regressor_dataset_all_diagnostic"
    material_id_to_parent = write_dataset(classifier_dataset, parent_to_cif, target_value="0")
    write_dataset(all_regressor_dataset, parent_to_cif, target_value="0.0")

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

    selected_parent_to_cif = {
        material_id_to_parent[material_id]: parent_to_cif[material_id_to_parent[material_id]]
        for material_id, row in classifier_rows.items()
        if int(row["predicted_class"]) == CLASS_LARGE
    }
    selected_regressor_dataset = run_dir / "top_bin_regressor_dataset_selected"
    selected_regressor_predictions = outputs_dir / "selected_top_bin_regressor_predictions.csv"
    selected_regressor_rows: dict[str, tuple[float, float]] = {}
    if selected_parent_to_cif:
        write_dataset(selected_regressor_dataset, selected_parent_to_cif, target_value="0.0")
        run_inference(
            dataset_dir=selected_regressor_dataset,
            model_path=linked_regressor,
            task="regression",
            output_csv=selected_regressor_predictions,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        selected_regressor_rows = load_regression_predictions(selected_regressor_predictions)
        add_regression_header(selected_regressor_predictions)
    else:
        selected_regressor_dataset.mkdir(parents=True, exist_ok=False)
        write_atom_init(selected_regressor_dataset / "atom_init.json")
        (selected_regressor_dataset / "id_prop.csv").write_text("")
        selected_regressor_predictions.write_text("material_id,target,predicted_magnetization_per_volume\n")

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
    add_regression_header(all_regressor_predictions)

    probability_fields = [
        f"class_{label}_{CLASS_NAMES[label]}_probability" for label in range(NUM_CLASSES)
    ]
    fields = [
        "parent",
        "material_id",
        "source_cif",
        "benchmark_rows",
        "total_moment_mu_b",
        "volume_angstrom3",
        "benchmark_magnetization_per_volume",
        "listed_outcar",
        "volume_structure",
        "volume_source",
        "listed_outcar_volume_error",
        "predicted_class",
        "predicted_class_name",
        "selected_for_top_bin_regressor",
        "routed_predicted_magnetization_per_volume",
        "routed_error",
        "all_regressor_predicted_magnetization_per_volume",
        "all_regressor_error",
    ] + probability_fields

    comparison_rows: list[dict[str, object]] = []
    for parent, cif_path in sorted(parent_to_cif.items()):
        material_id = cif_path.stem
        bench = benchmark[parent]
        classifier_row = classifier_rows[material_id]
        predicted_class = int(classifier_row["predicted_class"])
        probabilities = list(classifier_row["probabilities"])
        target = float(bench["benchmark_magnetization_per_volume"])
        routed = selected_regressor_rows.get(material_id)
        routed_prediction = None if routed is None else routed[1]
        all_prediction = all_regressor_rows[material_id][1]
        row: dict[str, object] = {
            "parent": parent,
            "material_id": material_id,
            "source_cif": str(cif_path),
            "benchmark_rows": bench["benchmark_rows"],
            "total_moment_mu_b": format_float(float(bench["total_moment_mu_b"])),
            "volume_angstrom3": format_float(float(bench["volume_angstrom3"])),
            "benchmark_magnetization_per_volume": format_float(target),
            "listed_outcar": bench["listed_outcar"],
            "volume_structure": bench["volume_structure"],
            "volume_source": bench["volume_source"],
            "listed_outcar_volume_error": bench["listed_outcar_volume_error"],
            "predicted_class": predicted_class,
            "predicted_class_name": CLASS_NAMES[predicted_class],
            "selected_for_top_bin_regressor": predicted_class == CLASS_LARGE,
            "routed_predicted_magnetization_per_volume": format_float(routed_prediction),
            "routed_error": format_float(None if routed_prediction is None else routed_prediction - target),
            "all_regressor_predicted_magnetization_per_volume": format_float(all_prediction),
            "all_regressor_error": format_float(all_prediction - target),
        }
        for label, field in enumerate(probability_fields):
            row[field] = f"{probabilities[label]:.16g}"
        comparison_rows.append(row)

    comparison_csv = outputs_dir / "magnetization_volume_comparison.csv"
    write_rows(comparison_csv, comparison_rows, fields)
    parity_plot = outputs_dir / "magnetization_volume_parity.png"
    metrics = write_parity_plot(parity_plot, comparison_rows)
    script_snapshot = copy_script_snapshot(run_dir)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_roots": [str(path) for path in source_roots],
        "benchmark_source_csv": str(benchmark_csv),
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
            "parents": len(parent_to_cif),
            "selected_for_top_bin_regressor": len(selected_parent_to_cif),
            "predicted_class_counts": {
                CLASS_NAMES[label]: sum(
                    1 for row in comparison_rows if int(row["predicted_class"]) == label
                )
                for label in range(NUM_CLASSES)
            },
        },
        "metrics": metrics,
        "inputs": {
            "benchmark_csv": str(linked_benchmark),
            "classifier_checkpoint": str(linked_classifier),
            "top_bin_regressor_checkpoint": str(linked_regressor),
            "classifier_dataset": str(classifier_dataset),
            "selected_regressor_dataset": str(selected_regressor_dataset),
            "all_regressor_dataset": str(all_regressor_dataset),
            "volume_structures": linked_volume_files,
            "script_snapshot": str(script_snapshot),
            "classifier_checkpoint_summary": summarize_checkpoint(linked_classifier),
            "top_bin_regressor_checkpoint_summary": summarize_checkpoint(linked_regressor),
        },
        "outputs": {
            "classifier_predictions": str(classifier_predictions),
            "selected_regressor_predictions": str(selected_regressor_predictions),
            "all_regressor_predictions": str(all_regressor_predictions),
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
