from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from analyze_parity import compute_metrics, invert_transform, make_loglog_plot, make_plot
from cgcnn.benchmark import format_metrics_report, run_benchmark
from cgcnn.data import CIFData
from cgcnn.inference import predict_model
from cgcnn.training import train_model


DEFAULT_SOURCE_ROOT = Path("/Users/qz/Downloads/cifs")
DEFAULT_SUMMARY_CSV = DEFAULT_SOURCE_ROOT / "mp_all_summary.csv"
DEFAULT_RUNS_ROOT = Path("/Users/qz/Downloads/runs")
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 10
DEFAULT_RANDOM_SEED = 123
DEFAULT_THRESHOLD = 1e-3
DEFAULT_TRANSFORM = "log"
DEFAULT_EPS = 1e-30


@dataclass(frozen=True)
class MaterialRecord:
    material_id: str
    raw_magnetization: float
    cif_path: Path


def build_run_dir(runs_root: Path, threshold: float) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"cgcnn_magnetization_threshold_split_{threshold:.0e}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collect_records(summary_csv: Path, source_root: Path) -> list[MaterialRecord]:
    records: list[MaterialRecord] = []
    missing_dirs: list[str] = []
    bad_rows: list[str] = []
    bad_cifs: list[str] = []

    with summary_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"material_id", "total_magnetization", "volume"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {summary_csv}: {sorted(missing_columns)}"
            )
        for row in reader:
            material_id = (row.get("material_id") or "").strip()
            if not material_id:
                bad_rows.append("<missing material_id>")
                continue
            try:
                magnetization = float(row["total_magnetization"]) / float(row["volume"])
            except (TypeError, ValueError, ZeroDivisionError):
                bad_rows.append(material_id)
                continue
            cif_dir = source_root / material_id
            if not cif_dir.is_dir():
                missing_dirs.append(material_id)
                continue
            cif_files = sorted(cif_dir.glob("*.cif"))
            if len(cif_files) != 1:
                bad_cifs.append(material_id)
                continue
            records.append(
                MaterialRecord(
                    material_id=material_id,
                    raw_magnetization=magnetization,
                    cif_path=cif_files[0],
                )
            )

    issues = []
    if bad_rows:
        issues.append(f"bad rows: {len(bad_rows)}")
    if missing_dirs:
        issues.append(f"missing material directories: {len(missing_dirs)}")
    if bad_cifs:
        issues.append(f"directories without exactly one CIF: {len(bad_cifs)}")
    if issues:
        raise RuntimeError("Dataset preparation failed: " + ", ".join(issues))
    if not records:
        raise RuntimeError("No usable materials found.")
    return records


def split_ids(
    records: list[MaterialRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, list[str]]:
    ids = [record.material_id for record in records]
    rng = random.Random(random_seed)
    rng.shuffle(ids)
    total_size = len(ids)
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    return {
        "train": ids[:train_size],
        "val": ids[-(valid_size + test_size) : -test_size],
        "test": ids[-test_size:],
    }


def write_split_csv(path: Path, material_ids: Iterable[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id"])
        for material_id in material_ids:
            writer.writerow([material_id])


def write_atom_init(path: Path) -> None:
    atom_init = {}
    width = 118
    for atomic_number in range(1, width + 1):
        vector = [0] * width
        vector[atomic_number - 1] = 1
        atom_init[str(atomic_number)] = vector
    with path.open("w") as handle:
        json.dump(atom_init, handle)


def apply_transform(value: float, transform: str) -> float:
    if transform == "raw":
        return value
    if transform == "sqrt":
        return math.sqrt(value)
    if transform == "cbrt":
        return value ** (1.0 / 3.0)
    if transform == "log":
        return math.log(max(value, DEFAULT_EPS))
    if transform == "log10":
        return math.log10(max(value, DEFAULT_EPS))
    raise ValueError(f"Unsupported transform: {transform}")


def write_classifier_dataset(stage_dir: Path, records: list[MaterialRecord], threshold: float) -> None:
    stage_dir.mkdir(parents=True, exist_ok=False)
    write_atom_init(stage_dir / "atom_init.json")
    with (stage_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for record in records:
            label = 1 if record.raw_magnetization > threshold else 0
            writer.writerow([record.material_id, label])
    for record in records:
        os.symlink(record.cif_path, stage_dir / f"{record.material_id}.cif")


def write_regressor_dataset(stage_dir: Path, records: list[MaterialRecord], transform: str) -> None:
    stage_dir.mkdir(parents=True, exist_ok=False)
    write_atom_init(stage_dir / "atom_init.json")
    with (stage_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for record in records:
            writer.writerow(
                [
                    record.material_id,
                    f"{apply_transform(record.raw_magnetization, transform):.16g}",
                ]
            )
    for record in records:
        os.symlink(record.cif_path, stage_dir / f"{record.material_id}.cif")


def run_training(
    stage_dir: Path,
    task: str,
    epochs: int,
    batch_size: int,
    workers: int,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> Path | None:
    checkpoint_dir = stage_dir / "checkpoints"
    metrics_history_path = stage_dir / "training_history.json"
    previous_cwd = Path.cwd()
    try:
        os.chdir(stage_dir)
        best_model = train_model(
            root_dir=str(stage_dir),
            task=task,
            epochs=epochs,
            batch_size=batch_size,
            workers=workers,
            cuda=False,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            print_freq=20,
            checkpoint_dir=str(checkpoint_dir),
            metrics_history_path=str(metrics_history_path),
        )
    finally:
        os.chdir(previous_cwd)
    return Path(best_model) if best_model else None


def run_inference(
    dataset_dir: Path,
    model_path: Path,
    task: str,
    output_csv: Path,
    batch_size: int,
    workers: int,
    include_ids: list[str] | None = None,
) -> Path:
    dataset = CIFData(str(dataset_dir), shuffle=False, include_ids=include_ids)
    predict_model(
        dataset=dataset,
        task=task,
        modelpath=str(model_path),
        batch_size=batch_size,
        workers=workers,
        cuda=False,
        print_freq=20,
        shuffle=False,
        output_csv=str(output_csv),
    )
    return output_csv


def load_regression_predictions(path: Path) -> dict[str, tuple[float, float]]:
    rows: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as handle:
        for material_id, target, prediction in csv.reader(handle):
            rows[material_id] = (float(target), float(prediction))
    return rows


def load_classification_predictions(path: Path) -> dict[str, tuple[int, float]]:
    rows: dict[str, tuple[int, float]] = {}
    with path.open(newline="") as handle:
        for material_id, target, probability in csv.reader(handle):
            rows[material_id] = (int(float(target)), float(probability))
    return rows


def compute_subset_metrics(
    rows: list[dict[str, float | int | str]],
    threshold: float,
) -> dict[str, dict[str, float]]:
    subsets = {
        "all": rows,
        "small": [
            row for row in rows if float(row["target_m"]) <= threshold
        ],
        "large": [
            row for row in rows if float(row["target_m"]) > threshold
        ],
    }
    output: dict[str, dict[str, float]] = {}
    for name, subset_rows in subsets.items():
        if not subset_rows:
            continue
        targets = [float(row["target_m"]) for row in subset_rows]
        predictions = [float(row["final_prediction"]) for row in subset_rows]
        output[name] = compute_metrics(np.array(targets), np.array(predictions))
    return output


def merge_predictions(
    classifier_predictions_csv: Path,
    small_predictions_csv: Path,
    large_predictions_csv: Path,
    output_csv: Path,
    output_metrics_json: Path,
    output_plot_png: Path,
    classifier_metrics_json: Path,
    small_metrics_json: Path,
    large_metrics_json: Path,
    raw_targets: dict[str, float],
    transform: str,
) -> None:
    classifier_rows = load_classification_predictions(classifier_predictions_csv)
    small_rows = load_regression_predictions(small_predictions_csv)
    large_rows = load_regression_predictions(large_predictions_csv)

    merged_rows: list[dict[str, float | int | str]] = []
    classifier_targets: list[int] = []
    classifier_probabilities: list[float] = []
    classifier_predicted_labels: list[int] = []

    for material_id in sorted(classifier_rows):
        if material_id not in raw_targets:
            raise KeyError(f"Missing raw target for {material_id}")
        classifier_target, positive_probability = classifier_rows[material_id]
        if material_id not in small_rows:
            raise KeyError(f"Missing small-regressor prediction for {material_id}")
        if material_id not in large_rows:
            raise KeyError(f"Missing large-regressor prediction for {material_id}")

        raw_target = raw_targets[material_id]
        small_target, small_prediction = small_rows[material_id]
        large_target, large_prediction = large_rows[material_id]
        small_target = apply_transform(raw_target, transform)
        large_target = apply_transform(raw_target, transform)
        small_prediction_raw = float(invert_transform(np.array([small_prediction]), transform)[0])
        large_prediction_raw = float(invert_transform(np.array([large_prediction]), transform)[0])
        predicted_label = 1 if positive_probability >= 0.5 else 0
        final_prediction = (
            large_prediction_raw if predicted_label == 1 else small_prediction_raw
        )
        merged_rows.append(
            {
                "material_id": material_id,
                "target_m": raw_target,
                "classifier_target": classifier_target,
                "classifier_predicted_class": predicted_label,
                "classifier_positive_probability": positive_probability,
                "small_regressor_prediction": small_prediction_raw,
                "large_regressor_prediction": large_prediction_raw,
                "final_prediction": final_prediction,
            }
        )
        classifier_targets.append(classifier_target)
        classifier_probabilities.append(positive_probability)
        classifier_predicted_labels.append(predicted_label)

    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "material_id",
                "target_m",
                "classifier_target",
                "classifier_predicted_class",
                "classifier_positive_probability",
                "small_regressor_prediction",
                "large_regressor_prediction",
                "final_prediction",
            ]
        )
        for row in merged_rows:
            writer.writerow(
                [
                    row["material_id"],
                    f"{row["target_m"]:.16g}",
                    row["classifier_target"],
                    row["classifier_predicted_class"],
                    f"{row["classifier_positive_probability"]:.16g}",
                    f"{row["small_regressor_prediction"]:.16g}",
                    f"{row["large_regressor_prediction"]:.16g}",
                    f"{row["final_prediction"]:.16g}",
                ]
            )

    final_targets = np.array([float(row["target_m"]) for row in merged_rows])
    final_predictions = np.array(
        [float(row["final_prediction"]) for row in merged_rows]
    )
    final_metrics = compute_metrics(final_targets, final_predictions)
    final_metrics["subsets"] = compute_subset_metrics(merged_rows, threshold=DEFAULT_THRESHOLD)
    with output_metrics_json.open("w") as handle:
        json.dump(final_metrics, handle, indent=2)
    make_plot(final_targets, final_predictions, final_metrics, output_plot_png)
    make_loglog_plot(final_targets, final_predictions, final_metrics, output_plot_png.with_name("parity_plot_loglog.png"))

    classifier_metrics = {
        "count": len(classifier_targets),
        "accuracy": float(sum(1 for t, p in zip(classifier_targets, classifier_predicted_labels) if t == p) / len(classifier_targets)),
        "precision": float(
            sum(1 for t, p in zip(classifier_targets, classifier_predicted_labels) if t == 1 and p == 1)
            / max(1, sum(1 for p in classifier_predicted_labels if p == 1))
        ),
        "recall": float(
            sum(1 for t, p in zip(classifier_targets, classifier_predicted_labels) if t == 1 and p == 1)
            / max(1, sum(1 for t in classifier_targets if t == 1))
        ),
    }
    with classifier_metrics_json.open("w") as handle:
        json.dump(classifier_metrics, handle, indent=2)

    if final_targets.size > 0:
        benchmark_input_csv = output_csv.with_name("benchmark_input.csv")
        processed_rows, benchmark_metrics = run_benchmark(str(output_csv), output_file=None)
        with output_csv.with_name("benchmark_metrics.json").open("w") as handle:
            json.dump(benchmark_metrics, handle, indent=2)
        with output_csv.with_name("benchmark_report.txt").open("w") as handle:
            handle.write(format_metrics_report(benchmark_metrics) + "\n")


def main() -> None:
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE_ROOT
    summary_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SUMMARY_CSV
    runs_root = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_RUNS_ROOT
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_EPOCHS
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_BATCH_SIZE
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_WORKERS
    threshold = float(sys.argv[7]) if len(sys.argv) > 7 else DEFAULT_THRESHOLD
    transform = sys.argv[8] if len(sys.argv) > 8 else DEFAULT_TRANSFORM

    records = collect_records(summary_csv, source_root)
    split_map = split_ids(records)
    raw_target_map = {record.material_id: record.raw_magnetization for record in records}

    run_dir = build_run_dir(runs_root, threshold)
    classifier_dir = run_dir / "classifier"
    small_regressor_dir = run_dir / "small_regressor"
    large_regressor_dir = run_dir / "large_regressor"
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=False)
    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=False)

    for split_name, ids in split_map.items():
        write_split_csv(splits_dir / f"{split_name}_ids.csv", ids)

    write_classifier_dataset(classifier_dir, records, threshold=threshold)
    write_regressor_dataset(small_regressor_dir, records, transform=transform)
    write_regressor_dataset(large_regressor_dir, records, transform=transform)

    small_records = [record for record in records if 0.0 < record.raw_magnetization <= threshold]
    large_records = [record for record in records if record.raw_magnetization > threshold]
    if not small_records:
        raise RuntimeError("No small-value records found for the threshold split.")
    if not large_records:
        raise RuntimeError("No large-value records found for the threshold split.")

    small_split_ids = {
        split_name: [material_id for material_id in ids if raw_target_map[material_id] > 0.0 and raw_target_map[material_id] <= threshold]
        for split_name, ids in split_map.items()
    }
    large_split_ids = {
        split_name: [material_id for material_id in ids if raw_target_map[material_id] > threshold]
        for split_name, ids in split_map.items()
    }

    classifier_model = run_training(
        classifier_dir,
        task="classification",
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        train_ids=split_map["train"],
        val_ids=split_map["val"],
        test_ids=split_map["test"],
    )
    small_regressor_model = run_training(
        small_regressor_dir,
        task="regression",
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        train_ids=small_split_ids["train"],
        val_ids=small_split_ids["val"],
        test_ids=small_split_ids["test"],
    )
    large_regressor_model = run_training(
        large_regressor_dir,
        task="regression",
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        train_ids=large_split_ids["train"],
        val_ids=large_split_ids["val"],
        test_ids=large_split_ids["test"],
    )

    if classifier_model is None or small_regressor_model is None or large_regressor_model is None:
        raise RuntimeError("Training did not produce all required checkpoints.")

    classifier_predictions_csv = run_inference(
        classifier_dir,
        classifier_model,
        task="classification",
        output_csv=output_dir / "classifier_test_results.csv",
        batch_size=batch_size,
        workers=workers,
        include_ids=split_map["test"],
    )
    small_regressor_predictions_csv = run_inference(
        small_regressor_dir,
        small_regressor_model,
        task="regression",
        output_csv=output_dir / "small_regressor_test_results.csv",
        batch_size=batch_size,
        workers=workers,
        include_ids=split_map["test"],
    )
    large_regressor_predictions_csv = run_inference(
        large_regressor_dir,
        large_regressor_model,
        task="regression",
        output_csv=output_dir / "large_regressor_test_results.csv",
        batch_size=batch_size,
        workers=workers,
        include_ids=split_map["test"],
    )

    merge_predictions(
        classifier_predictions_csv,
        small_regressor_predictions_csv,
        large_regressor_predictions_csv,
        output_dir / "final_test_results.csv",
        output_dir / "final_parity_metrics.json",
        output_dir / "parity_plot.png",
        output_dir / "classifier_metrics.json",
        output_dir / "small_regressor_metrics.json",
        output_dir / "large_regressor_metrics.json",
        raw_target_map,
        transform=transform,
    )

    metadata = {
        "summary_csv": str(summary_csv),
        "source_root": str(source_root),
        "dataset_size": len(records),
        "threshold": threshold,
        "target_transform": transform,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "cuda": False,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_seed": DEFAULT_RANDOM_SEED,
        },
        "paths": {
            "classifier_model": str(classifier_model),
            "small_regressor_model": str(small_regressor_model),
            "large_regressor_model": str(large_regressor_model),
            "classifier_predictions": str(classifier_predictions_csv),
            "small_regressor_predictions": str(small_regressor_predictions_csv),
            "large_regressor_predictions": str(large_regressor_predictions_csv),
            "final_predictions": str(output_dir / "final_test_results.csv"),
        },
    }
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(run_dir)


if __name__ == "__main__":
    main()
