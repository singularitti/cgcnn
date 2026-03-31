from __future__ import annotations

import csv
import json
import os
import random
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analyze_parity import compute_metrics, make_plot
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
SMALL_POSITIVE_MAX = 1e-4


warnings.filterwarnings(
    "ignore",
    message="Issues encountered while parsing CIF: .*",
)
warnings.filterwarnings(
    "ignore",
    message=".*not find enough neighbors to build graph.*",
)
warnings.filterwarnings(
    "ignore",
    message="No Pauling electronegativity.*",
)


@dataclass(frozen=True)
class MaterialRecord:
    material_id: str
    magnetization: float
    cif_path: Path


def build_run_dir(runs_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"cgcnn_magnetization_two_stage_{timestamp}"
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
                    magnetization=magnetization,
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


def link_cifs(stage_dir: Path, records: list[MaterialRecord]) -> None:
    for record in records:
        link_path = stage_dir / f"{record.material_id}.cif"
        os.symlink(record.cif_path, link_path)


def write_stage_dataset(
    stage_dir: Path,
    records: list[MaterialRecord],
    mode: str,
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=False)
    write_atom_init(stage_dir / "atom_init.json")
    link_cifs(stage_dir, records)
    id_prop_path = stage_dir / "id_prop.csv"
    with id_prop_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for record in records:
            if mode == "classification":
                writer.writerow([record.material_id, int(record.magnetization > 0.0)])
            elif mode == "regression":
                writer.writerow([record.material_id, f"{record.magnetization:.16g}"])
            else:
                raise ValueError(f"Unsupported mode: {mode}")


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
) -> Path:
    dataset = CIFData(str(dataset_dir), shuffle=False)
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


def compute_classification_metrics(
    targets: list[int],
    probabilities: list[float],
    predicted_labels: list[int],
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(targets, predicted_labels, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "count": len(targets),
        "accuracy": float(accuracy_score(targets, predicted_labels)),
        "precision": float(precision_score(targets, predicted_labels, zero_division=0)),
        "recall": float(recall_score(targets, predicted_labels, zero_division=0)),
        "f1": float(f1_score(targets, predicted_labels, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(set(targets)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
    return metrics


def compute_subset_regression_metrics(
    rows: list[dict[str, float | int | str]],
) -> dict[str, dict[str, float]]:
    subsets = {
        "all": rows,
        "zero": [row for row in rows if float(row["target_m"]) == 0.0],
        "positive": [row for row in rows if float(row["target_m"]) > 0.0],
        "small_positive": [
            row
            for row in rows
            if 0.0 < float(row["target_m"]) <= SMALL_POSITIVE_MAX
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
    regressor_predictions_csv: Path,
    output_csv: Path,
    output_metrics_json: Path,
    output_plot_png: Path,
    classifier_metrics_json: Path,
    regressor_metrics_json: Path,
) -> None:
    classifier_rows = load_classification_predictions(classifier_predictions_csv)
    regressor_rows = load_regression_predictions(regressor_predictions_csv)
    merged_rows: list[dict[str, float | int | str]] = []

    classifier_targets: list[int] = []
    classifier_probabilities: list[float] = []
    classifier_predicted_labels: list[int] = []
    regressor_positive_targets: list[float] = []
    regressor_positive_predictions: list[float] = []

    for material_id in sorted(classifier_rows):
        classifier_target, positive_probability = classifier_rows[material_id]
        if material_id not in regressor_rows:
            raise KeyError(f"Missing regressor prediction for {material_id}")
        target_m, regressor_prediction = regressor_rows[material_id]
        predicted_label = 1 if positive_probability >= 0.5 else 0
        final_prediction = regressor_prediction if predicted_label == 1 else 0.0
        merged_rows.append(
            {
                "material_id": material_id,
                "target_m": target_m,
                "classifier_target": classifier_target,
                "classifier_positive_probability": positive_probability,
                "classifier_predicted_class": predicted_label,
                "regressor_prediction": regressor_prediction,
                "final_prediction": final_prediction,
            }
        )
        classifier_targets.append(classifier_target)
        classifier_probabilities.append(positive_probability)
        classifier_predicted_labels.append(predicted_label)
        if target_m > 0.0:
            regressor_positive_targets.append(target_m)
            regressor_positive_predictions.append(regressor_prediction)

    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "material_id",
                "target_m",
                "classifier_target",
                "classifier_predicted_class",
                "classifier_positive_probability",
                "regressor_prediction",
                "final_prediction",
            ]
        )
        for row in merged_rows:
            writer.writerow(
                [
                    row["material_id"],
                    row["target_m"],
                    row["classifier_target"],
                    row["classifier_predicted_class"],
                    row["classifier_positive_probability"],
                    row["regressor_prediction"],
                    row["final_prediction"],
                ]
            )

    final_targets = np.array([float(row["target_m"]) for row in merged_rows])
    final_predictions = np.array([float(row["final_prediction"]) for row in merged_rows])
    final_metrics = compute_metrics(final_targets, final_predictions)
    final_metrics["subsets"] = compute_subset_regression_metrics(merged_rows)
    with output_metrics_json.open("w") as handle:
        json.dump(final_metrics, handle, indent=2)
    make_plot(final_targets, final_predictions, final_metrics, output_plot_png)

    classifier_metrics = compute_classification_metrics(
        classifier_targets, classifier_probabilities, classifier_predicted_labels
    )
    with classifier_metrics_json.open("w") as handle:
        json.dump(classifier_metrics, handle, indent=2)

    if not regressor_positive_targets:
        regressor_metrics = {"count": 0}
    else:
        regressor_metrics = compute_metrics(
            np.array(regressor_positive_targets),
            np.array(regressor_positive_predictions),
        )
    with regressor_metrics_json.open("w") as handle:
        json.dump(regressor_metrics, handle, indent=2)


def main() -> None:
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE_ROOT
    summary_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SUMMARY_CSV
    runs_root = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_RUNS_ROOT
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_EPOCHS
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_BATCH_SIZE
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_WORKERS

    records = collect_records(summary_csv, source_root)
    record_map = {record.material_id: record for record in records}
    split_map = split_ids(records)

    run_dir = build_run_dir(runs_root)
    classifier_dir = run_dir / "classifier"
    regressor_dir = run_dir / "regressor"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=False)

    positive_records = [record for record in records if record.magnetization > 0.0]
    if not positive_records:
        raise RuntimeError("Positive-only regression dataset is empty.")

    write_stage_dataset(classifier_dir, records, mode="classification")
    write_stage_dataset(regressor_dir, positive_records, mode="regression")

    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=False)
    for split_name, ids in split_map.items():
        write_split_csv(split_dir / f"{split_name}_ids.csv", ids)

    positive_split_ids = {
        split_name: [material_id for material_id in ids if record_map[material_id].magnetization > 0.0]
        for split_name, ids in split_map.items()
    }
    for split_name, ids in positive_split_ids.items():
        write_split_csv(split_dir / f"regressor_{split_name}_ids.csv", ids)

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
    regressor_model = run_training(
        regressor_dir,
        task="regression",
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        train_ids=positive_split_ids["train"],
        val_ids=positive_split_ids["val"],
        test_ids=positive_split_ids["test"],
    )
    if classifier_model is None or regressor_model is None:
        raise RuntimeError("Training did not produce model checkpoints.")

    classifier_test_records = [record_map[material_id] for material_id in split_map["test"]]
    regressor_test_records = classifier_test_records
    classifier_test_dir = eval_dir / "classifier_test_dataset"
    regressor_test_dir = eval_dir / "regressor_test_dataset"
    write_stage_dataset(classifier_test_dir, classifier_test_records, mode="classification")
    write_stage_dataset(regressor_test_dir, regressor_test_records, mode="regression")

    classifier_predictions_csv = run_inference(
        dataset_dir=classifier_test_dir,
        model_path=classifier_model,
        task="classification",
        output_csv=eval_dir / "classifier_test_results.csv",
        batch_size=batch_size,
        workers=workers,
    )
    regressor_predictions_csv = run_inference(
        dataset_dir=regressor_test_dir,
        model_path=regressor_model,
        task="regression",
        output_csv=eval_dir / "regressor_test_results.csv",
        batch_size=batch_size,
        workers=workers,
    )

    metadata = {
        "summary_csv": str(summary_csv),
        "source_root": str(source_root),
        "dataset_size": len(records),
        "positive_dataset_size": len(positive_records),
        "positive_threshold": "M > 0",
        "combine_rule": "hard_gate",
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
            "regressor_model": str(regressor_model),
            "classifier_predictions": str(classifier_predictions_csv),
            "regressor_predictions": str(regressor_predictions_csv),
        },
    }
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    merge_predictions(
        classifier_predictions_csv=classifier_predictions_csv,
        regressor_predictions_csv=regressor_predictions_csv,
        output_csv=eval_dir / "merged_predictions.csv",
        output_metrics_json=eval_dir / "final_metrics.json",
        output_plot_png=eval_dir / "final_parity_plot.png",
        classifier_metrics_json=eval_dir / "classifier_metrics.json",
        regressor_metrics_json=eval_dir / "regressor_positive_metrics.json",
    )

    print(run_dir)


if __name__ == "__main__":
    main()
