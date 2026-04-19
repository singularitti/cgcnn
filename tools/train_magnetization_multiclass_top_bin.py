from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from analyze_parity import compute_metrics, make_plot

from cgcnn.data import CIFData
from cgcnn.inference import predict_model
from cgcnn.training import train_model

DEFAULT_SOURCE_ROOT = Path("/Users/qz/Downloads/cifs")
DEFAULT_SUMMARY_CSV = DEFAULT_SOURCE_ROOT / "mp_all_summary.csv"
DEFAULT_RUNS_ROOT = REPO_ROOT / "tmp" / "training_runs"
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 10
DEFAULT_RANDOM_SEED = 123
SMALL_BIN_MAX = 1e-6
TOP_BIN_THRESHOLD = 1e-3
CLASS_ZERO = 0
CLASS_TINY = 1
CLASS_SMALL = 2
CLASS_LARGE = 3
NUM_CLASSES = 4
CLASS_NAMES = {
    CLASS_ZERO: "zero",
    CLASS_TINY: "tiny_positive",
    CLASS_SMALL: "small_positive",
    CLASS_LARGE: "large_positive",
}


@dataclass(frozen=True)
class MaterialRecord:
    material_id: str
    raw_magnetization: float
    cif_path: Path


def classify_magnetization(value: float) -> int:
    if value < 0.0:
        raise ValueError(f"Negative magnetization is unsupported: {value:.16g}")
    if value == 0.0:
        return CLASS_ZERO
    if value <= SMALL_BIN_MAX:
        return CLASS_TINY
    if value <= TOP_BIN_THRESHOLD:
        return CLASS_SMALL
    return CLASS_LARGE


def class_counts(records: Iterable[MaterialRecord]) -> dict[str, int]:
    counts = Counter(classify_magnetization(record.raw_magnetization) for record in records)
    return {CLASS_NAMES[label]: int(counts.get(label, 0)) for label in range(NUM_CLASSES)}


def build_run_dir(runs_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"cgcnn_magnetization_multiclass_top_regressor_{timestamp}"
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

    issues: list[str] = []
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


def compute_class_weights(
    records: list[MaterialRecord], train_ids: Iterable[str]
) -> tuple[list[float], dict[str, int]]:
    train_id_set = set(train_ids)
    train_records = [record for record in records if record.material_id in train_id_set]
    counts = Counter(classify_magnetization(record.raw_magnetization) for record in train_records)
    missing_labels = [label for label in range(NUM_CLASSES) if counts.get(label, 0) == 0]
    if missing_labels:
        missing_names = ", ".join(CLASS_NAMES[label] for label in missing_labels)
        raise RuntimeError(
            "Training split is missing classes required for multiclass classification: "
            f"{missing_names}"
        )
    total_count = sum(counts.values())
    weights = [total_count / (NUM_CLASSES * counts[label]) for label in range(NUM_CLASSES)]
    return weights, {CLASS_NAMES[label]: int(counts[label]) for label in range(NUM_CLASSES)}


def write_split_csv(path: Path, material_ids: Iterable[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id"])
        for material_id in material_ids:
            writer.writerow([material_id])


def read_split_csv(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["material_id"]:
            raise ValueError(f"Unexpected split CSV header in {path}: {header}")
        return [row[0] for row in reader if row]


def write_atom_init(path: Path) -> None:
    atom_init = {}
    width = 118
    for atomic_number in range(1, width + 1):
        vector = [0] * width
        vector[atomic_number - 1] = 1
        atom_init[str(atomic_number)] = vector
    with path.open("w") as handle:
        json.dump(atom_init, handle)


def link_cifs(stage_dir: Path, records: Iterable[MaterialRecord]) -> None:
    for record in records:
        os.symlink(record.cif_path, stage_dir / f"{record.material_id}.cif")


def write_stage_dataset(stage_dir: Path, records: list[MaterialRecord], mode: str) -> None:
    stage_dir.mkdir(parents=True, exist_ok=False)
    write_atom_init(stage_dir / "atom_init.json")
    link_cifs(stage_dir, records)
    with (stage_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for record in records:
            if mode == "classification":
                writer.writerow([record.material_id, classify_magnetization(record.raw_magnetization)])
            elif mode == "regression":
                writer.writerow([record.material_id, f"{record.raw_magnetization:.16g}"])
            else:
                raise ValueError(f"Unsupported mode: {mode}")


def ensure_stage_dataset(stage_dir: Path, records: list[MaterialRecord], mode: str) -> None:
    if stage_dir.exists():
        return
    write_stage_dataset(stage_dir, records, mode=mode)


def subset_ids_by_class(
    material_ids: Iterable[str],
    record_map: dict[str, MaterialRecord],
    class_label: int,
) -> list[str]:
    return [
        material_id
        for material_id in material_ids
        if classify_magnetization(record_map[material_id].raw_magnetization) == class_label
    ]


def run_training(
    stage_dir: Path,
    task: str,
    epochs: int,
    batch_size: int,
    workers: int,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    n_classes: int | None = None,
    class_weights: list[float] | None = None,
    classification_metric: str | None = None,
    classification_metric_class_index: int | None = None,
    resume: Path | None = None,
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
            n_classes=n_classes,
            class_weights=class_weights,
            classification_metric=classification_metric,
            classification_metric_class_index=classification_metric_class_index,
            resume=str(resume) if resume is not None else None,
        )
    finally:
        os.chdir(previous_cwd)
    return Path(best_model) if best_model else None


def load_training_history(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        history = json.load(handle)
    if not isinstance(history, list):
        raise ValueError(f"Malformed training history in {path}")
    return history


def resolve_stage_resume(stage_dir: Path, epochs: int) -> tuple[Path | None, bool]:
    history = load_training_history(stage_dir / "training_history.json")
    completed_epochs = 0
    for item in history:
        if isinstance(item, dict):
            try:
                completed_epochs = max(completed_epochs, int(item.get("epoch", 0)))
            except (TypeError, ValueError):
                continue
    best_model = stage_dir / "model_best.pth.tar"
    checkpoint = stage_dir / "checkpoint.pth.tar"
    if completed_epochs >= epochs and best_model.is_file():
        return best_model, True
    if checkpoint.is_file():
        return checkpoint, False
    return None, False


def load_split_map(split_dir: Path) -> dict[str, list[str]]:
    return {
        "train": read_split_csv(split_dir / "train_ids.csv"),
        "val": read_split_csv(split_dir / "val_ids.csv"),
        "test": read_split_csv(split_dir / "test_ids.csv"),
    }


def load_top_bin_split_map(split_dir: Path) -> dict[str, list[str]]:
    return {
        "train": read_split_csv(split_dir / "top_bin_train_ids.csv"),
        "val": read_split_csv(split_dir / "top_bin_val_ids.csv"),
        "test": read_split_csv(split_dir / "top_bin_test_ids.csv"),
    }


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


def load_classification_predictions(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) == 3:
                material_id, target_class, positive_probability = row
                probability = float(positive_probability)
                predicted_class = 1 if probability >= 0.5 else 0
                probabilities = [1.0 - probability, probability]
            elif len(row) >= 4:
                material_id = row[0]
                target_class = row[1]
                predicted_class = row[2]
                probabilities = [float(value) for value in row[3:]]
            else:
                raise ValueError(f"Malformed classification row in {path}: {row}")
            rows[material_id] = {
                "target_class": int(float(target_class)),
                "predicted_class": int(float(predicted_class)),
                "probabilities": probabilities,
            }
    return rows


def compute_classifier_metrics(rows: dict[str, dict[str, object]]) -> dict[str, object]:
    material_ids = sorted(rows)
    targets = np.array([rows[material_id]["target_class"] for material_id in material_ids], dtype=int)
    predictions = np.array([rows[material_id]["predicted_class"] for material_id in material_ids], dtype=int)
    probabilities = np.array([rows[material_id]["probabilities"] for material_id in material_ids], dtype=float)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(probabilities.shape[1])),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(probabilities.shape[1])),
        average="macro",
        zero_division=0,
    )

    per_class = {
        CLASS_NAMES[label]: {
            "label": label,
            "precision": float(precision[label]),
            "recall": float(recall[label]),
            "f1": float(f1[label]),
            "support": int(support[label]),
        }
        for label in range(probabilities.shape[1])
    }

    return {
        "count": int(len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "top_bin": per_class[CLASS_NAMES[CLASS_LARGE]],
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=list(range(probabilities.shape[1])),
        ).tolist(),
        "class_order": [CLASS_NAMES[label] for label in range(probabilities.shape[1])],
        "target_counts": {
            CLASS_NAMES[label]: int(np.sum(targets == label))
            for label in range(probabilities.shape[1])
        },
        "predicted_counts": {
            CLASS_NAMES[label]: int(np.sum(predictions == label))
            for label in range(probabilities.shape[1])
        },
    }


def write_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def evaluate_regressor_predictions(
    path: Path,
    metrics_path: Path,
    plot_path: Path | None = None,
) -> dict[str, float | int | None]:
    rows = load_regression_predictions(path)
    targets = np.array([target for target, _ in rows.values()], dtype=float)
    predictions = np.array([prediction for _, prediction in rows.values()], dtype=float)
    metrics = compute_metrics(targets, predictions)
    write_json(metrics_path, metrics)
    if plot_path is not None:
        make_plot(targets, predictions, metrics, plot_path)
    return metrics


def build_routing_diagnostic(
    classifier_rows: dict[str, dict[str, object]],
    regressor_rows: dict[str, tuple[float, float]],
) -> dict[str, object]:
    selected_ids = [
        material_id
        for material_id, row in classifier_rows.items()
        if int(row["predicted_class"]) == CLASS_LARGE
    ]
    if not selected_ids:
        return {"count": 0, "metrics": None}
    missing_ids = [material_id for material_id in selected_ids if material_id not in regressor_rows]
    if missing_ids:
        raise KeyError(f"Missing regressor diagnostic predictions for IDs: {missing_ids[:5]}")
    targets = np.array([regressor_rows[material_id][0] for material_id in selected_ids], dtype=float)
    predictions = np.array([regressor_rows[material_id][1] for material_id in selected_ids], dtype=float)
    return {
        "count": int(len(selected_ids)),
        "metrics": compute_metrics(targets, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a four-bin multiclass classifier and a top-bin regressor for magnetization."
    )
    parser.add_argument("source_root", nargs="?", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("summary_csv", nargs="?", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("runs_root", nargs="?", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--resume_run_dir", type=Path, default=None)
    args = parser.parse_args()

    records = collect_records(args.summary_csv, args.source_root)
    if args.max_records is not None and len(records) > args.max_records:
        rng = random.Random(args.seed)
        records = rng.sample(records, args.max_records)

    record_map = {record.material_id: record for record in records}
    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        split_dir = run_dir / "splits"
        split_map = load_split_map(split_dir)
        top_bin_split_ids = load_top_bin_split_map(split_dir)
    else:
        split_map = split_ids(records, random_seed=args.seed)
        top_bin_split_ids = {
            split_name: subset_ids_by_class(ids, record_map, CLASS_LARGE)
            for split_name, ids in split_map.items()
        }
    classifier_class_weights, train_class_counts = compute_class_weights(records, split_map["train"])
    top_bin_records = [
        record for record in records if classify_magnetization(record.raw_magnetization) == CLASS_LARGE
    ]
    if not top_bin_records:
        raise RuntimeError("No top-bin records found with M > 1e-3.")
    empty_splits = [split_name for split_name, ids in top_bin_split_ids.items() if not ids]
    if empty_splits:
        raise RuntimeError("Top-bin regression split is empty for: " + ", ".join(sorted(empty_splits)))

    if args.resume_run_dir is None:
        run_dir = build_run_dir(args.runs_root)
    classifier_dir = run_dir / "classifier"
    top_regressor_dir = run_dir / "top_bin_regressor"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    ensure_stage_dataset(classifier_dir, records, mode="classification")
    ensure_stage_dataset(top_regressor_dir, top_bin_records, mode="regression")

    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_run_dir is None:
        for split_name, ids in split_map.items():
            write_split_csv(split_dir / f"{split_name}_ids.csv", ids)
        for split_name, ids in top_bin_split_ids.items():
            write_split_csv(split_dir / f"top_bin_{split_name}_ids.csv", ids)

    classifier_resume, classifier_completed = resolve_stage_resume(classifier_dir, args.epochs)
    if classifier_completed:
        classifier_model = classifier_resume
    else:
        classifier_model = run_training(
            classifier_dir,
            task="classification",
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            train_ids=split_map["train"],
            val_ids=split_map["val"],
            test_ids=split_map["test"],
            n_classes=NUM_CLASSES,
            class_weights=classifier_class_weights,
            classification_metric="class_f1",
            classification_metric_class_index=CLASS_LARGE,
            resume=classifier_resume,
        )

    top_regressor_resume, top_regressor_completed = resolve_stage_resume(top_regressor_dir, args.epochs)
    if top_regressor_completed:
        top_regressor_model = top_regressor_resume
    else:
        top_regressor_model = run_training(
            top_regressor_dir,
            task="regression",
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            train_ids=top_bin_split_ids["train"],
            val_ids=top_bin_split_ids["val"],
            test_ids=top_bin_split_ids["test"],
            resume=top_regressor_resume,
        )
    if classifier_model is None or top_regressor_model is None:
        raise RuntimeError("Training did not produce the expected checkpoints.")

    classifier_test_records = [record_map[material_id] for material_id in split_map["test"]]
    top_bin_test_records = [record_map[material_id] for material_id in top_bin_split_ids["test"]]
    classifier_test_dir = eval_dir / "classifier_test_dataset"
    top_bin_regressor_test_dir = eval_dir / "top_bin_regressor_test_dataset"
    diagnostic_regressor_test_dir = eval_dir / "diagnostic_regressor_test_dataset"

    write_stage_dataset(classifier_test_dir, classifier_test_records, mode="classification")
    write_stage_dataset(top_bin_regressor_test_dir, top_bin_test_records, mode="regression")
    write_stage_dataset(diagnostic_regressor_test_dir, classifier_test_records, mode="regression")

    classifier_predictions_csv = run_inference(
        dataset_dir=classifier_test_dir,
        model_path=classifier_model,
        task="classification",
        output_csv=eval_dir / "classifier_test_results.csv",
        batch_size=args.batch_size,
        workers=args.workers,
    )
    top_bin_regressor_predictions_csv = run_inference(
        dataset_dir=top_bin_regressor_test_dir,
        model_path=top_regressor_model,
        task="regression",
        output_csv=eval_dir / "top_bin_regressor_test_results.csv",
        batch_size=args.batch_size,
        workers=args.workers,
    )
    diagnostic_regressor_predictions_csv = run_inference(
        dataset_dir=diagnostic_regressor_test_dir,
        model_path=top_regressor_model,
        task="regression",
        output_csv=eval_dir / "diagnostic_regressor_test_results.csv",
        batch_size=args.batch_size,
        workers=args.workers,
    )

    classifier_rows = load_classification_predictions(classifier_predictions_csv)
    classifier_metrics = compute_classifier_metrics(classifier_rows)
    write_json(eval_dir / "classifier_metrics.json", classifier_metrics)

    top_bin_regressor_metrics = evaluate_regressor_predictions(
        top_bin_regressor_predictions_csv,
        eval_dir / "top_bin_regressor_metrics.json",
        plot_path=eval_dir / "top_bin_regressor_parity.png",
    )
    diagnostic_regressor_rows = load_regression_predictions(diagnostic_regressor_predictions_csv)
    routing_diagnostic = build_routing_diagnostic(classifier_rows, diagnostic_regressor_rows)
    write_json(eval_dir / "routing_diagnostic.json", routing_diagnostic)

    metadata = {
        "summary_csv": str(args.summary_csv),
        "source_root": str(args.source_root),
        "binning": {
            "class_0": "M == 0",
            "class_1": "0 < M <= 1e-6",
            "class_2": "1e-6 < M <= 1e-3",
            "class_3": "M > 1e-3",
        },
        "dataset_counts": {
            "all_records": len(records),
            "top_bin_records": len(top_bin_records),
            "all_class_counts": class_counts(records),
            "train_class_counts": train_class_counts,
            "top_bin_split_sizes": {split_name: len(ids) for split_name, ids in top_bin_split_ids.items()},
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "seed": args.seed,
            "resume_run_dir": str(args.resume_run_dir) if args.resume_run_dir is not None else None,
            "classifier_class_weights": classifier_class_weights,
            "classifier_selection_metric": "class_f1",
            "classifier_selection_class": CLASS_LARGE,
        },
        "evaluation": {
            "classifier_metrics": str(eval_dir / "classifier_metrics.json"),
            "top_bin_regressor_metrics": str(eval_dir / "top_bin_regressor_metrics.json"),
            "routing_diagnostic": str(eval_dir / "routing_diagnostic.json"),
        },
        "paths": {
            "classifier_model": str(classifier_model),
            "top_bin_regressor_model": str(top_regressor_model),
            "classifier_predictions": str(classifier_predictions_csv),
            "top_bin_regressor_predictions": str(top_bin_regressor_predictions_csv),
            "diagnostic_regressor_predictions": str(diagnostic_regressor_predictions_csv),
        },
        "primary_outputs": {
            "classifier_task": "multiclass_magnitude_classification",
            "regressor_task": "raw_regression_on_true_top_bin_only",
            "merged_predictions": None,
            "top_bin_regressor_summary": top_bin_regressor_metrics,
        },
    }
    write_json(run_dir / "run_metadata.json", metadata)
    print(run_dir)


if __name__ == "__main__":
    main()
