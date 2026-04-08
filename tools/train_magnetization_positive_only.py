from __future__ import annotations

# ruff: noqa: I001

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


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))


DEFAULT_SOURCE_ROOT = Path("/Users/qz/Downloads/cifs")
DEFAULT_SUMMARY_CSV = DEFAULT_SOURCE_ROOT / "mp_all_summary.csv"
DEFAULT_RUNS_ROOT = Path("/Users/qz/Downloads/runs")
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 10
DEFAULT_RANDOM_SEED = 123
DEFAULT_TARGET_TRANSFORM = "raw"
DEFAULT_MIN_MAGNETIZATION = 0.0


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
    raw_magnetization: float
    cif_path: Path


def apply_transform(value: float, transform: str) -> float:
    if transform == "raw":
        return value
    if transform == "sqrt":
        return math.sqrt(value)
    if transform == "cbrt":
        return value ** (1.0 / 3.0)
    if transform == "log":
        return math.log(value)
    if transform == "log10":
        return math.log10(value)
    raise ValueError(f"Unsupported transform: {transform}")


def build_run_dir(runs_root: Path, transform: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"cgcnn_magnetization_positive_only_{transform}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collect_positive_records(
    summary_csv: Path, source_root: Path, min_magnetization: float = DEFAULT_MIN_MAGNETIZATION
) -> list[MaterialRecord]:
    records: list[MaterialRecord] = []
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
                continue
            try:
                magnetization = float(row["total_magnetization"]) / float(row["volume"])
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if magnetization <= min_magnetization:
                continue
            cif_dir = source_root / material_id
            if not cif_dir.is_dir():
                continue
            cif_files = sorted(cif_dir.glob("*.cif"))
            if len(cif_files) != 1:
                continue
            records.append(
                MaterialRecord(
                    material_id=material_id,
                    raw_magnetization=magnetization,
                    cif_path=cif_files[0],
                )
            )
    if not records:
        raise RuntimeError(
            f"No records found with magnetization_per_volume > {min_magnetization:.16g}."
        )
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


def write_atom_init(path: Path) -> None:
    atom_init = {}
    width = 118
    for atomic_number in range(1, width + 1):
        vector = [0] * width
        vector[atomic_number - 1] = 1
        atom_init[str(atomic_number)] = vector
    with path.open("w") as handle:
        json.dump(atom_init, handle)


def write_dataset(run_dir: Path, records: list[MaterialRecord], transform: str) -> None:
    write_atom_init(run_dir / "atom_init.json")
    with (run_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for record in records:
            writer.writerow(
                [record.material_id, f"{apply_transform(record.raw_magnetization, transform):.16g}"]
            )
    with (run_dir / "magnetization_positive_only.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "material_id",
                "magnetization_per_volume",
                f"{transform}_magnetization_per_volume",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.material_id,
                    f"{record.raw_magnetization:.16g}",
                    f"{apply_transform(record.raw_magnetization, transform):.16g}",
                ]
            )
    for record in records:
        os.symlink(record.cif_path, run_dir / f"{record.material_id}.cif")


def write_split_csv(path: Path, material_ids: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id"])
        for material_id in material_ids:
            writer.writerow([material_id])


def main() -> None:
    from analyze_parity import compute_metrics, invert_transform, make_loglog_plot, make_plot
    from cgcnn.benchmark import format_metrics_report, run_benchmark
    from cgcnn.data import CIFData
    from cgcnn.inference import predict_model
    from cgcnn.training import train_model

    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE_ROOT
    summary_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SUMMARY_CSV
    runs_root = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_RUNS_ROOT
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_EPOCHS
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_BATCH_SIZE
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_WORKERS
    target_transform = (
        sys.argv[7] if len(sys.argv) > 7 else DEFAULT_TARGET_TRANSFORM
    )
    min_magnetization = (
        float(sys.argv[8]) if len(sys.argv) > 8 else DEFAULT_MIN_MAGNETIZATION
    )

    records = collect_positive_records(
        summary_csv, source_root, min_magnetization=min_magnetization
    )
    split_map = split_ids(records)

    run_dir = build_run_dir(runs_root, target_transform)
    write_dataset(run_dir, records, target_transform)
    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=False)
    for split_name, material_ids in split_map.items():
        write_split_csv(splits_dir / f"{split_name}_ids.csv", material_ids)

    previous_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        best_model = train_model(
            root_dir=str(run_dir),
            task="regression",
            epochs=epochs,
            batch_size=batch_size,
            workers=workers,
            cuda=False,
            train_ids=split_map["train"],
            val_ids=split_map["val"],
            test_ids=split_map["test"],
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            print_freq=20,
            checkpoint_dir=str(run_dir / "checkpoints"),
            metrics_history_path=str(run_dir / "training_history.json"),
        )
        if best_model is None:
            raise RuntimeError("Training did not produce a checkpoint.")
        test_dataset = CIFData(
            str(run_dir),
            shuffle=False,
            include_ids=split_map["test"],
        )
        predict_model(
            dataset=test_dataset,
            task="regression",
            modelpath=str(best_model),
            batch_size=batch_size,
            workers=workers,
            cuda=False,
            print_freq=20,
            shuffle=False,
            output_csv=str(run_dir / "test_results.csv"),
        )
    finally:
        os.chdir(previous_cwd)

    import numpy as np

    ids = []
    targets = []
    predictions = []
    with (run_dir / "test_results.csv").open(newline="") as handle:
        for material_id, target, prediction in csv.reader(handle):
            ids.append(material_id)
            targets.append(float(target))
            predictions.append(float(prediction))
    targets_array = invert_transform(np.array(targets), target_transform)
    predictions_array = invert_transform(np.array(predictions), target_transform)
    raw_results_csv = run_dir / "test_results_raw.csv"
    with raw_results_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for material_id, target, prediction in zip(ids, targets_array, predictions_array):
            writer.writerow([material_id, f"{target:.16g}", f"{prediction:.16g}"])

    metrics = compute_metrics(targets_array, predictions_array)
    with (run_dir / "parity_metrics.json").open("w") as handle:
        json.dump({**metrics, "target_transform": target_transform}, handle, indent=2)
    make_plot(targets_array, predictions_array, metrics, run_dir / "parity_plot.png")
    make_loglog_plot(
        targets_array,
        predictions_array,
        metrics,
        run_dir / "parity_plot_loglog.png",
    )

    _, benchmark_metrics = run_benchmark(str(raw_results_csv))
    with (run_dir / "benchmark_metrics.json").open("w") as handle:
        json.dump(benchmark_metrics, handle, indent=2)
    with (run_dir / "benchmark_report.txt").open("w") as handle:
        handle.write(format_metrics_report(benchmark_metrics) + "\n")

    metadata = {
        "summary_csv": str(summary_csv),
        "source_root": str(source_root),
        "dataset_size": len(records),
        "filter": f"M > {min_magnetization:.16g}",
        "target_transform": target_transform,
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
        "value_range": {
            "min_M": min(record.raw_magnetization for record in records),
            "max_M": max(record.raw_magnetization for record in records),
        },
    }
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(run_dir)


if __name__ == "__main__":
    main()
