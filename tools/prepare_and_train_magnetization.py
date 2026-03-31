from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cgcnn.training import train_model


DEFAULT_SOURCE_ROOT = Path("/Users/qz/Downloads/cifs")
DEFAULT_SUMMARY_CSV = DEFAULT_SOURCE_ROOT / "mp_all_summary.csv"
DEFAULT_RUNS_ROOT = Path("/Users/qz/Downloads/runs")


warnings.filterwarnings(
    "ignore",
    message="Issues encountered while parsing CIF: .*",
)
warnings.filterwarnings(
    "ignore",
    message=".*not find enough neighbors to build graph.*",
)


def apply_transform(value: float, transform: str) -> float:
    if transform == "raw":
        return value
    if transform == "sqrt":
        return math.sqrt(value)
    if transform == "cbrt":
        return value ** (1.0 / 3.0)
    raise ValueError(f"Unsupported transform: {transform}")


def build_run_dir(runs_root: Path, transform: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"cgcnn_magnetization_{transform}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collect_records(summary_csv: Path, source_root: Path) -> list[dict]:
    records: list[dict] = []
    missing_dirs: list[str] = []
    missing_cifs: list[str] = []
    bad_rows: list[str] = []

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
                total_magnetization = float(row["total_magnetization"])
                volume = float(row["volume"])
            except (TypeError, ValueError):
                bad_rows.append(material_id)
                continue

            if volume == 0:
                bad_rows.append(material_id)
                continue

            cif_dir = source_root / material_id
            if not cif_dir.is_dir():
                missing_dirs.append(material_id)
                continue

            cif_files = sorted(cif_dir.glob("*.cif"))
            if len(cif_files) != 1:
                missing_cifs.append(material_id)
                continue

            raw_m = total_magnetization / volume
            records.append(
                {
                    "material_id": material_id,
                    "raw_m": raw_m,
                    "cif_path": cif_files[0],
                }
            )

    problems: list[str] = []
    if bad_rows:
        problems.append(f"bad rows: {len(bad_rows)}")
    if missing_dirs:
        problems.append(f"missing material directories: {len(missing_dirs)}")
    if missing_cifs:
        problems.append(f"directories without exactly one CIF: {len(missing_cifs)}")
    if problems:
        raise RuntimeError("Dataset preparation failed: " + ", ".join(problems))

    if not records:
        raise RuntimeError("No valid records were found in the summary CSV.")

    return records


def write_targets(records: list[dict], run_dir: Path, transform: str) -> None:
    id_prop_path = run_dir / "id_prop.csv"
    transformed_path = run_dir / "magnetization_per_volume.csv"

    with id_prop_path.open("w", newline="") as id_prop_handle:
        writer = csv.writer(id_prop_handle)
        for record in records:
            transformed_m = apply_transform(record["raw_m"], transform)
            record["transformed_m"] = transformed_m
            writer.writerow([record["material_id"], f"{transformed_m:.16g}"])

    with transformed_path.open("w", newline="") as transformed_handle:
        writer = csv.writer(transformed_handle)
        writer.writerow(
            ["material_id", "magnetization_per_volume", f"{transform}_magnetization"]
        )
        for record in records:
            writer.writerow(
                [
                    record["material_id"],
                    f"{record['raw_m']:.16g}",
                    f"{record['transformed_m']:.16g}",
                ]
            )


def write_atom_init(run_dir: Path) -> None:
    atom_init = {}
    width = 118
    for atomic_number in range(1, width + 1):
        vector = [0] * width
        vector[atomic_number - 1] = 1
        atom_init[str(atomic_number)] = vector

    with (run_dir / "atom_init.json").open("w") as handle:
        json.dump(atom_init, handle)


def link_cifs(records: list[dict], run_dir: Path) -> None:
    for record in records:
        link_path = run_dir / f"{record['material_id']}.cif"
        os.symlink(record["cif_path"], link_path)


def write_metadata(
    records: list[dict],
    run_dir: Path,
    summary_csv: Path,
    source_root: Path,
    transform: str,
    epochs: int,
    batch_size: int,
    workers: int,
) -> None:
    raw_values = [record["raw_m"] for record in records]
    metadata = {
        "summary_csv": str(summary_csv),
        "source_root": str(source_root),
        "dataset_size": len(records),
        "target": "total_magnetization / volume",
        "target_transform": transform,
        "value_range": {
            "min_M": min(raw_values),
            "max_M": max(raw_values),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "cuda": False,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
        },
    }
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


def prepare_dataset(
    source_root: Path,
    summary_csv: Path,
    runs_root: Path,
    transform: str,
    epochs: int,
    batch_size: int,
    workers: int,
) -> Path:
    print(f"Preparing dataset from {summary_csv}", flush=True)
    records = collect_records(summary_csv, source_root)
    print(f"Validated {len(records)} materials", flush=True)

    run_dir = build_run_dir(runs_root, transform)
    write_targets(records, run_dir, transform)
    write_atom_init(run_dir)
    link_cifs(records, run_dir)
    write_metadata(
        records=records,
        run_dir=run_dir,
        summary_csv=summary_csv,
        source_root=source_root,
        transform=transform,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
    )
    print(f"Prepared run directory: {run_dir}", flush=True)
    return run_dir


def train(run_dir: Path, epochs: int, batch_size: int, workers: int) -> Path | None:
    print("Starting CGCNN training", flush=True)
    os.chdir(run_dir)
    resume_path = run_dir / "checkpoint.pth.tar"
    resume = str(resume_path) if resume_path.exists() else None
    best_model = train_model(
        root_dir=str(run_dir),
        task="regression",
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        cuda=False,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        print_freq=20,
        resume=resume,
    )
    print(f"Training finished. Best model: {best_model}", flush=True)
    return Path(best_model) if best_model else None


if __name__ == "__main__":
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE_ROOT
    summary_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SUMMARY_CSV
    runs_root = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_RUNS_ROOT
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    transform = sys.argv[7] if len(sys.argv) > 7 else "raw"

    run_dir = prepare_dataset(
        source_root=source_root,
        summary_csv=summary_csv,
        runs_root=runs_root,
        transform=transform,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
    )
    train(run_dir=run_dir, epochs=epochs, batch_size=batch_size, workers=workers)
