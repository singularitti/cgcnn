from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

for warning_message in (
    r".*not find enough neighbors to build graph.*",
    r".*Issues encountered while parsing CIF.*",
    r".*No Pauling electronegativity.*",
):
    warnings.filterwarnings("ignore", message=warning_message, category=UserWarning)

os.environ["PYTHONWARNINGS"] = ",".join(
    [
        "ignore:.*not find enough neighbors to build graph.*:UserWarning",
        "ignore:.*Issues encountered while parsing CIF.*:UserWarning",
        "ignore:.*No Pauling electronegativity.*:UserWarning",
    ]
)

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cgcnn.device import get_env_device
from cgcnn.training import train_model


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return Path(value).expanduser().resolve()


def append_log(run_dir: Path, message: str) -> None:
    path = run_dir / "RUN_LOG.md"
    with path.open("a") as handle:
        handle.write(f"- `{now_iso()}` {message}\n")


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def read_records(source_csv: Path, cif_root: Path) -> tuple[list[tuple[str, float]], dict]:
    records: list[tuple[str, float]] = []
    total_rows = 0
    valid_numbers = 0
    invalid_numbers = 0
    missing_cifs = 0
    missing_examples: list[str] = []

    with source_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"material_id", "energy_above_hull"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Missing required CSV columns: {sorted(missing)}")
        for row in reader:
            total_rows += 1
            material_id = (row.get("material_id") or "").strip()
            value = parse_float(row.get("energy_above_hull") or "")
            if value is None:
                invalid_numbers += 1
                continue
            valid_numbers += 1
            cif_path = cif_root / f"{material_id}.cif"
            if not material_id or not cif_path.is_file():
                missing_cifs += 1
                if len(missing_examples) < 20:
                    missing_examples.append(str(cif_path))
                continue
            records.append((material_id, value))

    return records, {
        "total_csv_data_rows": total_rows,
        "valid_energy_above_hull_rows": valid_numbers,
        "invalid_energy_above_hull_rows": invalid_numbers,
        "rows_with_valid_target_and_existing_cif": len(records),
        "missing_cif_rows": missing_cifs,
        "missing_cif_examples": missing_examples,
    }


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        raise FileExistsError(f"Refusing to replace symlink: {link_path}")
    if link_path.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link_path}")
    link_path.symlink_to(target_path)


def write_split(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id"])
        writer.writerows([[item] for item in ids])


def prepare_dataset(run_dir: Path, source_csv: Path, cif_root: Path, atom_init: Path) -> dict:
    prepared_marker = run_dir / ".dataset_prepared"
    stats_path = run_dir / "id_prop_stats.json"
    if prepared_marker.exists() and stats_path.exists():
        append_log(run_dir, "Dataset preparation already exists; reusing staged inputs.")
        return json.loads(stats_path.read_text())

    append_log(run_dir, "Reading source CSV and validating energy_above_hull values.")
    records, stats = read_records(source_csv, cif_root)
    if not records:
        raise RuntimeError("No rows had both a valid target and a matching CIF file.")

    append_log(run_dir, f"Writing id_prop.csv with {len(records)} full-dataset rows.")
    with (run_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(records)

    ensure_symlink(run_dir / "atom_init.json", atom_init)
    append_log(run_dir, "Symlinking CIF inputs from WORK into the SCRATCH run folder.")
    for index, (material_id, _) in enumerate(records, start=1):
        ensure_symlink(run_dir / f"{material_id}.cif", cif_root / f"{material_id}.cif")
        if index % 25000 == 0:
            append_log(run_dir, f"Symlinked {index} CIF files.")

    seed = int(os.environ.get("SPLIT_SEED", "20260624"))
    train_ratio = float(os.environ.get("TRAIN_RATIO", "0.8"))
    val_ratio = float(os.environ.get("VAL_RATIO", "0.1"))
    test_ratio = float(os.environ.get("TEST_RATIO", "0.1"))
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise RuntimeError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    ids = [material_id for material_id, _ in records]
    random.Random(seed).shuffle(ids)
    train_count = int(len(ids) * train_ratio)
    val_count = int(len(ids) * val_ratio)
    train_ids = ids[:train_count]
    val_ids = ids[train_count : train_count + val_count]
    test_ids = ids[train_count + val_count :]

    split_dir = run_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    write_split(split_dir / "train_ids.csv", train_ids)
    write_split(split_dir / "val_ids.csv", val_ids)
    write_split(split_dir / "test_ids.csv", test_ids)

    stats.update(
        {
            "split_seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
        }
    )
    stats_path.write_text(json.dumps(stats, indent=2))
    prepared_marker.write_text(now_iso() + "\n")
    append_log(run_dir, "Dataset preparation complete.")
    return stats


def read_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return [row[0] for row in reader if row]


def write_readme(run_dir: Path, source_csv: Path, cif_root: Path, atom_init: Path, copy_back_dir: Path, stats: dict) -> None:
    metadata = {
        "purpose": "Train a CGCNN regression model on Materials Project CIFs using energy_above_hull.",
        "created_at": now_iso(),
        "run_dir": str(run_dir),
        "copy_back_dir": str(copy_back_dir),
        "repo_root": str(REPO_ROOT),
        "source_csv": str(source_csv),
        "cif_root": str(cif_root),
        "atom_init_source": str(atom_init),
        "target_column": "energy_above_hull",
        "id_column": "material_id",
        "input_provenance": {
            "id_prop.csv": "Generated from source_csv rows with finite energy_above_hull and matching CIF files.",
            "atom_init.json": "Symlink to data/sample-regression/atom_init.json in the repo.",
            "cif_files": "Symlinks to flat CIF files in the WORK mp-cif folder.",
        },
        "outputs": [
            "model_best.pth.tar",
            "checkpoint.pth.tar",
            "training_history.json",
            "test_results.csv",
            "run_metadata.json",
            "id_prop_stats.json",
            "RUN_LOG.md",
            "slurm-*.out",
            "slurm-*.err",
        ],
        "counts": stats,
        "training_options": {
            "epochs": int(os.environ.get("EPOCHS", "1")),
            "batch_size": int(os.environ.get("BATCH_SIZE", "256")),
            "workers": int(os.environ.get("WORKERS", "24")),
            "optimizer": os.environ.get("OPTIM", "SGD"),
            "lr": float(os.environ.get("LR", "0.02")),
            "atom_fea_len": int(os.environ.get("ATOM_FEA_LEN", "64")),
            "h_fea_len": int(os.environ.get("H_FEA_LEN", "32")),
            "n_conv": int(os.environ.get("N_CONV", "4")),
            "n_h": int(os.environ.get("N_H", "1")),
            "weight_decay": float(os.environ.get("WEIGHT_DECAY", "0.0")),
            "initialize_from": os.environ.get("INITIALIZE_FROM", ""),
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "README.md").write_text(
        f"""# CGCNN energy_above_hull training run

Created: `{metadata["created_at"]}`

This SCRATCH folder stages and trains a CGCNN regression model for
`energy_above_hull` using the full valid row set from the Materials Project
summary CSV.

## Inputs

- Source CSV: `{source_csv}`
- ID column: `material_id`
- Target column: `energy_above_hull`
- CIF files: symlinks to `{cif_root}/<material_id>.cif`
- Atom features: symlink `{run_dir / "atom_init.json"}` -> `{atom_init}`
- Code: editable checkout at `{REPO_ROOT}`
- Pretrained reference/checkpoint, when used: `{os.environ.get("INITIALIZE_FROM", "")}`

## Outputs

- `id_prop.csv`: full CGCNN target table generated from valid source rows.
- `<material_id>.cif`: symlinked CIF inputs, not copied result artifacts.
- `splits/`: train/validation/test ID lists.
- `model_best.pth.tar`, `checkpoint.pth.tar`: trained model checkpoints.
- `training_history.json`: per-epoch validation metrics.
- `test_results.csv`: held-out predictions from the best model.
- `run_metadata.json`, `id_prop_stats.json`, `RUN_LOG.md`: provenance and audit logs.
- Slurm stdout/stderr logs.

Selected result artifacts are copied to `{copy_back_dir}`. Temporary staged CIF
symlinks are intentionally not copied back.
"""
    )


def copy_results(run_dir: Path, copy_back_dir: Path, status: str) -> None:
    copy_back_dir.mkdir(parents=True, exist_ok=True)
    result_names = [
        "README.md",
        "RUN_LOG.md",
        "run_metadata.json",
        "id_prop_stats.json",
        "training_history.json",
        "test_results.csv",
        "model_best.pth.tar",
        "checkpoint.pth.tar",
    ]
    for name in result_names:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, copy_back_dir / name)
    split_source = run_dir / "splits"
    if split_source.exists():
        split_target = copy_back_dir / "splits"
        if split_target.exists():
            shutil.rmtree(split_target)
        shutil.copytree(split_source, split_target)
    for pattern in ("slurm-*.out", "slurm-*.err"):
        for source in run_dir.glob(pattern):
            shutil.copy2(source, copy_back_dir / source.name)
    (copy_back_dir / "COPY_STATUS.txt").write_text(f"{status}\n{now_iso()}\n")


if __name__ == "__main__":
    run_dir = env_path("RUN_DIR")
    source_csv = env_path("SOURCE_CSV")
    cif_root = env_path("CIF_ROOT")
    atom_init = env_path("ATOM_INIT")
    copy_back_dir = env_path("COPY_BACK_DIR")

    run_dir.mkdir(parents=True, exist_ok=True)
    append_log(run_dir, "Started Stampede3 CGCNN energy_above_hull workflow.")
    status = "failed"
    try:
        stats = prepare_dataset(run_dir, source_csv, cif_root, atom_init)
        write_readme(run_dir, source_csv, cif_root, atom_init, copy_back_dir, stats)

        train_ids = read_ids(run_dir / "splits" / "train_ids.csv")
        val_ids = read_ids(run_dir / "splits" / "val_ids.csv")
        test_ids = read_ids(run_dir / "splits" / "test_ids.csv")

        os.chdir(run_dir)
        append_log(run_dir, "Starting CGCNN training.")
        start = time.time()
        best = train_model(
            root_dir=str(run_dir),
            task="regression",
            epochs=int(os.environ.get("EPOCHS", "1")),
            batch_size=int(os.environ.get("BATCH_SIZE", "256")),
            lr=float(os.environ.get("LR", "0.02")),
            optim_name=os.environ.get("OPTIM", "SGD"),
            atom_fea_len=int(os.environ.get("ATOM_FEA_LEN", "64")),
            h_fea_len=int(os.environ.get("H_FEA_LEN", "32")),
            n_conv=int(os.environ.get("N_CONV", "4")),
            n_h=int(os.environ.get("N_H", "1")),
            device=get_env_device(),
            workers=int(os.environ.get("WORKERS", "24")),
            weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.0")),
            momentum=float(os.environ.get("MOMENTUM", "0.9")),
            print_freq=int(os.environ.get("PRINT_FREQ", "100")),
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            initialize_from=os.environ.get("INITIALIZE_FROM") or None,
            checkpoint_dir=str(run_dir / "checkpoints"),
            metrics_history_path=str(run_dir / "training_history.json"),
        )
        append_log(run_dir, f"Training completed in {time.time() - start:.1f} seconds.")
        append_log(run_dir, f"Best checkpoint: {best}")

        metadata_path = run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["completed_at"] = now_iso()
        metadata["best_checkpoint"] = str(best)
        metadata_path.write_text(json.dumps(metadata, indent=2))

        missing = [
            name
            for name in ("model_best.pth.tar", "checkpoint.pth.tar", "test_results.csv")
            if not (run_dir / name).exists()
        ]
        if missing:
            raise RuntimeError(f"Missing expected training outputs: {missing}")
        status = "success"
        append_log(run_dir, "Copying selected result artifacts back to WORK.")
        copy_results(run_dir, copy_back_dir, status)
        append_log(run_dir, "Workflow finished successfully.")
    except Exception:
        append_log(run_dir, "Workflow failed; copying available diagnostics.")
        (run_dir / "ERROR.txt").write_text(traceback.format_exc())
        copy_results(run_dir, copy_back_dir, status)
        raise
