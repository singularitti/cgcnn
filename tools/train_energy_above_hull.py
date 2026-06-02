from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from cgcnn.training import train_model


DEFAULT_SOURCE_CSV = Path.home() / "Downloads" / "cifs" / "mp_all_summary.csv"
DEFAULT_CIF_ROOT = Path.home() / "Downloads" / "cifs"
DEFAULT_ATOM_INIT = REPO_ROOT / "data" / "sample-regression" / "atom_init.json"
JOB_TAG_SCRIPT = (
    Path.home()
    / ".ghq"
    / "github.com"
    / "singularitti"
    / ".agents"
    / "skills"
    / "job-tags"
    / "scripts"
    / "job_tag.sh"
)


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        with redirect_stdout(Tee(sys.stdout, handle)), redirect_stderr(
            Tee(sys.stderr, handle)
        ):
            yield


def tag_job(status: str, *paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if not existing:
        return
    try:
        subprocess.run(
            [str(JOB_TAG_SCRIPT), status, *existing],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[job-tags] could not mark {status}: {exc}", file=sys.stderr)


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def read_records(source_csv: Path, cif_root: Path) -> tuple[list[tuple[str, float]], dict]:
    records: list[tuple[str, float]] = []
    skipped_empty_or_nan = 0
    skipped_missing_cif = 0
    missing_examples: list[str] = []

    with source_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"material_id", "energy_above_hull"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        for row in reader:
            material_id = (row.get("material_id") or "").strip()
            value = parse_float(row.get("energy_above_hull") or "")
            if not material_id or value is None:
                skipped_empty_or_nan += 1
                continue
            cif_path = cif_root / material_id / f"{material_id}.cif"
            if not cif_path.is_file():
                skipped_missing_cif += 1
                if len(missing_examples) < 20:
                    missing_examples.append(str(cif_path))
                continue
            records.append((material_id, value))

    stats = {
        "source_csv_rows_with_usable_target_and_cif": len(records),
        "skipped_empty_or_nan_energy_above_hull": skipped_empty_or_nan,
        "skipped_missing_cif": skipped_missing_cif,
        "missing_cif_examples": missing_examples,
    }
    return records, stats


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        raise FileExistsError(f"Refusing to replace existing symlink: {link_path}")
    if link_path.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link_path}")
    link_path.symlink_to(target_path)


def write_csv_ids(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id"])
        writer.writerows([[item] for item in ids])


def prepare_run_dir(
    run_dir: Path,
    source_csv: Path,
    cif_root: Path,
    atom_init: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    args_payload: dict,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=False)
    tag_job("pending", run_dir)

    records, stats = read_records(source_csv, cif_root)
    if not records:
        raise RuntimeError("No usable energy_above_hull rows with matching CIF files.")

    with (run_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(records)

    ensure_symlink(run_dir / "atom_init.json", atom_init)
    for material_id, _ in records:
        ensure_symlink(
            run_dir / f"{material_id}.cif",
            cif_root / material_id / f"{material_id}.cif",
        )

    ids = [material_id for material_id, _ in records]
    rng = random.Random(seed)
    rng.shuffle(ids)
    total = len(ids)
    train_count = int(train_ratio * total)
    val_count = int(val_ratio * total)
    test_count = total - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            "Split ratios produced an empty split: "
            f"train={train_count}, val={val_count}, test={test_count}"
        )

    split_dir = run_dir / "splits"
    split_dir.mkdir()
    train_ids = ids[:train_count]
    val_ids = ids[train_count : train_count + val_count]
    test_ids = ids[train_count + val_count :]
    write_csv_ids(split_dir / "train_ids.csv", train_ids)
    write_csv_ids(split_dir / "val_ids.csv", val_ids)
    write_csv_ids(split_dir / "test_ids.csv", test_ids)

    metadata = {
        "purpose": "Train a CGCNN regression model for energy_above_hull from mp_all_summary.csv.",
        "created_at": datetime.now().astimezone().isoformat(),
        "workspace": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "source_csv": str(source_csv),
        "source_columns": {
            "id": "material_id",
            "target": "energy_above_hull",
        },
        "target_transform": "raw",
        "cif_root": str(cif_root),
        "cif_provenance": "Each top-level <material_id>.cif in the run folder is a symlink to ~/Downloads/cifs/<material_id>/<material_id>.cif.",
        "atom_init": {
            "link": str(run_dir / "atom_init.json"),
            "target": str(atom_init),
            "provenance": "Symlink to the repo sample-regression atom initialization file.",
        },
        "counts": {
            **stats,
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "split_files": {
            "train_ids": str(split_dir / "train_ids.csv"),
            "val_ids": str(split_dir / "val_ids.csv"),
            "test_ids": str(split_dir / "test_ids.csv"),
        },
        "command_line_options": args_payload,
        "outputs": {
            "id_prop_csv": str(run_dir / "id_prop.csv"),
            "checkpoints": str(run_dir / "checkpoints"),
            "training_history": str(run_dir / "training_history.json"),
            "test_results": str(run_dir / "test_results.csv"),
            "epoch_parity": str(run_dir / "epoch_parity"),
            "log": str(run_dir / "run.log"),
        },
    }
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    readme = f"""# CGCNN energy_above_hull run

Created: `{metadata["created_at"]}`

This folder trains a CPU-only CGCNN regression model to predict `energy_above_hull`
from `/Users/qz/Downloads/cifs/mp_all_summary.csv`.

## Inputs

- Source CSV: `{source_csv}`
- ID column: `material_id`
- Target column: `energy_above_hull`
- CIF inputs: symlinked from `{cif_root}/<material_id>/<material_id>.cif`
- Atom features: symlink `{run_dir / "atom_init.json"}` -> `{atom_init}`
- Split seed: `{seed}`
- Split counts: train `{len(train_ids)}`, validation `{len(val_ids)}`, test `{len(test_ids)}`
- Training options: see `run_metadata.json`

Rows with empty or NaN `energy_above_hull` were excluded. Rows whose expected
CIF file was missing were also excluded.

## Outputs

- `id_prop.csv`: CGCNN dataset target table.
- `<material_id>.cif`: symlinks to the source CIF files.
- `splits/`: explicit train/validation/test material IDs.
- `checkpoints/`: per-epoch model checkpoints.
- `checkpoint.pth.tar` and `model_best.pth.tar`: latest and best checkpoints.
- `training_history.json`: validation MAE history written during training.
- `test_results.csv`: held-out predictions for the best model.
- `epoch_parity/`: per-epoch test predictions, metrics, and parity plots.
- `run.log`: combined workflow log.
"""
    (run_dir / "README.md").write_text(readme)
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and train a CGCNN energy_above_hull regression run."
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--cif-root", type=Path, default=DEFAULT_CIF_ROOT)
    parser.add_argument("--atom-init", type=Path, default=DEFAULT_ATOM_INIT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "runs" / f"mp_all_ehull_cpu_{timestamp}",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optim", default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--atom-fea-len", type=int, default=96)
    parser.add_argument("--h-fea-len", type=int, default=256)
    parser.add_argument("--n-conv", type=int, default=4)
    parser.add_argument("--n-h", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.source_csv = args.source_csv.expanduser().resolve()
    args.cif_root = args.cif_root.expanduser().resolve()
    args.atom_init = args.atom_init.expanduser().resolve()

    if args.run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {args.run_dir}")
    if not args.source_csv.is_file():
        raise FileNotFoundError(args.source_csv)
    if not args.cif_root.is_dir():
        raise FileNotFoundError(args.cif_root)
    if not args.atom_init.is_file():
        raise FileNotFoundError(args.atom_init)
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    args_payload = vars(args).copy()
    args_payload = {key: str(value) if isinstance(value, Path) else value for key, value in args_payload.items()}

    try:
        prepared = prepare_run_dir(
            run_dir=args.run_dir,
            source_csv=args.source_csv,
            cif_root=args.cif_root,
            atom_init=args.atom_init,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            args_payload=args_payload,
        )
        with tee_log(args.run_dir / "run.log"):
            print(json.dumps(prepared["metadata"], indent=2))
            tag_job("running", args.run_dir)
            os.chdir(args.run_dir)
            best_checkpoint = train_model(
                root_dir=str(args.run_dir),
                task="regression",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                optim_name=args.optim,
                atom_fea_len=args.atom_fea_len,
                h_fea_len=args.h_fea_len,
                n_conv=args.n_conv,
                n_h=args.n_h,
                cuda=False,
                workers=args.workers,
                weight_decay=args.weight_decay,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                train_ids=prepared["train_ids"],
                val_ids=prepared["val_ids"],
                test_ids=prepared["test_ids"],
                checkpoint_dir=str(args.run_dir / "checkpoints"),
                metrics_history_path=str(args.run_dir / "training_history.json"),
                print_freq=args.print_freq,
            )
            print(f"Best checkpoint: {best_checkpoint}")

            from generate_epoch_parity_plots import generate_plots_for_run

            generate_plots_for_run(
                args.run_dir,
                batch_size=args.batch_size,
                workers=args.workers,
                cuda=False,
                checkpoint_dir=args.run_dir / "checkpoints",
            )

            metadata_path = args.run_dir / "run_metadata.json"
            with metadata_path.open() as handle:
                metadata = json.load(handle)
            metadata["completed_at"] = datetime.now().astimezone().isoformat()
            metadata["best_checkpoint"] = str(best_checkpoint)
            with metadata_path.open("w") as handle:
                json.dump(metadata, handle, indent=2)

            required_outputs = [
                args.run_dir / "id_prop.csv",
                args.run_dir / "model_best.pth.tar",
                args.run_dir / "test_results.csv",
                args.run_dir / "epoch_parity" / "epoch_parity_summary.json",
            ]
            missing = [str(path) for path in required_outputs if not path.exists()]
            if missing:
                raise RuntimeError(f"Missing expected outputs: {missing}")
            tag_job("finished", args.run_dir)
            print(f"Finished run: {args.run_dir}")
    except Exception:
        if args.run_dir.exists():
            tag_job("failed", args.run_dir)
        raise


if __name__ == "__main__":
    main()
