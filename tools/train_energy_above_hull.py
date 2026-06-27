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

from cgcnn.device import get_env_device
from cgcnn.training import train_model


DEFAULT_SOURCE_CSV = Path.home() / "run" / "mp-cif" / "mp_all_summary.csv"
DEFAULT_CIF_ROOT = Path.home() / "run" / "mp-cif"
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
    if not math.isfinite(parsed):
        return None
    return parsed


def transform_target(value: float, transform: str, scale: float) -> float:
    if transform == "raw":
        return value
    if transform == "scaled_log1p":
        if value < 0.0:
            raise ValueError("scaled_log1p requires non-negative targets.")
        if scale <= 0.0:
            raise ValueError("--target-transform-scale must be positive.")
        return math.log1p(value / scale)
    raise ValueError(f"Unsupported target transform: {transform}")


def transform_metadata(transform: str, scale: float) -> dict[str, object]:
    payload: dict[str, object] = {"target_transform": transform}
    if transform == "scaled_log1p":
        payload.update(
            {
                "target_transform_formula": "z = log1p(energy_above_hull / scale)",
                "target_transform_inverse_formula": (
                    "energy_above_hull = scale * expm1(max(z, 0))"
                ),
                "target_transform_scale_eV_per_atom": scale,
                "prediction_inverse_lower_bound": 0.0,
            }
        )
    return payload


def candidate_cif_paths(cif_root: Path, material_id: str) -> tuple[Path, Path]:
    return (
        cif_root / f"{material_id}.cif",
        cif_root / material_id / f"{material_id}.cif",
    )


def find_cif_path(cif_root: Path, material_id: str) -> Path | None:
    for path in candidate_cif_paths(cif_root, material_id):
        if path.is_file():
            return path
    return None


def read_records(source_csv: Path, cif_root: Path) -> tuple[list[tuple[str, float, Path]], dict]:
    records: list[tuple[str, float, Path]] = []
    total_rows = 0
    valid_numeric_energy_above_hull = 0
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
            total_rows += 1
            material_id = (row.get("material_id") or "").strip()
            value = parse_float(row.get("energy_above_hull") or "")
            if not material_id or value is None:
                skipped_empty_or_nan += 1
                continue
            valid_numeric_energy_above_hull += 1
            cif_path = find_cif_path(cif_root, material_id)
            if cif_path is None:
                skipped_missing_cif += 1
                if len(missing_examples) < 20:
                    flat_path, nested_path = candidate_cif_paths(cif_root, material_id)
                    missing_examples.append(f"{flat_path} or {nested_path}")
                continue
            records.append((material_id, value, cif_path))

    stats = {
        "source_csv_rows": total_rows,
        "valid_numeric_energy_above_hull": valid_numeric_energy_above_hull,
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


def write_id_prop(path: Path, records: list[tuple[str, float, Path]], transform: str, scale: float) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for material_id, value, _ in records:
            writer.writerow([material_id, f"{transform_target(value, transform, scale):.16g}"])


def write_physical_results(
    transformed_results_csv: Path,
    physical_results_csv: Path,
    transform: str,
    scale: float,
) -> None:
    rows: list[tuple[str, float, float]] = []
    with transformed_results_csv.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            material_id = row[0]
            target = transform_target_inverse(float(row[1]), transform, scale)
            prediction = transform_target_inverse(float(row[2]), transform, scale)
            rows.append((material_id, target, prediction))
    if not rows:
        raise ValueError(f"No usable rows found in {transformed_results_csv}")
    with physical_results_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            (material_id, f"{target:.16g}", f"{prediction:.16g}")
            for material_id, target, prediction in rows
        )


def transform_target_inverse(value: float, transform: str, scale: float) -> float:
    if transform == "raw":
        return value
    if transform == "scaled_log1p":
        return scale * math.expm1(max(value, 0.0))
    raise ValueError(f"Unsupported target transform: {transform}")


def read_csv_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if value.lower() == "material_id":
                continue
            ids.append(value)
    if not ids:
        raise ValueError(f"No IDs found in {path}")
    return ids


def load_prepared_run(run_dir: Path) -> dict:
    metadata_path = run_dir / "run_metadata.json"
    split_dir = run_dir / "splits"
    required_paths = [
        run_dir / "id_prop.csv",
        run_dir / "atom_init.json",
        metadata_path,
        split_dir / "train_ids.csv",
        split_dir / "val_ids.csv",
        split_dir / "test_ids.csv",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Prepared run is incomplete: {missing}")
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    return {
        "train_ids": read_csv_ids(split_dir / "train_ids.csv"),
        "val_ids": read_csv_ids(split_dir / "val_ids.csv"),
        "test_ids": read_csv_ids(split_dir / "test_ids.csv"),
        "metadata": metadata,
    }


def prepare_run_dir(
    run_dir: Path,
    source_csv: Path,
    cif_root: Path,
    atom_init: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    target_transform: str,
    target_transform_scale: float,
    create_cif_symlinks: bool,
    args_payload: dict,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = [
        run_dir / "id_prop.csv",
        run_dir / "run_metadata.json",
        run_dir / "README.md",
        run_dir / "TRAINING_LOG.md",
        run_dir / "splits",
        run_dir / "checkpoint.pth.tar",
        run_dir / "model_best.pth.tar",
        run_dir / "checkpoints",
        run_dir / "epoch_parity",
    ]
    existing_generated = [
        str(path) for path in generated_paths if path.exists() or path.is_symlink()
    ]
    if existing_generated:
        raise FileExistsError(
            "Refusing to overwrite existing generated training artifacts: "
            f"{existing_generated}"
        )
    tag_job("pending", run_dir)

    records, stats = read_records(source_csv, cif_root)
    if not records:
        raise RuntimeError("No usable energy_above_hull rows with matching CIF files.")

    if target_transform != "raw":
        write_id_prop(run_dir / "id_prop_raw.csv", records, "raw", target_transform_scale)
    write_id_prop(run_dir / "id_prop.csv", records, target_transform, target_transform_scale)

    ensure_symlink(run_dir / "atom_init.json", atom_init)
    if create_cif_symlinks:
        for material_id, _, cif_path in records:
            ensure_symlink(run_dir / f"{material_id}.cif", cif_path)

    ids = [material_id for material_id, _, _ in records]
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
        **transform_metadata(target_transform, target_transform_scale),
        "cif_root": str(cif_root),
        "cif_provenance": (
            "Each top-level <material_id>.cif in the run folder is a symlink "
            "to the matching flat or nested CIF under cif_root."
            if create_cif_symlinks
            else "No CIF symlinks were created in this run folder; graph inputs are read from the graph cache."
        ),
        "create_cif_symlinks": create_cif_symlinks,
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
            "id_prop_raw_csv": (
                str(run_dir / "id_prop_raw.csv") if target_transform != "raw" else None
            ),
            "checkpoints": str(run_dir / "checkpoints"),
            "training_history": str(run_dir / "training_history.json"),
            "test_results": str(run_dir / "test_results.csv"),
            "test_results_physical": str(run_dir / "test_results_physical.csv"),
            "parity_metrics_physical": str(run_dir / "parity_metrics_physical.json"),
            "epoch_parity": str(run_dir / "epoch_parity"),
            "log": str(run_dir / "run.log"),
            "markdown_log": str(run_dir / "TRAINING_LOG.md"),
        },
    }
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    readme = f"""# CGCNN energy_above_hull run

Created: `{metadata["created_at"]}`

This folder trains a CPU-only CGCNN regression model to predict `energy_above_hull`
from `{source_csv}`.

## Inputs

- Source CSV: `{source_csv}`
- ID column: `material_id`
- Target column: `energy_above_hull`
- Target transform: `{target_transform}`{f" with scale `{target_transform_scale}` eV/atom" if target_transform == "scaled_log1p" else ""}
- CIF inputs: {f"symlinked from matching flat or nested CIF paths under `{cif_root}`" if create_cif_symlinks else "not symlinked; graph inputs are read from the graph cache"}
- Atom features: symlink `{run_dir / "atom_init.json"}` -> `{atom_init}`
- Split seed: `{seed}`
- Split counts: train `{len(train_ids)}`, validation `{len(val_ids)}`, test `{len(test_ids)}`
- Training options: see `run_metadata.json`

Rows with empty or NaN `energy_above_hull` were excluded. Rows whose expected
CIF file was missing were also excluded.

## Outputs

- `id_prop.csv`: active CGCNN target table after the configured target transform.
- `id_prop_raw.csv`: raw `energy_above_hull` labels, present for transformed runs.
- `<material_id>.cif`: {("symlinks to the source CIF files." if create_cif_symlinks else "not present for this graph-cache run.")}
- `splits/`: explicit train/validation/test material IDs.
- `checkpoints/`: per-epoch model checkpoints.
- `checkpoint.pth.tar` and `model_best.pth.tar`: symlink aliases to the latest and best files under `checkpoints/`.
- `training_history.json`: validation MAE history written during training.
- `test_results.csv`: held-out predictions for the best model.
- `epoch_parity/`: per-epoch test predictions, metrics, and parity plots.
- `run.log`: combined workflow log.
"""
    (run_dir / "README.md").write_text(readme)
    log = f"""# CGCNN energy_above_hull training log

Created: `{metadata["created_at"]}`

## Purpose

Train a CGCNN regression model on Materials Project CIF structures from
`{cif_root}` using `energy_above_hull` from `{source_csv}` as the target.

## Inputs

- Source CSV: `{source_csv}`
- CIF root: `{cif_root}`
- Target transform: `{target_transform}`{f" with scale `{target_transform_scale}` eV/atom" if target_transform == "scaled_log1p" else ""}
- Atom features: `{run_dir / "atom_init.json"}` symlinked to `{atom_init}`
- Source code: `{REPO_ROOT}`
- Command-line options: see `run_metadata.json`
- CIF provenance: {("symlinks are created in this run folder; source CIF files are not copied." if create_cif_symlinks else "no CIF symlinks are created; graph inputs are read from the graph cache.")}

## Counts

- Source CSV data rows: `{stats["source_csv_rows"]}`
- Rows with finite numeric `energy_above_hull`: `{stats["valid_numeric_energy_above_hull"]}`
- Rows with finite numeric target and matching CIF: `{len(records)}`
- Rows skipped because target was empty, non-numeric, or NaN: `{stats["skipped_empty_or_nan_energy_above_hull"]}`
- Rows skipped because CIF was missing: `{stats["skipped_missing_cif"]}`
- Train/validation/test split: `{len(train_ids)}` / `{len(val_ids)}` / `{len(test_ids)}`

## Generated Inputs

- `id_prop.csv`: two-column active CGCNN target table after the configured target transform.
- `id_prop_raw.csv`: raw `energy_above_hull` labels, present for transformed runs.
- `atom_init.json`: symlink to repo sample regression atom features.
- `<material_id>.cif`: {("symlinks to source CIFs under `" + str(cif_root) + "`." if create_cif_symlinks else "not present for this graph-cache run.")}
- `splits/*.csv`: explicit split IDs generated with seed `{seed}`.

## Outputs

- `run.log`: combined training stdout/stderr.
- `checkpoint.pth.tar`, `model_best.pth.tar`: symlink aliases to the latest and best files under `checkpoints/`.
- `checkpoints/epoch_*.pth.tar`: per-epoch checkpoints.
- `training_history.json`: validation metric by epoch.
- `test_results.csv`: held-out predictions from the best checkpoint.
- `test_results_physical.csv`: held-out predictions inverse-transformed to eV/atom.
- `parity_metrics_physical.json`: physical-scale metrics for the best checkpoint.
- `epoch_parity/`: per-epoch prediction CSVs, metrics JSON, and parity plots.

## Timeline

- `{metadata["created_at"]}`: Created run folder and generated input symlinks.
"""
    (run_dir / "TRAINING_LOG.md").write_text(log)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--cif-root", type=Path, default=DEFAULT_CIF_ROOT)
    parser.add_argument("--atom-init", type=Path, default=DEFAULT_ATOM_INIT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path.home() / "run" / f"mp_ehull_training_{timestamp}",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--optim", default="SGD", choices=["Adam", "SGD"])
    parser.add_argument("--atom-fea-len", type=int, default=64)
    parser.add_argument("--h-fea-len", type=int, default=32)
    parser.add_argument("--n-conv", type=int, default=4)
    parser.add_argument("--n-h", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Stop after this many epochs without validation improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement required to reset early-stopping patience.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--dataset-format",
        choices=["cif", "graph_cache", "auto"],
        default="cif",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Graph cache root to use when --dataset-format=graph_cache.",
    )
    parser.add_argument(
        "--graph-cache-max-cached-shards",
        type=int,
        default=4,
        help="Maximum graph cache shards retained per dataset object.",
    )
    parser.add_argument(
        "--id-prop-file",
        type=Path,
        help="Optional id_prop.csv for graph cache training.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["raw", "scaled_log1p"],
        default="raw",
        help="Target transform used in the active id_prop.csv.",
    )
    parser.add_argument(
        "--target-transform-scale",
        type=float,
        default=0.05,
        help="Scale in eV/atom for --target-transform=scaled_log1p.",
    )
    parser.add_argument(
        "--skip-epoch-parity",
        action="store_true",
        help="Skip per-epoch parity generation after training.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create id_prop.csv, symlinks, splits, and metadata without training.",
    )
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="Train using an existing prepared run directory.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume training from a checkpoint in the prepared run directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.source_csv = args.source_csv.expanduser().resolve()
    args.cif_root = args.cif_root.expanduser().resolve()
    args.atom_init = args.atom_init.expanduser().resolve()
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.id_prop_file is not None:
        args.id_prop_file = args.id_prop_file.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()

    if args.prepare_only and args.reuse_prepared:
        raise ValueError("--prepare-only and --reuse-prepared are mutually exclusive")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    if args.target_transform == "scaled_log1p" and args.target_transform_scale <= 0.0:
        raise ValueError("--target-transform-scale must be positive")
    if args.id_prop_file is not None and args.target_transform != "raw":
        raise ValueError(
            "--id-prop-file cannot be combined with --target-transform; "
            "let this script generate the transformed label file in the run directory."
        )
    if args.run_dir.exists() and not (args.prepare_only or args.reuse_prepared):
        raise FileExistsError(
            f"Run directory already exists: {args.run_dir}. "
            "Use --reuse-prepared for an existing prepared run."
        )
    if not args.source_csv.is_file():
        raise FileNotFoundError(args.source_csv)
    if not args.cif_root.is_dir():
        raise FileNotFoundError(args.cif_root)
    if not args.atom_init.is_file():
        raise FileNotFoundError(args.atom_init)
    if args.dataset_format == "graph_cache":
        if args.cache_dir is None:
            raise ValueError("--cache-dir is required with --dataset-format=graph_cache")
        if not args.cache_dir.is_dir():
            raise FileNotFoundError(args.cache_dir)
        if args.graph_cache_max_cached_shards < 1:
            raise ValueError("--graph-cache-max-cached-shards must be at least 1")
        if args.id_prop_file is not None and not args.id_prop_file.is_file():
            raise FileNotFoundError(args.id_prop_file)
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    args_payload = vars(args).copy()
    args_payload = {key: str(value) if isinstance(value, Path) else value for key, value in args_payload.items()}

    try:
        if args.reuse_prepared:
            prepared = load_prepared_run(args.run_dir)
        else:
            prepared = prepare_run_dir(
                run_dir=args.run_dir,
                source_csv=args.source_csv,
                cif_root=args.cif_root,
                atom_init=args.atom_init,
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                target_transform=args.target_transform,
                target_transform_scale=args.target_transform_scale,
                create_cif_symlinks=args.dataset_format != "graph_cache",
                args_payload=args_payload,
            )
        if args.prepare_only:
            print(json.dumps(prepared["metadata"], indent=2))
            print(f"Prepared run: {args.run_dir}")
            return
        with tee_log(args.run_dir / "run.log"):
            print(json.dumps(prepared["metadata"], indent=2))
            tag_job("running", args.run_dir)
            os.chdir(args.run_dir)
            training_root = args.cache_dir if args.dataset_format == "graph_cache" else args.run_dir
            training_id_prop_file = args.id_prop_file
            if (
                training_id_prop_file is None
                and args.dataset_format == "graph_cache"
                and args.target_transform != "raw"
            ):
                training_id_prop_file = args.run_dir / "id_prop.csv"
            best_checkpoint = train_model(
                root_dir=str(training_root),
                task="regression",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                optim_name=args.optim,
                atom_fea_len=args.atom_fea_len,
                h_fea_len=args.h_fea_len,
                n_conv=args.n_conv,
                n_h=args.n_h,
                device=get_env_device(),
                workers=args.workers,
                weight_decay=args.weight_decay,
                resume=str(args.resume) if args.resume is not None else None,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                train_ids=prepared["train_ids"],
                val_ids=prepared["val_ids"],
                test_ids=prepared["test_ids"],
                checkpoint_dir=str(args.run_dir / "checkpoints"),
                metrics_history_path=str(args.run_dir / "training_history.json"),
                print_freq=args.print_freq,
                dataset_format=args.dataset_format,
                graph_cache_max_cached_shards=args.graph_cache_max_cached_shards,
                id_prop_file=(
                    str(training_id_prop_file)
                    if training_id_prop_file is not None
                    else None
                ),
            )
            print(f"Best checkpoint: {best_checkpoint}")

            from analyze_parity import compute_metrics, load_results, make_plot

            physical_results_csv = args.run_dir / "test_results_physical.csv"
            write_physical_results(
                args.run_dir / "test_results.csv",
                physical_results_csv,
                args.target_transform,
                args.target_transform_scale,
            )
            _, physical_targets, physical_predictions = load_results(physical_results_csv)
            physical_metrics = compute_metrics(physical_targets, physical_predictions)
            with (args.run_dir / "parity_metrics_physical.json").open("w") as handle:
                json.dump(
                    {
                        **physical_metrics,
                        **transform_metadata(
                            args.target_transform,
                            args.target_transform_scale,
                        ),
                    },
                    handle,
                    indent=2,
                )
            make_plot(
                physical_targets,
                physical_predictions,
                physical_metrics,
                args.run_dir / "parity_plot_physical.png",
            )
            if not args.skip_epoch_parity:
                from generate_epoch_parity_plots import generate_plots_for_run

                generate_plots_for_run(
                    args.run_dir,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    device=get_env_device(),
                    checkpoint_dir=args.run_dir / "checkpoints",
                    dataset_format=(
                        "graph_cache" if args.dataset_format == "graph_cache" else "cif"
                    ),
                    cache_dir=args.cache_dir,
                    id_prop_file=training_id_prop_file,
                    max_cached_shards=args.graph_cache_max_cached_shards,
                )

            metadata_path = args.run_dir / "run_metadata.json"
            with metadata_path.open() as handle:
                metadata = json.load(handle)
            metadata["completed_at"] = datetime.now().astimezone().isoformat()
            metadata["best_checkpoint"] = str(best_checkpoint)
            metadata["epoch_parity_skipped"] = bool(args.skip_epoch_parity)
            with metadata_path.open("w") as handle:
                json.dump(metadata, handle, indent=2)
            with (args.run_dir / "TRAINING_LOG.md").open("a") as handle:
                handle.write(
                    f"- `{metadata['completed_at']}`: Training completed successfully. "
                    f"Best checkpoint: `{best_checkpoint}`\\n"
                )

            required_outputs = [
                args.run_dir / "id_prop.csv",
                args.run_dir / "model_best.pth.tar",
                args.run_dir / "test_results.csv",
                args.run_dir / "test_results_physical.csv",
                args.run_dir / "parity_metrics_physical.json",
            ]
            if not args.skip_epoch_parity:
                required_outputs.append(
                    args.run_dir / "epoch_parity" / "epoch_parity_summary.json"
                )
            missing = [str(path) for path in required_outputs if not path.exists()]
            if missing:
                raise RuntimeError(f"Missing expected outputs: {missing}")
            tag_job("finished", args.run_dir)
            print(f"Finished run: {args.run_dir}")
    except Exception:
        if args.run_dir.exists():
            tag_job("failed", args.run_dir)
            with (args.run_dir / "TRAINING_LOG.md").open("a") as handle:
                handle.write(
                    f"- `{datetime.now().astimezone().isoformat()}`: Training failed; "
                    "see `run.log` and terminal traceback.\\n"
                )
        raise


if __name__ == "__main__":
    main()
