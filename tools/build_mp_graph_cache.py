from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cgcnn.data import (  # noqa: E402
    AtomCustomJSONInitializer,
    GaussianDistance,
    build_crystal_graph,
    load_cif_structure,
)


SCHEMA_VERSION = "cgcnn_graph_cache_v2"
DEFAULT_DATA_ROOT = Path.home() / "Downloads" / "download" / "mp-cif"
DEFAULT_ATOM_INIT = REPO_ROOT / "data" / "sample-regression" / "atom_init.json"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "mp_graph_cache_v2"
DEFAULT_SPLIT_SEED = 20260624
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


@dataclass(frozen=True)
class GraphRecord:
    material_id: str
    cif_path: str


@dataclass(frozen=True)
class LabelRecord:
    material_id: str
    values: tuple[float, ...]


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def find_cif_path(cif_root: Path, material_id: str, layout: str) -> Path | None:
    candidates: list[Path] = []
    if layout in {"auto", "flat"}:
        candidates.append(cif_root / f"{material_id}.cif")
    if layout in {"auto", "nested"}:
        candidates.append(cif_root / material_id / f"{material_id}.cif")
    for candidate in candidates:
        if candidate.name.startswith("._"):
            continue
        if candidate.is_file():
            return candidate
    return None


def read_records(
    source_csv: Path,
    cif_root: Path,
    id_column: str,
    target_columns: Sequence[str],
    cif_layout: str,
) -> tuple[list[GraphRecord], list[LabelRecord], dict[str, Any]]:
    records: list[GraphRecord] = []
    label_records: list[LabelRecord] = []
    skipped_empty_or_nan_target = 0
    skipped_missing_cif = 0
    missing_examples: list[str] = []

    with source_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, *target_columns}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        source_rows = 0
        for row in reader:
            source_rows += 1
            material_id = (row.get(id_column) or "").strip()
            if not material_id:
                continue
            cif_path = find_cif_path(cif_root, material_id, cif_layout)
            if cif_path is None:
                skipped_missing_cif += 1
                if len(missing_examples) < 20:
                    missing_examples.append(str(cif_root / f"{material_id}.cif"))
                continue
            records.append(
                GraphRecord(
                    material_id=material_id,
                    cif_path=str(cif_path),
                )
            )
            target_values: list[float] = []
            for target_column in target_columns:
                value = parse_float(row.get(target_column) or "")
                if value is None:
                    skipped_empty_or_nan_target += 1
                    target_values = []
                    break
                target_values.append(value)
            if target_values:
                label_records.append(
                    LabelRecord(material_id=material_id, values=tuple(target_values))
                )

    stats = {
        "source_csv_rows": source_rows,
        "graph_rows": len(records),
        "label_rows": len(label_records),
        "target_columns": list(target_columns),
        "skipped_empty_or_nan_target": skipped_empty_or_nan_target,
        "skipped_missing_cif": skipped_missing_cif,
        "missing_cif_examples": missing_examples,
    }
    return records, label_records, stats


def read_label_records(
    source_csv: Path,
    graph_ids: set[str],
    id_column: str,
    target_columns: Sequence[str],
) -> tuple[list[LabelRecord], dict[str, Any]]:
    label_records: list[LabelRecord] = []
    skipped_missing_graph = 0
    skipped_empty_or_nan_target = 0
    missing_graph_examples: list[str] = []

    with source_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, *target_columns}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        source_rows = 0
        for row in reader:
            source_rows += 1
            material_id = (row.get(id_column) or "").strip()
            if not material_id:
                continue
            if material_id not in graph_ids:
                skipped_missing_graph += 1
                if len(missing_graph_examples) < 20:
                    missing_graph_examples.append(material_id)
                continue
            target_values: list[float] = []
            for target_column in target_columns:
                value = parse_float(row.get(target_column) or "")
                if value is None:
                    skipped_empty_or_nan_target += 1
                    target_values = []
                    break
                target_values.append(value)
            if target_values:
                label_records.append(
                    LabelRecord(material_id=material_id, values=tuple(target_values))
                )

    stats = {
        "source_csv_rows": source_rows,
        "label_rows": len(label_records),
        "target_columns": list(target_columns),
        "skipped_empty_or_nan_target": skipped_empty_or_nan_target,
        "skipped_missing_graph": skipped_missing_graph,
        "missing_graph_examples": missing_graph_examples,
    }
    return label_records, stats


def write_id_prop(path: Path, records: Sequence[LabelRecord]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for record in records:
            writer.writerow([
                record.material_id,
                *[f"{value:.9g}" for value in record.values],
            ])


def write_graph_ids(path: Path, records: Sequence[GraphRecord]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id"])
        writer.writerows([[record.material_id] for record in records])


def write_csv_ids(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id"])
        writer.writerows([[item] for item in ids])


def write_splits(
    split_dir: Path,
    records: Sequence[GraphRecord | LabelRecord],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[str]]:
    ids = [record.material_id for record in records]
    rng = random.Random(seed)
    rng.shuffle(ids)
    train_count = int(train_ratio * len(ids))
    val_count = int(val_ratio * len(ids))
    test_count = len(ids) - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            "Split ratios produced an empty split: "
            f"train={train_count}, val={val_count}, test={test_count}"
        )

    split_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": ids[:train_count],
        "val": ids[train_count : train_count + val_count],
        "test": ids[train_count + val_count :],
    }
    write_csv_ids(split_dir / "train_ids.csv", splits["train"])
    write_csv_ids(split_dir / "val_ids.csv", splits["val"])
    write_csv_ids(split_dir / "test_ids.csv", splits["test"])
    return splits


def label_name_from_columns(target_columns: Sequence[str]) -> str:
    cleaned = []
    for column in target_columns:
        label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in column)
        cleaned.append(label.strip("_") or "target")
    return "__".join(cleaned)


def relative_to_cache(path: Path, cache_dir: Path) -> str:
    try:
        return str(path.relative_to(cache_dir))
    except ValueError:
        return str(path)


def write_label_task(
    cache_dir: Path,
    label_records: Sequence[LabelRecord],
    target_columns: Sequence[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Any]:
    label_name = label_name_from_columns(target_columns)
    label_dir = cache_dir / "labels" / label_name
    split_dir = label_dir / "splits"
    label_dir.mkdir(parents=True, exist_ok=True)

    id_prop_file = label_dir / "id_prop.csv"
    write_id_prop(id_prop_file, label_records)
    splits = write_splits(
        split_dir,
        label_records,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    split_counts = {name: len(ids) for name, ids in splits.items()}
    split_files = {
        "train": split_dir / "train_ids.csv",
        "val": split_dir / "val_ids.csv",
        "test": split_dir / "test_ids.csv",
    }
    return {
        "name": label_name,
        "target_columns": list(target_columns),
        "id_prop_file": relative_to_cache(id_prop_file, cache_dir),
        "split_counts": split_counts,
        "split_files": {
            name: relative_to_cache(path, cache_dir)
            for name, path in split_files.items()
        },
    }


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


def append_log(cache_dir: Path, message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat()
    with (cache_dir / "RUN_LOG.md").open("a") as handle:
        handle.write(f"- `{timestamp}` {message}\n")


def git_info() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "status_short": status,
        "dirty": bool(status),
    }


def record_to_payload(record: GraphRecord) -> tuple[str, str]:
    return record.material_id, record.cif_path


def decode_np_id(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def read_existing_shard(shard_path: Path, shard_index: int) -> dict[str, Any]:
    with np.load(shard_path, allow_pickle=False) as data:
        ids = [decode_np_id(item) for item in data["ids"]]
        atom_offsets = data["atom_offsets"]
        atom_count = int(atom_offsets[-1])
    return {
        "shard_index": shard_index,
        "name": shard_path.name,
        "count": len(ids),
        "atom_count": atom_count,
        "byte_size": shard_path.stat().st_size,
        "ids": ids,
        "resumed": True,
    }


def build_shard_worker(payload: dict[str, Any]) -> dict[str, Any]:
    shard_path = Path(payload["shard_path"])
    shard_index = int(payload["shard_index"])
    if payload["resume"] and shard_path.is_file():
        return read_existing_shard(shard_path, shard_index)

    for warning_message in (
        r".*not find enough neighbors to build graph.*",
        r".*Issues encountered while parsing CIF.*",
        r".*No Pauling electronegativity.*",
    ):
        warnings.filterwarnings(
            "ignore",
            message=warning_message,
            category=UserWarning,
        )

    atom_initializer = AtomCustomJSONInitializer(payload["atom_init"])
    gaussian_distance = GaussianDistance(
        dmin=payload["dmin"],
        dmax=payload["radius"],
        step=payload["step"],
    )

    ids: list[str] = []
    atom_chunks: list[np.ndarray] = []
    nbr_chunks: list[np.ndarray] = []
    nbr_idx_chunks: list[np.ndarray] = []
    atom_offsets = [0]

    for material_id, cif_path in payload["records"]:
        crystal = load_cif_structure(cif_path)
        atom_fea, nbr_fea, nbr_fea_idx = build_crystal_graph(
            crystal,
            atom_initializer,
            gaussian_distance,
            max_num_nbr=payload["max_num_nbr"],
            radius=payload["radius"],
            cif_id=material_id,
        )
        ids.append(material_id)
        atom_chunks.append(atom_fea.astype("<f4", copy=False))
        nbr_chunks.append(nbr_fea.astype("<f4", copy=False))
        nbr_idx_chunks.append(nbr_fea_idx.astype("<i8", copy=False))
        atom_offsets.append(atom_offsets[-1] + atom_fea.shape[0])

    max_id_len = max(len(item.encode("utf-8")) for item in ids)
    ids_array = np.asarray(
        [item.encode("utf-8") for item in ids],
        dtype=f"S{max_id_len}",
    )
    arrays = {
        "ids": ids_array,
        "atom_fea": np.concatenate(atom_chunks, axis=0).astype("<f4", copy=False),
        "nbr_fea": np.concatenate(nbr_chunks, axis=0).astype("<f4", copy=False),
        "nbr_fea_idx": np.concatenate(nbr_idx_chunks, axis=0).astype(
            "<i8",
            copy=False,
        ),
        "atom_offsets": np.asarray(atom_offsets, dtype="<i8"),
    }

    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    if payload["compressed"]:
        np.savez_compressed(tmp_path, **arrays)
    else:
        np.savez(tmp_path, **arrays)
    tmp_npz_path = tmp_path
    tmp_npz_candidate = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    if not tmp_npz_path.exists() and tmp_npz_candidate.exists():
        tmp_npz_path = tmp_npz_candidate
    tmp_npz_path.replace(shard_path)

    return {
        "shard_index": shard_index,
        "name": shard_path.name,
        "count": len(ids),
        "atom_count": int(atom_offsets[-1]),
        "byte_size": shard_path.stat().st_size,
        "ids": ids,
        "resumed": False,
    }


def write_readme(
    cache_dir: Path,
    created_at: str,
    source_csv: Path,
    cif_root: Path,
    atom_init: Path,
    copied_atom_init: Path,
    label_meta: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    label_section = "No label task was written for this graph cache."
    if label_meta is not None:
        label_section = "\n".join([
            f"- Label task: `{label_meta['name']}`",
            f"- Target columns: `{', '.join(label_meta['target_columns'])}`",
            f"- Label file: `{label_meta['id_prop_file']}`",
            "- Split files:",
            f"  - Train: `{label_meta['split_files']['train']}`",
            f"  - Validation: `{label_meta['split_files']['val']}`",
            f"  - Test: `{label_meta['split_files']['test']}`",
        ])

    readme = f"""# CGCNN graph cache

Created: `{created_at}`

This folder contains a local CGCNN graph cache. It was created to move the slow
CIF parsing and neighbor-graph construction step off the remote Linux Skylake
filesystem. The graph shards are independent of the training target. Training
should pair this graph cache with a task-specific `id_prop.csv` label file.

## Inputs

- Source CSV: `{source_csv}`
- ID column: `{args.id_column}`
- CIF inputs: read in place from `{cif_root}` using layout `{args.cif_layout}`.
- Atom features: copied from `{atom_init}` to `{copied_atom_init}`.
- Graph parameters: `max_num_nbr={args.max_num_nbr}`, `radius={args.radius}`,
  `dmin={args.dmin}`, `step={args.step}`.
- Split seed: `{args.split_seed}`
- Shard size: `{args.shard_size}`
- Command-line options: see `manifest.json`.

No CIF files are copied or symlinked into this cache. The source CIFs are read
in place, and the graph outputs are primitive NumPy arrays in `.npz` shard files.

## Label Tasks

{label_section}

## Outputs

- `manifest.json`: cache schema, source hashes, graph parameters, split counts,
  shard inventory, and material-ID-to-shard index.
- `graph_ids.csv`: material IDs with cached graph tensors.
- `labels/`: optional task-specific target files and split files.
- `atom_init.json`: copied atom feature input used during graph construction.
- `shards/`: sharded `.npz` graph arrays.
- `RUN_LOG.md`: status log for this cache build.
"""
    (cache_dir / "README.md").write_text(readme)


def validate_cache(
    cache_dir: Path,
    expected_count: int,
    label_meta: dict[str, Any] | None = None,
) -> None:
    from cgcnn.data import CachedGraphData

    if label_meta is None:
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        if len(manifest["index"]) != expected_count:
            raise RuntimeError(
                "Cache index length mismatch: "
                f"{len(manifest['index'])} != {expected_count}"
            )
        for shard_meta in [manifest["shards"][0], manifest["shards"][-1]]:
            shard_path = cache_dir / "shards" / shard_meta["file"]
            with np.load(shard_path, allow_pickle=False) as data:
                for key in ["ids", "atom_fea", "nbr_fea", "nbr_fea_idx", "atom_offsets"]:
                    if key not in data:
                        raise RuntimeError(f"{shard_path} is missing {key}")
                    if data[key].dtype.hasobject:
                        raise RuntimeError(f"{shard_path}:{key} has object dtype")
            if int(shard_meta["count"]) <= 0:
                raise RuntimeError(f"Empty shard in manifest: {shard_meta['file']}")
        return

    id_prop_file = label_meta["id_prop_file"]
    expected_dataset_count = sum(label_meta["split_counts"].values())
    split_counts = label_meta["split_counts"]

    dataset = CachedGraphData(
        cache_dir,
        id_prop_file=id_prop_file,
        shuffle=False,
        max_cached_shards=1,
    )
    if len(dataset.manifest["index"]) != expected_count:
        raise RuntimeError(
            "Cache index length mismatch: "
            f"{len(dataset.manifest['index'])} != {expected_count}"
        )
    if len(dataset) != expected_dataset_count:
        raise RuntimeError(
            f"Dataset length mismatch: {len(dataset)} != {expected_dataset_count}"
        )
    for idx in sorted({0, len(dataset) // 2, len(dataset) - 1}):
        (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id = dataset[idx]
        if atom_fea.ndim != 2 or nbr_fea.ndim != 3 or nbr_fea_idx.ndim != 2:
            raise RuntimeError(f"Invalid cached tensor shapes for {cif_id}")
        if target.ndim != 1:
            raise RuntimeError(f"Invalid cached target shape for {cif_id}")

    manifest = dataset.manifest
    index = manifest["index"]
    for split_name, expected in split_counts.items():
        split_path = Path(label_meta["split_files"][split_name])
        if not split_path.is_absolute():
            split_path = cache_dir / split_path
        with split_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            ids = [row["material_id"] for row in reader]
        if len(ids) != expected:
            raise RuntimeError(
                f"{split_name} split count mismatch: {len(ids)} != {expected}"
            )
        missing = [item for item in ids if item not in index]
        if missing:
            raise RuntimeError(
                f"{split_name} split contains IDs missing from cache index: {missing[:5]}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable sharded CGCNN graph cache for MP CIF data."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-csv", type=Path, default=None)
    parser.add_argument("--atom-init", type=Path, default=DEFAULT_ATOM_INIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--id-column", default="material_id")
    parser.add_argument(
        "--target-column",
        dest="target_columns",
        nargs="+",
        default=["energy_above_hull"],
        help="One or more target columns to write into a task-specific label file.",
    )
    parser.add_argument(
        "--cif-layout",
        choices=["auto", "flat", "nested"],
        default="flat",
    )
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-num-nbr", type=int, default=12)
    parser.add_argument("--radius", type=float, default=8)
    parser.add_argument("--dmin", type=float, default=0)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Add a label task to an existing cache without rebuilding graph shards.",
    )
    parser.add_argument(
        "--set-default-label",
        action="store_true",
        help="When used with --labels-only, make the written label task the cache default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.source_csv = (
        args.source_csv.expanduser().resolve()
        if args.source_csv is not None
        else args.data_root / "mp_all_summary.csv"
    )
    args.atom_init = args.atom_init.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if not args.source_csv.is_file():
        raise FileNotFoundError(args.source_csv)
    manifest_path = args.output_dir / "manifest.json"

    if args.labels_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Existing manifest required: {manifest_path}")
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        graph_ids = set(manifest.get("index", {}))
        if not graph_ids:
            raise ValueError(f"Manifest has no graph index: {manifest_path}")
        label_records, label_stats = read_label_records(
            args.source_csv,
            graph_ids,
            args.id_column,
            args.target_columns,
        )
        if not label_records:
            raise RuntimeError("No usable label rows matched the graph cache.")
        label_meta = write_label_task(
            args.output_dir,
            label_records,
            args.target_columns,
            seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        manifest.setdefault("labels", {})[label_meta["name"]] = {
            **label_meta,
            "source_csv": str(args.source_csv),
            "source_csv_sha256": sha256_file(args.source_csv),
            "source_stats": label_stats,
            "created_at": datetime.now().astimezone().isoformat(),
        }
        if args.set_default_label or not manifest.get("default_label"):
            manifest["default_label"] = label_meta["name"]
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        with (args.output_dir / "manifest.json.tmp").open("w") as handle:
            json.dump(manifest, handle, indent=2)
        (args.output_dir / "manifest.json.tmp").replace(manifest_path)
        append_log(
            args.output_dir,
            f"Added label task {label_meta['name']} with {len(label_records)} rows.",
        )
        validate_cache(
            args.output_dir,
            expected_count=len(graph_ids),
            label_meta=label_meta,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "label_task": label_meta,
                    "label_rows": len(label_records),
                    "manifest": str(manifest_path),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if not args.atom_init.is_file():
        raise FileNotFoundError(args.atom_init)
    if args.output_dir.exists() and not args.resume:
        existing = list(args.output_dir.iterdir())
        if existing:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {args.output_dir}. "
                "Use --resume only for an interrupted cache build."
            )

    created_at = datetime.now().astimezone().isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = args.output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    tag_job("pending", args.output_dir)

    copied_atom_init = args.output_dir / "atom_init.json"
    if copied_atom_init.exists():
        if sha256_file(copied_atom_init) != sha256_file(args.atom_init):
            raise FileExistsError(
                f"Existing atom_init copy differs from requested input: {copied_atom_init}"
            )
    else:
        shutil.copy2(args.atom_init, copied_atom_init)

    append_log(args.output_dir, "Cache directory prepared.")

    records, label_records, source_stats = read_records(
        args.source_csv,
        args.data_root,
        args.id_column,
        args.target_columns,
        args.cif_layout,
    )
    if not records:
        raise RuntimeError("No usable rows with matching CIF files.")
    if not label_records:
        raise RuntimeError("No usable label rows with matching cached graph IDs.")
    write_graph_ids(args.output_dir / "graph_ids.csv", records)
    label_meta = write_label_task(
        args.output_dir,
        label_records,
        args.target_columns,
        seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_readme(
        args.output_dir,
        created_at,
        args.source_csv,
        args.data_root,
        args.atom_init,
        copied_atom_init,
        label_meta,
        args,
    )

    atom_init_hash = sha256_file(args.atom_init)
    source_csv_hash = sha256_file(args.source_csv)
    command_line_options = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    shard_payloads: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(records), args.shard_size)):
        chunk = records[start : start + args.shard_size]
        shard_name = f"shard_{shard_index:05d}.npz"
        shard_payloads.append(
            {
                "shard_index": shard_index,
                "shard_path": str(shards_dir / shard_name),
                "records": [record_to_payload(record) for record in chunk],
                "atom_init": str(args.atom_init),
                "max_num_nbr": args.max_num_nbr,
                "radius": args.radius,
                "dmin": args.dmin,
                "step": args.step,
                "compressed": args.compressed,
                "resume": args.resume,
            }
        )

    append_log(
        args.output_dir,
        f"Starting graph cache build for {len(records)} records in "
        f"{len(shard_payloads)} shards with {args.workers} workers.",
    )
    tag_job("running", args.output_dir)

    results: list[dict[str, Any]] = []
    try:
        if args.workers == 1:
            for payload in shard_payloads:
                result = build_shard_worker(payload)
                results.append(result)
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"shard {result['shard_index'] + 1}/{len(shard_payloads)} "
                    f"count={result['count']} atoms={result['atom_count']} "
                    f"resumed={result['resumed']}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_index = {
                    executor.submit(build_shard_worker, payload): payload["shard_index"]
                    for payload in shard_payloads
                }
                completed = 0
                for future in as_completed(future_to_index):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    print(
                        f"[{datetime.now().isoformat(timespec='seconds')}] "
                        f"shard {result['shard_index'] + 1}/{len(shard_payloads)} "
                        f"done ({completed}/{len(shard_payloads)}) "
                        f"count={result['count']} atoms={result['atom_count']} "
                        f"resumed={result['resumed']}",
                        flush=True,
                    )
    except Exception:
        append_log(args.output_dir, "Cache build failed.")
        tag_job("failed", args.output_dir)
        raise

    results.sort(key=lambda row: row["shard_index"])
    manifest_shards: list[dict[str, Any]] = []
    index: dict[str, dict[str, int | str]] = {}
    for result in results:
        manifest_shards.append(
            {
                "file": result["name"],
                "count": result["count"],
                "atom_count": result["atom_count"],
                "byte_size": result["byte_size"],
            }
        )
        for row_index, material_id in enumerate(result["ids"]):
            index[material_id] = {"shard": result["name"], "row": row_index}

    if len(index) != len(records):
        append_log(args.output_dir, "Cache build failed validation: duplicate or missing IDs.")
        tag_job("failed", args.output_dir)
        raise RuntimeError(f"Cache index has {len(index)} IDs for {len(records)} records.")

    graph_parameters = {
        "max_num_nbr": args.max_num_nbr,
        "radius": args.radius,
        "dmin": args.dmin,
        "step": args.step,
        "gaussian_filter_length": len(
            np.arange(args.dmin, args.radius + args.step, args.step)
        ),
    }
    label_meta_with_source = {
        **label_meta,
        "source_csv": str(args.source_csv),
        "source_csv_sha256": source_csv_hash,
        "source_stats": source_stats,
        "created_at": created_at,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "completed_at": datetime.now().astimezone().isoformat(),
        "workspace": str(REPO_ROOT),
        "output_dir": str(args.output_dir),
        "source_csv": str(args.source_csv),
        "source_csv_sha256": source_csv_hash,
        "source_id_prop": str(args.data_root / "id_prop.csv"),
        "cif_source": str(args.data_root),
        "cif_layout": args.cif_layout,
        "atom_init": str(args.atom_init),
        "atom_init_copy": str(copied_atom_init),
        "atom_init_sha256": atom_init_hash,
        "id_column": args.id_column,
        "graph_count": len(records),
        "valid_row_count": len(label_records),
        "source_stats": source_stats,
        "graph_parameters": graph_parameters,
        "dtype_policy": {
            "atom_fea": "little-endian float32",
            "nbr_fea": "little-endian float32",
            "nbr_fea_idx": "little-endian int64",
            "atom_offsets": "little-endian int64",
            "ids": "UTF-8 bytes",
        },
        "endianness": sys.byteorder,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "pymatgen_version": package_version("pymatgen"),
        "split_seed": args.split_seed,
        "split_counts": label_meta["split_counts"],
        "split_files": label_meta["split_files"],
        "labels": {label_meta["name"]: label_meta_with_source},
        "default_label": label_meta["name"],
        "shard_size": args.shard_size,
        "compressed": args.compressed,
        "shards": manifest_shards,
        "index": index,
        "command_line_options": command_line_options,
        "code": git_info(),
    }
    tmp_manifest = args.output_dir / "manifest.json.tmp"
    with tmp_manifest.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    tmp_manifest.replace(manifest_path)

    try:
        validate_cache(
            args.output_dir,
            expected_count=len(records),
            label_meta=label_meta,
        )
    except Exception:
        append_log(args.output_dir, "Cache validation failed.")
        tag_job("failed", args.output_dir)
        raise

    append_log(
        args.output_dir,
        f"Cache build finished successfully: {len(records)} records, "
        f"{len(manifest_shards)} shards.",
    )
    tag_job("finished", args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "graph_count": len(records),
                "label_rows": len(label_records),
                "label_task": label_meta,
                "shards": len(manifest_shards),
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
