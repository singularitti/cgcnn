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
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Sequence

import ijson
import numpy as np
from pymatgen.core import Structure

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cgcnn.data import (  # noqa: E402
    AtomCustomJSONInitializer,
    GaussianDistance,
    build_crystal_graph,
)
from cgcnn.device import get_env_device  # noqa: E402
from cgcnn.training import train_model  # noqa: E402


SCHEMA_VERSION = "cgcnn_graph_cache_v2"
DEFAULT_DATASET = Path(
    "/work2/04996/tg842951/stampede3/MPtrj_2022.9_full_dataset/data/MPtrj_2022.9_full.json"
)
DEFAULT_SCHEMA = Path(
    "/work2/04996/tg842951/stampede3/MPtrj_2022.9_full_dataset/schema.json"
)
DEFAULT_ATOM_INIT = REPO_ROOT / "data" / "sample-regression" / "atom_init.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs"
DEFAULT_SPLIT_SEED = 20260624
EXPECTED_MATERIALS = 145_923
EXPECTED_RECORDS = 1_580_395


@dataclass(frozen=True)
class StreamRecord:
    graph_id: str
    material_key: str
    frame_key: str
    mp_id: str
    target: float
    structure_dict: dict[str, Any]
    n_atoms: int
    energies: dict[str, float]


@dataclass(frozen=True)
class GraphPayloadRecord:
    graph_id: str
    mp_id: str
    target: float
    structure_dict: dict[str, Any]


class NumericSummary:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.nonfinite = 0

    def add(self, value: Any) -> None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            self.nonfinite += 1
            return
        if math.isfinite(parsed):
            self.values.append(parsed)
        else:
            self.nonfinite += 1

    def as_dict(self) -> dict[str, Any]:
        if not self.values:
            return {
                "count": 0,
                "nonfinite": self.nonfinite,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "quantiles": {},
            }
        arr = np.asarray(self.values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "nonfinite": self.nonfinite,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "quantiles": {
                "0.001": float(np.quantile(arr, 0.001)),
                "0.01": float(np.quantile(arr, 0.01)),
                "0.05": float(np.quantile(arr, 0.05)),
                "0.5": float(np.quantile(arr, 0.5)),
                "0.95": float(np.quantile(arr, 0.95)),
                "0.99": float(np.quantile(arr, 0.99)),
                "0.999": float(np.quantile(arr, 0.999)),
            },
        }


class DistributionAccumulator:
    def __init__(self, target_column: str) -> None:
        self.target_column = target_column
        self.material_count = 0
        self.record_count = 0
        self.usable_target_count = 0
        self.nonfinite_target_count = 0
        self.total_atoms = 0
        self.atom_counts = NumericSummary()
        self.frames_per_material = NumericSummary()
        self.energy_summaries = {
            "energy_per_atom": NumericSummary(),
            "ef_per_atom": NumericSummary(),
            "e_per_atom_relaxed": NumericSummary(),
            "ef_per_atom_relaxed": NumericSummary(),
        }
        self.relaxed_equal_count = 0
        self.relaxed_compare_count = 0
        self.target_ranges_by_mp_id: dict[str, list[float]] = {}

    def add_material(self, frame_count: int) -> None:
        self.material_count += 1
        self.frames_per_material.add(frame_count)

    def add_record(self, record: StreamRecord) -> None:
        self.record_count += 1
        self.usable_target_count += 1
        self.total_atoms += record.n_atoms
        self.atom_counts.add(record.n_atoms)
        for column, summary in self.energy_summaries.items():
            summary.add(record.energies.get(column))
        e_relaxed = record.energies.get("e_per_atom_relaxed")
        ef_relaxed = record.energies.get("ef_per_atom_relaxed")
        if e_relaxed is not None and ef_relaxed is not None:
            self.relaxed_compare_count += 1
            if math.isclose(e_relaxed, ef_relaxed, rel_tol=0.0, abs_tol=1e-12):
                self.relaxed_equal_count += 1
        target_range = self.target_ranges_by_mp_id.setdefault(
            record.mp_id,
            [record.target, record.target, 0.0],
        )
        target_range[0] = min(target_range[0], record.target)
        target_range[1] = max(target_range[1], record.target)
        target_range[2] += 1.0

    def add_nonfinite_target(self) -> None:
        self.record_count += 1
        self.nonfinite_target_count += 1

    def add_nonfinite_targets(self, count: int) -> None:
        self.record_count += count
        self.nonfinite_target_count += count

    def as_dict(self) -> dict[str, Any]:
        varying = 0
        repeated = 0
        max_span = 0.0
        for minimum, maximum, count in self.target_ranges_by_mp_id.values():
            span = maximum - minimum
            max_span = max(max_span, span)
            if count > 1:
                repeated += 1
            if span > 1e-12:
                varying += 1
        relaxed_equal_fraction = (
            self.relaxed_equal_count / self.relaxed_compare_count
            if self.relaxed_compare_count
            else None
        )
        return {
            "target_column": self.target_column,
            "material_count": self.material_count,
            "record_count": self.record_count,
            "usable_target_count": self.usable_target_count,
            "nonfinite_target_count": self.nonfinite_target_count,
            "total_atoms": self.total_atoms,
            "atom_counts": self.atom_counts.as_dict(),
            "frames_per_material": self.frames_per_material.as_dict(),
            "energies": {
                column: summary.as_dict()
                for column, summary in self.energy_summaries.items()
            },
            "ef_per_atom_relaxed_equals_e_per_atom_relaxed": {
                "count": self.relaxed_equal_count,
                "compared": self.relaxed_compare_count,
                "fraction": relaxed_equal_fraction,
            },
            "ef_per_atom_variation_by_mp_id": {
                "mp_id_count": len(self.target_ranges_by_mp_id),
                "mp_ids_with_multiple_records": repeated,
                "mp_ids_with_varying_target": varying,
                "max_span": max_span,
            },
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


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


def append_log(run_dir: Path, message: str) -> None:
    with (run_dir / "RUN_LOG.md").open("a") as handle:
        handle.write(f"- `{now_iso()}` {message}\n")


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def stream_records(
    dataset: Path,
    target_column: str,
    limit: int | None = None,
) -> Iterable[tuple[str, list[StreamRecord], int, int]]:
    emitted = 0
    with dataset.open("rb") as handle:
        for material_key, frames in ijson.kvitems(handle, "", use_float=True):
            material_records: list[StreamRecord] = []
            nonfinite_targets = 0
            for frame_key, raw_record in frames.items():
                target = safe_float(raw_record.get(target_column))
                if target is None:
                    nonfinite_targets += 1
                    continue
                structure_dict = raw_record["structure"]
                sites = structure_dict.get("sites") or []
                energies = {
                    column: safe_float(raw_record.get(column))
                    for column in [
                        "energy_per_atom",
                        "ef_per_atom",
                        "e_per_atom_relaxed",
                        "ef_per_atom_relaxed",
                    ]
                }
                material_records.append(
                    StreamRecord(
                        graph_id=f"{material_key}::{frame_key}",
                        material_key=material_key,
                        frame_key=frame_key,
                        mp_id=str(raw_record.get("mp_id") or material_key),
                        target=target,
                        structure_dict=structure_dict,
                        n_atoms=len(sites),
                        energies=energies,
                    )
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    break
            yield material_key, material_records, len(frames), nonfinite_targets
            if limit is not None and emitted >= limit:
                break


def graph_worker(payload: dict[str, Any]) -> dict[str, Any]:
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
    mp_ids: list[str] = []
    targets: list[float] = []
    atom_chunks: list[np.ndarray] = []
    nbr_chunks: list[np.ndarray] = []
    nbr_idx_chunks: list[np.ndarray] = []
    atom_offsets = [0]
    graph_nbytes = 0

    for item in payload["records"]:
        structure = Structure.from_dict(item["structure_dict"])
        atom_fea, nbr_fea, nbr_fea_idx = build_crystal_graph(
            structure,
            atom_initializer,
            gaussian_distance,
            max_num_nbr=payload["max_num_nbr"],
            radius=payload["radius"],
            cif_id=item["graph_id"],
        )
        atom_fea = atom_fea.astype("<f4", copy=False)
        nbr_fea = nbr_fea.astype("<f4", copy=False)
        nbr_fea_idx = nbr_fea_idx.astype("<i8", copy=False)
        ids.append(item["graph_id"])
        mp_ids.append(item["mp_id"])
        targets.append(float(item["target"]))
        atom_chunks.append(atom_fea)
        nbr_chunks.append(nbr_fea)
        nbr_idx_chunks.append(nbr_fea_idx)
        atom_offsets.append(atom_offsets[-1] + atom_fea.shape[0])
        graph_nbytes += atom_fea.nbytes + nbr_fea.nbytes + nbr_fea_idx.nbytes

    shard_name = payload.get("shard_name")
    shard_path = payload.get("shard_path")
    byte_size = 0
    if shard_path is not None:
        max_id_len = max(len(item.encode("utf-8")) for item in ids)
        ids_array = np.asarray(
            [item.encode("utf-8") for item in ids],
            dtype=f"S{max_id_len}",
        )
        arrays = {
            "ids": ids_array,
            "atom_fea": np.concatenate(atom_chunks, axis=0).astype("<f4", copy=False),
            "nbr_fea": np.concatenate(nbr_chunks, axis=0).astype("<f4", copy=False),
            "nbr_fea_idx": np.concatenate(nbr_idx_chunks, axis=0).astype("<i8", copy=False),
            "atom_offsets": np.asarray(atom_offsets, dtype="<i8"),
        }
        path = Path(shard_path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            if payload["compressed"]:
                np.savez_compressed(handle, **arrays)
            else:
                np.savez(handle, **arrays)
        tmp_path.replace(path)
        byte_size = path.stat().st_size

    return {
        "shard_index": payload["shard_index"],
        "shard_name": shard_name,
        "count": len(ids),
        "atom_count": int(atom_offsets[-1]),
        "byte_size": byte_size,
        "graph_nbytes": graph_nbytes,
        "ids": ids,
        "mp_ids": mp_ids,
        "targets": targets,
    }


def records_to_worker_dicts(records: Sequence[StreamRecord]) -> list[dict[str, Any]]:
    return [
        {
            "graph_id": record.graph_id,
            "mp_id": record.mp_id,
            "target": record.target,
            "structure_dict": record.structure_dict,
        }
        for record in records
    ]


def base_worker_payload(
    args: argparse.Namespace,
    shard_index: int,
    records: Sequence[StreamRecord],
    shard_path: Path | None,
) -> dict[str, Any]:
    return {
        "shard_index": shard_index,
        "shard_name": shard_path.name if shard_path is not None else None,
        "shard_path": str(shard_path) if shard_path is not None else None,
        "records": records_to_worker_dicts(records),
        "atom_init": str(args.atom_init),
        "max_num_nbr": args.max_num_nbr,
        "radius": args.radius,
        "dmin": args.dmin,
        "step": args.step,
        "compressed": args.compressed,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def write_graph_ids(path: Path, rows: Sequence[tuple[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id", "mp_id"])
        writer.writerows(rows)


def write_id_prop(path: Path, rows: Sequence[tuple[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for graph_id, target in rows:
            writer.writerow([graph_id, f"{target:.16g}"])


def write_split(path: Path, ids: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id"])
        writer.writerows([[item] for item in ids])


def make_group_splits(
    graph_rows: Sequence[tuple[str, str, float]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[str]]:
    group_ids = sorted({mp_id for _, mp_id, _ in graph_rows})
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    train_group_count = int(train_ratio * len(group_ids))
    val_group_count = int(val_ratio * len(group_ids))
    train_groups = set(group_ids[:train_group_count])
    val_groups = set(group_ids[train_group_count : train_group_count + val_group_count])
    test_groups = set(group_ids[train_group_count + val_group_count :])

    splits = {"train": [], "val": [], "test": []}
    for graph_id, mp_id, _ in graph_rows:
        if mp_id in train_groups:
            splits["train"].append(graph_id)
        elif mp_id in val_groups:
            splits["val"].append(graph_id)
        elif mp_id in test_groups:
            splits["test"].append(graph_id)
        else:
            raise RuntimeError(f"Graph row has unknown group: {mp_id}")
    return splits


def validate_split_groups(
    graph_rows: Sequence[tuple[str, str, float]],
    splits: dict[str, list[str]],
) -> None:
    mp_by_graph_id = {graph_id: mp_id for graph_id, mp_id, _ in graph_rows}
    seen: dict[str, str] = {}
    for split_name, ids in splits.items():
        for graph_id in ids:
            mp_id = mp_by_graph_id[graph_id]
            previous = seen.setdefault(mp_id, split_name)
            if previous != split_name:
                raise RuntimeError(
                    f"mp_id {mp_id} appears in both {previous} and {split_name}"
                )


def read_split_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return [row[0] for row in reader if row]


def validate_cache(cache_dir: Path, expected_count: int) -> None:
    from cgcnn.data import CachedGraphData

    dataset = CachedGraphData(
        cache_dir,
        id_prop_file=cache_dir / "labels" / "ef_per_atom" / "id_prop.csv",
        shuffle=False,
        max_cached_shards=1,
    )
    if len(dataset) != expected_count:
        raise RuntimeError(f"Dataset length mismatch: {len(dataset)} != {expected_count}")
    for idx in sorted({0, len(dataset) // 2, len(dataset) - 1}):
        (atom_fea, nbr_fea, nbr_fea_idx), target, graph_id = dataset[idx]
        if atom_fea.ndim != 2 or nbr_fea.ndim != 3 or nbr_fea_idx.ndim != 2:
            raise RuntimeError(f"Invalid cached tensor shapes for {graph_id}")
        if target.ndim != 1 or target.numel() != 1:
            raise RuntimeError(f"Invalid target shape for {graph_id}")


def build_readme_text(
    run_dir: Path,
    args: argparse.Namespace,
    created_at: str,
    command_line: Sequence[str],
) -> str:
    return f"""# MPtrj ef_per_atom CGCNN run

Created: `{created_at}`

## Purpose

Train and evaluate CGCNN on every usable structure record in the MPtrj 2022.9
full JSON dataset with `ef_per_atom` as the per-frame formation-energy target.

## Inputs

- Dataset JSON: `{args.dataset}`
- Dataset schema: `{args.schema}`
- Target column: `ef_per_atom`
- Structure provenance: structures are read in place from the JSON and loaded
  with `pymatgen.core.Structure.from_dict`; the 12 GB JSON is not copied.
- Atom features: `{args.atom_init}`
- Repository: `{REPO_ROOT}`
- Command-line options: `{" ".join(command_line)}`

## Outputs

- `distribution_report.json`: full target/energy/atom-count audit.
- `benchmark_10k/benchmark_report.json`: temporary 10k CPU graph-build
  benchmark used to estimate full-cache scale.
- `graph_cache/`: sharded graph cache and `labels/ef_per_atom/id_prop.csv`.
- `graph_cache/labels/ef_per_atom/splits/`: group-safe train/val/test splits.
- `training/`: checkpoints, metrics history, test predictions, and summary.
- `slurm-*.out` and `slurm-*.err`: Slurm logs for submitted jobs.
- `RUN_LOG.md`: chronological run notes.

Temporary benchmark artifacts are isolated under `benchmark_10k/` so they can be
removed after the full run succeeds unless they are intentionally retained.
"""


def run_benchmark(args: argparse.Namespace) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = args.run_dir / "benchmark_10k"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    append_log(args.run_dir, f"Starting {args.limit}-record CPU graph benchmark.")
    created_at = now_iso()
    start = time.perf_counter()
    distribution = DistributionAccumulator(args.target_column)
    payloads: list[dict[str, Any]] = []
    shard_records: list[StreamRecord] = []
    shard_index = 0
    for _, records, frame_count, nonfinite_targets in stream_records(
        args.dataset,
        args.target_column,
        args.limit,
    ):
        distribution.add_material(frame_count)
        distribution.add_nonfinite_targets(nonfinite_targets)
        for record in records:
            distribution.add_record(record)
            shard_records.append(record)
            if len(shard_records) >= args.shard_size:
                payloads.append(base_worker_payload(args, shard_index, shard_records, None))
                shard_records = []
                shard_index += 1
    if shard_records:
        payloads.append(base_worker_payload(args, shard_index, shard_records, None))

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(graph_worker, payload) for payload in payloads]
        for future in futures:
            result = future.result()
            results.append(result)
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"benchmark shard {result['shard_index'] + 1}/{len(payloads)} "
                f"records={result['count']} atoms={result['atom_count']}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    records = sum(result["count"] for result in results)
    atoms = sum(result["atom_count"] for result in results)
    graph_nbytes = sum(result["graph_nbytes"] for result in results)
    bytes_per_record = graph_nbytes / records if records else None
    bytes_per_atom = graph_nbytes / atoms if atoms else None
    report = {
        "created_at": created_at,
        "completed_at": now_iso(),
        "dataset": str(args.dataset),
        "target_column": args.target_column,
        "limit": args.limit,
        "workers": args.workers,
        "shard_size": args.shard_size,
        "elapsed_seconds": elapsed,
        "records": records,
        "atoms": atoms,
        "records_per_second": records / elapsed if elapsed else None,
        "atoms_per_second": atoms / elapsed if elapsed else None,
        "graph_nbytes": graph_nbytes,
        "bytes_per_record": bytes_per_record,
        "bytes_per_atom": bytes_per_atom,
        "estimated_full": {
            "expected_records": EXPECTED_RECORDS,
            "expected_atoms_from_atom_rate": (
                EXPECTED_RECORDS * atoms / records if records else None
            ),
            "seconds_from_record_rate": (
                EXPECTED_RECORDS / (records / elapsed) if records and elapsed else None
            ),
            "graph_nbytes_from_record_rate": (
                EXPECTED_RECORDS * bytes_per_record if bytes_per_record is not None else None
            ),
        },
        "distribution_sample": distribution.as_dict(),
        "package_versions": package_versions(),
        "code": git_info(),
    }
    write_json(benchmark_dir / "benchmark_report.json", report)
    append_log(args.run_dir, "Benchmark complete.")
    print(json.dumps(report, indent=2), flush=True)


def package_versions() -> dict[str, str | None]:
    return {
        "python": sys.version,
        "ijson": package_version("ijson"),
        "numpy": np.__version__,
        "pymatgen": package_version("pymatgen"),
        "torch": package_version("torch"),
    }


def submit_limited(
    executor: ProcessPoolExecutor,
    pending: dict[Any, int],
    payload: dict[str, Any],
    max_pending: int,
    results: list[dict[str, Any]],
) -> None:
    while len(pending) >= max_pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            pending.pop(future)
            result = future.result()
            results.append(result)
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"shard {result['shard_index']} done records={result['count']} "
                f"atoms={result['atom_count']}",
                flush=True,
            )
    future = executor.submit(graph_worker, payload)
    pending[future] = payload["shard_index"]


def build_cache(args: argparse.Namespace) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir
    shards_dir = cache_dir / "shards"
    label_dir = cache_dir / "labels" / "ef_per_atom"
    split_dir = label_dir / "splits"
    shards_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    if not (cache_dir / "atom_init.json").exists():
        shutil.copy2(args.atom_init, cache_dir / "atom_init.json")

    append_log(args.run_dir, "Starting full MPtrj distribution scan and graph cache build.")
    created_at = now_iso()
    start = time.perf_counter()
    distribution = DistributionAccumulator(args.target_column)
    results: list[dict[str, Any]] = []
    pending: dict[Any, int] = {}
    shard_records: list[StreamRecord] = []
    shard_index = 0
    max_pending = max(args.workers * 2, 1)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _, records, frame_count, nonfinite_targets in stream_records(
            args.dataset,
            args.target_column,
        ):
            distribution.add_material(frame_count)
            distribution.add_nonfinite_targets(nonfinite_targets)
            for record in records:
                distribution.add_record(record)
                shard_records.append(record)
                if len(shard_records) >= args.shard_size:
                    shard_path = shards_dir / f"shard_{shard_index:05d}.npz"
                    payload = base_worker_payload(args, shard_index, shard_records, shard_path)
                    submit_limited(executor, pending, payload, max_pending, results)
                    shard_records = []
                    shard_index += 1
        if shard_records:
            shard_path = shards_dir / f"shard_{shard_index:05d}.npz"
            payload = base_worker_payload(args, shard_index, shard_records, shard_path)
            submit_limited(executor, pending, payload, max_pending, results)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                results.append(result)
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"shard {result['shard_index']} done records={result['count']} "
                    f"atoms={result['atom_count']}",
                    flush=True,
                )

    results.sort(key=lambda row: row["shard_index"])
    elapsed = time.perf_counter() - start
    index: dict[str, dict[str, int | str]] = {}
    shards: list[dict[str, Any]] = []
    graph_rows: list[tuple[str, str, float]] = []
    for result in results:
        shard_name = result["shard_name"]
        shards.append(
            {
                "file": shard_name,
                "count": result["count"],
                "atom_count": result["atom_count"],
                "byte_size": result["byte_size"],
            }
        )
        for row_index, graph_id in enumerate(result["ids"]):
            if graph_id in index:
                raise RuntimeError(f"Duplicate graph ID: {graph_id}")
            index[graph_id] = {"shard": shard_name, "row": row_index}
            graph_rows.append(
                (graph_id, result["mp_ids"][row_index], result["targets"][row_index])
            )

    distribution_payload = distribution.as_dict()
    if args.expected_materials and distribution.material_count != args.expected_materials:
        raise RuntimeError(
            f"Material count mismatch: {distribution.material_count} != {args.expected_materials}"
        )
    if args.expected_records and distribution.record_count != args.expected_records:
        raise RuntimeError(
            f"Record count mismatch: {distribution.record_count} != {args.expected_records}"
        )
    if distribution.nonfinite_target_count:
        raise RuntimeError(f"Found nonfinite {args.target_column} targets.")
    if len(index) != distribution.usable_target_count:
        raise RuntimeError(
            f"Graph count mismatch: {len(index)} != {distribution.usable_target_count}"
        )

    splits = make_group_splits(
        graph_rows,
        seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    validate_split_groups(graph_rows, splits)
    for split_name, ids in splits.items():
        write_split(split_dir / f"{split_name}_ids.csv", ids)
    write_id_prop(label_dir / "id_prop.csv", [(row[0], row[2]) for row in graph_rows])
    write_graph_ids(cache_dir / "graph_ids.csv", [(row[0], row[1]) for row in graph_rows])

    label_meta = {
        "name": "ef_per_atom",
        "target_columns": ["ef_per_atom"],
        "id_prop_file": "labels/ef_per_atom/id_prop.csv",
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "split_files": {
            name: f"labels/ef_per_atom/splits/{name}_ids.csv"
            for name in splits
        },
        "split_group": "mp_id",
    }
    graph_parameters = {
        "max_num_nbr": args.max_num_nbr,
        "radius": args.radius,
        "dmin": args.dmin,
        "step": args.step,
        "gaussian_filter_length": len(np.arange(args.dmin, args.radius + args.step, args.step)),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "completed_at": now_iso(),
        "workspace": str(REPO_ROOT),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset) if args.hash_dataset else None,
        "schema": str(args.schema),
        "target_column": args.target_column,
        "atom_init": str(args.atom_init),
        "atom_init_copy": "atom_init.json",
        "atom_init_sha256": sha256_file(args.atom_init),
        "graph_count": len(index),
        "distribution": distribution_payload,
        "graph_parameters": graph_parameters,
        "split_seed": args.split_seed,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "labels": {"ef_per_atom": label_meta},
        "default_label": "ef_per_atom",
        "shard_size": args.shard_size,
        "compressed": args.compressed,
        "shards": shards,
        "index": index,
        "elapsed_seconds": elapsed,
        "package_versions": package_versions(),
        "code": git_info(),
        "command_line_options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    write_json(args.run_dir / "distribution_report.json", distribution_payload)
    write_json(cache_dir / "manifest.json", manifest)
    validate_cache(cache_dir, expected_count=len(index))
    append_log(args.run_dir, "Full graph cache build and validation complete.")
    print(json.dumps({"cache_dir": str(cache_dir), "graph_count": len(index)}, indent=2))


def train_from_cache(args: argparse.Namespace) -> None:
    training_dir = args.run_dir / "training"
    checkpoint_dir = training_dir / "checkpoints"
    training_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    label_dir = args.cache_dir / "labels" / "ef_per_atom"
    split_dir = label_dir / "splits"
    append_log(args.run_dir, "Starting CGCNN training from MPtrj graph cache.")
    os.chdir(training_dir)
    best_checkpoint = train_model(
        str(args.cache_dir),
        task="regression",
        dataset_format="graph_cache",
        id_prop_file=str(label_dir / "id_prop.csv"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        workers=args.workers,
        device=get_env_device() if args.device == "auto" else args.device,
        resume=str(args.resume) if args.resume is not None else None,
        resume_lr=args.resume_lr,
        checkpoint_dir=str(checkpoint_dir),
        metrics_history_path=str(training_dir / "metrics_history.json"),
        print_freq=args.print_freq,
        train_ids=read_split_ids(split_dir / "train_ids.csv"),
        val_ids=read_split_ids(split_dir / "val_ids.csv"),
        test_ids=read_split_ids(split_dir / "test_ids.csv"),
        graph_cache_max_cached_shards=args.max_cached_shards,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    summary = {
        "completed_at": now_iso(),
        "cache_dir": str(args.cache_dir),
        "target_column": "ef_per_atom",
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "resume": str(args.resume) if args.resume is not None else None,
        "resume_lr": args.resume_lr,
        "workers": args.workers,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "max_cached_shards": args.max_cached_shards,
        "device": args.device,
    }
    write_json(training_dir / "training_summary.json", summary)
    append_log(args.run_dir, "Training command complete.")
    print(json.dumps(summary, indent=2), flush=True)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--atom-init", type=Path, default=DEFAULT_ATOM_INIT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-column", default="ef_per_atom")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 2)))
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-num-nbr", type=int, default=12)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--dmin", type=float, default=0.0)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--compressed", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream MPtrj JSON records into a CGCNN graph cache and train CGCNN."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    add_common_args(benchmark_parser)
    benchmark_parser.add_argument("--limit", type=int, default=10_000)

    build_parser = subparsers.add_parser("build-cache")
    add_common_args(build_parser)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    build_parser.add_argument("--train-ratio", type=float, default=0.8)
    build_parser.add_argument("--val-ratio", type=float, default=0.1)
    build_parser.add_argument("--test-ratio", type=float, default=0.1)
    build_parser.add_argument("--expected-materials", type=int, default=EXPECTED_MATERIALS)
    build_parser.add_argument("--expected-records", type=int, default=EXPECTED_RECORDS)
    build_parser.add_argument("--hash-dataset", action="store_true")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--run-dir", type=Path, required=True)
    train_parser.add_argument("--cache-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=0.01)
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.add_argument("--resume-lr", type=float, default=None)
    train_parser.add_argument("--workers", type=int, default=16)
    train_parser.add_argument("--persistent-workers", action="store_true")
    train_parser.add_argument("--prefetch-factor", type=int, default=None)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--print-freq", type=int, default=50)
    train_parser.add_argument("--max-cached-shards", type=int, default=512)

    args = parser.parse_args()
    if hasattr(args, "dataset"):
        args.dataset = args.dataset.expanduser().resolve()
        args.schema = args.schema.expanduser().resolve()
        args.atom_init = args.atom_init.expanduser().resolve()
        if not args.dataset.is_file():
            raise FileNotFoundError(args.dataset)
        if not args.atom_init.is_file():
            raise FileNotFoundError(args.atom_init)
        if args.shard_size <= 0:
            raise ValueError("--shard-size must be positive.")
        if args.workers <= 0:
            raise ValueError("--workers must be positive.")
    args.run_dir = args.run_dir.expanduser().resolve()
    if hasattr(args, "output_dir"):
        args.output_dir = args.output_dir.expanduser().resolve()
        ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(ratio_sum - 1.0) > 1e-8:
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
    if hasattr(args, "cache_dir"):
        args.cache_dir = args.cache_dir.expanduser().resolve()
    if getattr(args, "resume", None) is not None:
        args.resume = args.resume.expanduser().resolve()
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.command == "benchmark":
        run_benchmark(cli_args)
    elif cli_args.command == "build-cache":
        build_cache(cli_args)
    elif cli_args.command == "train":
        train_from_cache(cli_args)
    else:
        raise SystemExit(f"Unknown command: {cli_args.command}")
