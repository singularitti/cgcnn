from __future__ import annotations

import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pymatgen.core.structure import Structure

from cgcnn.data import (
    AtomCustomJSONInitializer,
    GaussianDistance,
)
from cgcnn.device import get_env_device
from cgcnn.inference import predict_model


@dataclass(frozen=True)
class ChunkTask:
    chunk_index: int
    ids_path: Path
    chunk_dir: Path


class ExternalCIFData:
    """CGCNN-compatible dataset that reads CIFs from an external directory tree."""

    def __init__(
        self,
        structures_dir: Path,
        atom_init_path: Path,
        id_prop_path: Path,
        max_num_nbr: int = 12,
        radius: float = 8.0,
        dmin: float = 0.0,
        step: float = 0.2,
    ):
        self.structures_dir = structures_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.ari = AtomCustomJSONInitializer(str(atom_init_path))
        with id_prop_path.open() as handle:
            reader = csv.reader(handle)
            self.id_prop_data = [row for row in reader if row]
        if not self.id_prop_data:
            raise ValueError(f"id_prop.csv is empty: {id_prop_path}")
        self.n_targets = len(self.id_prop_data[0]) - 1

    def __len__(self) -> int:
        return len(self.id_prop_data)

    def __getitem__(self, idx: int):
        import numpy as np
        import torch

        cif_id, target_values = self.id_prop_data[idx][0], self.id_prop_data[idx][1:]
        crystal = Structure.from_file(resolve_cif_path(self.structures_dir, cif_id))
        atom_fea = np.vstack([
            self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))
        ])
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr))
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        target = [float(value) for value in target_values]
        return (
            torch.tensor(atom_fea, dtype=torch.float32),
            torch.tensor(nbr_fea, dtype=torch.float32),
            torch.tensor(nbr_fea_idx, dtype=torch.long),
        ), torch.tensor(target, dtype=torch.float32), cif_id


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.symlink_to(target_path)


def write_chunk_ids(chunk_path: Path, chunk_ids: list[str]) -> None:
    with chunk_path.open("w") as handle:
        for cif_id in chunk_ids:
            handle.write(f"{cif_id}\n")


def write_chunk_targets(id_prop_path: Path, chunk_ids: list[str]) -> None:
    with id_prop_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for cif_id in chunk_ids:
            writer.writerow([cif_id, 0.0])


def predict_chunk(
    task: ChunkTask,
    *,
    structures_dir: str,
    atom_init_path: str,
    model_path: str,
    batch_size: int,
    print_freq: int,
) -> dict[str, object]:
    start = time.time()
    chunk_dir = task.chunk_dir
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ids = [line.strip() for line in task.ids_path.read_text().splitlines() if line.strip()]
    id_prop_path = chunk_dir / "id_prop.csv"
    write_chunk_targets(id_prop_path, ids)
    output_csv = chunk_dir / "predictions.csv"
    dataset = ExternalCIFData(
        structures_dir=Path(structures_dir),
        atom_init_path=Path(atom_init_path),
        id_prop_path=id_prop_path,
    )
    predict_model(
        dataset=dataset,
        task="regression",
        modelpath=model_path,
        batch_size=batch_size,
        workers=0,
        device=get_env_device(),
        print_freq=print_freq,
        shuffle=False,
        output_csv=str(output_csv),
    )
    prediction_values = []
    with output_csv.open() as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                prediction_values.append(float(row[2]))
    elapsed = time.time() - start
    status = {
        "chunk_index": task.chunk_index,
        "chunk_dir": str(chunk_dir),
        "ids_file": str(task.ids_path),
        "prediction_csv": str(output_csv),
        "n_structures": len(ids),
        "elapsed_seconds": elapsed,
        "prediction_min": min(prediction_values) if prediction_values else None,
        "prediction_max": max(prediction_values) if prediction_values else None,
        "prediction_mean": (
            sum(prediction_values) / len(prediction_values) if prediction_values else None
        ),
    }
    (chunk_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def resolve_cif_path(structures_dir: Path, cif_id: str) -> Path:
    flat_path = structures_dir / f"{cif_id}.cif"
    if flat_path.exists():
        return flat_path
    if "_" in cif_id:
        nested_path = structures_dir / cif_id.split("_", 1)[0] / f"{cif_id}.cif"
        if nested_path.exists():
            return nested_path
    matches = list(structures_dir.rglob(f"{cif_id}.cif"))
    if not matches:
        raise FileNotFoundError(f"Could not find CIF for structure id {cif_id} under {structures_dir}")
    if len(matches) > 1:
        raise ValueError(f"Found multiple CIFs for structure id {cif_id} under {structures_dir}")
    return matches[0]


def load_structure_ids(structures_dir: Path) -> list[str]:
    structure_ids = []
    seen_ids = set()
    for path in structures_dir.rglob("*.cif"):
        cif_id = path.stem
        if cif_id in seen_ids:
            raise ValueError(f"Duplicate structure id discovered: {cif_id}")
        seen_ids.add(cif_id)
        structure_ids.append(cif_id)
    return sorted(structure_ids)


def create_chunk_tasks(base_dir: Path, structure_ids: list[str], chunk_size: int) -> list[ChunkTask]:
    ids_dir = base_dir / "chunk_ids"
    chunks_dir = base_dir / "chunks"
    ids_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for chunk_index in range(math.ceil(len(structure_ids) / chunk_size)):
        start = chunk_index * chunk_size
        end = min(len(structure_ids), start + chunk_size)
        chunk_ids = structure_ids[start:end]
        chunk_ids_path = ids_dir / f"chunk_{chunk_index:04d}_ids.txt"
        write_chunk_ids(chunk_ids_path, chunk_ids)
        tasks.append(
            ChunkTask(
                chunk_index=chunk_index,
                ids_path=chunk_ids_path,
                chunk_dir=chunks_dir / f"chunk_{chunk_index:04d}",
            )
        )
    return tasks


def merge_predictions(base_dir: Path, chunk_statuses: list[dict[str, object]]) -> dict[str, object]:
    merged_dir = base_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = merged_dir / "predictions.csv"
    row_count = 0
    prediction_sum = 0.0
    prediction_min = None
    prediction_max = None
    with merged_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["structure_id", "dummy_target", "predicted_formation_energy_per_atom"])
        for status in sorted(chunk_statuses, key=lambda item: int(item["chunk_index"])):
            with Path(str(status["prediction_csv"])).open() as chunk_handle:
                reader = csv.reader(chunk_handle)
                for row in reader:
                    if not row:
                        continue
                    writer.writerow(row)
                    prediction = float(row[2])
                    prediction_sum += prediction
                    prediction_min = prediction if prediction_min is None else min(prediction_min, prediction)
                    prediction_max = prediction if prediction_max is None else max(prediction_max, prediction)
                    row_count += 1
    summary = {
        "merged_csv": str(merged_csv),
        "prediction_count": row_count,
        "prediction_mean": (prediction_sum / row_count) if row_count else None,
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
    }
    (merged_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_filtered_predictions(merged_csv: Path, threshold: float) -> dict[str, object]:
    output_csv = merged_csv.with_name(f"{merged_csv.stem}.lt_neg_{abs(threshold):g}.sorted.csv")
    filtered_rows = []
    with merged_csv.open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if row and float(row[2]) < threshold:
                filtered_rows.append(row)
    filtered_rows.sort(key=lambda row: float(row[2]))
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(filtered_rows)
    summary = {
        "threshold": threshold,
        "filtered_csv": str(output_csv),
        "row_count": len(filtered_rows),
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_run_readme(
    output_dir: Path,
    *,
    structures_dir: Path,
    training_dataset_dir: Path,
    model_path: Path,
    training_summary_path: Path,
    chunk_size: int,
    max_workers: int,
    batch_size: int,
    structure_count: int,
    chunk_count: int,
    merged_summary: dict[str, object],
    filtered_summary: dict[str, object],
) -> None:
    readme = f"""# CGCNN formation-energy inference on external ternary structures

## What this run did

This folder applies the saved CGCNN regression checkpoint at `{model_path}` to the CIF corpus under `{structures_dir}`.
The checkpoint provenance comes from `{training_summary_path}`, whose summary reports:

- target: `formation_energy_per_atom`
- training rows: `154879`
- best epoch: `15`
- best validation MAE: `0.09187468886375427`
- test MAE: `0.09025649543994237`

The inference job used the atom embeddings from `{training_dataset_dir / "atom_init.json"}` and processed the external corpus in `{chunk_count}` chunks of up to `{chunk_size}` structures each, with up to `{max_workers}` Python worker processes running chunks in parallel. Inside each chunk, CGCNN inference used `workers=0` to avoid macOS multiprocessing issues during data loading.

## Inputs linked here

The `links/` subfolder contains symlinks to the source inputs and provenance files used for this run:

- `model_best.pth.tar`
- `training_summary.json`
- `training_dataset_atom_init.json`
- `training_dataset_id_prop.csv`
- `source_structures`

## Outputs

- `merged/predictions.csv`: combined prediction table with columns `structure_id`, `dummy_target`, and `predicted_formation_energy_per_atom`
- `merged/merge_summary.json`: global prediction count and aggregate statistics
- `merged/{Path(str(filtered_summary["filtered_csv"])).name}`: rows with predicted formation energy per atom below `{filtered_summary["threshold"]}`, sorted ascending by prediction
- `chunks/chunk_*/predictions.csv`: per-chunk CGCNN outputs
- `chunks/chunk_*/status.json`: per-chunk status and summary statistics
- `chunk_ids/`: the structure IDs assigned to each chunk
- `run_summary.json`: top-level run metadata for this folder

## Result summary

- structures discovered: `{structure_count}`
- merged prediction rows: `{merged_summary["prediction_count"]}`
- predicted formation-energy minimum: `{merged_summary["prediction_min"]}`
- predicted formation-energy maximum: `{merged_summary["prediction_max"]}`
- predicted formation-energy mean: `{merged_summary["prediction_mean"]}`
- filtered rows below `{filtered_summary["threshold"]}`: `{filtered_summary["row_count"]}`
"""
    (output_dir / "README.md").write_text(readme)


def write_run_summary(output_dir: Path, payload: dict[str, object]) -> None:
    (output_dir / "run_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def parse_args(argv: list[str]) -> dict[str, object]:
    if len(argv) != 9:
        raise SystemExit(
            "Usage: uv run python tools/predict_external_formation_energy.py "
            "<structures_dir> <training_dataset_dir> <training_summary.json> "
            "<model_best.pth.tar> <output_dir> <chunk_size> <max_workers> <batch_size>"
        )
    return {
        "structures_dir": Path(argv[1]).expanduser().resolve(),
        "training_dataset_dir": Path(argv[2]).expanduser().resolve(),
        "training_summary_path": Path(argv[3]).expanduser().resolve(),
        "model_path": Path(argv[4]).expanduser().resolve(),
        "output_dir": Path(argv[5]).expanduser().resolve(),
        "chunk_size": int(argv[6]),
        "max_workers": int(argv[7]),
        "batch_size": int(argv[8]),
    }


if __name__ == "__main__":
    args = parse_args(sys.argv)
    structures_dir = args["structures_dir"]
    training_dataset_dir = args["training_dataset_dir"]
    training_summary_path = args["training_summary_path"]
    model_path = args["model_path"]
    output_dir = args["output_dir"]
    chunk_size = args["chunk_size"]
    max_workers = args["max_workers"]
    batch_size = args["batch_size"]
    atom_init_path = training_dataset_dir / "atom_init.json"
    id_prop_path = training_dataset_dir / "id_prop.csv"

    if output_dir.exists():
        raise SystemExit(f"Refusing to reuse existing output dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=False)
    links_dir = output_dir / "links"
    links_dir.mkdir()
    ensure_symlink(links_dir / "model_best.pth.tar", model_path)
    ensure_symlink(links_dir / "training_summary.json", training_summary_path)
    ensure_symlink(links_dir / "training_dataset_atom_init.json", atom_init_path)
    ensure_symlink(links_dir / "training_dataset_id_prop.csv", id_prop_path)
    ensure_symlink(links_dir / "source_structures", structures_dir)

    structure_ids = load_structure_ids(structures_dir)
    chunk_tasks = create_chunk_tasks(output_dir, structure_ids, chunk_size)
    print(
        f"Prepared {len(chunk_tasks)} chunks for {len(structure_ids)} structures "
        f"with chunk_size={chunk_size}"
    )

    chunk_statuses = []
    start = time.time()
    print_freq = max(1, math.ceil(chunk_size / batch_size / 20))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                predict_chunk,
                task,
                structures_dir=str(structures_dir),
                atom_init_path=str(atom_init_path),
                model_path=str(model_path),
                batch_size=batch_size,
                print_freq=print_freq,
            )
            for task in chunk_tasks
        ]
        for future in as_completed(futures):
            status = future.result()
            chunk_statuses.append(status)
            print(
                f"Completed chunk {status['chunk_index']:04d} "
                f"({status['n_structures']} structures, {status['elapsed_seconds']:.1f}s)"
            )

    merged_summary = merge_predictions(output_dir, chunk_statuses)
    filtered_summary = write_filtered_predictions(Path(str(merged_summary["merged_csv"])), threshold=-0.2)
    elapsed = time.time() - start
    run_summary = {
        "structures_dir": str(structures_dir),
        "training_dataset_dir": str(training_dataset_dir),
        "training_summary_path": str(training_summary_path),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "chunk_size": chunk_size,
        "max_workers": max_workers,
        "batch_size": batch_size,
        "structure_count": len(structure_ids),
        "chunk_count": len(chunk_tasks),
        "elapsed_seconds": elapsed,
        "chunk_statuses": sorted(chunk_statuses, key=lambda item: int(item["chunk_index"])),
        "merged_summary": merged_summary,
        "filtered_summary": filtered_summary,
    }
    write_run_summary(output_dir, run_summary)
    write_run_readme(
        output_dir,
        structures_dir=structures_dir,
        training_dataset_dir=training_dataset_dir,
        model_path=model_path,
        training_summary_path=training_summary_path,
        chunk_size=chunk_size,
        max_workers=max_workers,
        batch_size=batch_size,
        structure_count=len(structure_ids),
        chunk_count=len(chunk_tasks),
        merged_summary=merged_summary,
        filtered_summary=filtered_summary,
    )
    print(
        f"Finished {len(structure_ids)} predictions in {elapsed:.1f}s. "
        f"Merged CSV: {merged_summary['merged_csv']}"
    )
