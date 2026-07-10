from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cgcnn.device import get_env_device
from cgcnn.inference import predict_model, predict_regression_models
from predict_external_formation_energy import ExternalCIFData, resolve_cif_path


DEFAULT_STRUCTURES_DIR = Path(
    "/scratch/04996/tg842951/exa_amd_MnBiO_20260709/work_dir/Mn-Bi-O/structures"
)
DEFAULT_MPTRJ_MODEL = Path(
    "/work2/04996/tg842951/stampede3/run/mptrj_ef_per_atom_20260705_044309_results/checkpoints/best_validation.pth.tar"
)
DEFAULT_MY_MODEL = Path(
    "/work2/04996/tg842951/stampede3/run/MnBiO/predict/my new/my_formation_energy_mp_all.pth.tar"
)
DEFAULT_XIE_MODEL = Path(
    "/work2/04996/tg842951/stampede3/run/MnBiO/predict/xie old/xie-formation-energy-per-atom.pth.tar"
)
DEFAULT_ATOM_INIT = REPO_ROOT / "data" / "sample-regression" / "atom_init.json"
DEFAULT_MPTRJ_SUMMARY = Path(
    "/work2/04996/tg842951/stampede3/run/mptrj_ef_per_atom_20260705_044309_results/final_training_summary.json"
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    path: Path


@dataclass(frozen=True)
class ChunkTask:
    model_key: str
    model_label: str
    model_path: str
    chunk_index: int
    ids_path: Path
    chunk_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict Mn-Bi-O formation energies with three CGCNN checkpoints. "
            "Use --mode shard inside Slurm array jobs, then --mode merge after all shards finish."
        )
    )
    parser.add_argument("--mode", choices=["shard", "merge"], required=True)
    parser.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCTURES_DIR)
    parser.add_argument(
        "--structure-ids-file",
        type=Path,
        default=None,
        help="Optional precomputed structure-ID list to avoid rescanning the CIF tree.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atom-init", type=Path, default=DEFAULT_ATOM_INIT)
    parser.add_argument("--mptrj-model", type=Path, default=DEFAULT_MPTRJ_MODEL)
    parser.add_argument("--my-model", type=Path, default=DEFAULT_MY_MODEL)
    parser.add_argument("--xie-model", type=Path, default=DEFAULT_XIE_MODEL)
    parser.add_argument("--mptrj-summary", type=Path, default=DEFAULT_MPTRJ_SUMMARY)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--max-workers", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument(
        "--execution-mode",
        choices=["per-model", "shared-graphs"],
        default="per-model",
        help="Reuse each parsed graph batch across all models with shared-graphs.",
    )
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow faster TF32 matrix multiplication on supported CUDA GPUs.",
    )
    return parser.parse_args()


def model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    return [
        ModelSpec("mptrj_20260705", "MPTrj 20260705 best validation", args.mptrj_model),
        ModelSpec("my_mp_all", "my_formation_energy_mp_all", args.my_model),
        ModelSpec("xie_old", "Xie old formation-energy-per-atom", args.xie_model),
    ]


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if link_path.is_symlink():
            if os.readlink(link_path) == str(target_path):
                return
            link_path.unlink()
        elif link_path.exists():
            return
        link_path.symlink_to(target_path)
    except FileExistsError:
        return


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return value


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def checkpoint_summary(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    if not path.exists():
        return summary
    checkpoint = torch.load(path, map_location="cpu")
    summary.update(
        {
            "epoch": checkpoint.get("epoch"),
            "best_mae_error": checkpoint.get("best_mae_error"),
            "best_validation_score": checkpoint.get("best_validation_score"),
            "normalizer": jsonable(checkpoint.get("normalizer")),
            "args": jsonable(checkpoint.get("args", {})),
        }
    )
    return jsonable(summary)


def discover_structure_ids(structures_dir: Path) -> list[str]:
    structure_ids = []
    seen = set()
    for path in structures_dir.rglob("*.cif"):
        structure_id = path.stem
        if structure_id in seen:
            raise ValueError(f"Duplicate CIF stem discovered: {structure_id}")
        seen.add(structure_id)
        structure_ids.append(structure_id)
    return sorted(structure_ids)


def load_structure_ids(path: Path) -> list[str]:
    structure_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not structure_ids:
        raise ValueError(f"Structure ID file is empty: {path}")
    if len(set(structure_ids)) != len(structure_ids):
        raise ValueError(f"Structure ID file contains duplicates: {path}")
    return structure_ids


def select_shard(ids: list[str], shard_index: int, shard_count: int) -> list[str]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    return [structure_id for i, structure_id in enumerate(ids) if i % shard_count == shard_index]


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for structure_id in ids:
            handle.write(f"{structure_id}\n")


def write_id_prop(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for structure_id in ids:
            writer.writerow([structure_id, "0.0"])


def make_chunk_tasks(
    shard_dir: Path,
    ids: list[str],
    chunk_size: int,
    models: list[ModelSpec],
) -> list[ChunkTask]:
    tasks: list[ChunkTask] = []
    chunk_count = math.ceil(len(ids) / chunk_size)
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_size
        end = min(len(ids), start + chunk_size)
        chunk_ids = ids[start:end]
        ids_path = shard_dir / "chunk_ids" / f"chunk_{chunk_index:05d}.txt"
        write_ids(ids_path, chunk_ids)
        for model in models:
            tasks.append(
                ChunkTask(
                    model_key=model.key,
                    model_label=model.label,
                    model_path=str(model.path),
                    chunk_index=chunk_index,
                    ids_path=ids_path,
                    chunk_dir=shard_dir / "chunks" / model.key / f"chunk_{chunk_index:05d}",
                )
            )
    return tasks


def predict_chunk(
    task: ChunkTask,
    *,
    structures_dir: str,
    atom_init: str,
    batch_size: int,
    print_freq: int,
) -> dict[str, object]:
    torch.set_num_threads(1)
    start = time.time()
    ids = [line.strip() for line in task.ids_path.read_text().splitlines() if line.strip()]
    task.chunk_dir.mkdir(parents=True, exist_ok=True)
    id_prop = task.chunk_dir / "id_prop.csv"
    raw_csv = task.chunk_dir / "raw_predictions.csv"
    merged_csv = task.chunk_dir / "predictions.csv"
    write_id_prop(id_prop, ids)

    dataset = ExternalCIFData(
        structures_dir=Path(structures_dir),
        atom_init_path=Path(atom_init),
        id_prop_path=id_prop,
    )
    predict_model(
        dataset=dataset,
        task="regression",
        modelpath=task.model_path,
        batch_size=batch_size,
        workers=0,
        device=get_env_device(),
        print_freq=print_freq,
        shuffle=False,
        output_csv=str(raw_csv),
    )

    values = []
    with raw_csv.open(newline="") as src, merged_csv.open("w", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        writer.writerow(["structure_id", "dummy_target", "predicted_formation_energy_per_atom"])
        for row in reader:
            if not row:
                continue
            writer.writerow(row)
            values.append(float(row[2]))

    status = {
        "model_key": task.model_key,
        "model_label": task.model_label,
        "chunk_index": task.chunk_index,
        "ids_path": str(task.ids_path),
        "chunk_dir": str(task.chunk_dir),
        "predictions_csv": str(merged_csv),
        "raw_predictions_csv": str(raw_csv),
        "n_structures": len(ids),
        "elapsed_seconds": time.time() - start,
        "prediction_min": min(values) if values else None,
        "prediction_max": max(values) if values else None,
        "prediction_mean": float(np.mean(values)) if values else None,
    }
    write_json(task.chunk_dir / "status.json", status)
    return status


def merge_model_chunks(shard_dir: Path, model: ModelSpec) -> dict[str, object]:
    model_dir = shard_dir / "chunks" / model.key
    output_csv = shard_dir / "predictions" / f"{model.key}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    values = []
    with output_csv.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["structure_id", "dummy_target", "predicted_formation_energy_per_atom"])
        for chunk_csv in sorted(model_dir.glob("chunk_*/predictions.csv")):
            with chunk_csv.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    value = float(row["predicted_formation_energy_per_atom"])
                    writer.writerow([row["structure_id"], row["dummy_target"], f"{value:.16g}"])
                    values.append(value)
                    count += 1
    return {
        "model_key": model.key,
        "model_label": model.label,
        "predictions_csv": str(output_csv),
        "count": count,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": float(np.mean(values)) if values else None,
    }


def summarize_model_predictions(path: Path, model: ModelSpec) -> dict[str, object]:
    count = 0
    value_sum = 0.0
    value_min = math.inf
    value_max = -math.inf
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = float(row["predicted_formation_energy_per_atom"])
            count += 1
            value_sum += value
            value_min = min(value_min, value)
            value_max = max(value_max, value)
    return {
        "model_key": model.key,
        "model_label": model.label,
        "predictions_csv": str(path),
        "count": count,
        "min": value_min if count else None,
        "max": value_max if count else None,
        "mean": value_sum / count if count else None,
    }


def run_shared_graph_predictions(
    args: argparse.Namespace,
    shard_dir: Path,
    shard_ids: list[str],
    structures_dir: Path,
    atom_init: Path,
    models: list[ModelSpec],
) -> list[dict[str, object]]:
    id_prop = shard_dir / "id_prop.csv"
    write_id_prop(id_prop, shard_ids)
    dataset = ExternalCIFData(
        structures_dir=structures_dir,
        atom_init_path=atom_init,
        id_prop_path=id_prop,
    )
    predictions_dir = shard_dir / "predictions"
    output_csvs = {
        model.key: predictions_dir / f"{model.key}.csv" for model in models
    }

    if args.allow_tf32:
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    predict_regression_models(
        dataset,
        {model.key: model.path for model in models},
        output_csvs=output_csvs,
        batch_size=args.batch_size,
        workers=args.loader_workers,
        device=get_env_device(),
        print_freq=args.print_freq,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        write_header=True,
    )
    return [
        summarize_model_predictions(output_csvs[model.key], model) for model in models
    ]


def prepare_links(output_dir: Path, structures_dir: Path, atom_init: Path, models: list[ModelSpec]) -> None:
    links_dir = output_dir / "links"
    ensure_symlink(links_dir / "source_structures", structures_dir)
    ensure_symlink(links_dir / "atom_init.json", atom_init)
    ensure_symlink(links_dir / "cgcnn_repo", REPO_ROOT)
    ensure_symlink(links_dir / "script", Path(__file__).resolve())
    for model in models:
        ensure_symlink(links_dir / f"{model.key}.pth.tar", model.path)


def link_optional_provenance(output_dir: Path, args: argparse.Namespace) -> None:
    if args.mptrj_summary.exists():
        ensure_symlink(output_dir / "links" / "mptrj_final_training_summary.json", args.mptrj_summary.resolve())


def run_shard(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    structures_dir = args.structures_dir.resolve()
    atom_init = args.atom_init.resolve()
    models = [ModelSpec(item.key, item.label, item.path.resolve()) for item in model_specs(args)]
    for path in [structures_dir, atom_init, *[model.path for model in models]]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_links(output_dir, structures_dir, atom_init, models)
    link_optional_provenance(output_dir, args)
    if args.structure_ids_file is not None:
        ensure_symlink(
            output_dir / "links" / "source_structure_ids.txt",
            args.structure_ids_file.resolve(),
        )

    structure_ids_file = (
        args.structure_ids_file.resolve() if args.structure_ids_file is not None else None
    )
    if structure_ids_file is not None and not structure_ids_file.is_file():
        raise FileNotFoundError(structure_ids_file)
    all_ids = (
        load_structure_ids(structure_ids_file)
        if structure_ids_file is not None
        else discover_structure_ids(structures_dir)
    )
    shard_ids = select_shard(all_ids, args.shard_index, args.shard_count)
    shard_dir = output_dir / "shards" / f"shard_{args.shard_index:04d}_of_{args.shard_count:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_ids(shard_dir / "structure_ids.txt", shard_ids)
    tasks = (
        make_chunk_tasks(shard_dir, shard_ids, args.chunk_size, models)
        if args.execution_mode == "per-model"
        else []
    )

    run_metadata = {
        "mode": "shard",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "repo": str(REPO_ROOT),
        "git_revision": git_revision(),
        "structures_dir": str(structures_dir),
        "structure_ids_file": str(structure_ids_file) if structure_ids_file else None,
        "atom_init": str(atom_init),
        "output_dir": str(output_dir),
        "shard_dir": str(shard_dir),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "total_structure_count": len(all_ids),
        "shard_structure_count": len(shard_ids),
        "chunk_size": args.chunk_size,
        "chunk_count": math.ceil(len(shard_ids) / args.chunk_size),
        "execution_mode": args.execution_mode,
        "max_workers": args.max_workers,
        "loader_workers": args.loader_workers,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.persistent_workers,
        "batch_size": args.batch_size,
        "allow_tf32": args.allow_tf32,
        "models": [checkpoint_summary(model.path) for model in models],
    }
    write_json(shard_dir / "shard_input_summary.json", run_metadata)

    statuses = []
    start = time.time()
    if args.execution_mode == "shared-graphs":
        model_summaries = run_shared_graph_predictions(
            args,
            shard_dir,
            shard_ids,
            structures_dir,
            atom_init,
            models,
        )
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    predict_chunk,
                    task,
                    structures_dir=str(structures_dir),
                    atom_init=str(atom_init),
                    batch_size=args.batch_size,
                    print_freq=args.print_freq,
                )
                for task in tasks
            ]
            for future in as_completed(futures):
                status = future.result()
                statuses.append(status)
                print(
                    f"done {status['model_key']} chunk {status['chunk_index']:05d} "
                    f"n={status['n_structures']} elapsed={status['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        model_summaries = [merge_model_chunks(shard_dir, model) for model in models]
    run_metadata.update(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - start,
            "chunk_statuses": sorted(
                statuses,
                key=lambda item: (str(item["model_key"]), int(item["chunk_index"])),
            ),
            "model_summaries": model_summaries,
        }
    )
    write_json(shard_dir / "shard_summary.json", run_metadata)
    print(json.dumps({"shard_summary": str(shard_dir / "shard_summary.json")}, indent=2))


def load_prediction_csv(path: Path) -> dict[str, float]:
    predictions: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            structure_id = row["structure_id"]
            if structure_id in predictions:
                raise ValueError(f"Duplicate structure_id {structure_id!r} in {path}")
            predictions[structure_id] = float(row["predicted_formation_energy_per_atom"])
    return predictions


def write_merged_model_csv(output_dir: Path, models: list[ModelSpec]) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        predictions: dict[str, float] = {}
        shard_csvs = sorted((output_dir / "shards").glob(f"shard_*_of_*/predictions/{model.key}.csv"))
        if not shard_csvs:
            raise FileNotFoundError(f"No shard prediction CSVs found for {model.key}")
        for shard_csv in shard_csvs:
            shard_predictions = load_prediction_csv(shard_csv)
            overlap = set(predictions).intersection(shard_predictions)
            if overlap:
                raise ValueError(f"{model.key} has duplicate IDs across shards, first={sorted(overlap)[:5]}")
            predictions.update(shard_predictions)
        merged_csv = merged_dir / f"{model.key}_predictions.csv"
        values = np.array(list(predictions.values()), dtype=float)
        with merged_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["structure_id", "predicted_formation_energy_per_atom"])
            for structure_id in sorted(predictions):
                writer.writerow([structure_id, f"{predictions[structure_id]:.16g}"])
        merged[model.key] = {
            "model_label": model.label,
            "predictions_csv": str(merged_csv),
            "count": int(len(predictions)),
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
        }
    return merged


def compare_against_my(
    output_dir: Path,
    structures_dir: Path,
    predictions_by_model: dict[str, dict[str, float]],
) -> dict[str, object]:
    ids = sorted(set.intersection(*(set(values) for values in predictions_by_model.values())))
    if not ids:
        raise ValueError("No overlapping structure IDs across the three prediction sets.")
    missing = {
        key: len(set(ids).symmetric_difference(values))
        for key, values in predictions_by_model.items()
    }
    comparison_csv = output_dir / "merged" / "formation_energy_model_overlap.csv"
    fields = [
        "structure_id",
        "source_cif",
        "my_mp_all_predicted_formation_energy_per_atom",
        "mptrj_20260705_predicted_formation_energy_per_atom",
        "xie_old_predicted_formation_energy_per_atom",
        "mptrj_minus_my_mp_all",
        "xie_old_minus_my_mp_all",
    ]
    with comparison_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for structure_id in ids:
            my_value = predictions_by_model["my_mp_all"][structure_id]
            mptrj_value = predictions_by_model["mptrj_20260705"][structure_id]
            xie_value = predictions_by_model["xie_old"][structure_id]
            writer.writerow(
                {
                    "structure_id": structure_id,
                    "source_cif": str(resolve_cif_path(structures_dir, structure_id)),
                    "my_mp_all_predicted_formation_energy_per_atom": f"{my_value:.16g}",
                    "mptrj_20260705_predicted_formation_energy_per_atom": f"{mptrj_value:.16g}",
                    "xie_old_predicted_formation_energy_per_atom": f"{xie_value:.16g}",
                    "mptrj_minus_my_mp_all": f"{mptrj_value - my_value:.16g}",
                    "xie_old_minus_my_mp_all": f"{xie_value - my_value:.16g}",
                }
            )

    my_values = np.array([predictions_by_model["my_mp_all"][structure_id] for structure_id in ids])
    mptrj_values = np.array([predictions_by_model["mptrj_20260705"][structure_id] for structure_id in ids])
    xie_values = np.array([predictions_by_model["xie_old"][structure_id] for structure_id in ids])
    return {
        "comparison_csv": str(comparison_csv),
        "overlap_count": len(ids),
        "missing_or_extra_counts_relative_to_overlap": missing,
        "metrics": {
            "mptrj_20260705_vs_my_mp_all": pair_metrics(my_values, mptrj_values),
            "xie_old_vs_my_mp_all": pair_metrics(my_values, xie_values),
        },
    }


def pair_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    diff = y - x
    return {
        "pearson_r": float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan"),
        "mae_y_minus_x": float(np.mean(np.abs(diff))),
        "rmse_y_minus_x": float(np.sqrt(np.mean(diff**2))),
        "bias_y_minus_x": float(np.mean(diff)),
        "max_abs_y_minus_x": float(np.max(np.abs(diff))),
    }


def plot_overlap(comparison_csv: Path, output_dir: Path) -> dict[str, str]:
    ids = []
    my_values = []
    mptrj_values = []
    xie_values = []
    with comparison_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ids.append(row["structure_id"])
            my_values.append(float(row["my_mp_all_predicted_formation_energy_per_atom"]))
            mptrj_values.append(float(row["mptrj_20260705_predicted_formation_energy_per_atom"]))
            xie_values.append(float(row["xie_old_predicted_formation_energy_per_atom"]))
    x = np.asarray(my_values, dtype=float)
    y_mptrj = np.asarray(mptrj_values, dtype=float)
    y_xie = np.asarray(xie_values, dtype=float)
    all_values = np.concatenate([x, y_mptrj, y_xie])
    lo = float(np.nanmin(all_values))
    hi = float(np.nanmax(all_values))
    pad = max((hi - lo) * 0.06, 1e-6)
    lo -= pad
    hi += pad

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / "formation_energy_overlap_vs_my_mp_all.png"
    pdf = figures_dir / "formation_energy_overlap_vs_my_mp_all.pdf"

    plt.rcParams["agg.path.chunksize"] = 20000
    fig, ax = plt.subplots(figsize=(7.4, 7.0), dpi=220)
    ax.scatter(
        x,
        y_mptrj,
        s=1.0,
        alpha=0.08,
        color="#2563eb",
        linewidths=0,
        rasterized=True,
        label="MPTrj 20260705 vs my MP-all",
    )
    ax.scatter(
        x,
        y_xie,
        s=1.0,
        alpha=0.08,
        color="#dc2626",
        linewidths=0,
        rasterized=True,
        label="Xie old vs my MP-all",
    )
    ax.plot([lo, hi], [lo, hi], color="#111827", linewidth=1.0, linestyle="--", label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("my_formation_energy_mp_all prediction (eV/atom)")
    ax.set_ylabel("comparison model prediction (eV/atom)")
    ax.set_title(f"Mn-Bi-O formation-energy prediction overlap ({len(ids):,} structures)")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.legend(loc="best", frameon=True, markerscale=6)
    fig.tight_layout()
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def write_readme(output_dir: Path, summary: dict[str, object]) -> None:
    models = summary["models"]
    comparison = summary["comparison"]
    metrics = comparison["metrics"]
    readme = "\n".join(
        [
            "# Mn-Bi-O Formation-Energy Model Overlap",
            "",
            "## Purpose",
            "",
            "Predict formation energies for the EXA AMD Mn-Bi-O CIF corpus with three CGCNN checkpoints, then compare the two surrounding models against `my_formation_energy_mp_all.pth.tar` on the horizontal axis.",
            "",
            "## Inputs",
            "",
            f"- Structures: `{summary['structures_dir']}`",
            f"- Atom features: `{summary['atom_init']}`",
            f"- MPTrj checkpoint: `{models['mptrj_20260705']['path']}`",
            f"- My MP-all checkpoint: `{models['my_mp_all']['path']}`",
            f"- Xie old checkpoint: `{models['xie_old']['path']}`",
            f"- Repository: `{summary['repo']}`",
            f"- Git revision: `{summary['git_revision']}`",
            "",
            "The `links/` folder symlinks the structures, checkpoints, atom initializer, repository, script, and Slurm job files used for this run.",
            "",
            "## Outputs",
            "",
            "- Shard-level predictions: `shards/shard_*/predictions/*.csv`",
            "- Merged per-model predictions: `merged/*_predictions.csv`",
            "- Joined overlap table: `merged/formation_energy_model_overlap.csv`",
            "- Color-labeled overlap plot: `figures/formation_energy_overlap_vs_my_mp_all.png` and `.pdf`",
            "- Run summary: `run_summary.json`",
            "- Slurm logs: `logs/`",
            "",
            "## Summary",
            "",
            f"- Overlapping structures: `{comparison['overlap_count']}`",
            f"- MPTrj vs my MP-all Pearson r: `{metrics['mptrj_20260705_vs_my_mp_all']['pearson_r']}`",
            f"- MPTrj vs my MP-all MAE difference: `{metrics['mptrj_20260705_vs_my_mp_all']['mae_y_minus_x']}`",
            f"- Xie old vs my MP-all Pearson r: `{metrics['xie_old_vs_my_mp_all']['pearson_r']}`",
            f"- Xie old vs my MP-all MAE difference: `{metrics['xie_old_vs_my_mp_all']['mae_y_minus_x']}`",
            "",
        ]
    )
    (output_dir / "README.md").write_text(readme)


def run_merge(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    structures_dir = args.structures_dir.resolve()
    atom_init = args.atom_init.resolve()
    models = [ModelSpec(item.key, item.label, item.path.resolve()) for item in model_specs(args)]
    prepare_links(output_dir, structures_dir, atom_init, models)
    link_optional_provenance(output_dir, args)

    merged_summaries = write_merged_model_csv(output_dir, models)
    predictions_by_model = {
        model.key: {
            structure_id: value
            for structure_id, value in load_prediction_csv(Path(merged_summaries[model.key]["predictions_csv"])).items()
        }
        for model in models
    }
    comparison = compare_against_my(output_dir, structures_dir, predictions_by_model)
    figures = plot_overlap(Path(comparison["comparison_csv"]), output_dir)
    summary = {
        "mode": "merge",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "repo": str(REPO_ROOT),
        "git_revision": git_revision(),
        "structures_dir": str(structures_dir),
        "atom_init": str(atom_init),
        "output_dir": str(output_dir),
        "models": {
            model.key: {
                "label": model.label,
                "path": str(model.path),
                "checkpoint": checkpoint_summary(model.path),
                "merged": merged_summaries[model.key],
            }
            for model in models
        },
        "comparison": comparison,
        "figures": figures,
    }
    write_json(output_dir / "run_summary.json", summary)
    write_readme(output_dir, summary)
    print(json.dumps({"run_summary": str(output_dir / "run_summary.json"), "figures": figures}, indent=2))


def main() -> None:
    args = parse_args()
    if args.loader_workers < 0:
        raise ValueError("--loader-workers must be non-negative")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    if args.mode == "shard":
        run_shard(args)
    elif args.mode == "merge":
        run_merge(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
