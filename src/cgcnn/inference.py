"""
Inference utilities for CGCNN; refactors predict.py into importable functions.
"""

import csv
import os
import sys
import time
from contextlib import ExitStack
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CIFData, collate_pool, collate_pool_vectorized
from .device import resolve_device, use_pinned_memory
from .model import CrystalGraphConvNet
from .process_cleanup import (
    cleanup_orphaned_python_workers,
    cleanup_stale_torch_shm_managers,
    ensure_orphaned_worker_reaper,
)
from .utils import Normalizer, _validate

__all__ = ["predict_model", "predict_regression_models"]


def _checkpoint_arg(args, name, default):
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _load_regression_model(
    dataset,
    modelpath,
    orig_atom_fea_len,
    nbr_fea_len,
    device,
):
    checkpoint = torch.load(modelpath, map_location="cpu")
    args = checkpoint.get("args", {})
    task = _checkpoint_arg(args, "task", "regression")
    if task != "regression":
        raise ValueError(f"Expected a regression checkpoint, got task={task!r}: {modelpath}")
    n_targets = int(_checkpoint_arg(args, "n_targets", dataset.n_targets))
    if n_targets != dataset.n_targets:
        raise ValueError(
            f"Dataset has {dataset.n_targets} targets but checkpoint expects {n_targets}: "
            f"{modelpath}"
        )
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=int(_checkpoint_arg(args, "atom_fea_len", 64)),
        n_conv=int(_checkpoint_arg(args, "n_conv", 3)),
        h_fea_len=int(_checkpoint_arg(args, "h_fea_len", 128)),
        n_h=int(_checkpoint_arg(args, "n_h", 1)),
        classification=False,
        n_targets=n_targets,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    normalizer = Normalizer(torch.zeros(n_targets))
    normalizer.load_state_dict(checkpoint.get("normalizer", normalizer.state_dict()))
    normalizer.mean = normalizer.mean.to(device)
    normalizer.std = normalizer.std.to(device)
    return model, normalizer


def predict_regression_models(
    dataset,
    modelpaths: Mapping[str, str | os.PathLike],
    *,
    output_csvs: Mapping[str, str | os.PathLike],
    batch_size: int = 256,
    workers: int = 0,
    device: str | torch.device | None = None,
    print_freq: int = 10,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    write_header: bool = False,
):
    """Predict several regression checkpoints from one graph-loading pass.

    Every batch of crystal graphs is transferred to the accelerator once and
    reused by all models. This is intended for comparing small checkpoints on
    CIF-heavy datasets where parsing and graph construction dominate runtime.
    """
    if set(modelpaths) != set(output_csvs):
        raise ValueError("modelpaths and output_csvs must have identical keys.")
    if not modelpaths:
        return {}
    if workers < 0:
        raise ValueError("workers must be non-negative.")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive.")

    device = resolve_device(device=device)
    cleaned_pids = cleanup_orphaned_python_workers(sys.executable)
    if cleaned_pids:
        print(f"Cleaned orphaned Python worker processes: {cleaned_pids}")
    ensure_orphaned_worker_reaper(
        sys.executable,
        torch_shm_retain_count=max(workers * 2, 0),
    )

    sample, _, _ = dataset[0]
    orig_atom_fea_len = sample[0].shape[-1]
    nbr_fea_len = sample[1].shape[-1]
    loaded = {
        key: _load_regression_model(
            dataset,
            str(modelpath),
            orig_atom_fea_len,
            nbr_fea_len,
            device,
        )
        for key, modelpath in modelpaths.items()
    }

    loader_kwargs = {}
    if workers > 0:
        loader_kwargs.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_pool_vectorized,
        pin_memory=use_pinned_memory(device),
        **loader_kwargs,
    )

    total = 0
    started = time.time()
    with ExitStack() as stack:
        writers = {}
        for key, output_csv in output_csvs.items():
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(output_path.open("w", newline=""))
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(
                    ["structure_id", "dummy_target", "predicted_formation_energy_per_atom"]
                )
            writers[key] = writer

        non_blocking = device.type != "cpu"
        with torch.inference_mode():
            for batch_index, (inputs, targets, cif_ids) in enumerate(loader):
                crystal_mapping = (
                    inputs[3][0].to(device, non_blocking=non_blocking),
                    inputs[3][1].to(device, non_blocking=non_blocking),
                )
                model_inputs = (
                    inputs[0].to(device, non_blocking=non_blocking),
                    inputs[1].to(device, non_blocking=non_blocking),
                    inputs[2].to(device, non_blocking=non_blocking),
                    crystal_mapping,
                )
                predictions = {
                    key: normalizer.denorm(model(*model_inputs)).cpu()
                    for key, (model, normalizer) in loaded.items()
                }
                target_rows = targets.tolist()
                for key, prediction in predictions.items():
                    for cif_id, target_row, prediction_row in zip(
                        cif_ids, target_rows, prediction.tolist()
                    ):
                        writers[key].writerow([cif_id, *target_row, *prediction_row])
                total += len(cif_ids)
                if batch_index % print_freq == 0:
                    elapsed = time.time() - started
                    print(
                        f"Predict: [{batch_index}/{len(loader)}] structures={total} "
                        f"elapsed={elapsed:.1f}s rate={total / max(elapsed, 1e-9):.1f}/s",
                        flush=True,
                    )
    return {key: str(Path(path)) for key, path in output_csvs.items()}


def predict_model(
    dataset: CIFData,
    task: str | None = None,
    atom_fea_len: int | None = None,
    n_conv: int | None = None,
    h_fea_len: int | None = None,
    n_h: int | None = None,
    n_targets: int | None = None,
    n_classes: int | None = None,
    model: CrystalGraphConvNet | None = None,
    normalizer: Normalizer | None = None,
    modelpath: str | None = None,
    batch_size: int = 256,
    workers: int = 0,
    cuda: bool | None = None,
    device: str | torch.device | None = None,
    print_freq: int = 10,
    shuffle: bool = False,
    output_csv: str = "test_results.csv",
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
):
    """Load a model from a saved checkpoint or use provided model and predict on CIF files in `dataset`.

    Returns the path to the CSV file written (`test_results.csv`) or None on failure.
    """
    device = resolve_device(device=device, cuda=cuda)
    cleaned_pids = cleanup_orphaned_python_workers(sys.executable)
    if cleaned_pids:
        print(f"Cleaned orphaned Python worker processes: {cleaned_pids}")
    torch_shm_retain_count = max(workers * 2, 0)
    if torch_shm_retain_count > 0:
        cleaned_torch_shm = cleanup_stale_torch_shm_managers(
            retain_count=torch_shm_retain_count,
        )
        if cleaned_torch_shm:
            print(f"Cleaned stale torch_shm_manager processes: {cleaned_torch_shm}")
    ensure_orphaned_worker_reaper(
        sys.executable,
        torch_shm_retain_count=torch_shm_retain_count,
    )
    if model is None:
        if modelpath is None or not os.path.isfile(modelpath):
            raise ValueError("Either model or valid modelpath must be provided")
        checkpoint = torch.load(modelpath, map_location="cpu")
        args = checkpoint.get("args", {})
        task = task or args.get("task", "regression")
        atom_fea_len = atom_fea_len or args.get("atom_fea_len", 64)
        n_conv = n_conv or args.get("n_conv", 3)
        h_fea_len = h_fea_len or args.get("h_fea_len", 128)
        n_h = n_h or args.get("n_h", 1)
        n_targets = n_targets or args.get("n_targets", dataset.n_targets)
        n_classes = n_classes or args.get("n_classes", 2)
    else:
        checkpoint = None
        task = task or "regression"
        atom_fea_len = atom_fea_len or 64
        n_conv = n_conv or 3
        h_fea_len = h_fea_len or 128
        n_h = n_h or 1
        n_targets = n_targets or dataset.n_targets
        n_classes = n_classes or getattr(model, "n_classes", 2)

    collate_fn = collate_pool
    loader_kwargs = {}
    if workers > 0:
        loader_kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=use_pinned_memory(device),
        **loader_kwargs,
    )
    model_target_dim = n_targets
    if task == "regression" and dataset.n_targets != model_target_dim:
        raise ValueError("Dataset target dimensionality doesn't match model.")

    if model is None:
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(
            orig_atom_fea_len,
            nbr_fea_len,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            classification=True if task == "classification" else False,
            n_targets=model_target_dim,
            n_classes=n_classes or 2,
        )
        model.load_state_dict(checkpoint["state_dict"])  # will raise on mismatch
    model.to(device)

    if task == "classification":
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    if normalizer is None:
        if checkpoint is None:
            raise ValueError(
                "Normalizer must be provided if not loading from checkpoint"
            )
        normalizer = Normalizer(
            torch.zeros(model_target_dim if task == "regression" else 1)
        )
        normalizer.load_state_dict(
            checkpoint.get("normalizer", normalizer.state_dict())
        )

    out_csv = _validate(
        test_loader,
        model,
        criterion,
        normalizer,
        device,
        task,
        test=True,
        print_freq=print_freq,
        output_csv=output_csv,
    )
    return out_csv
