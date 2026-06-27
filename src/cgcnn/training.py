"""
Training utilities and high-level train function for CGCNN.
This refactors the previous `main.py` logic into importable functions.
"""

import json
import os
import sys
import time
import warnings
from random import sample

import torch
import torch.multiprocessing as torch_mp
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from .data import CachedGraphData, CIFData, collate_pool, get_train_val_test_loader
from .device import resolve_device, use_pinned_memory
from .model import CrystalGraphConvNet
from .process_cleanup import (
    cleanup_orphaned_python_workers,
    cleanup_stale_torch_shm_managers,
    ensure_orphaned_worker_reaper,
)
from .utils import (
    AverageMeter,
    Normalizer,
    _forward_and_loss,
    _prepare_inputs_targets,
    _print_progress,
    _update_metrics,
    _validate,
    save_checkpoint,
)

__all__ = ["train_model"]


def _replace_symlink(link_path: str, target_path: str) -> None:
    link_abs = os.path.abspath(link_path)
    link_dir = os.path.dirname(link_abs) or "."
    target_abs = os.path.abspath(target_path)
    relative_target = os.path.relpath(target_abs, link_dir)
    tmp_link = f"{link_abs}.tmp-{os.getpid()}"
    if os.path.lexists(tmp_link):
        os.unlink(tmp_link)
    os.symlink(relative_target, tmp_link)
    os.replace(tmp_link, link_abs)


def train_model(
    root_dir: str,
    task: str = "regression",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 0.01,
    lr_milestones: list | None = None,
    optim_name: str = "SGD",
    atom_fea_len: int = 64,
    h_fea_len: int = 128,
    n_conv: int = 3,
    n_h: int = 1,
    cuda: bool | None = None,
    device: str | torch.device | None = None,
    workers: int = 0,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    start_epoch: int = 0,
    print_freq: int = 10,
    train_ratio: float | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    resume: str | None = None,
    initialize_from: str | None = None,
    checkpoint_dir: str | None = None,
    metrics_history_path: str | None = None,
    train_ids: list[str] | None = None,
    val_ids: list[str] | None = None,
    test_ids: list[str] | None = None,
    n_classes: int | None = None,
    class_weights: list[float] | None = None,
    classification_metric: str | None = None,
    classification_metric_class_index: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    dataset_format: str = "cif",
    id_prop_file: str | None = None,
    graph_cache_max_cached_shards: int = 512,
):
    """Train a CGCNN model.

    Parameters
    ----------
    root_dir: str
        Path to CIF dataset directory (root of dataset)
    task: str
        'regression' or 'classification'
    Other args are analogous to the previous CLI arguments.

    Returns
    -------
    path to the best saved model file (model_best.pth.tar) if saved, else None.
    """
    device = resolve_device(device=device, cuda=cuda)
    if workers > 0:
        try:
            torch_mp.set_sharing_strategy("file_system")
            print("Using torch multiprocessing sharing strategy: file_system")
        except RuntimeError as exc:
            print(f"Could not set torch sharing strategy to file_system: {exc}")
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
    if lr_milestones is None:
        lr_milestones = [100]
    if task == "classification":
        n_classes = n_classes or 2
        classification_metric = classification_metric or (
            "auc" if n_classes == 2 else "macro_f1"
        )
    explicit_split_ids = any(ids is not None for ids in [train_ids, val_ids, test_ids])
    if dataset_format == "cif":
        if id_prop_file is not None:
            raise ValueError("id_prop_file is only supported for graph_cache datasets.")
        dataset = CIFData(root_dir, shuffle=not explicit_split_ids)
    elif dataset_format == "graph_cache":
        dataset = CachedGraphData(
            root_dir,
            id_prop_file=id_prop_file,
            shuffle=not explicit_split_ids,
            max_cached_shards=graph_cache_max_cached_shards,
        )
    elif dataset_format == "auto":
        manifest_path = os.path.join(root_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            dataset = CachedGraphData(
                root_dir,
                id_prop_file=id_prop_file,
                shuffle=not explicit_split_ids,
                max_cached_shards=graph_cache_max_cached_shards,
            )
        else:
            if id_prop_file is not None:
                raise ValueError(
                    "id_prop_file is only supported when auto resolves to graph_cache."
                )
            dataset = CIFData(root_dir, shuffle=not explicit_split_ids)
    else:
        raise ValueError(
            "dataset_format must be one of 'cif', 'graph_cache', or 'auto'."
        )
    train_indices = val_indices = test_indices = None
    if explicit_split_ids:
        if train_ids is None or val_ids is None or test_ids is None:
            raise ValueError(
                "train_ids, val_ids, and test_ids must all be provided when using explicit splits."
            )
        id_to_index = {
            row[0]: idx for idx, row in enumerate(dataset.id_prop_data) if row
        }
        try:
            train_indices = [id_to_index[cif_id] for cif_id in train_ids]
            val_indices = [id_to_index[cif_id] for cif_id in val_ids]
            test_indices = [id_to_index[cif_id] for cif_id in test_ids]
        except KeyError as exc:
            raise ValueError(f"Unknown CIF id in explicit split: {exc.args[0]}") from exc
    collate_fn = collate_pool
    returned_loaders = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        train_ratio=train_ratio,
        num_workers=workers,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        pin_memory=use_pinned_memory(device),
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        return_test=True,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
    if isinstance(returned_loaders, tuple) and len(returned_loaders) == 3:
        train_loader, val_loader, test_loader = returned_loaders
    else:
        train_loader, val_loader = returned_loaders
        test_loader = None
    n_targets = dataset.n_targets

    # obtain target value normalizer
    if task == "classification":
        normalizer = Normalizer(torch.zeros(n_targets))
        normalizer.load_state_dict({"mean": 0.0, "std": 1.0})
    else:
        if len(dataset) < 500:
            warnings.warn(
                "Dataset has less than 500 data points. Lower accuracy is expected. "
            )
            target_rows = dataset.id_prop_data
        else:
            target_rows = [dataset.id_prop_data[i] for i in sample(range(len(dataset)), 500)]
        sample_target = torch.tensor(
            [[float(value) for value in row[1:]] for row in target_rows],
            dtype=torch.float32,
        )
        normalizer = Normalizer(sample_target)

    # build model
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
        n_targets=n_targets,
        n_classes=n_classes or 2,
    )
    model.to(device)

    # define loss func and optimizer
    if task == "classification":
        criterion_weight = None
        if class_weights is not None:
            if len(class_weights) != n_classes:
                raise ValueError(
                    f"Expected {n_classes} class weights, received {len(class_weights)}."
                )
            criterion_weight = torch.tensor(class_weights, dtype=torch.float)
            if device.type != "cpu":
                criterion_weight = criterion_weight.to(device)
        criterion = nn.NLLLoss(weight=criterion_weight)
    else:
        criterion = nn.MSELoss()
    if optim_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optim_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    else:
        raise NameError("Only SGD or Adam is allowed as optim_name")

    # optionally initialize model weights without restoring optimizer/normalizer
    if initialize_from and not resume:
        if os.path.isfile(initialize_from):
            checkpoint = torch.load(initialize_from, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(initialize_from)

    # optionally resume from a checkpoint
    best_validation_score = 1e10 if task == "regression" else float("-inf")
    if resume:
        if os.path.isfile(resume):
            checkpoint = torch.load(resume, map_location="cpu")
            start_epoch = checkpoint.get("epoch", start_epoch)
            best_validation_score = checkpoint.get(
                "best_validation_score",
                checkpoint.get("best_mae_error", best_validation_score),
            )
            if isinstance(best_validation_score, torch.Tensor):
                best_validation_score = float(best_validation_score.item())
            model.load_state_dict(checkpoint["state_dict"])  # raises if mismatch
            optimizer.load_state_dict(checkpoint["optimizer"])
            normalizer.load_state_dict(checkpoint["normalizer"])

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    history = []
    if metrics_history_path and os.path.isfile(metrics_history_path):
        try:
            with open(metrics_history_path) as f:
                loaded_history = json.load(f)
            if isinstance(loaded_history, list):
                history = loaded_history
        except json.JSONDecodeError:
            history = []

    # training loop
    best_checkpoint_path = os.path.abspath("model_best.pth.tar")
    stale_epochs = 0
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        _train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            normalizer,
            device,
            task,
            print_freq,
        )
        val_metric = _validate(
            val_loader,
            model,
            criterion,
            normalizer,
            device,
            task,
            test=False,
            print_freq=10,
            classification_metric=classification_metric or "auc",
            classification_metric_class_index=classification_metric_class_index,
        )
        if isinstance(val_metric, torch.Tensor):
            val_metric = float(val_metric.item())
        else:
            val_metric = float(val_metric)
        if val_metric != val_metric:
            if task == "classification":
                val_metric = 0.0
            else:
                raise RuntimeError("Training diverged: validation MAE is NaN")
        scheduler.step()
        previous_best = best_validation_score
        if task == "regression":
            is_best = val_metric < best_validation_score
            best_validation_score = min(val_metric, best_validation_score)
            significant_improvement = val_metric < previous_best - early_stopping_min_delta
        else:
            is_best = val_metric > best_validation_score
            best_validation_score = max(val_metric, best_validation_score)
            significant_improvement = val_metric > previous_best + early_stopping_min_delta
        if significant_improvement:
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint_state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_validation_score": best_validation_score,
            "best_mae_error": best_validation_score,
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "args": {
                "task": task,
                "atom_fea_len": atom_fea_len,
                "n_conv": n_conv,
                "h_fea_len": h_fea_len,
                "n_h": n_h,
                "n_targets": n_targets,
                "n_classes": n_classes if task == "classification" else None,
            },
        }
        if checkpoint_dir is not None:
            epoch_checkpoint_path = os.path.join(
                checkpoint_dir, f"epoch_{epoch + 1:03d}.pth.tar"
            )
            save_checkpoint(checkpoint_state, False, filename=epoch_checkpoint_path)
            _replace_symlink("checkpoint.pth.tar", epoch_checkpoint_path)
            if is_best:
                _replace_symlink("model_best.pth.tar", epoch_checkpoint_path)
        else:
            save_checkpoint(checkpoint_state, is_best)
            epoch_checkpoint_path = "checkpoint.pth.tar"
        history.append(
            {
                "epoch": epoch + 1,
                "val_metric": float(val_metric),
                "best_val_metric": float(best_validation_score),
                "is_best": is_best,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_sec": time.time() - epoch_start,
                "stale_epochs": stale_epochs,
                "checkpoint": (
                    os.path.abspath(epoch_checkpoint_path)
                    if checkpoint_dir is not None
                    else os.path.abspath("checkpoint.pth.tar")
                ),
            }
        )
        if metrics_history_path:
            with open(metrics_history_path, "w") as f:
                json.dump(history, f, indent=2)
        if is_best:
            best_checkpoint_path = os.path.abspath("model_best.pth.tar")
        if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
            print(
                "Early stopping triggered after "
                f"{stale_epochs} stale epochs. Best validation metric: "
                f"{best_validation_score:.6f}"
            )
            break

    if (
        (not best_checkpoint_path or not os.path.exists(best_checkpoint_path))
        and os.path.exists("checkpoint.pth.tar")
    ):
        best_checkpoint_path = os.path.abspath("checkpoint.pth.tar")

    # test best model if requested
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        _validate(
            test_loader,
            model,
            criterion,
            normalizer,
            device,
            task,
            test=True,
            print_freq=10,
            classification_metric=classification_metric or "auc",
            classification_metric_class_index=classification_metric_class_index,
        )

    return best_checkpoint_path


def _train_epoch(
    train_loader, model, criterion, optimizer, epoch, normalizer, device, task, print_freq
):
    # Simple version of original main.train that uses local arguments
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # initialize all metrics so they're available for both tasks
    mae_errors = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_var, target_var = _prepare_inputs_targets(
            input, target, normalizer, device, task
        )
        output, loss = _forward_and_loss(model, input_var, target_var, criterion)
        _update_metrics(
            loss,
            output,
            target,
            task,
            losses,
            mae_errors,
            accuracies,
            precisions,
            recalls,
            fscores,
            auc_scores,
            test=False,
            test_preds=None,
            test_targets=None,
            test_cif_ids=None,
            batch_cif_ids=None,
            normalizer=normalizer,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        _print_progress(
            i,
            len(train_loader),
            batch_time,
            losses,
            mae_errors,
            accuracies,
            precisions,
            recalls,
            fscores,
            auc_scores,
            task,
            print_freq,
            prefix=f"Epoch: [{epoch}]",
        )
