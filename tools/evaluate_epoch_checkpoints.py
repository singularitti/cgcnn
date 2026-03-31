from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet
from cgcnn.utils import Normalizer, _prepare_inputs_targets, _forward_and_loss
from analyze_parity import compute_metrics, make_plot


warnings.filterwarnings(
    "ignore",
    message="Issues encountered while parsing CIF: .*",
)
warnings.filterwarnings(
    "ignore",
    message=".*not find enough neighbors to build graph.*",
)
warnings.filterwarnings(
    "ignore",
    message="No Pauling electronegativity.*",
)


def build_test_loader(
    run_dir: Path,
    batch_size: int = 64,
    workers: int = 10,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    dataset = CIFData(str(run_dir))
    total_size = len(dataset)
    indices = list(range(total_size))
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    test_indices = indices[-test_size:]
    _ = valid_size
    _ = train_size
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=workers,
        collate_fn=collate_pool,
        pin_memory=False,
    )
    return dataset, test_loader


def load_model(dataset: CIFData, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False,
        n_targets=dataset.n_targets,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalizer = Normalizer(torch.zeros(dataset.n_targets))
    normalizer.load_state_dict(checkpoint["normalizer"])
    return model, normalizer


def evaluate_checkpoint(
    dataset: CIFData,
    test_loader: DataLoader,
    checkpoint_path: Path,
    epoch: int,
    output_dir: Path,
) -> dict:
    model, normalizer = load_model(dataset, checkpoint_path)
    criterion = nn.MSELoss()

    targets: list[float] = []
    predictions: list[float] = []
    ids: list[str] = []
    losses: list[float] = []

    with torch.no_grad():
        for input_batch, target_batch, batch_ids in test_loader:
            input_var, target_var = _prepare_inputs_targets(
                input_batch, target_batch, normalizer, False, "regression"
            )
            output, loss = _forward_and_loss(model, input_var, target_var, criterion)
            pred = normalizer.denorm(output.data.cpu()).view(-1).tolist()
            true = target_batch.view(-1).tolist()
            predictions.extend(pred)
            targets.extend(true)
            ids.extend(batch_ids)
            losses.append(float(loss.data.cpu().item()))

    target_arr = np.array(targets)
    pred_arr = np.array(predictions)
    metrics = compute_metrics(target_arr, pred_arr)
    metrics["epoch"] = epoch
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["mse_loss_avg"] = float(sum(losses) / len(losses))

    epoch_results_csv = output_dir / f"test_results_epoch_{epoch:03d}.csv"
    with epoch_results_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for cif_id, target, pred in zip(ids, targets, predictions):
            writer.writerow([cif_id, target, pred])

    epoch_plot_png = output_dir / f"parity_plot_epoch_{epoch:03d}.png"
    make_plot(target_arr, pred_arr, metrics, epoch_plot_png)

    epoch_metrics_json = output_dir / f"parity_metrics_epoch_{epoch:03d}.json"
    with epoch_metrics_json.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def plot_metric_trends(metrics_history: list[dict], output_path: Path) -> None:
    epochs = [item["epoch"] for item in metrics_history]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), dpi=180, sharex=True)
    metric_specs = [
        ("pearson_r", "Pearson R", "#1f77b4"),
        ("r2_score", "R^2", "#2ca02c"),
        ("mae", "MAE", "#d62728"),
    ]
    for ax, (key, label, color) in zip(axes, metric_specs):
        values = [item[key] for item in metrics_history]
        ax.scatter(epochs, values, color=color, s=28)
        ax.plot(epochs, values, color=color, linewidth=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Epoch")
    axes[0].set_title("Per-Epoch Test Metrics")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    checkpoints_dir = run_dir / "checkpoints"
    output_dir = run_dir / "epoch_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = sorted(checkpoints_dir.glob("epoch_*.pth.tar"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoints_dir}")

    dataset, test_loader = build_test_loader(
        run_dir=run_dir,
        batch_size=batch_size,
        workers=workers,
    )

    metrics_history: list[dict] = []
    for checkpoint_path in checkpoint_paths:
        epoch = int(checkpoint_path.stem.split("_")[1].split(".")[0])
        metrics = evaluate_checkpoint(
            dataset=dataset,
            test_loader=test_loader,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            output_dir=output_dir,
        )
        metrics_history.append(metrics)

    metrics_history.sort(key=lambda item: item["epoch"])
    metrics_history_path = output_dir / "parity_metrics_history.json"
    with metrics_history_path.open("w") as handle:
        json.dump(metrics_history, handle, indent=2)

    plot_metric_trends(metrics_history, output_dir / "metric_trends.png")
    print(output_dir)
