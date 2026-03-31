from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

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
from cgcnn.utils import Normalizer, _forward_and_loss, _prepare_inputs_targets
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


def split_indices(
    dataset_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[int]]:
    indices = list(range(dataset_size))
    train_size = int(train_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    valid_size = int(val_ratio * dataset_size)
    return {
        "train": indices[:train_size],
        "val": indices[-(valid_size + test_size) : -test_size],
        "test": indices[-test_size:],
    }


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


def evaluate_split(
    run_dir: Path,
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
    batch_size: int,
    workers: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    dataset = CIFData(str(run_dir))
    split_map = split_indices(
        dataset_size=len(dataset),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    selected_indices = split_map[split]
    sampler = SubsetRandomSampler(selected_indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_pool,
        pin_memory=False,
    )

    model, normalizer = load_model(dataset, checkpoint_path)
    criterion = nn.MSELoss()

    ids: list[str] = []
    targets: list[float] = []
    predictions: list[float] = []
    losses: list[float] = []

    with torch.no_grad():
        for input_batch, target_batch, batch_ids in loader:
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
    metrics["split"] = split
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["dataset_size"] = len(dataset)
    metrics["split_size"] = len(selected_indices)
    metrics["train_ratio"] = train_ratio
    metrics["val_ratio"] = val_ratio
    metrics["test_ratio"] = test_ratio
    metrics["split_index_start"] = min(selected_indices)
    metrics["split_index_end"] = max(selected_indices)
    metrics["mse_loss_avg"] = float(sum(losses) / len(losses))
    metrics["random_seed"] = 123

    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / f"{split}_results.csv"
    with results_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id", "target", "prediction", "residual", "abs_error"])
        for cif_id, target, pred in zip(ids, targets, predictions):
            residual = pred - target
            writer.writerow([cif_id, target, pred, residual, abs(residual)])

    split_ids_csv = output_dir / f"{split}_ids.csv"
    with split_ids_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["material_id"])
        for idx in selected_indices:
            writer.writerow([dataset.id_prop_data[idx][0]])

    hard_examples = sorted(
        (
            {
                "material_id": cif_id,
                "target": float(target),
                "prediction": float(pred),
                "residual": float(pred - target),
                "abs_error": float(abs(pred - target)),
            }
            for cif_id, target, pred in zip(ids, targets, predictions)
        ),
        key=lambda item: item["abs_error"],
        reverse=True,
    )

    hard_examples_json = output_dir / f"{split}_hard_examples_top200.json"
    with hard_examples_json.open("w") as handle:
        json.dump(hard_examples[:200], handle, indent=2)

    metrics_json = output_dir / f"{split}_metrics.json"
    with metrics_json.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    parity_png = output_dir / f"{split}_parity_plot.png"
    make_plot(target_arr, pred_arr, metrics, parity_png)

    print(results_csv)
    print(split_ids_csv)
    print(hard_examples_json)
    print(metrics_json)
    print(parity_png)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CGCNN checkpoint on a dataset split.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or (args.run_dir / "model_best.pth.tar")
    evaluate_split(
        run_dir=args.run_dir,
        checkpoint_path=checkpoint_path,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
