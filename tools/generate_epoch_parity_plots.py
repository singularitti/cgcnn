from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from analyze_parity import (
    compute_metrics,
    invert_transform,
    load_transform,
    load_results,
    make_plot,
)
from cgcnn.data import CIFData
from cgcnn.inference import predict_model

EPOCH_PATTERN = re.compile(r"epoch_(\d+)\.pth\.tar$")


def natural_sort_key(value: str) -> tuple:
    parts = re.split(r"(\d+)", value)
    return tuple(int(x) if x.isdigit() else x for x in parts)


def read_id_csv(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split id file: {path}")
    ids: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == "material_id":
                continue
            ids.append(row[0].strip())
    if not ids:
        raise ValueError(f"No IDs found in {path}")
    return ids


def epoch_from_checkpoint(path: Path) -> int:
    match = EPOCH_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse epoch number from checkpoint: {path}")
    return int(match.group(1))


def make_fixed_loglog_plot(
    targets: np.ndarray,
    predictions: np.ndarray,
    metrics: dict,
    output_png: Path,
    min_val: float = 1e-11,
    max_val: float = 1.0,
) -> None:
    valid = (
        (targets > 0)
        & (predictions > 0)
        & (targets >= min_val)
        & (targets <= max_val)
        & (predictions >= min_val)
        & (predictions <= max_val)
    )
    filtered_targets = targets[valid]
    filtered_predictions = predictions[valid]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=180)
    if filtered_targets.size > 0:
        ax.scatter(filtered_targets, filtered_predictions, s=8, alpha=0.35, linewidths=0, color="#1f6feb")
    ax.plot([min_val, max_val], [min_val, max_val], color="#d62728", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title("CGCNN Parity Plot (log-log, fixed scale)")

    locator = ticker.LogLocator(base=10.0)
    formatter = ticker.LogFormatterSciNotation(base=10.0)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    text = "\n".join(
        [
            f"N = {metrics['count']}",
            f"R = {metrics['pearson_r']:.4f}" if metrics["pearson_r"] is not None else "R = n/a",
            f"R^2 = {metrics['r2_score']:.4f}" if metrics["r2_score"] is not None else "R^2 = n/a",
            f"MAE = {metrics['mae']:.4f}",
            f"RMSE = {metrics['rmse']:.4f}",
        ]
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_metrics(summary: list[dict[str, str | int | float]], output_root: Path) -> None:
    epochs = [int(record["epoch"]) for record in summary]
    mae_values = [float(record["mae"]) for record in summary]
    r2_values = [float(record["r2_score"]) if record["r2_score"] is not None else float("nan") for record in summary]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
    ax.plot(epochs, mae_values, marker="o", linestyle="-", color="#1f77b4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title("Epoch vs MAE")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_root / "epoch_mae.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
    ax.plot(epochs, r2_values, marker="o", linestyle="-", color="#2ca02c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R^2")
    ax.set_title("Epoch vs R^2")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_root / "epoch_r2.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
    ax.plot(epochs, mae_values, marker="o", linestyle="-", color="#1f77b4", label="MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax2 = ax.twinx()
    ax2.plot(epochs, r2_values, marker="s", linestyle="-", color="#2ca02c", label="R^2")
    ax2.set_ylabel("R^2")
    ax.set_title("Epoch Metrics")
    ax.grid(True, linestyle="--", alpha=0.4)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(output_root / "epoch_metrics.png", bbox_inches="tight")
    plt.close(fig)


def generate_plots_for_run(
    run_dir: Path,
    batch_size: int,
    workers: int,
    cuda: bool,
    checkpoint_dir: Path | None = None,
    max_epochs: int | None = None,
) -> None:
    checkpoint_root = checkpoint_dir or (run_dir / "checkpoints")
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_root}")

    splits_dir = run_dir / "splits"
    if not splits_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {splits_dir}")

    test_ids = read_id_csv(splits_dir / "test_ids.csv")
    dataset = CIFData(str(run_dir), shuffle=False, include_ids=test_ids)
    transform = load_transform(run_dir)
    output_root = run_dir / "epoch_parity"
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(
        checkpoint_root.glob("epoch_*.pth.tar"),
        key=lambda p: (epoch_from_checkpoint(p), p.name),
    )
    if max_epochs is not None:
        checkpoints = [ckpt for ckpt in checkpoints if epoch_from_checkpoint(ckpt) <= max_epochs]
    if not checkpoints:
        raise RuntimeError(f"No epoch checkpoints found in {checkpoint_root}")

    summary: list[dict[str, str | int | float]] = []
    for checkpoint in checkpoints:
        epoch = epoch_from_checkpoint(checkpoint)
        print(f"Generating epoch {epoch} parity...")
        epoch_csv = output_root / f"test_results_epoch_{epoch:03d}.csv"
        if not epoch_csv.exists():
            predict_model(
                dataset=dataset,
                task="regression",
                modelpath=str(checkpoint),
                batch_size=batch_size,
                workers=workers,
                cuda=cuda,
                print_freq=20,
                shuffle=False,
                output_csv=str(epoch_csv),
            )
        else:
            print(f"Reusing existing results for epoch {epoch}.")

        _, targets, predictions = load_results(epoch_csv)
        targets = invert_transform(targets, transform)
        predictions = invert_transform(predictions, transform)
        metrics = compute_metrics(targets, predictions)
        metrics_path = output_root / f"parity_metrics_epoch_{epoch:03d}.json"
        plot_path = output_root / f"parity_plot_epoch_{epoch:03d}.png"
        logplot_path = output_root / f"parity_plot_loglog_epoch_{epoch:03d}.png"

        with metrics_path.open("w") as handle:
            json.dump({**metrics, "target_transform": transform, "epoch": epoch}, handle, indent=2)

        make_plot(targets, predictions, metrics, plot_path)
        make_fixed_loglog_plot(targets, predictions, metrics, logplot_path)

        summary.append(
            {
                "epoch": epoch,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2_score": metrics["r2_score"],
                "pearson_r": metrics["pearson_r"],
                "spearman_rho": metrics["spearman_rho"],
                "metric_file": str(metrics_path.name),
                "plot_file": str(plot_path.name),
                "logplot_file": str(logplot_path.name) if logplot_path.exists() else "",
            }
        )

    summary_path = output_root / "epoch_parity_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    plot_epoch_metrics(summary, output_root)

    print(f"Generated epoch parity plots in {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate parity plots for each epoch checkpoint."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing checkpoints and splits.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--max_epoch", type=int)
    args = parser.parse_args()
    generate_plots_for_run(
        args.run_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        cuda=args.cuda,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.max_epoch,
    )


if __name__ == "__main__":
    main()
