from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def invert_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    if transform == "sqrt":
        return np.square(values)
    if transform == "cbrt":
        return np.power(values, 3)
    raise ValueError(f"Unsupported transform: {transform}")


def load_transform(run_dir: Path) -> str:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return "raw"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    return metadata.get("target_transform", "raw")


def load_results(test_results_csv: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    material_ids: list[str] = []
    targets: list[float] = []
    predictions: list[float] = []

    with test_results_csv.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            material_ids.append(row[0])
            targets.append(float(row[1]))
            predictions.append(float(row[2]))

    if not targets:
        raise ValueError(f"No usable rows found in {test_results_csv}")

    return material_ids, np.array(targets), np.array(predictions)


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if targets.size == 0 or predictions.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")
    residuals = predictions - targets
    pearson_r = None
    spearman_r = None
    r2 = None
    if targets.size >= 2 and predictions.size >= 2:
        pearson_matrix = np.corrcoef(targets, predictions)
        pearson_value = pearson_matrix[0, 1]
        if not np.isnan(pearson_value):
            pearson_r = float(pearson_value)

        target_ranks = np.argsort(np.argsort(targets))
        prediction_ranks = np.argsort(np.argsort(predictions))
        spearman_value = np.corrcoef(target_ranks, prediction_ranks)[0, 1]
        if not np.isnan(spearman_value):
            spearman_r = float(spearman_value)
        try:
            r2_value = r2_score(targets, predictions)
        except ValueError:
            r2_value = np.nan
        if not np.isnan(r2_value):
            r2 = float(r2_value)

    return {
        "count": int(targets.shape[0]),
        "pearson_r": pearson_r,
        "pearson_r_squared": pearson_r**2 if pearson_r is not None else None,
        "spearman_rho": spearman_r,
        "r2_score": r2,
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(targets, predictions))),
        "bias": float(np.mean(residuals)),
        "target_mean": float(np.mean(targets)),
        "prediction_mean": float(np.mean(predictions)),
    }


def make_plot(
    targets: np.ndarray, predictions: np.ndarray, metrics: dict[str, float], output_png: Path
) -> None:
    def format_metric(value) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    combined_min = float(min(np.min(targets), np.min(predictions)))
    combined_max = float(max(np.max(targets), np.max(predictions)))
    padding = 0.03 * (combined_max - combined_min) if combined_max > combined_min else 0.1
    axis_min = combined_min - padding
    axis_max = combined_max + padding

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=180)
    ax.scatter(targets, predictions, s=8, alpha=0.35, linewidths=0, color="#1f6feb")
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="#d62728", linewidth=1.2)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title("CGCNN Parity Plot")

    text = "\n".join(
        [
            f"N = {metrics['count']}",
            f"R = {format_metric(metrics['pearson_r'])}",
            f"R^2 = {format_metric(metrics['r2_score'])}",
            f"MAE = {format_metric(metrics['mae'])}",
            f"RMSE = {format_metric(metrics['rmse'])}",
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


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    test_results_csv = run_dir / "test_results.csv"
    metrics_json = run_dir / "parity_metrics.json"
    output_png = run_dir / "parity_plot.png"

    transform = load_transform(run_dir)
    _, targets, predictions = load_results(test_results_csv)
    targets = invert_transform(targets, transform)
    predictions = invert_transform(predictions, transform)
    metrics = compute_metrics(targets, predictions)
    make_plot(targets, predictions, metrics, output_png)

    with metrics_json.open("w") as handle:
        json.dump({**metrics, "target_transform": transform}, handle, indent=2)

    print(output_png)
    print(metrics_json)
