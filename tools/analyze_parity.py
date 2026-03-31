from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    residuals = predictions - targets
    pearson_r = float(np.corrcoef(targets, predictions)[0, 1])

    target_ranks = np.argsort(np.argsort(targets))
    prediction_ranks = np.argsort(np.argsort(predictions))
    spearman_r = float(np.corrcoef(target_ranks, prediction_ranks)[0, 1])

    return {
        "count": int(targets.shape[0]),
        "pearson_r": pearson_r,
        "pearson_r_squared": pearson_r**2,
        "spearman_rho": spearman_r,
        "r2_score": float(r2_score(targets, predictions)),
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(targets, predictions))),
        "bias": float(np.mean(residuals)),
        "target_mean": float(np.mean(targets)),
        "prediction_mean": float(np.mean(predictions)),
    }


def make_plot(
    targets: np.ndarray, predictions: np.ndarray, metrics: dict[str, float], output_png: Path
) -> None:
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
            f"R = {metrics['pearson_r']:.4f}",
            f"R^2 = {metrics['r2_score']:.4f}",
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


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    test_results_csv = run_dir / "test_results.csv"
    metrics_json = run_dir / "parity_metrics.json"
    output_png = run_dir / "parity_plot.png"

    _, targets, predictions = load_results(test_results_csv)
    metrics = compute_metrics(targets, predictions)
    make_plot(targets, predictions, metrics, output_png)

    with metrics_json.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    print(output_png)
    print(metrics_json)
