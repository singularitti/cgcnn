from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_predictions(path: Path) -> dict[str, float]:
    predictions: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            structure_id = row["structure_id"]
            predictions[structure_id] = float(row["predicted_formation_energy_per_atom"])
    return predictions


def parse_args(argv: list[str]) -> dict[str, object]:
    if len(argv) not in {5, 6}:
        raise SystemExit(
            "Usage: uv run python tools/plot_model_comparison_parity.py "
            "<old_predictions.csv> <new_predictions.csv> <output.png> <summary.json> [both_below_threshold]"
        )
    threshold = float(argv[5]) if len(argv) == 6 else None
    return {
        "old_csv": Path(argv[1]).expanduser().resolve(),
        "new_csv": Path(argv[2]).expanduser().resolve(),
        "output_png": Path(argv[3]).expanduser().resolve(),
        "summary_json": Path(argv[4]).expanduser().resolve(),
        "both_below_threshold": threshold,
    }


if __name__ == "__main__":
    args = parse_args(sys.argv)
    old_predictions = load_predictions(args["old_csv"])
    new_predictions = load_predictions(args["new_csv"])
    overlap_ids = sorted(set(old_predictions) & set(new_predictions))
    if not overlap_ids:
        raise SystemExit("No overlapping structure IDs were found.")
    threshold = args["both_below_threshold"]
    if threshold is not None:
        overlap_ids = [
            structure_id
            for structure_id in overlap_ids
            if old_predictions[structure_id] < threshold
            and new_predictions[structure_id] < threshold
        ]
    if not overlap_ids:
        raise SystemExit("No overlapping structure IDs remained after threshold filtering.")

    old_values = np.array([old_predictions[structure_id] for structure_id in overlap_ids])
    new_values = np.array([new_predictions[structure_id] for structure_id in overlap_ids])

    min_axis = float(min(old_values.min(), new_values.min()))
    max_axis = float(max(old_values.max(), new_values.max()))
    pearson_r = float(np.corrcoef(old_values, new_values)[0, 1])
    mae = float(np.mean(np.abs(new_values - old_values)))
    rmse = float(math.sqrt(np.mean((new_values - old_values) ** 2)))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    hb = ax.hexbin(
        old_values,
        new_values,
        gridsize=220,
        mincnt=1,
        cmap="inferno",
        bins="log",
        linewidths=0,
    )
    ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("Old model formation energy per atom")
    ax.set_ylabel("New model formation energy per atom")
    title = f"MnBiF overlap parity plot ({len(overlap_ids):,} shared IDs)"
    if threshold is not None:
        title = f"MnBiF overlap parity plot ({len(overlap_ids):,} shared IDs, both < {threshold})"
    ax.set_title(title)
    ax.set_xlim(min_axis, max_axis)
    ax.set_ylim(min_axis, max_axis)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    colorbar = fig.colorbar(hb, ax=ax)
    colorbar.set_label("log10(count per hexbin)")
    fig.tight_layout()
    fig.savefig(args["output_png"], bbox_inches="tight")
    plt.close(fig)

    summary = {
        "old_csv": str(args["old_csv"]),
        "new_csv": str(args["new_csv"]),
        "output_png": str(args["output_png"]),
        "overlap_count": len(overlap_ids),
        "both_below_threshold": threshold,
        "old_min": float(old_values.min()),
        "old_max": float(old_values.max()),
        "new_min": float(new_values.min()),
        "new_max": float(new_values.max()),
        "pearson_r": pearson_r,
        "mae_new_minus_old": mae,
        "rmse_new_minus_old": rmse,
    }
    args["summary_json"].write_text(json.dumps(summary, indent=2) + "\n")
    print(args["output_png"])
