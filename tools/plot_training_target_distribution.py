from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLIT_ORDER = ["train", "val", "test"]
THRESHOLDS = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
QUANTILES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]


def load_id_prop(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["material_id", "target"])
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["material_id", "target"]).copy()
    return df


def load_split_ids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "material_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a material_id column")
        return {row["material_id"] for row in reader if row.get("material_id")}


def attach_splits(df: pd.DataFrame, split_dir: Path) -> pd.DataFrame:
    split_map: dict[str, str] = {}
    for split in SPLIT_ORDER:
        for material_id in load_split_ids(split_dir / f"{split}_ids.csv"):
            if material_id in split_map:
                raise ValueError(f"Material ID {material_id} appears in multiple splits")
            split_map[material_id] = split

    df = df.copy()
    df["split"] = df["material_id"].map(split_map).fillna("unassigned")
    return df


def write_summary(df: pd.DataFrame, output_path: Path) -> None:
    rows: list[dict[str, float | int | str]] = []
    groups = [("all", df)] + [(split, df[df["split"] == split]) for split in SPLIT_ORDER]
    for name, group in groups:
        values = group["target"].to_numpy(dtype=float)
        row: dict[str, float | int | str] = {
            "split": name,
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "zero_count": int(np.sum(values == 0.0)),
            "zero_fraction": float(np.mean(values == 0.0)),
        }
        for q in QUANTILES:
            row[f"q{q:g}"] = float(np.quantile(values, q))
        for threshold in THRESHOLDS:
            row[f"le_{threshold:g}"] = int(np.sum(values <= threshold))
            row[f"frac_le_{threshold:g}"] = float(np.mean(values <= threshold))
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_distribution(df: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "all": "#2f3a45",
        "train": "#0072b2",
        "val": "#d55e00",
        "test": "#009e73",
    }
    values = df["target"].to_numpy(dtype=float)
    nonzero = values[values > 0]
    log_bins = np.logspace(np.log10(max(nonzero.min(), 1e-8)), np.log10(values.max()), 90)
    linear_bins = np.linspace(0, min(1.0, values.max()), 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.hist(values, bins=linear_bins, color=colors["all"], alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("All samples: bulk target distribution")
    ax.set_xlabel("energy_above_hull (eV/atom), clipped to 0-1")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for split in SPLIT_ORDER:
        split_values = df.loc[df["split"] == split, "target"].to_numpy(dtype=float)
        ax.hist(
            split_values[split_values > 0],
            bins=log_bins,
            histtype="step",
            linewidth=1.8,
            label=split,
            color=colors[split],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Nonzero tail by split")
    ax.set_xlabel("energy_above_hull (eV/atom), log scale")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for name in ["all", *SPLIT_ORDER]:
        series = df["target"] if name == "all" else df.loc[df["split"] == name, "target"]
        sorted_values = np.sort(series.to_numpy(dtype=float))
        ecdf = np.arange(1, sorted_values.size + 1) / sorted_values.size
        ax.plot(sorted_values, ecdf, label=name, linewidth=1.7, color=colors[name])
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_title("Cumulative distribution")
    ax.set_xlabel("energy_above_hull (eV/atom), symlog scale")
    ax.set_ylabel("fraction <= x")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    labels = ["all", *SPLIT_ORDER]
    box_values = [
        (df["target"] if label == "all" else df.loc[df["split"] == label, "target"]).to_numpy(dtype=float)
        for label in labels
    ]
    ax.boxplot(box_values, tick_labels=labels, showfliers=False, patch_artist=True)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("Split quantiles, outliers hidden")
    ax.set_ylabel("energy_above_hull (eV/atom), symlog scale")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("CGCNN training target distribution from id_prop.csv", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot target distribution for a prepared CGCNN run.")
    parser.add_argument("--id-prop", type=Path, required=True, help="Prepared CGCNN id_prop.csv")
    parser.add_argument("--split-dir", type=Path, required=True, help="Directory containing train/val/test ID CSVs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plot and CSV outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = attach_splits(load_id_prop(args.id_prop), args.split_dir)
    if (df["split"] == "unassigned").any():
        count = int((df["split"] == "unassigned").sum())
        raise ValueError(f"{count} samples in id_prop.csv were not assigned to a split")

    plot_path = args.output_dir / "energy_above_hull_distribution_by_split.png"
    summary_path = args.output_dir / "energy_above_hull_distribution_summary.csv"
    values_path = args.output_dir / "energy_above_hull_values_by_split.csv"

    plot_distribution(df, plot_path)
    write_summary(df, summary_path)
    df.sort_values(["split", "material_id"]).to_csv(values_path, index=False)

    print(f"Saved plot to {plot_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved values to {values_path}")


if __name__ == "__main__":
    main()
