from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_values(csv_path: Path, column: str) -> list[float]:
    values: list[float] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Column {column!r} not found in {csv_path}")
        for row in reader:
            raw_value = row.get(column, "")
            if not raw_value:
                continue
            value = float(raw_value)
            if math.isfinite(value):
                values.append(value)
    if not values:
        raise ValueError(f"No finite values found in column {column!r}")
    return values


def add_histogram(ax: plt.Axes, values: list[float], title: str, xlabel: str) -> None:
    ax.hist(values, bins=200, color="#1f77b4", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)


def plot_distributions(values: list[float], output_path: Path) -> None:
    sqrt_values = [math.sqrt(value) for value in values]
    cbrt_values = [math.copysign(abs(value) ** (1.0 / 3.0), value) for value in values]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    add_histogram(
        axes[0],
        values,
        "magnetization_per_volume",
        "magnetization_per_volume",
    )
    add_histogram(
        axes[1],
        sqrt_values,
        "sqrt(magnetization_per_volume)",
        "sqrt(magnetization_per_volume)",
    )
    add_histogram(
        axes[2],
        cbrt_values,
        "cbrt(magnetization_per_volume)",
        "cbrt(magnetization_per_volume)",
    )

    fig.suptitle("Distribution of magnetization_per_volume and transformed values")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def export_values(values: list[float], output_path: Path, column: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([column, f"sqrt_{column}", f"cbrt_{column}"])
        for value in values:
            writer.writerow(
                [
                    value,
                    math.sqrt(value),
                    math.copysign(abs(value) ** (1.0 / 3.0), value),
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot histograms of magnetization_per_volume and its transforms."
    )
    parser.add_argument("csv_path", type=Path, help="Input CSV file")
    parser.add_argument(
        "--column",
        default="magnetization_per_volume",
        help="Column to extract from the CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/magnetization_per_volume_distribution.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--export-values",
        type=Path,
        default=Path("analysis/magnetization_per_volume_values.csv"),
        help="Optional output CSV path for extracted and transformed values",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    values = load_values(args.csv_path, args.column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_distributions(values, args.output)
    export_values(values, args.export_values, args.column)
    print(
        f"Saved plot for {len(values)} values from {args.column!r} to {args.output}"
    )
    print(f"Saved extracted values to {args.export_values}")


if __name__ == "__main__":
    main()
