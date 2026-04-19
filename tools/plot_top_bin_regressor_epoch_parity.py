from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from analyze_parity import compute_metrics, make_loglog_plot, make_plot
from cgcnn.data import CIFData
from cgcnn.inference import predict_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate parity plots for every saved top-bin regressor checkpoint."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=1000)
    return parser.parse_args()


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    targets: list[float] = []
    predictions: list[float] = []
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            targets.append(float(row[1]))
            predictions.append(float(row[2]))
    return np.array(targets, dtype=float), np.array(predictions, dtype=float)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoints_dir = run_dir / "top_bin_regressor" / "checkpoints"
    dataset_dir = run_dir / "evaluation" / "top_bin_regressor_test_dataset"
    output_dir = run_dir / "evaluation" / "top_bin_regressor_epoch_parity"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CIFData(str(dataset_dir), shuffle=False)
    summary_rows: list[dict[str, object]] = []

    for checkpoint_path in sorted(checkpoints_dir.glob("epoch_*.pth.tar")):
        epoch_label = checkpoint_path.stem.replace(".pth", "")
        csv_path = output_dir / f"{epoch_label}_test_results.csv"
        png_path = output_dir / f"{epoch_label}_parity.png"
        loglog_png_path = output_dir / f"{epoch_label}_parity_loglog.png"
        metrics_path = output_dir / f"{epoch_label}_metrics.json"

        predict_model(
            dataset=dataset,
            task="regression",
            modelpath=str(checkpoint_path),
            batch_size=args.batch_size,
            workers=args.workers,
            cuda=False,
            print_freq=args.print_freq,
            shuffle=False,
            output_csv=str(csv_path),
        )

        targets, predictions = load_predictions(csv_path)
        metrics = compute_metrics(targets, predictions)
        make_plot(targets, predictions, metrics, png_path)
        made_loglog = make_loglog_plot(targets, predictions, metrics, loglog_png_path)
        if not made_loglog and loglog_png_path.exists():
            loglog_png_path.unlink()

        metrics_payload = dict(metrics)
        metrics_payload["checkpoint"] = str(checkpoint_path)
        metrics_payload["test_results_csv"] = str(csv_path)
        metrics_payload["parity_plot"] = str(png_path)
        metrics_payload["parity_plot_loglog"] = (
            str(loglog_png_path) if loglog_png_path.exists() else None
        )
        with metrics_path.open("w") as handle:
            json.dump(metrics_payload, handle, indent=2)

        summary_rows.append(
            {
                "epoch": epoch_label.split("_")[-1],
                "checkpoint": str(checkpoint_path),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2_score": metrics["r2_score"],
                "pearson_r": metrics["pearson_r"],
                "parity_plot": str(png_path),
                "parity_plot_loglog": (
                    str(loglog_png_path) if loglog_png_path.exists() else ""
                ),
                "metrics_json": str(metrics_path),
                "test_results_csv": str(csv_path),
            }
        )
        print(f"finished {epoch_label}", flush=True)

    summary_csv = output_dir / "epoch_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "checkpoint",
                "mae",
                "rmse",
                "r2_score",
                "pearson_r",
                "parity_plot",
                "parity_plot_loglog",
                "metrics_json",
                "test_results_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"summary_csv={summary_csv}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
