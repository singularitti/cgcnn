from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


MAD_SCALE_FACTOR = 1.4826


@dataclass(frozen=True)
class EpochOutlierRule:
    epoch: int
    abs_error_q99: float
    residual_median: float
    residual_mad: float
    robust_sigma: float


def load_run_metadata(run_dir: Path) -> dict:
    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open() as handle:
        return json.load(handle)


def load_epoch_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "epoch_analysis" / "parity_metrics_history.json"
    with metrics_path.open() as handle:
        return json.load(handle)


def load_summary_rows(summary_csv: Path) -> dict[str, dict[str, str]]:
    with summary_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "material_id",
            "formula_pretty",
            "total_magnetization",
            "energy_above_hull",
            "is_stable",
            "volume",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns in {summary_csv}: {sorted(missing)}"
            )
        return {row["material_id"]: row for row in reader}


def load_epoch_results(run_dir: Path, epoch: int) -> dict[str, dict[str, float]]:
    results_path = run_dir / "epoch_analysis" / f"test_results_epoch_{epoch:03d}.csv"
    rows: dict[str, dict[str, float]] = {}
    with results_path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            material_id = row[0]
            target = float(row[1])
            prediction = float(row[2])
            rows[material_id] = {
                "target_norm": target,
                "prediction_norm": prediction,
                "residual": prediction - target,
                "abs_error": abs(prediction - target),
            }
    if not rows:
        raise ValueError(f"No usable rows found in {results_path}")
    return rows


def quantile(values: list[float], q: float) -> float:
    sorted_values = sorted(values)
    index = min(
        len(sorted_values) - 1,
        max(0, math.ceil(q * len(sorted_values)) - 1),
    )
    return sorted_values[index]


def build_epoch_rule(rows: dict[str, dict[str, float]], epoch: int) -> EpochOutlierRule:
    abs_errors = [row["abs_error"] for row in rows.values()]
    residuals = [row["residual"] for row in rows.values()]
    residual_median = statistics.median(residuals)
    residual_mad = statistics.median(
        abs(residual - residual_median) for residual in residuals
    )
    robust_sigma = MAD_SCALE_FACTOR * residual_mad if residual_mad else 1.0
    return EpochOutlierRule(
        epoch=epoch,
        abs_error_q99=quantile(abs_errors, 0.99),
        residual_median=residual_median,
        residual_mad=residual_mad,
        robust_sigma=robust_sigma,
    )


def robust_zscore(residual: float, rule: EpochOutlierRule) -> float:
    if rule.robust_sigma == 0:
        return 0.0
    return abs(residual - rule.residual_median) / rule.robust_sigma


def median(values: list[float]) -> float:
    return statistics.median(values)


def summarize_range(values: list[float]) -> tuple[float, float, float]:
    return min(values), median(values), max(values)


def classify_behavior(residuals: list[float]) -> str:
    if all(residual > 0 for residual in residuals):
        return "persistent_overshoot"
    if all(residual < 0 for residual in residuals):
        return "persistent_undershoot"
    return "mixed_sign"


def analyze_outliers(
    run_dir: Path,
    summary_csv: Path,
    source_root: Path,
    stable_epochs: list[int],
    min_persistent_epochs: int,
) -> dict:
    metadata = load_run_metadata(run_dir)
    norm = metadata["normalization"]
    mean = float(norm["mean"])
    std = float(norm["std"])
    summary_rows = load_summary_rows(summary_csv)
    epoch_metrics = {item["epoch"]: item for item in load_epoch_metrics(run_dir)}

    stable_metrics = []
    epoch_results: dict[int, dict[str, dict[str, float]]] = {}
    epoch_rules: dict[int, EpochOutlierRule] = {}

    for epoch in stable_epochs:
        if epoch not in epoch_metrics:
            raise KeyError(f"Epoch {epoch} is missing from parity_metrics_history.json")
        stable_metrics.append(epoch_metrics[epoch])
        results = load_epoch_results(run_dir, epoch)
        epoch_results[epoch] = results
        epoch_rules[epoch] = build_epoch_rule(results, epoch)

    stable_metrics.sort(key=lambda item: item["epoch"])

    material_sets = [set(results) for results in epoch_results.values()]
    common_material_ids = set.intersection(*material_sets)
    if not common_material_ids:
        raise RuntimeError("No common material IDs found across stable epochs.")

    missing_by_epoch = {
        epoch: len(set(common_material_ids) ^ set(results))
        for epoch, results in epoch_results.items()
    }
    if any(count != 0 for count in missing_by_epoch.values()):
        raise RuntimeError(
            "Stable-epoch test result files do not contain identical material ID sets."
        )

    outliers: list[dict] = []
    stripe_outliers: list[dict] = []
    tail_outliers: list[dict] = []

    for material_id in sorted(common_material_ids):
        summary_row = summary_rows.get(material_id)
        if summary_row is None:
            raise KeyError(f"{material_id} is missing from {summary_csv}")

        epoch_records = []
        flagged_epochs = []
        flagged_residuals = []
        for epoch in stable_epochs:
            row = epoch_results[epoch][material_id]
            rule = epoch_rules[epoch]
            row_with_epoch = {
                "epoch": epoch,
                **row,
                "robust_zscore": robust_zscore(row["residual"], rule),
            }
            epoch_records.append(row_with_epoch)
            if (
                row["abs_error"] >= rule.abs_error_q99
                and row_with_epoch["robust_zscore"] >= 3.5
            ):
                flagged_epochs.append(epoch)
                flagged_residuals.append(row["residual"])

        if len(flagged_epochs) < min_persistent_epochs:
            continue

        behavior = classify_behavior(flagged_residuals)
        if behavior == "mixed_sign":
            continue

        cif_dir = source_root / material_id
        cif_files = sorted(cif_dir.glob("*.cif"))
        if len(cif_files) != 1:
            raise RuntimeError(
                f"{material_id} must have exactly one CIF file in {cif_dir}, found {len(cif_files)}"
            )

        total_magnetization = float(summary_row["total_magnetization"])
        volume = float(summary_row["volume"])
        recomputed_raw_m = total_magnetization / volume

        targets_norm = [record["target_norm"] for record in epoch_records]
        predictions_norm = [record["prediction_norm"] for record in epoch_records]
        residuals = [record["residual"] for record in epoch_records]

        target_raw_values = [value * std + mean for value in targets_norm]
        prediction_raw_values = [value * std + mean for value in predictions_norm]

        target_norm_min, target_norm_median, target_norm_max = summarize_range(
            targets_norm
        )
        prediction_norm_min, prediction_norm_median, prediction_norm_max = summarize_range(
            predictions_norm
        )
        residual_min, residual_median, residual_max = summarize_range(residuals)
        target_raw_min, target_raw_median, target_raw_max = summarize_range(
            target_raw_values
        )
        prediction_raw_min, prediction_raw_median, prediction_raw_max = summarize_range(
            prediction_raw_values
        )

        outlier = {
            "material_id": material_id,
            "formula_pretty": summary_row["formula_pretty"],
            "is_stable": summary_row["is_stable"],
            "energy_above_hull": float(summary_row["energy_above_hull"]),
            "total_magnetization": total_magnetization,
            "volume": volume,
            "recomputed_raw_m": recomputed_raw_m,
            "normalization_mean": mean,
            "normalization_std": std,
            "flagged_epoch_count": len(flagged_epochs),
            "flagged_epochs": ",".join(str(epoch) for epoch in flagged_epochs),
            "behavior": behavior,
            "target_norm_min": target_norm_min,
            "target_norm_median": target_norm_median,
            "target_norm_max": target_norm_max,
            "prediction_norm_min": prediction_norm_min,
            "prediction_norm_median": prediction_norm_median,
            "prediction_norm_max": prediction_norm_max,
            "residual_min": residual_min,
            "residual_median": residual_median,
            "residual_max": residual_max,
            "target_raw_m_min": target_raw_min,
            "target_raw_m_median": target_raw_median,
            "target_raw_m_max": target_raw_max,
            "prediction_raw_m_min": prediction_raw_min,
            "prediction_raw_m_median": prediction_raw_median,
            "prediction_raw_m_max": prediction_raw_max,
            "cif_path": str(cif_files[0]),
        }
        outliers.append(outlier)

        if (
            behavior == "persistent_overshoot"
            and 0.4 <= target_norm_median <= 0.6
        ):
            stripe_outliers.append(outlier)
        if behavior == "persistent_undershoot" and target_norm_median >= 5.0:
            tail_outliers.append(outlier)

    metric_ranges = {}
    for key in ("pearson_r", "r2_score", "mae", "rmse", "bias"):
        values = [float(item[key]) for item in stable_metrics]
        metric_ranges[key] = {
            "min": min(values),
            "max": max(values),
            "spread": max(values) - min(values),
        }

    rules_json = {
        str(epoch): {
            "abs_error_q99": rule.abs_error_q99,
            "residual_median": rule.residual_median,
            "residual_mad": rule.residual_mad,
            "robust_sigma": rule.robust_sigma,
            "mad_scale_factor": MAD_SCALE_FACTOR,
        }
        for epoch, rule in epoch_rules.items()
    }

    return {
        "summary": {
            "run_dir": str(run_dir),
            "summary_csv": str(summary_csv),
            "source_root": str(source_root),
            "stable_epochs": stable_epochs,
            "min_persistent_epochs": min_persistent_epochs,
            "candidate_count": len(outliers),
            "stripe_count": len(stripe_outliers),
            "tail_count": len(tail_outliers),
            "all_stable_epochs_have_same_material_ids": True,
            "common_material_count": len(common_material_ids),
            "metric_ranges": metric_ranges,
        },
        "stable_epoch_metrics": stable_metrics,
        "epoch_outlier_rules": rules_json,
        "persistent_outliers": outliers,
        "stripe_outliers": stripe_outliers,
        "tail_outliers": tail_outliers,
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        output_path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze persistent magnetization outliers across stable epochs."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/Users/qz/Downloads/cifs/mp_all_summary.csv"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/Users/qz/Downloads/cifs"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[12, 13, 14, 15],
    )
    parser.add_argument(
        "--min-persistent-epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    report = analyze_outliers(
        run_dir=args.run_dir,
        summary_csv=args.summary_csv,
        source_root=args.source_root,
        stable_epochs=args.epochs,
        min_persistent_epochs=args.min_persistent_epochs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = args.output_dir / "summary.json"
    persistent_csv_path = args.output_dir / "persistent_outliers.csv"
    stripe_csv_path = args.output_dir / "stripe_outliers.csv"
    tail_csv_path = args.output_dir / "tail_outliers.csv"

    with summary_json_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    write_csv(report["persistent_outliers"], persistent_csv_path)
    write_csv(report["stripe_outliers"], stripe_csv_path)
    write_csv(report["tail_outliers"], tail_csv_path)

    print(summary_json_path)
    print(persistent_csv_path)
    print(stripe_csv_path)
    print(tail_csv_path)


if __name__ == "__main__":
    main()
