import csv
import math
from collections.abc import Iterable, Sequence

__all__ = [
    "compute_metrics",
    "format_metrics_report",
    "write_output_csv",
    "run_benchmark",
]

OUTPUT_HEADER = [
    "ID",
    "Target",
    "Prediction",
    "Difference",
    "AbsDifference",
    "PercentError(%)",
]
BenchmarkRow = tuple[str, float, float, float, float, float]
BenchmarkMetrics = dict[str, float]


def read_csv_rows(input_file: str) -> list[list[str]]:
    with open(input_file, newline="") as handle:
        return list(csv.reader(handle))


def try_parse_result_row(row: Sequence[str]) -> BenchmarkRow | None:
    if len(row) < 3:
        return None

    try:
        float(row[1])
    except ValueError:
        return None

    cif_id = row[0]
    try:
        target = float(row[1])
        prediction = float(row[2])
    except ValueError:
        return None

    diff = prediction - target
    abs_diff = abs(diff)
    percent_error = abs_diff / abs(target) * 100.0 if target != 0 else float("inf")
    return (cif_id, target, prediction, diff, abs_diff, percent_error)


def build_processed_rows(rows: Iterable[Sequence[str]]) -> list[BenchmarkRow]:
    processed_rows: list[BenchmarkRow] = []
    for row in rows:
        parsed = try_parse_result_row(row)
        if parsed is not None:
            processed_rows.append(parsed)
    return processed_rows


def compute_metrics(processed_rows: Sequence[BenchmarkRow]) -> BenchmarkMetrics:
    if not processed_rows:
        raise ValueError("Cannot compute metrics from empty input.")

    n = len(processed_rows)
    diffs = [row[3] for row in processed_rows]
    abs_diffs = [row[4] for row in processed_rows]
    squared_diffs = [diff**2 for diff in diffs]
    percent_errors = [row[5] for row in processed_rows]
    valid_percent_errors = [value for value in percent_errors if not math.isinf(value)]

    metrics = {
        "n": float(n),
        "mean_diff": sum(diffs) / n,
        "mae": sum(abs_diffs) / n,
        "rmse": math.sqrt(sum(squared_diffs) / n),
    }
    if valid_percent_errors:
        metrics["mean_percent_error"] = sum(valid_percent_errors) / len(
            valid_percent_errors
        )
    else:
        metrics["mean_percent_error"] = float("nan")
    return metrics


def format_metrics_report(metrics: BenchmarkMetrics) -> str:
    lines = [
        f"Statistics (N={int(metrics['n'])}):",
        f"Mean Difference (Pred - Target): {metrics['mean_diff']:.6f}",
        f"MAE (Mean Absolute Error):       {metrics['mae']:.6f}",
        f"RMSE (Root Mean Squared Error):  {metrics['rmse']:.6f}",
    ]
    if math.isnan(metrics["mean_percent_error"]):
        lines.append("Mean Percent Error: undefined (all targets are zero)")
    else:
        lines.append(
            "Mean Percent Error (excluding targets==0):"
            f" {metrics['mean_percent_error']:.6f}%"
        )
    return "\n".join(lines)


def write_output_csv(output_file: str, processed_rows: Sequence[BenchmarkRow]) -> None:
    with open(output_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_HEADER)
        writer.writerows(processed_rows)


def run_benchmark(
    input_file: str, output_file: str | None = None
) -> tuple[list[BenchmarkRow], BenchmarkMetrics]:
    rows = read_csv_rows(input_file)

    if not rows:
        raise ValueError("Input file is empty.")

    processed_rows = build_processed_rows(rows)
    if not processed_rows:
        raise ValueError("No valid data found.")

    metrics = compute_metrics(processed_rows)
    if output_file is not None:
        write_output_csv(output_file, processed_rows)
    return processed_rows, metrics
