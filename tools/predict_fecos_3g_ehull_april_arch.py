from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cgcnn.data import CIFData
from cgcnn.inference import predict_model


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(
    "/work2/04996/tg842951/stampede3/run/FeCoS_3G_benchmark_ehull_20260630_090754"
)
RUN_DIR = BENCHMARK_ROOT / "cgcnn_ehull_april_arch"
DATASET_DIR = RUN_DIR / "dataset"
MODEL_PATH = RUN_DIR / "model_best.pth.tar"
BENCHMARK_CSV = BENCHMARK_ROOT / "benchmark_ehull_3g.csv"
RAW_OUTPUT = RUN_DIR / "test_results.csv"
PREDICTIONS_CSV = RUN_DIR / "ehull_predictions.csv"
COMPARISON_CSV = RUN_DIR / "ehull_3g_benchmark_vs_cgcnn_by_candidate.csv"
FORMULA_SUMMARY_CSV = RUN_DIR / "ehull_3g_formula_summary.csv"
SUMMARY_TXT = RUN_DIR / "ehull_3g_benchmark_vs_cgcnn_summary.txt"
PARITY_PNG = RUN_DIR / "ehull_3g_parity_predicted_vs_benchmark.png"
PARITY_PDF = RUN_DIR / "ehull_3g_parity_predicted_vs_benchmark.pdf"
METADATA_PATH = RUN_DIR / "prediction_metadata.json"
SCALE_EV_PER_ATOM = 0.05
STABLE_THRESHOLD_EV_PER_ATOM = 0.1


def inverse_scaled_log1p(value: float) -> float:
    return SCALE_EV_PER_ATOM * math.expm1(max(value, 0.0))


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else float("nan")


def load_benchmark_rows() -> list[dict[str, str]]:
    with BENCHMARK_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_raw_predictions() -> dict[str, dict[str, float]]:
    predictions: dict[str, dict[str, float]] = {}
    with RAW_OUTPUT.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            predicted_z = float(row[2])
            predictions[row[0]] = {
                "dummy_target_z": float(row[1]),
                "predicted_ehull_z": predicted_z,
                "predicted_ehull_eV_per_atom": inverse_scaled_log1p(predicted_z),
            }
    return predictions


def write_predictions_csv(predictions: dict[str, dict[str, float]]) -> None:
    with (DATASET_DIR / "candidate_metadata.csv").open(newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    with PREDICTIONS_CSV.open("w", newline="") as handle:
        fieldnames = [
            "candidate_id",
            "source_dataset",
            "relative_workdir",
            "formula",
            "dummy_target_z",
            "predicted_ehull_z",
            "predicted_ehull_eV_per_atom",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata_rows:
            pred = predictions[row["candidate_id"]]
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames[:4]},
                    "dummy_target_z": f"{pred['dummy_target_z']:.16g}",
                    "predicted_ehull_z": f"{pred['predicted_ehull_z']:.16g}",
                    "predicted_ehull_eV_per_atom": (
                        f"{pred['predicted_ehull_eV_per_atom']:.16g}"
                    ),
                }
            )


def write_comparison(predictions: dict[str, dict[str, float]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for bench in load_benchmark_rows():
        candidate_id = f"{bench['source_dataset']}__{bench['relative_workdir']}"
        pred = predictions[candidate_id]
        ref = float(bench["e_above_hull_eV_per_atom"])
        pred_ev = pred["predicted_ehull_eV_per_atom"]
        error = pred_ev - ref
        predicted_stable = pred_ev <= STABLE_THRESHOLD_EV_PER_ATOM
        row = {
            "candidate_id": candidate_id,
            "source_dataset": bench["source_dataset"],
            "relative_workdir": bench["relative_workdir"],
            "formula": bench["formula"],
            "e_above_hull_eV_per_atom": f"{ref:.16g}",
            "predicted_ehull_eV_per_atom": f"{pred_ev:.16g}",
            "predicted_ehull_z": f"{pred['predicted_ehull_z']:.16g}",
            "error_pred_minus_ref_eV_per_atom": f"{error:.16g}",
            "abs_error_eV_per_atom": f"{abs(error):.16g}",
            "is_stable": bench["is_stable"],
            "predicted_stable_at_0p1_eV_bool": str(predicted_stable),
            "stable_match_at_0p1_eV_bool": str(
                (bench["is_stable"].strip().lower() == "true") == predicted_stable
            ),
            "decomposition": bench["decomposition"],
        }
        rows.append(row)
    rows.sort(key=lambda item: float(item["abs_error_eV_per_atom"]), reverse=True)
    with COMPARISON_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_formula_summary(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["formula"]].append(row)
    summary_rows: list[dict[str, str]] = []
    for formula, items in grouped.items():
        errors = [float(row["error_pred_minus_ref_eV_per_atom"]) for row in items]
        abs_errors = [abs(value) for value in errors]
        refs = [float(row["e_above_hull_eV_per_atom"]) for row in items]
        preds = [float(row["predicted_ehull_eV_per_atom"]) for row in items]
        matches = [row["stable_match_at_0p1_eV_bool"] == "True" for row in items]
        summary_rows.append(
            {
                "formula": formula,
                "n_candidates": len(items),
                "mean_reference_ehull_eV_per_atom": f"{sum(refs) / len(refs):.16g}",
                "mean_predicted_ehull_eV_per_atom": f"{sum(preds) / len(preds):.16g}",
                "mae_eV_per_atom": f"{sum(abs_errors) / len(abs_errors):.16g}",
                "mean_error_pred_minus_ref_eV_per_atom": (
                    f"{sum(errors) / len(errors):.16g}"
                ),
                "stable_match_fraction_at_0p1_eV": (
                    f"{sum(matches) / len(matches):.16g}"
                ),
            }
        )
    summary_rows.sort(key=lambda item: (-int(item["n_candidates"]), item["formula"]))
    with FORMULA_SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_summary_and_plot(rows: list[dict[str, str]]) -> None:
    refs = [float(row["e_above_hull_eV_per_atom"]) for row in rows]
    preds = [float(row["predicted_ehull_eV_per_atom"]) for row in rows]
    errors = [pred - ref for ref, pred in zip(refs, preds)]
    abs_errors = [abs(value) for value in errors]
    matches = [row["stable_match_at_0p1_eV_bool"] == "True" for row in rows]
    lines = [
        f"benchmark_rows: {len(rows)}",
        f"prediction_rows: {len(rows)}",
        f"unique_formulas: {len(set(row['formula'] for row in rows))}",
        f"mae_eV_per_atom: {sum(abs_errors) / len(abs_errors)}",
        f"rmse_eV_per_atom: {math.sqrt(sum(value * value for value in errors) / len(errors))}",
        f"mean_error_pred_minus_ref_eV_per_atom: {sum(errors) / len(errors)}",
        f"median_abs_error_eV_per_atom: {statistics.median(abs_errors)}",
        f"max_abs_error_eV_per_atom: {max(abs_errors)}",
        f"pearson_r: {pearson(refs, preds)}",
        f"reference_stable_count: {sum(row['is_stable'].strip().lower() == 'true' for row in rows)}",
        f"predicted_at_or_below_0p1_count: {sum(pred <= STABLE_THRESHOLD_EV_PER_ATOM for pred in preds)}",
        "",
        "stable_classification_at_0p1_eV:",
        f"threshold_eV_per_atom: {STABLE_THRESHOLD_EV_PER_ATOM}",
        f"matched_rows: {sum(matches)}",
        f"mismatched_rows: {len(matches) - sum(matches)}",
        f"match_fraction: {sum(matches) / len(matches):.6g}",
        "",
        "largest_abs_errors:",
    ]
    for row in rows[:15]:
        lines.append(
            f"{row['candidate_id']} ({row['formula']}): "
            f"ref={row['e_above_hull_eV_per_atom']}, "
            f"pred={row['predicted_ehull_eV_per_atom']}, "
            f"error={row['error_pred_minus_ref_eV_per_atom']}"
        )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")

    max_val = max(max(refs), max(preds), STABLE_THRESHOLD_EV_PER_ATOM)
    limit = math.ceil((max_val * 1.05) / 0.1) * 0.1
    fig, ax = plt.subplots(figsize=(6.4, 5.8), dpi=180)
    colors = ["#2f6fbb" if match else "#c43b3b" for match in matches]
    ax.scatter(refs, preds, c=colors, s=18, edgecolor="white", linewidth=0.25, alpha=0.78)
    ax.plot([0, limit], [0, limit], color="#222222", linewidth=1.2)
    ax.axhline(STABLE_THRESHOLD_EV_PER_ATOM, color="#777777", linestyle="--", linewidth=1.0)
    ax.axvline(STABLE_THRESHOLD_EV_PER_ATOM, color="#777777", linestyle="--", linewidth=1.0)
    for row in rows[:8]:
        ax.annotate(
            row["relative_workdir"],
            (
                float(row["e_above_hull_eV_per_atom"]),
                float(row["predicted_ehull_eV_per_atom"]),
            ),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.4,
        )
    ax.set_xlim(-0.01, limit)
    ax.set_ylim(-0.01, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Benchmark E_hull (eV/atom)")
    ax.set_ylabel("Predicted E_hull (eV/atom)")
    ax.set_title("CGCNN E_hull Parity: Fe-Co-S 3G Benchmark")
    ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.8)
    ax.text(
        0.03,
        0.97,
        (
            f"n = {len(rows)}\n"
            f"MAE = {sum(abs_errors) / len(abs_errors):.3f} eV/atom\n"
            f"RMSE = {math.sqrt(sum(value * value for value in errors) / len(errors)):.3f} eV/atom\n"
            f"r = {pearson(refs, preds):.3f}\n"
            f"stable match = {sum(matches)}/{len(matches)}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.92,
        },
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2f6fbb",
            markeredgecolor="white",
            markersize=7,
            label="stable label matched",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#c43b3b",
            markeredgecolor="white",
            markersize=7,
            label="stable label mismatched",
        ),
        Line2D(
            [0],
            [0],
            color="#777777",
            linestyle="--",
            linewidth=1.0,
            label="0.10 eV/atom threshold",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.4, frameon=True)
    fig.tight_layout()
    fig.savefig(PARITY_PNG, bbox_inches="tight")
    fig.savefig(PARITY_PDF, bbox_inches="tight")
    plt.close(fig)


def update_metadata(row_count: int) -> None:
    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    metadata["prediction_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["outputs"] = {
        "test_results_csv": str(RAW_OUTPUT),
        "ehull_predictions_csv": str(PREDICTIONS_CSV),
        "comparison_csv": str(COMPARISON_CSV),
        "formula_summary_csv": str(FORMULA_SUMMARY_CSV),
        "summary_txt": str(SUMMARY_TXT),
        "parity_png": str(PARITY_PNG),
        "parity_pdf": str(PARITY_PDF),
        "prediction_rows": row_count,
        "interactive_node": Path("/proc/sys/kernel/hostname").read_text().strip(),
        "command": "uv run python tools/predict_fecos_3g_ehull_april_arch.py",
        "batch_size": 256,
        "workers": 0,
        "device": "cpu",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    dataset = CIFData(str(DATASET_DIR), shuffle=False)
    predict_model(
        dataset,
        modelpath=str(MODEL_PATH),
        task="regression",
        batch_size=256,
        workers=0,
        cuda=False,
        print_freq=1,
        output_csv=str(RAW_OUTPUT),
    )
    raw_predictions = load_raw_predictions()
    write_predictions_csv(raw_predictions)
    comparison_rows = write_comparison(raw_predictions)
    write_formula_summary(comparison_rows)
    write_summary_and_plot(comparison_rows)
    update_metadata(len(comparison_rows))
    print(f"Wrote {len(comparison_rows)} candidate predictions to {PREDICTIONS_CSV}")
