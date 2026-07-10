from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/work2/04996/tg842951/stampede3/run/FeCoS_paper/MPRelaxSet")
DATASET_DIR = ROOT / "cgcnn_ehull"
BENCHMARK_CSV = (
    ROOT
    / "fe_co_s_ehull_mp_relaxset_20260630_022854"
    / "fe_co_s_ehull.csv"
)
OLD_PREDICTIONS = DATASET_DIR / "ehull_predictions.csv"
NEW_PREDICTIONS = (
    DATASET_DIR / "ehull_predictions_mp_all_ehull_cpu_20260430_045246.csv"
)
LATEST_PREDICTIONS = (
    DATASET_DIR
    / "ehull_predictions_scaled_log1p_s0p05_april_arch_20260630_0736.csv"
)
COMPARISON_CSV = DATASET_DIR / "ehull_mp_vs_cgcnn_two_models_by_formula.csv"
SUMMARY_TXT = DATASET_DIR / "ehull_mp_vs_cgcnn_two_models_summary.txt"
PLOT_PNG = DATASET_DIR / "ehull_parity_old_and_mp_all_models.png"
PLOT_PDF = DATASET_DIR / "ehull_parity_old_and_mp_all_models.pdf"
STABILITY_THRESHOLD = 0.1

MODELS = {
    "old_scaled_log1p_s0p05": {
        "path": OLD_PREDICTIONS,
        "legend": "Old log1p",
        "color": "#2f6fbb",
        "marker": "o",
    },
    "mp_all_raw_20260430": {
        "path": NEW_PREDICTIONS,
        "legend": "April raw",
        "color": "#d17b25",
        "marker": "^",
    },
    "scaled_log1p_april_arch_20260630": {
        "path": LATEST_PREDICTIONS,
        "legend": "Latest log1p",
        "color": "#4c9a52",
        "marker": "s",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require_unique(rows: list[dict[str, str]], key: str, label: str) -> None:
    counts = Counter(row[key] for row in rows)
    duplicates = [value for value, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"{label} has duplicate {key} values: {duplicates}")


def metrics(refs: list[float], preds: list[float]) -> dict[str, float]:
    errors = [pred - ref for ref, pred in zip(refs, preds)]
    abs_errors = [abs(error) for error in errors]
    ref_mean = statistics.fmean(refs)
    pred_mean = statistics.fmean(preds)
    denom_ref = math.sqrt(sum((ref - ref_mean) ** 2 for ref in refs))
    denom_pred = math.sqrt(sum((pred - pred_mean) ** 2 for pred in preds))
    pearson = (
        sum((ref - ref_mean) * (pred - pred_mean) for ref, pred in zip(refs, preds))
        / (denom_ref * denom_pred)
        if denom_ref and denom_pred
        else float("nan")
    )
    return {
        "mae": statistics.fmean(abs_errors),
        "rmse": math.sqrt(statistics.fmean([error * error for error in errors])),
        "mean_error": statistics.fmean(errors),
        "median_abs_error": statistics.median(abs_errors),
        "max_abs_error": max(abs_errors),
        "pearson_r": pearson,
    }


def format_float(value: float) -> str:
    return f"{value:.16g}"


benchmark_rows = read_csv(BENCHMARK_CSV)
require_unique(benchmark_rows, "formula", "benchmark")
benchmark_by_formula = {row["formula"]: row for row in benchmark_rows}

prediction_by_model: dict[str, dict[str, dict[str, str]]] = {}
for model_name, config in MODELS.items():
    rows = read_csv(config["path"])
    for row in rows:
        row["formula"] = row.pop("material_id")
    require_unique(rows, "formula", model_name)
    prediction_by_model[model_name] = {row["formula"]: row for row in rows}

all_formulas = set(benchmark_by_formula)
for rows_by_formula in prediction_by_model.values():
    all_formulas.update(rows_by_formula)

fieldnames = [
    "id",
    "formula",
    "benchmark_ehull_eV_per_atom",
    "benchmark_is_stable",
    "decomposition",
]
for model_name in MODELS:
    fieldnames.extend(
        [
            f"{model_name}_predicted_ehull_eV_per_atom",
            f"{model_name}_error_pred_minus_benchmark_eV_per_atom",
            f"{model_name}_abs_error_eV_per_atom",
            f"{model_name}_predicted_stable_at_0p1_eV_bool",
            f"{model_name}_stable_match_at_0p1_eV_bool",
        ]
    )
fieldnames.append("_merge")

joined: list[dict[str, str]] = []
for formula in sorted(all_formulas):
    ref_row = benchmark_by_formula.get(formula)
    row = {
        "id": ref_row.get("id", "") if ref_row else "",
        "formula": formula,
        "benchmark_ehull_eV_per_atom": (
            ref_row.get("e_above_hull_eV_per_atom", "") if ref_row else ""
        ),
        "benchmark_is_stable": ref_row.get("is_stable", "") if ref_row else "",
        "decomposition": ref_row.get("decomposition", "") if ref_row else "",
        "_merge": "both",
    }
    if ref_row is None:
        row["_merge"] = "prediction_only"
    ref_value = (
        float(ref_row["e_above_hull_eV_per_atom"]) if ref_row is not None else None
    )
    ref_stable = (
        ref_row.get("is_stable", "").strip().lower() == "true"
        if ref_row is not None
        else None
    )
    for model_name, predictions in prediction_by_model.items():
        pred_row = predictions.get(formula)
        if pred_row is None:
            if row["_merge"] == "both":
                row["_merge"] = "benchmark_only"
            for suffix in [
                "predicted_ehull_eV_per_atom",
                "error_pred_minus_benchmark_eV_per_atom",
                "abs_error_eV_per_atom",
                "predicted_stable_at_0p1_eV_bool",
                "stable_match_at_0p1_eV_bool",
            ]:
                row[f"{model_name}_{suffix}"] = ""
            continue
        pred_value = float(pred_row["predicted_ehull_eV_per_atom"])
        row[f"{model_name}_predicted_ehull_eV_per_atom"] = format_float(pred_value)
        if ref_value is None:
            row[f"{model_name}_error_pred_minus_benchmark_eV_per_atom"] = ""
            row[f"{model_name}_abs_error_eV_per_atom"] = ""
            row[f"{model_name}_predicted_stable_at_0p1_eV_bool"] = ""
            row[f"{model_name}_stable_match_at_0p1_eV_bool"] = ""
            continue
        error = pred_value - ref_value
        pred_stable = pred_value <= STABILITY_THRESHOLD
        row[f"{model_name}_error_pred_minus_benchmark_eV_per_atom"] = format_float(
            error
        )
        row[f"{model_name}_abs_error_eV_per_atom"] = format_float(abs(error))
        row[f"{model_name}_predicted_stable_at_0p1_eV_bool"] = str(pred_stable)
        row[f"{model_name}_stable_match_at_0p1_eV_bool"] = str(
            ref_stable == pred_stable
        )
    joined.append(row)

with COMPARISON_CSV.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(joined)

matched = [row for row in joined if row["_merge"] == "both"]
refs = [float(row["benchmark_ehull_eV_per_atom"]) for row in matched]
summary_lines = [
    f"benchmark_rows: {len(benchmark_rows)}",
    *[
        f"{model_name}_prediction_rows: {len(prediction_by_model[model_name])}"
        for model_name in MODELS
    ],
    f"matched_rows: {len(matched)}",
    f"stability_threshold_eV_per_atom: {STABILITY_THRESHOLD}",
    "",
]

model_metrics: dict[str, dict[str, float]] = {}
for model_name in MODELS:
    preds = [
        float(row[f"{model_name}_predicted_ehull_eV_per_atom"]) for row in matched
    ]
    model_metrics[model_name] = metrics(refs, preds)
    match_count = sum(
        1 for row in matched if row[f"{model_name}_stable_match_at_0p1_eV_bool"] == "True"
    )
    summary_lines.extend(
        [
            f"{model_name}:",
            f"  mae_eV_per_atom: {model_metrics[model_name]['mae']}",
            f"  rmse_eV_per_atom: {model_metrics[model_name]['rmse']}",
            f"  mean_error_pred_minus_benchmark_eV_per_atom: {model_metrics[model_name]['mean_error']}",
            f"  median_abs_error_eV_per_atom: {model_metrics[model_name]['median_abs_error']}",
            f"  max_abs_error_eV_per_atom: {model_metrics[model_name]['max_abs_error']}",
            f"  pearson_r: {model_metrics[model_name]['pearson_r']}",
            f"  predicted_at_or_below_0p1_count: {sum(1 for pred in preds if pred <= STABILITY_THRESHOLD)}",
            f"  stable_classification_matches: {match_count}",
            f"  stable_classification_mismatches: {len(matched) - match_count}",
            f"  stable_classification_match_fraction: {match_count / len(matched):.6g}",
            "",
        ]
    )

summary_lines.append("largest_abs_errors_by_model:")
for model_name in MODELS:
    summary_lines.append(f"{model_name}:")
    for row in sorted(
        matched,
        key=lambda item: float(item[f"{model_name}_abs_error_eV_per_atom"]),
        reverse=True,
    )[:8]:
        summary_lines.append(
            "  "
            f"{row['formula']}: ref={row['benchmark_ehull_eV_per_atom']}, "
            f"pred={row[f'{model_name}_predicted_ehull_eV_per_atom']}, "
            f"error={row[f'{model_name}_error_pred_minus_benchmark_eV_per_atom']}"
        )
SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n")

max_value = max(
    [STABILITY_THRESHOLD]
    + refs
    + [
        float(row[f"{model_name}_predicted_ehull_eV_per_atom"])
        for row in matched
        for model_name in MODELS
    ]
)
limit = math.ceil((max_value * 1.08) / 0.05) * 0.05
fig, ax = plt.subplots(figsize=(6.7, 5.9), dpi=180)
for model_name, config in MODELS.items():
    preds = [
        float(row[f"{model_name}_predicted_ehull_eV_per_atom"]) for row in matched
    ]
    ax.scatter(
        refs,
        preds,
        label=(
            config["legend"]
        ),
        marker=config["marker"],
        s=58,
        c=config["color"],
        edgecolor="white",
        linewidth=0.7,
        alpha=0.86,
        zorder=3,
    )
ax.plot([0, limit], [0, limit], color="#222222", linewidth=1.2, zorder=2)
ax.axhline(STABILITY_THRESHOLD, color="#777777", linewidth=1.0, linestyle="--")
ax.axvline(STABILITY_THRESHOLD, color="#777777", linewidth=1.0, linestyle="--")
ax.set_xlim(-0.01, limit)
ax.set_ylim(-0.01, limit)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Benchmark E_hull (eV/atom)")
ax.set_ylabel("Predicted E_hull (eV/atom)")
ax.set_title("CGCNN E_hull Parity: Three CGCNN Models")
ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.8)
handles, labels = ax.get_legend_handles_labels()
handles.append(
    plt.Line2D(
        [0],
        [0],
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        label="0.10 eV/atom threshold",
    )
)
ax.legend(handles=handles, loc="upper left", fontsize=8.0, frameon=True, framealpha=0.92)
fig.tight_layout()
fig.savefig(PLOT_PNG, bbox_inches="tight")
fig.savefig(PLOT_PDF, bbox_inches="tight")
plt.close(fig)

run_md = DATASET_DIR / "RUN.md"
if run_md.exists():
    text = run_md.read_text().rstrip()
    note = f"""

## Two-Model Comparison

The April MP-all raw-target checkpoint `{MODELS['mp_all_raw_20260430']['path']}` and the larger scaled-log1p April-architecture checkpoint `{MODELS['scaled_log1p_april_arch_20260630']['path']}` were applied to the same 32 CIF files and compared with the benchmark table by `formula`.

Outputs:

- `ehull_predictions_mp_all_ehull_cpu_20260430_045246.csv`: raw-model physical E_hull predictions in eV/atom.
- `ehull_predictions_scaled_log1p_s0p05_april_arch_20260630_0736.csv`: larger scaled-log1p April-architecture physical E_hull predictions in eV/atom.
- `ehull_mp_vs_cgcnn_two_models_by_formula.csv`: benchmark and all model predictions joined by `formula`.
- `ehull_mp_vs_cgcnn_two_models_summary.txt`: metrics for all models using the `{STABILITY_THRESHOLD:.2f}` eV/atom predicted-stability threshold.
- `ehull_parity_old_and_mp_all_models.png` / `.pdf`: combined parity plot with all model series.
"""
    marker = "\n## Two-Model Comparison\n"
    if marker in text:
        text = text.split(marker)[0].rstrip() + note
    else:
        text = text + note
    run_md.write_text(text + "\n")

print(COMPARISON_CSV)
print(SUMMARY_TXT)
print(PLOT_PNG)
