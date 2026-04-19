from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly


DEFAULT_CIF_ID_REGEX = r"pos_pos_(\d+)_relaxed\.cif$"
DEFAULT_POS_DIR_PATTERN = "pos_{id}"
DEFAULT_CIF_NAME_PATTERN = "pos_pos_{id}_relaxed.cif"


@dataclass(frozen=True)
class ParityConfig:
    predictions_csv: Path
    output_html: Path
    output_summary: Path | None
    pos_root: Path | None
    pos_dir_pattern: str
    cif_name_pattern: str
    cif_id_regex: str
    selected_only: bool
    include_plotlyjs: bool | str
    distance_percentile: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Plotly parity plot whose points link to source CIF files."
        )
    )
    parser.add_argument(
        "predictions_csv",
        type=Path,
        help=(
            "Merged prediction CSV with true_m, predicted_m, and optionally "
            "source_path/is_selected columns."
        ),
    )
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--pos-root",
        type=Path,
        default=None,
        help=(
            "Parent directory containing pos_<id>/ folders. When provided, CIF "
            "links are built from this root plus --pos-dir-pattern and "
            "--cif-name-pattern."
        ),
    )
    parser.add_argument(
        "--pos-dir-pattern",
        default=DEFAULT_POS_DIR_PATTERN,
        help="Format string for position directories. Default: pos_{id}",
    )
    parser.add_argument(
        "--cif-name-pattern",
        default=DEFAULT_CIF_NAME_PATTERN,
        help="Format string for CIF filenames. Default: pos_pos_{id}_relaxed.cif",
    )
    parser.add_argument(
        "--cif-id-regex",
        default=DEFAULT_CIF_ID_REGEX,
        help=(
            "Regex used to extract the id from a source_path/CIF filename. The "
            "first capture group is used as id."
        ),
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Plot all rows with predicted_m instead of only is_selected == 1.",
    )
    parser.add_argument(
        "--distance-percentile",
        type=float,
        default=95.0,
        help="Percentile cutoff for perpendicular-distance outliers.",
    )
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--plotlyjs",
        default="inline",
        choices=["inline", "cdn", "directory"],
        help=(
            "How Plotly JS is included in the HTML. Use inline for a standalone "
            "file, cdn for smaller HTML, or directory to write plotly.min.js "
            "beside the HTML."
        ),
    )
    return parser.parse_args()


def plotlyjs_option(value: str) -> bool | str:
    if value == "inline":
        return True
    if value == "cdn":
        return "cdn"
    return "directory"


def file_uri(path: Path) -> str:
    return "file://" + quote(str(path))


def format_pattern(pattern: str, cif_id: str) -> str:
    return pattern.format(id=cif_id, pos_id=cif_id)


def infer_id(row: dict[str, str], cif_id_regex: re.Pattern[str]) -> str:
    for key in ("source_path", "structure_id"):
        value = row.get(key, "")
        if not value:
            continue
        match = cif_id_regex.search(Path(value).name)
        if match:
            return match.group(1)
        match = re.search(r"pos_(\d+)$", value)
        if match:
            return match.group(1)
    structure_id = row.get("structure_id", "")
    if structure_id.startswith("pos_"):
        return structure_id.removeprefix("pos_")
    return structure_id


def resolve_cif_path(
    row: dict[str, str],
    cif_id: str,
    pos_root: Path | None,
    pos_dir_pattern: str,
    cif_name_pattern: str,
) -> Path:
    if pos_root is None:
        source_path = row.get("source_path", "")
        if not source_path:
            raise ValueError(
                "source_path column is required when --pos-root is not provided."
            )
        return Path(source_path).expanduser()
    return (
        pos_root.expanduser()
        / format_pattern(pos_dir_pattern, cif_id)
        / format_pattern(cif_name_pattern, cif_id)
    )


def load_parity_rows(config: ParityConfig) -> pd.DataFrame:
    cif_id_regex = re.compile(config.cif_id_regex)
    rows: list[dict[str, object]] = []
    with config.predictions_csv.expanduser().open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"true_m", "predicted_m"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns in {config.predictions_csv}: {sorted(missing)}"
            )
        for row in reader:
            if config.selected_only and row.get("is_selected") != "1":
                continue
            if not row.get("predicted_m"):
                continue
            cif_id = infer_id(row, cif_id_regex)
            cif_path = resolve_cif_path(
                row,
                cif_id=cif_id,
                pos_root=config.pos_root,
                pos_dir_pattern=config.pos_dir_pattern,
                cif_name_pattern=config.cif_name_pattern,
            )
            true_m = float(row["true_m"])
            predicted_m = float(row["predicted_m"])
            distance = abs(predicted_m - true_m) / math.sqrt(2.0)
            foot = (true_m + predicted_m) / 2.0
            rows.append(
                {
                    "structure_id": row.get("structure_id", cif_id),
                    "id": cif_id,
                    "source_path": str(cif_path),
                    "file_url": file_uri(cif_path),
                    "true_m": true_m,
                    "predicted_m": predicted_m,
                    "p_large": float(row["p_large"]) if row.get("p_large") else np.nan,
                    "d": distance,
                    "foot_x": foot,
                    "foot_y": foot,
                }
            )
    if not rows:
        raise ValueError("No rows with true_m and predicted_m were found.")
    return pd.DataFrame(rows)


def add_distance_outlier_flag(
    df: pd.DataFrame, distance_percentile: float
) -> tuple[pd.DataFrame, float]:
    threshold = float(np.percentile(df["d"].to_numpy(), distance_percentile))
    flagged = df.copy()
    flagged["is_outlier"] = flagged["d"] >= threshold
    return flagged, threshold


def parity_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["true_m"].to_numpy(dtype=float)
    y_pred = df["predicted_m"].to_numpy(dtype=float)
    residual = y_pred - y_true
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "R": float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(df) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan"),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
    }


def axis_range(df: pd.DataFrame) -> tuple[float, float]:
    low = float(np.nanmin([df["true_m"].min(), df["predicted_m"].min()]))
    high = float(np.nanmax([df["true_m"].max(), df["predicted_m"].max()]))
    pad = max((high - low) * 0.07, 1e-6)
    return low - pad, high + pad


def customdata(df: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            df["id"],
            df["d"],
            df["source_path"],
            df["file_url"],
        ],
        axis=-1,
    )


def add_point_traces(fig: go.Figure, df: pd.DataFrame) -> None:
    normal = df[~df["is_outlier"]]
    outliers = df[df["is_outlier"]].sort_values("d", ascending=False)
    hovertemplate = (
        "id: %{customdata[0]}<br>"
        "true M: %{x:.8g}<br>"
        "predicted M: %{y:.8g}<br>"
        "d: %{customdata[1]:.8g}<br>"
        "CIF: %{customdata[2]}"
        "<extra></extra>"
    )
    fig.add_trace(
        go.Scatter(
            x=normal["true_m"],
            y=normal["predicted_m"],
            mode="markers",
            name="points",
            customdata=customdata(normal),
            marker={
                "color": "#2364AA",
                "size": 7,
                "opacity": 0.62,
                "line": {"width": 0},
            },
            hovertemplate=hovertemplate,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=outliers["true_m"],
            y=outliers["predicted_m"],
            mode="markers",
            name="distance outliers",
            customdata=customdata(outliers),
            marker={
                "color": "#F3A712",
                "size": 9,
                "opacity": 0.96,
                "line": {"color": "#8B1E3F", "width": 1.2},
            },
            hovertemplate=hovertemplate,
        )
    )


def add_reference_traces(
    fig: go.Figure, df: pd.DataFrame, range_min: float, range_max: float
) -> None:
    outliers = df[df["is_outlier"]].sort_values("d", ascending=False)
    drop_x: list[float | None] = []
    drop_y: list[float | None] = []
    for row in outliers.itertuples(index=False):
        drop_x.extend([row.true_m, row.foot_x, None])
        drop_y.extend([row.predicted_m, row.foot_y, None])
    fig.add_trace(
        go.Scatter(
            x=drop_x,
            y=drop_y,
            mode="lines",
            name="perpendicular d",
            line={"color": "rgba(139, 30, 63, 0.55)", "width": 1.1},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[range_min, range_max],
            y=[range_min, range_max],
            mode="lines",
            name="y = x",
            line={"color": "#222222", "width": 2},
            hoverinfo="skip",
        )
    )


def metrics_annotation(
    df: pd.DataFrame, metrics: dict[str, float], distance_threshold: float
) -> dict[str, object]:
    return {
        "x": 0.985,
        "y": 0.145,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "right",
        "yanchor": "bottom",
        "align": "left",
        "showarrow": False,
        "bgcolor": "rgba(255,255,255,0.88)",
        "bordercolor": "#888",
        "borderwidth": 1,
        "borderpad": 6,
        "font": {"size": 13, "color": "#111"},
        "text": (
            f"N = {len(df)}&nbsp;&nbsp;&nbsp;&nbsp;R^2 = {metrics['R2']:.4f}<br>"
            f"outliers = {int(df['is_outlier'].sum())}"
            f"&nbsp;&nbsp;&nbsp;&nbsp;MAE = {metrics['MAE']:.6g}<br>"
            f"d_95 = {distance_threshold:.6g}"
            f"&nbsp;&nbsp;&nbsp;&nbsp;RMSE = {metrics['RMSE']:.6g}<br>"
            f"R = {metrics['R']:.4f}"
        ),
    }


def build_figure(
    df: pd.DataFrame,
    metrics: dict[str, float],
    distance_threshold: float,
    width: int,
    height: int,
) -> go.Figure:
    range_min, range_max = axis_range(df)
    fig = go.Figure()
    add_point_traces(fig, df)
    add_reference_traces(fig, df, range_min=range_min, range_max=range_max)
    fig.update_layout(
        title={"text": "FeCoS Top-Bin Regressor Parity", "x": 0.5, "xanchor": "center"},
        width=width,
        height=height,
        template="plotly_white",
        margin={"l": 80, "r": 40, "t": 70, "b": 80},
        xaxis={
            "title": "True M = magnetization / volume",
            "range": [range_min, range_max],
            "scaleanchor": "y",
            "scaleratio": 1,
            "showgrid": True,
            "gridcolor": "#d7d7d7",
            "zeroline": False,
        },
        yaxis={
            "title": "Predicted M",
            "range": [range_min, range_max],
            "showgrid": True,
            "gridcolor": "#d7d7d7",
            "zeroline": False,
        },
        legend={
            "x": 0.99,
            "y": 0.01,
            "xanchor": "right",
            "yanchor": "bottom",
            "orientation": "h",
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "#888",
            "borderwidth": 1,
            "traceorder": "normal",
            "itemsizing": "constant",
        },
        annotations=[metrics_annotation(df, metrics, distance_threshold)],
    )
    return fig


def write_plot_html(
    fig: go.Figure, output_html: Path, include_plotlyjs: bool | str
) -> None:
    post_script = """
const plot = document.getElementById('{plot_id}');
plot.on('plotly_click', function(eventData) {
  const point = eventData.points && eventData.points[0];
  if (!point || !point.customdata || !point.customdata[3]) return;
  window.open(point.customdata[3], '_blank');
});
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_html,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        div_id="linked-parity-plot",
        post_script=post_script,
        config={"responsive": True, "displaylogo": False},
    )


def write_summary(
    path: Path | None,
    config: ParityConfig,
    df: pd.DataFrame,
    metrics: dict[str, float],
    distance_threshold: float,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "html": str(config.output_html),
        "library": "plotly",
        "plotly_version": plotly.__version__,
        "predictions_csv": str(config.predictions_csv),
        "pos_root": None if config.pos_root is None else str(config.pos_root),
        "pos_dir_pattern": config.pos_dir_pattern,
        "cif_name_pattern": config.cif_name_pattern,
        "cif_id_regex": config.cif_id_regex,
        "selected_only": config.selected_only,
        "n_points": int(len(df)),
        "linked_points": int(len(df)),
        "distance_outliers": int(df["is_outlier"].sum()),
        "distance_percentile": config.distance_percentile,
        "distance_threshold": distance_threshold,
        "aspect_ratio": f"{config.width}:{config.height}",
        "metrics": metrics,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def run(config: ParityConfig) -> dict[str, float]:
    df = load_parity_rows(config)
    df, distance_threshold = add_distance_outlier_flag(
        df, distance_percentile=config.distance_percentile
    )
    metrics = parity_metrics(df)
    fig = build_figure(
        df,
        metrics=metrics,
        distance_threshold=distance_threshold,
        width=config.width,
        height=config.height,
    )
    write_plot_html(fig, config.output_html, include_plotlyjs=config.include_plotlyjs)
    write_summary(config.output_summary, config, df, metrics, distance_threshold)
    return {
        "n_points": float(len(df)),
        "distance_outliers": float(df["is_outlier"].sum()),
        "distance_threshold": distance_threshold,
        **metrics,
    }


def main() -> int:
    args = parse_args()
    config = ParityConfig(
        predictions_csv=args.predictions_csv,
        output_html=args.output_html,
        output_summary=args.output_summary,
        pos_root=args.pos_root,
        pos_dir_pattern=args.pos_dir_pattern,
        cif_name_pattern=args.cif_name_pattern,
        cif_id_regex=args.cif_id_regex,
        selected_only=not args.all_rows,
        include_plotlyjs=plotlyjs_option(args.plotlyjs),
        distance_percentile=args.distance_percentile,
        width=args.width,
        height=args.height,
    )
    result = run(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
