from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def load_predictions(path: Path) -> dict[str, float]:
    predictions: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            predictions[row["structure_id"]] = float(row["predicted_formation_energy_per_atom"])
    return predictions


def compute_density_colors(
    x_values: np.ndarray,
    y_values: np.ndarray,
    bins: int = 320,
) -> tuple[np.ndarray, float, float]:
    x_edges = np.linspace(float(x_values.min()), float(x_values.max()), bins + 1)
    y_edges = np.linspace(float(y_values.min()), float(y_values.max()), bins + 1)
    hist, _, _ = np.histogram2d(x_values, y_values, bins=[x_edges, y_edges])
    x_idx = np.clip(np.digitize(x_values, x_edges) - 1, 0, bins - 1)
    y_idx = np.clip(np.digitize(y_values, y_edges) - 1, 0, bins - 1)
    counts = hist[x_idx, y_idx]
    log_counts = np.log10(np.maximum(counts, 1))
    return log_counts, float(log_counts.min()), float(log_counts.max())


def structure_id_to_file_uri(structures_root: str, structure_id: str) -> str:
    prefix = structure_id.split("_", 1)[0]
    cif_path = Path(structures_root) / prefix / f"{structure_id}.cif"
    return "file://" + quote(str(cif_path))


def parse_args(argv: list[str]) -> dict[str, object]:
    if len(argv) != 7:
        raise SystemExit(
            "Usage: uv run python tools/plot_model_comparison_interactive.py "
            "<old_predictions.csv> <new_predictions.csv> <structures_root> "
            "<threshold> <output.html> <summary.json>"
        )
    return {
        "old_csv": Path(argv[1]).expanduser().resolve(),
        "new_csv": Path(argv[2]).expanduser().resolve(),
        "structures_root": Path(argv[3]).expanduser().resolve(),
        "threshold": float(argv[4]),
        "output_html": Path(argv[5]).expanduser().resolve(),
        "summary_json": Path(argv[6]).expanduser().resolve(),
    }


if __name__ == "__main__":
    args = parse_args(sys.argv)
    old_predictions = load_predictions(args["old_csv"])
    new_predictions = load_predictions(args["new_csv"])
    threshold = float(args["threshold"])
    overlap_ids = sorted(set(old_predictions) & set(new_predictions))
    overlap_ids = [
        structure_id
        for structure_id in overlap_ids
        if old_predictions[structure_id] < threshold and new_predictions[structure_id] < threshold
    ]
    if not overlap_ids:
        raise SystemExit("No overlapping structure IDs remained after threshold filtering.")

    old_values = np.array([old_predictions[structure_id] for structure_id in overlap_ids])
    new_values = np.array([new_predictions[structure_id] for structure_id in overlap_ids])
    density_values, density_min, density_max = compute_density_colors(old_values, new_values)

    min_axis = float(min(old_values.min(), new_values.min()))
    max_axis = float(max(old_values.max(), new_values.max()))
    pearson_r = float(np.corrcoef(old_values, new_values)[0, 1])
    mae = float(np.mean(np.abs(new_values - old_values)))
    rmse = float(math.sqrt(np.mean((new_values - old_values) ** 2)))

    hover_text = [
        (
            f"structure_id={structure_id}<br>"
            f"old={old_value:.6f}<br>"
            f"new={new_value:.6f}<br>"
            f"log10(bin count)={density_value:.3f}<extra></extra>"
        )
        for structure_id, old_value, new_value, density_value in zip(
            overlap_ids, old_values, new_values, density_values
        )
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=old_values,
            y=new_values,
            mode="markers",
            customdata=np.array(overlap_ids, dtype=object),
            hovertemplate="%{text}",
            text=hover_text,
            marker={
                "size": 3,
                "opacity": 0.75,
                "color": density_values,
                "colorscale": "Inferno",
                "cmin": density_min,
                "cmax": density_max,
                "colorbar": {"title": "log10(bin count)"},
            },
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=[min_axis, max_axis],
            y=[min_axis, max_axis],
            mode="lines",
            line={"color": "black", "dash": "dash", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"MnBiF old vs new model parity (both < {threshold}, {len(overlap_ids):,} shared IDs)",
        xaxis={
            "title": "Old model formation energy per atom",
            "range": [min_axis, max_axis],
            "scaleanchor": "y",
            "scaleratio": 1,
            "constrain": "domain",
        },
        yaxis={
            "title": "New model formation energy per atom",
            "range": [min_axis, max_axis],
            "constrain": "domain",
        },
        width=900,
        height=900,
        template="simple_white",
        hovermode="closest",
        clickmode="event",
        margin={"l": 80, "r": 30, "t": 70, "b": 80},
        showlegend=False,
        annotations=[
            {
                "text": "Click a point to open its CIF. Zoom in before clicking when points are dense.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 1.08,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12},
            }
        ],
    )

    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True, div_id="parity_plot")
    structures_root = str(args["structures_root"])
    click_script = f"""
<script>
document.addEventListener("DOMContentLoaded", function() {{
  const plot = document.getElementById("parity_plot");
  if (!plot) return;
  const structuresRoot = {json.dumps(structures_root)};
  function structureIdToFileUrl(structureId) {{
    const prefix = structureId.split("_", 1)[0];
    const rawPath = `${{structuresRoot}}/${{prefix}}/${{structureId}}.cif`;
    return "file://" + encodeURI(rawPath);
  }}
  plot.on("plotly_click", function(event) {{
    if (!event || !event.points || event.points.length === 0) return;
    const structureId = event.points[0].customdata;
    if (!structureId) return;
    window.open(structureIdToFileUrl(structureId), "_blank");
  }});
}});
</script>
"""
    args["output_html"].write_text(html.replace("</body>", click_script + "\n</body>"))

    summary = {
        "old_csv": str(args["old_csv"]),
        "new_csv": str(args["new_csv"]),
        "structures_root": str(args["structures_root"]),
        "threshold": threshold,
        "output_html": str(args["output_html"]),
        "overlap_count": len(overlap_ids),
        "old_min": float(old_values.min()),
        "old_max": float(old_values.max()),
        "new_min": float(new_values.min()),
        "new_max": float(new_values.max()),
        "pearson_r": pearson_r,
        "mae_new_minus_old": mae,
        "rmse_new_minus_old": rmse,
        "density_bins": 320,
        "note": "Click opens the CIF file for the selected point. Zoom first in dense regions.",
    }
    args["summary_json"].write_text(json.dumps(summary, indent=2) + "\n")
    print(args["output_html"])
