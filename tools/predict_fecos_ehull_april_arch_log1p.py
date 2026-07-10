from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from cgcnn.data import CIFData
from cgcnn.inference import predict_model


DATASET_DIR = Path(
    "/work2/04996/tg842951/stampede3/run/FeCoS_paper/MPRelaxSet/cgcnn_ehull"
)
MODEL_SOURCE = Path(
    "/work2/04996/tg842951/stampede3/run/mp_ehull_scaled_log1p_s0p05_april_arch_20260630_0736/model_best.pth.tar"
)
MODEL_LINK = DATASET_DIR / "model_best_scaled_log1p_s0p05_april_arch_20260630_0736.pth.tar"
RAW_OUTPUT = DATASET_DIR / "test_results_scaled_log1p_s0p05_april_arch_20260630_0736.csv"
PREDICTION_OUTPUT = (
    DATASET_DIR / "ehull_predictions_scaled_log1p_s0p05_april_arch_20260630_0736.csv"
)
METADATA_PATH = (
    DATASET_DIR / "prediction_metadata_scaled_log1p_s0p05_april_arch_20260630_0736.json"
)
SCALE_EV_PER_ATOM = 0.05


def inverse_scaled_log1p(value: float) -> float:
    return SCALE_EV_PER_ATOM * math.expm1(max(value, 0.0))


def ensure_model_link() -> None:
    if MODEL_LINK.is_symlink() and MODEL_LINK.resolve(strict=False) == MODEL_SOURCE:
        return
    if MODEL_LINK.exists() or MODEL_LINK.is_symlink():
        raise FileExistsError(f"{MODEL_LINK} already exists and points elsewhere")
    MODEL_LINK.symlink_to(MODEL_SOURCE)


def write_predictions(raw_output: Path, prediction_output: Path) -> int:
    rows: list[dict[str, str]] = []
    with raw_output.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            predicted_z = float(row[2])
            rows.append(
                {
                    "material_id": row[0],
                    "dummy_target_z": f"{float(row[1]):.16g}",
                    "predicted_ehull_z": f"{predicted_z:.16g}",
                    "predicted_ehull_eV_per_atom": f"{inverse_scaled_log1p(predicted_z):.16g}",
                }
            )
    if not rows:
        raise RuntimeError(f"No prediction rows were written to {raw_output}")
    with prediction_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_metadata(row_count: int) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Apply the larger April-architecture scaled-log1p MP energy_above_hull CGCNN model to the Fe-Co-S MPRelaxSet CIFs.",
        "target_directory": str(DATASET_DIR),
        "inputs": {
            "cif_directory": str(DATASET_DIR),
            "id_prop_csv": str(DATASET_DIR / "id_prop.csv"),
            "atom_init": str(DATASET_DIR / "atom_init.json"),
            "model_source": str(MODEL_SOURCE),
            "model_link": str(MODEL_LINK),
            "target_transform": "scaled_log1p",
            "target_transform_scale_eV_per_atom": SCALE_EV_PER_ATOM,
        },
        "outputs": {
            "test_results_csv": str(RAW_OUTPUT),
            "ehull_predictions_csv": str(PREDICTION_OUTPUT),
            "prediction_rows": row_count,
            "interactive_node": Path("/proc/sys/kernel/hostname").read_text().strip(),
            "command": "uv run python tools/predict_fecos_ehull_april_arch_log1p.py",
            "batch_size": 256,
            "workers": 0,
            "device": "cpu",
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    ensure_model_link()
    dataset = CIFData(str(DATASET_DIR), shuffle=False)
    predict_model(
        dataset,
        modelpath=str(MODEL_LINK),
        task="regression",
        batch_size=256,
        workers=0,
        cuda=False,
        print_freq=1,
        output_csv=str(RAW_OUTPUT),
    )
    count = write_predictions(RAW_OUTPUT, PREDICTION_OUTPUT)
    write_metadata(count)
    print(f"Wrote {count} predictions to {PREDICTION_OUTPUT}")
