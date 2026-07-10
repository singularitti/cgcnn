from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from cgcnn.data import CIFData
from cgcnn.inference import predict_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(
    "/work2/04996/tg842951/stampede3/run/FeCoS_paper/MPRelaxSet/cgcnn_ehull"
)
MODEL_PATH = DATASET_DIR / "model_best.pth.tar"
RAW_OUTPUT = DATASET_DIR / "test_results.csv"
PHYSICAL_OUTPUT = DATASET_DIR / "ehull_predictions.csv"
METADATA_PATH = DATASET_DIR / "prediction_metadata.json"
SCALE_EV_PER_ATOM = 0.05


def inverse_scaled_log1p(value: float) -> float:
    return SCALE_EV_PER_ATOM * math.expm1(max(value, 0.0))


def write_physical_predictions(raw_output: Path, physical_output: Path) -> int:
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
    with physical_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def update_metadata(row_count: int) -> None:
    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    metadata.setdefault("outputs", {})
    metadata["prediction_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["outputs"].update(
        {
            "test_results_csv": str(RAW_OUTPUT),
            "ehull_predictions_csv": str(PHYSICAL_OUTPUT),
            "prediction_rows": row_count,
            "interactive_node": Path("/proc/sys/kernel/hostname").read_text().strip(),
            "command": "uv run python tools/predict_fecos_ehull.py",
            "batch_size": 256,
            "workers": 0,
            "device": "cpu",
        }
    )
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
    count = write_physical_predictions(RAW_OUTPUT, PHYSICAL_OUTPUT)
    update_metadata(count)
    print(f"Wrote {count} predictions to {PHYSICAL_OUTPUT}")
