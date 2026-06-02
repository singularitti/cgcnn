from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cgcnn.data import CIFData
from train_magnetization_multiclass_top_bin import (
    CLASS_LARGE,
    CLASS_NAMES,
    NUM_CLASSES,
    load_classification_predictions,
    load_regression_predictions,
    run_inference,
    write_atom_init,
)

SOURCE_DIR = Path(
    "~/Library/CloudStorage/CloudMounter-StampedeWork/run/MnBiO/predict/magnetization"
).expanduser()
CLASSIFIER_CHECKPOINT = SOURCE_DIR / "classifier_model_best.pth.tar"
TOP_BIN_REGRESSOR_CHECKPOINT = SOURCE_DIR / "epoch_029.pth.tar"
RUNS_ROOT = REPO_ROOT / "tmp" / "prediction_runs"
DEFAULT_BATCH_SIZE = 256
DEFAULT_WORKERS = 0


def make_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"mnbio_magnetization_combined_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def symlink_force(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


def collect_cifs(source_dir: Path) -> list[Path]:
    cifs = sorted(source_dir.glob("*.cif"))
    if not cifs:
        raise RuntimeError(f"No CIF files found in {source_dir}")
    return cifs


def material_id_from_cif(path: Path) -> str:
    return path.stem


def write_dataset(dataset_dir: Path, cifs: list[Path], target_value: str) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=False)
    write_atom_init(dataset_dir / "atom_init.json")
    with (dataset_dir / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for cif_path in cifs:
            material_id = material_id_from_cif(cif_path)
            symlink_force(cif_path, dataset_dir / f"{material_id}.cif")
            writer.writerow([material_id, target_value])


def summarize_checkpoint(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        args = {}
    normalizer = checkpoint.get("normalizer", {})
    if hasattr(normalizer, "keys"):
        normalizer_summary = sorted(str(key) for key in normalizer.keys())
    else:
        normalizer_summary = str(type(normalizer).__name__)
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "best_validation_score": checkpoint.get("best_validation_score"),
        "args": args,
        "normalizer_keys": normalizer_summary,
    }


def copy_script_snapshot(run_dir: Path) -> Path:
    snapshot = run_dir / "inputs" / "scripts" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), snapshot)
    return snapshot


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(run_dir: Path, metadata: dict[str, object]) -> None:
    readme = run_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# MnBiO Magnetization Combined Prediction",
                "",
                f"Created: {metadata['created_at_utc']}",
                "",
                "## Purpose",
                "",
                "Predict magnetization per volume for the supplied MnBiO CIF structures using the saved combined workflow: a four-class magnetization classifier routes structures predicted as class 3 / large_positive / M > 1e-3 to a dedicated top-bin regressor.",
                "",
                "## Inputs",
                "",
                f"- Source folder: `{metadata['source_dir']}`",
                f"- Classifier checkpoint symlink: `{metadata['inputs']['classifier_checkpoint']}`",
                f"- Top-bin regressor checkpoint symlink: `{metadata['inputs']['top_bin_regressor_checkpoint']}`",
                f"- CIF symlink dataset for classification: `{metadata['inputs']['classifier_dataset']}`",
                f"- CIF symlink dataset for selected class-3 regression: `{metadata['inputs']['top_bin_regressor_dataset']}`",
                f"- Script snapshot: `{metadata['inputs']['script_snapshot']}`",
                f"- Batch size: `{metadata['parameters']['batch_size']}`",
                f"- Workers: `{metadata['parameters']['workers']}`",
                "",
                "All CIFs and checkpoints are symlinked from the source folder; they are not copied into this run folder.",
                "",
                "## Outputs",
                "",
                f"- Classifier raw predictions: `{metadata['outputs']['classifier_predictions']}`",
                f"- Selected candidate regressor raw predictions: `{metadata['outputs']['top_bin_regressor_predictions']}`",
                f"- Final merged predictions for all structures: `{metadata['outputs']['merged_predictions']}`",
                f"- Selected class-3 predictions only: `{metadata['outputs']['selected_predictions']}`",
                f"- Run metadata: `{metadata['outputs']['metadata']}`",
                "",
                "For structures not routed to class 3, the continuous regressor prediction is intentionally blank because the saved workflow only applies the top-bin regressor to class-3 candidates.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    if not CLASSIFIER_CHECKPOINT.is_file():
        raise FileNotFoundError(CLASSIFIER_CHECKPOINT)
    if not TOP_BIN_REGRESSOR_CHECKPOINT.is_file():
        raise FileNotFoundError(TOP_BIN_REGRESSOR_CHECKPOINT)

    source_cifs = collect_cifs(SOURCE_DIR)
    run_dir = make_run_dir()
    inputs_dir = run_dir / "inputs"
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    linked_classifier = inputs_dir / "checkpoints" / CLASSIFIER_CHECKPOINT.name
    linked_regressor = inputs_dir / "checkpoints" / TOP_BIN_REGRESSOR_CHECKPOINT.name
    symlink_force(CLASSIFIER_CHECKPOINT, linked_classifier)
    symlink_force(TOP_BIN_REGRESSOR_CHECKPOINT, linked_regressor)

    classifier_dataset = run_dir / "classifier_dataset_all"
    write_dataset(classifier_dataset, source_cifs, target_value="0")

    classifier_predictions = outputs_dir / "classifier_predictions.csv"
    run_inference(
        dataset_dir=classifier_dataset,
        model_path=linked_classifier,
        task="classification",
        output_csv=classifier_predictions,
        batch_size=DEFAULT_BATCH_SIZE,
        workers=DEFAULT_WORKERS,
    )
    classifier_rows = load_classification_predictions(classifier_predictions)
    selected_ids = sorted(
        material_id
        for material_id, row in classifier_rows.items()
        if int(row["predicted_class"]) == CLASS_LARGE
    )
    selected_cifs = [
        cif_path for cif_path in source_cifs if material_id_from_cif(cif_path) in selected_ids
    ]

    top_bin_dataset = run_dir / "top_bin_regressor_dataset_selected"
    regressor_predictions = outputs_dir / "top_bin_regressor_predictions.csv"
    regressor_rows: dict[str, tuple[float, float]] = {}
    if selected_cifs:
        write_dataset(top_bin_dataset, selected_cifs, target_value="0.0")
        run_inference(
            dataset_dir=top_bin_dataset,
            model_path=linked_regressor,
            task="regression",
            output_csv=regressor_predictions,
            batch_size=DEFAULT_BATCH_SIZE,
            workers=DEFAULT_WORKERS,
        )
        regressor_rows = load_regression_predictions(regressor_predictions)
    else:
        top_bin_dataset.mkdir(parents=True, exist_ok=False)
        write_atom_init(top_bin_dataset / "atom_init.json")
        (top_bin_dataset / "id_prop.csv").write_text("")
        regressor_predictions.write_text("")

    class_probability_fields = [
        f"class_{label}_{CLASS_NAMES[label]}_probability" for label in range(NUM_CLASSES)
    ]
    merged_fields = [
        "material_id",
        "source_cif",
        "predicted_class",
        "predicted_class_name",
        "selected_for_top_bin_regressor",
        "predicted_magnetization_per_volume",
        "dummy_regression_target",
    ] + class_probability_fields
    merged_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for cif_path in source_cifs:
        material_id = material_id_from_cif(cif_path)
        classifier_row = classifier_rows[material_id]
        predicted_class = int(classifier_row["predicted_class"])
        probabilities = list(classifier_row["probabilities"])
        regression = regressor_rows.get(material_id)
        prediction = "" if regression is None else f"{regression[1]:.16g}"
        dummy_target = "" if regression is None else f"{regression[0]:.16g}"
        row = {
            "material_id": material_id,
            "source_cif": str(cif_path),
            "predicted_class": predicted_class,
            "predicted_class_name": CLASS_NAMES[predicted_class],
            "selected_for_top_bin_regressor": predicted_class == CLASS_LARGE,
            "predicted_magnetization_per_volume": prediction,
            "dummy_regression_target": dummy_target,
        }
        for label, field in enumerate(class_probability_fields):
            row[field] = f"{probabilities[label]:.16g}"
        merged_rows.append(row)
        if predicted_class == CLASS_LARGE:
            selected_rows.append(row)

    merged_predictions = outputs_dir / "combined_predictions.csv"
    selected_predictions = outputs_dir / "selected_top_bin_predictions.csv"
    write_rows(merged_predictions, merged_rows, merged_fields)
    write_rows(selected_predictions, selected_rows, merged_fields)

    script_snapshot = copy_script_snapshot(run_dir)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(SOURCE_DIR),
        "class_definitions": {
            "class_0": "M == 0",
            "class_1": "0 < M <= 1e-6",
            "class_2": "1e-6 < M <= 1e-3",
            "class_3": "M > 1e-3",
        },
        "parameters": {
            "batch_size": DEFAULT_BATCH_SIZE,
            "workers": DEFAULT_WORKERS,
            "cuda": False,
        },
        "counts": {
            "input_cifs": len(source_cifs),
            "selected_for_top_bin_regressor": len(selected_rows),
            "predicted_class_counts": {
                CLASS_NAMES[label]: sum(
                    1 for row in merged_rows if int(row["predicted_class"]) == label
                )
                for label in range(NUM_CLASSES)
            },
        },
        "inputs": {
            "classifier_checkpoint": str(linked_classifier),
            "top_bin_regressor_checkpoint": str(linked_regressor),
            "classifier_dataset": str(classifier_dataset),
            "top_bin_regressor_dataset": str(top_bin_dataset),
            "script_snapshot": str(script_snapshot),
            "classifier_checkpoint_summary": summarize_checkpoint(linked_classifier),
            "top_bin_regressor_checkpoint_summary": summarize_checkpoint(linked_regressor),
        },
        "outputs": {
            "classifier_predictions": str(classifier_predictions),
            "top_bin_regressor_predictions": str(regressor_predictions),
            "merged_predictions": str(merged_predictions),
            "selected_predictions": str(selected_predictions),
            "metadata": str(run_dir / "run_metadata.json"),
        },
    }
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    write_readme(run_dir, metadata)
    print(run_dir)
