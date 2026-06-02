from __future__ import annotations

import csv
import json
import os
import re
import sys
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from analyze_parity import compute_metrics, load_results, make_plot  # noqa: E402
from cgcnn.data import CIFData, collate_pool  # noqa: E402
from cgcnn.model import CrystalGraphConvNet  # noqa: E402
from cgcnn.utils import Normalizer, _validate  # noqa: E402


DATASET_DIR = REPO_ROOT / "data" / "mp-all-formation-energy"
RUN_DIR = Path(
    "/Users/qz/Downloads/training_runs/"
    "mp_all_formation_energy_full_from_subset_20260420"
)
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
EVALUATION_DIR = RUN_DIR / "evaluation"
TEST_DATASET_DIR = EVALUATION_DIR / "test_dataset"
PARITY_DIR = EVALUATION_DIR / "epoch_parity"
BATCH_SIZE = 256


def symlink_force(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return
        raise FileExistsError(f"{link} already exists and does not point to {target}")
    link.symlink_to(target)


def write_readme(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def build_test_dataset() -> int:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    PARITY_DIR.mkdir(parents=True, exist_ok=True)

    full_dataset = CIFData(str(DATASET_DIR), shuffle=True)
    total_size = len(full_dataset.id_prop_data)
    test_size = int(0.1 * total_size)
    test_rows = full_dataset.id_prop_data[-test_size:]

    with (TEST_DATASET_DIR / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(test_rows)

    symlink_force(DATASET_DIR / "atom_init.json", TEST_DATASET_DIR / "atom_init.json")
    for row in test_rows:
        cif_id = row[0]
        symlink_force(DATASET_DIR / f"{cif_id}.cif", TEST_DATASET_DIR / f"{cif_id}.cif")

    write_readme(
        EVALUATION_DIR / "README.md",
        """
        # Full formation-energy evaluation

        Created on 2026-04-20 for post-training evaluation of the full
        CGCNN formation-energy run. This folder reconstructs the held-out
        test split from the same shuffled `CIFData` ordering used during
        training and stores per-epoch parity diagnostics.
        """,
    )
    write_readme(
        TEST_DATASET_DIR / "README.md",
        f"""
        # Reconstructed test dataset

        Created on 2026-04-20. This flat CGCNN dataset contains the {test_size}
        held-out test rows from the full 154,879-row formation-energy dataset.
        `id_prop.csv` was regenerated from the shuffled training dataset, and
        CIF files plus `atom_init.json` are symlinks back to
        `{DATASET_DIR}`.
        """,
    )
    write_readme(
        PARITY_DIR / "README.md",
        """
        # Epoch parity diagnostics

        Created on 2026-04-20. This folder contains one held-out test-set
        prediction CSV, metrics JSON, and equal-aspect parity plot PNG for
        each archived epoch checkpoint from the full formation-energy CGCNN
        training run.
        """,
    )
    return test_size


def epoch_from_checkpoint(path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.pth\.tar$", path.name)
    if not match:
        raise ValueError(f"Cannot parse epoch from {path}")
    return int(match.group(1))


def load_model(checkpoint_path: Path, dataset: CIFData) -> tuple[CrystalGraphConvNet, Normalizer]:
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    args = checkpoint.get("args", {})
    task = args.get("task", "regression")
    if task != "regression":
        raise ValueError(f"Expected regression checkpoint, got task={task!r}")

    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    n_targets = args.get("n_targets", dataset.n_targets)
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.get("atom_fea_len", 64),
        n_conv=args.get("n_conv", 3),
        h_fea_len=args.get("h_fea_len", 128),
        n_h=args.get("n_h", 1),
        classification=False,
        n_targets=n_targets,
        n_classes=args.get("n_classes", 2),
    )
    model.load_state_dict(checkpoint["state_dict"])

    normalizer = Normalizer(torch.zeros(n_targets))
    normalizer.load_state_dict(checkpoint.get("normalizer", normalizer.state_dict()))
    return model, normalizer


def evaluate_epoch(checkpoint_path: Path, dataset: CIFData, loader: DataLoader) -> dict[str, object]:
    epoch = epoch_from_checkpoint(checkpoint_path)
    csv_path = PARITY_DIR / f"epoch_{epoch:03d}_test_results.csv"
    metrics_path = PARITY_DIR / f"epoch_{epoch:03d}_metrics.json"
    plot_path = PARITY_DIR / f"epoch_{epoch:03d}_parity.png"

    model, normalizer = load_model(checkpoint_path, dataset)
    criterion = nn.MSELoss()
    _validate(
        loader,
        model,
        criterion,
        normalizer,
        cuda=False,
        task="regression",
        test=True,
        print_freq=1000,
        output_csv=str(csv_path),
    )

    _, targets, predictions = load_results(csv_path)
    metrics = compute_metrics(targets, predictions)
    metrics_with_context = {
        "epoch": epoch,
        "checkpoint": str(checkpoint_path),
        **metrics,
        "test_results_csv": str(csv_path),
        "parity_plot": str(plot_path),
    }
    make_plot(targets, predictions, metrics_with_context, plot_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_with_context, handle, indent=2)
    return {
        **metrics_with_context,
        "metrics_json": str(metrics_path),
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    summary_csv = PARITY_DIR / "epoch_summary.csv"
    summary_json = PARITY_DIR / "epoch_summary.json"
    fields = [
        "epoch",
        "count",
        "mae",
        "rmse",
        "pearson_r",
        "pearson_r_squared",
        "r2_score",
        "spearman_rho",
        "bias",
        "checkpoint",
        "test_results_csv",
        "metrics_json",
        "parity_plot",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main() -> None:
    warnings.filterwarnings("ignore", message=".*No oxidation states specified.*")
    warnings.filterwarnings("ignore", message=".*CrystalNN.*")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    test_size = build_test_dataset()
    print(f"Reconstructed held-out test dataset with {test_size} rows.")

    checkpoints = sorted(CHECKPOINT_DIR.glob("epoch_*.pth.tar"), key=epoch_from_checkpoint)
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found in {CHECKPOINT_DIR}")
    print(f"Evaluating {len(checkpoints)} epoch checkpoints.")

    dataset = CIFData(str(TEST_DATASET_DIR), shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pool,
        pin_memory=False,
    )

    rows = []
    for checkpoint in checkpoints:
        epoch = epoch_from_checkpoint(checkpoint)
        print(f"Evaluating epoch {epoch:03d}: {checkpoint}")
        metrics = evaluate_epoch(checkpoint, dataset, loader)
        rows.append(metrics)
        print(
            "Epoch {epoch:03d}: MAE={mae:.6f}, RMSE={rmse:.6f}, "
            "R={r:.6f}, R2={r2:.6f}".format(
                epoch=epoch,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r=metrics["pearson_r"],
                r2=metrics["r2_score"],
            )
        )

    write_summary(rows)
    best = min(rows, key=lambda row: row["mae"])
    print(
        "Best test MAE epoch: {epoch:03d} "
        "MAE={mae:.6f} RMSE={rmse:.6f} R={r:.6f} R2={r2:.6f}".format(
            epoch=best["epoch"],
            mae=best["mae"],
            rmse=best["rmse"],
            r=best["pearson_r"],
            r2=best["r2_score"],
        )
    )
    print(PARITY_DIR)


if __name__ == "__main__":
    main()
