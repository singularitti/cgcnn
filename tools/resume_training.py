from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

warnings.filterwarnings(
    "ignore",
    message="Issues encountered while parsing CIF: .*",
)
warnings.filterwarnings(
    "ignore",
    message=".*not find enough neighbors to build graph.*",
)

from cgcnn.training import train_model
from analyze_parity import compute_metrics, load_results, make_plot


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    checkpoint = run_dir / "checkpoint.pth.tar"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    checkpoint_dir = run_dir / "checkpoints"
    metrics_history_path = run_dir / "training_history.json"

    os.chdir(run_dir)
    train_model(
        root_dir=str(run_dir),
        task="regression",
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        cuda=False,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        print_freq=20,
        resume=str(checkpoint),
        checkpoint_dir=str(checkpoint_dir),
        metrics_history_path=str(metrics_history_path),
    )

    epoch = None
    try:
        import torch

        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        epoch = int(checkpoint_data.get("epoch"))
    except Exception:
        epoch = None

    _, targets, predictions = load_results(run_dir / "test_results.csv")
    parity_metrics = compute_metrics(targets, predictions)
    if epoch is not None:
        parity_metrics["epoch"] = epoch
    with (run_dir / "parity_metrics.json").open("w") as handle:
        json.dump(parity_metrics, handle, indent=2)
    make_plot(targets, predictions, parity_metrics, run_dir / "parity_plot.png")

    parity_history_path = run_dir / "parity_metrics_history.json"
    parity_history = []
    if parity_history_path.exists():
        with parity_history_path.open() as handle:
            loaded_history = json.load(handle)
        if isinstance(loaded_history, list):
            parity_history = loaded_history
    if epoch is not None:
        parity_history = [
            item for item in parity_history if int(item.get("epoch", -1)) != epoch
        ]
    parity_history.append(parity_metrics)
    parity_history.sort(key=lambda item: int(item.get("epoch", -1)))
    with parity_history_path.open("w") as handle:
        json.dump(parity_history, handle, indent=2)
