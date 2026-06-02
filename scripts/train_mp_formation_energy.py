from pathlib import Path
import os
import warnings

for warning_message in (
    r".*not find enough neighbors to build graph.*",
    r".*Issues encountered while parsing CIF.*",
    r".*No Pauling electronegativity.*",
):
    warnings.filterwarnings(
        "ignore",
        message=warning_message,
        category=UserWarning,
    )
os.environ["PYTHONWARNINGS"] = ",".join([
    "ignore:.*not find enough neighbors to build graph.*:UserWarning",
    "ignore:.*Issues encountered while parsing CIF.*:UserWarning",
    "ignore:.*No Pauling electronegativity.*:UserWarning",
])

from cgcnn.training import train_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "data" / "mp-all-formation-energy"
RUN_DIR = REPO_ROOT / "runs" / "mp_all_formation_energy_epoch1_cpu"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
METRICS_PATH = RUN_DIR / "metrics_history.json"


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(RUN_DIR)
    best = train_model(
        str(DATASET),
        task="regression",
        epochs=1,
        batch_size=256,
        workers=4,
        cuda=False,
        checkpoint_dir=str(CHECKPOINT_DIR),
        metrics_history_path=str(METRICS_PATH),
        print_freq=10,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    print(f"best_checkpoint={best}")
    print(f"run_dir={RUN_DIR}")


if __name__ == "__main__":
    main()
