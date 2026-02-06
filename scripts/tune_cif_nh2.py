import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from cgcnn.training import train_model


@dataclass
class TrialConfig:
    name: str
    epochs: int
    batch_size: int
    lr: float
    optim_name: str
    atom_fea_len: int
    h_fea_len: int
    n_conv: int
    weight_decay: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_trial(data_dir: Path, out_dir: Path, cfg: TrialConfig, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(out_dir)
    try:
        set_seed(seed)
        best_path = train_model(
            root_dir=str(data_dir),
            task="regression",
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            optim_name=cfg.optim_name,
            atom_fea_len=cfg.atom_fea_len,
            h_fea_len=cfg.h_fea_len,
            n_conv=cfg.n_conv,
            n_h=2,
            cuda=False,
            workers=0,
            weight_decay=cfg.weight_decay,
            momentum=0.9,
            print_freq=20,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        if best_path and Path(best_path).exists():
            ckpt = torch.load(best_path, map_location="cpu")
            best_val_mae = float(ckpt["best_mae_error"])
        else:
            best_val_mae = float("inf")
        return {
            "trial": cfg.name,
            "best_val_mae": best_val_mae,
            "best_model": str(Path(best_path).resolve()) if best_path else None,
            **asdict(cfg),
        }
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    data_dir = Path.home() / "Downloads" / "cif"
    root_out = Path("runs") / "cif_nh2_tuning"
    root_out.mkdir(parents=True, exist_ok=True)

    trials = [
        TrialConfig("t01", 8, 32, 0.005, "Adam", 64, 128, 3, 0.0),
        TrialConfig("t02", 8, 64, 0.005, "Adam", 64, 128, 3, 0.0),
        TrialConfig("t03", 8, 32, 0.01, "Adam", 64, 128, 3, 0.0),
        TrialConfig("t04", 8, 64, 0.01, "Adam", 64, 128, 3, 0.0),
        TrialConfig("t05", 8, 32, 0.005, "Adam", 64, 256, 3, 1e-6),
        TrialConfig("t06", 8, 32, 0.005, "Adam", 96, 128, 4, 1e-6),
        TrialConfig("t07", 8, 64, 0.003, "Adam", 64, 256, 4, 1e-6),
        TrialConfig("t08", 8, 64, 0.01, "SGD", 64, 128, 3, 0.0),
    ]

    results: list[dict] = []
    for i, cfg in enumerate(trials):
        trial_out = root_out / cfg.name
        result = run_trial(data_dir=data_dir, out_dir=trial_out, cfg=cfg, seed=2026 + i)
        results.append(result)
        print(
            f"{cfg.name}: val_mae={result['best_val_mae']:.6f} "
            f"lr={cfg.lr} batch={cfg.batch_size} conv={cfg.n_conv} "
            f"atom_fea={cfg.atom_fea_len} h_fea={cfg.h_fea_len} opt={cfg.optim_name}"
        )

    ranked = sorted(results, key=lambda x: x["best_val_mae"])
    best = ranked[0]

    with (root_out / "tuning_results.json").open("w") as f:
        json.dump({"ranked": ranked, "best": best}, f, indent=2)

    final_cfg = TrialConfig(
        name="final_best",
        epochs=20,
        batch_size=int(best["batch_size"]),
        lr=float(best["lr"]),
        optim_name=str(best["optim_name"]),
        atom_fea_len=int(best["atom_fea_len"]),
        h_fea_len=int(best["h_fea_len"]),
        n_conv=int(best["n_conv"]),
        weight_decay=float(best["weight_decay"]),
    )
    final_out = root_out / "final_best"
    final_result = run_trial(
        data_dir=data_dir, out_dir=final_out, cfg=final_cfg, seed=3030
    )

    with (root_out / "final_result.json").open("w") as f:
        json.dump(final_result, f, indent=2)

    print("Best tuning trial:", best["trial"], "val_mae=", best["best_val_mae"])
    print("Final training model:", final_result["best_model"])
