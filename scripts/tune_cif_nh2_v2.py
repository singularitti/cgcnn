import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from cgcnn.training import train_model

DEFAULT_SEARCH_SPACE = {
    "optim_name": ["SGD", "Adam"],
    "lr": [0.03, 0.01, 0.005, 0.003, 0.001],
    "batch_size": [32, 64, 128],
    "n_conv": [2, 3, 4, 5],
    "atom_fea_len": [64, 96, 128],
    "h_fea_len": [64, 128, 256, 384],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "momentum_if_sgd": [0.8, 0.9, 0.95],
    "n_h": [2],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_test_results(test_results_csv):
    diffs = []
    if not test_results_csv.exists():
        return {
            "test_mae": None,
            "outlier_abs_gt_10_count": None,
            "max_abs_diff": None,
            "p95_abs_diff": None,
        }
    with test_results_csv.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                target = float(row[1].strip())
                pred = float(row[2].strip())
            except ValueError:
                continue
            diffs.append(abs(target - pred))
    if not diffs:
        return {
            "test_mae": None,
            "outlier_abs_gt_10_count": None,
            "max_abs_diff": None,
            "p95_abs_diff": None,
        }
    arr = np.array(diffs, dtype=float)
    return {
        "test_mae": float(arr.mean()),
        "outlier_abs_gt_10_count": int((arr > 10.0).sum()),
        "max_abs_diff": float(arr.max()),
        "p95_abs_diff": float(np.percentile(arr, 95)),
    }


def run_single_trial(data_dir, out_dir, cfg, seed, epochs):
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(out_dir)
    try:
        set_seed(seed)
        best_path = train_model(
            root_dir=str(data_dir),
            task="regression",
            epochs=epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            optim_name=cfg.optim_name,
            atom_fea_len=cfg.atom_fea_len,
            h_fea_len=cfg.h_fea_len,
            n_conv=cfg.n_conv,
            n_h=cfg.n_h,
            cuda=False,
            workers=0,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
            print_freq=20,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        if not best_path or not Path(best_path).exists():
            raise RuntimeError("No best checkpoint produced")
        ckpt = torch.load(best_path, map_location="cpu")
        best_val_mae = float(ckpt["best_mae_error"])
        test_metrics = parse_test_results(Path("test_results.csv"))
        result = {
            "status": "ok",
            "seed": seed,
            "epochs": epochs,
            "best_val_mae": best_val_mae,
            "best_model": str(Path(best_path).resolve()),
            **test_metrics,
            **cfg.__dict__,
        }
    finally:
        os.chdir(cwd)
    return result


def write_csv_with_header_and_diff(src):
    rows = []
    with src.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                cif_id = row[0].strip()
                target = float(row[1].strip())
                pred = float(row[2].strip())
            except ValueError:
                continue
            rows.append((cif_id, target, pred, target - pred))

    with src.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "target", "prediction", "diff"])
        for r in rows:
            writer.writerow([r[0], f"{r[1]:.10f}", f"{r[2]:.10f}", f"{r[3]:.10f}"])


def make_cfg(**kwargs):
    cfg = type("TrialConfig", (), {})()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def build_candidates(sample_size, seed):
    rng = random.Random(seed)
    all_cfgs = []

    for opt in DEFAULT_SEARCH_SPACE["optim_name"]:
        moms = (
            DEFAULT_SEARCH_SPACE["momentum_if_sgd"] if opt == "SGD" else [0.9]
        )
        for lr in DEFAULT_SEARCH_SPACE["lr"]:
            for bsz in DEFAULT_SEARCH_SPACE["batch_size"]:
                for nc in DEFAULT_SEARCH_SPACE["n_conv"]:
                    for al in DEFAULT_SEARCH_SPACE["atom_fea_len"]:
                        for hl in DEFAULT_SEARCH_SPACE["h_fea_len"]:
                            for wd in DEFAULT_SEARCH_SPACE["weight_decay"]:
                                for mom in moms:
                                    all_cfgs.append(
                                        make_cfg(
                                            optim_name=opt,
                                            lr=lr,
                                            batch_size=bsz,
                                            n_conv=nc,
                                            atom_fea_len=al,
                                            h_fea_len=hl,
                                            weight_decay=wd,
                                            momentum=mom,
                                            n_h=2,
                                        )
                                    )

    if sample_size >= len(all_cfgs):
        return all_cfgs
    return rng.sample(all_cfgs, sample_size)


def aggregate_config_results(config_dir):
    run_results = []
    for p in sorted(config_dir.glob("seed_*/result.json")):
        with p.open() as f:
            run_results.append(json.load(f))
    ok_runs = [r for r in run_results if r.get("status") == "ok"]
    if not ok_runs:
        return None

    maes = np.array([r["best_val_mae"] for r in ok_runs], dtype=float)
    outlier10 = np.array([r["outlier_abs_gt_10_count"] for r in ok_runs], dtype=float)
    maxabs = np.array([r["max_abs_diff"] for r in ok_runs], dtype=float)
    p95 = np.array([r["p95_abs_diff"] for r in ok_runs], dtype=float)
    test_mae = np.array([r["test_mae"] for r in ok_runs], dtype=float)

    base = dict(ok_runs[0])
    return {
        "config_id": config_dir.name,
        "num_success": len(ok_runs),
        "num_total": len(run_results),
        "mean_val_mae": float(maes.mean()),
        "std_val_mae": float(maes.std()),
        "mean_test_mae": float(test_mae.mean()),
        "mean_outlier_abs_gt_10_count": float(outlier10.mean()),
        "mean_max_abs_diff": float(maxabs.mean()),
        "mean_p95_abs_diff": float(p95.mean()),
        "best_model_example": base["best_model"],
            "config": {
                "optim_name": base["optim_name"],
                "lr": base["lr"],
                "batch_size": base["batch_size"],
            "n_conv": base["n_conv"],
            "atom_fea_len": base["atom_fea_len"],
            "h_fea_len": base["h_fea_len"],
            "weight_decay": base["weight_decay"],
            "momentum": base["momentum"],
            "n_h": base["n_h"],
        },
    }


def run_trial_with_timeout(script_path, data_dir, run_dir, seed, epochs, cfg, timeout_sec):
    cmd = [
        sys.executable,
        str(script_path),
        "run",
        str(data_dir),
        str(run_dir),
        str(seed),
        str(epochs),
        json.dumps(cfg.__dict__),
    ]
    timed_out = False
    try:
        subprocess.run(cmd, check=False, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True

    result_path = run_dir / "result.json"
    if timed_out:
        with result_path.open("w") as f:
            json.dump(
                {"status": "timeout", "seed": seed, "timeout_sec": timeout_sec},
                f,
                indent=2,
            )
        return
    if not result_path.exists():
        with result_path.open("w") as f:
            json.dump({"status": "failed", "seed": seed}, f, indent=2)


def orchestrate():
    data_dir = Path.home() / "Downloads" / "cif"
    root_out = Path("runs") / "cif_nh2_tuning_v2"
    root_out.mkdir(parents=True, exist_ok=True)

    search_epochs = 12
    final_epochs = 30
    seeds = [2026, 2027, 2028]
    timeout_sec = 15 * 60

    candidates = build_candidates(sample_size=12, seed=42)
    with (root_out / "search_space_and_sample.json").open("w") as f:
        json.dump(
            {
                "search_space": DEFAULT_SEARCH_SPACE,
                "sample_size": len(candidates),
                "timeout_sec_per_run": timeout_sec,
                "search_epochs": search_epochs,
                "final_epochs": final_epochs,
                "seeds": seeds,
            },
            f,
            indent=2,
        )

    script_path = Path(__file__).resolve()

    for idx, cfg in enumerate(candidates, start=1):
        cfg_dir = root_out / f"cfg_{idx:02d}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with (cfg_dir / "config.json").open("w") as f:
            json.dump(cfg.__dict__, f, indent=2)

        for seed in seeds:
            run_dir = cfg_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_trial_with_timeout(
                script_path=script_path,
                data_dir=data_dir,
                run_dir=run_dir,
                seed=seed,
                epochs=search_epochs,
                cfg=cfg,
                timeout_sec=timeout_sec,
            )

    aggregated = []
    for cfg_dir in sorted(root_out.glob("cfg_*")):
        agg = aggregate_config_results(cfg_dir)
        if agg is not None:
            aggregated.append(agg)

    # Primary: mean validation MAE; Secondary: fewer outliers and lower worst-case error.
    ranked = sorted(
        aggregated,
        key=lambda x: (
            x["mean_val_mae"],
            x["mean_outlier_abs_gt_10_count"],
            x["mean_max_abs_diff"],
            x["std_val_mae"],
        ),
    )
    with (root_out / "ranked_search_results.json").open("w") as f:
        json.dump(ranked, f, indent=2)

    top2 = ranked[:2]
    final_candidates = []
    for i, entry in enumerate(top2, start=1):
        cfg = make_cfg(**entry["config"])
        final_dir = root_out / f"final_cfg_{i}"
        final_seed = 4040 + i
        final_dir.mkdir(parents=True, exist_ok=True)
        run_trial_with_timeout(
            script_path=script_path,
            data_dir=data_dir,
            run_dir=final_dir,
            seed=final_seed,
            epochs=final_epochs,
            cfg=cfg,
            timeout_sec=timeout_sec,
        )
        result_path = final_dir / "result.json"
        if result_path.exists():
            with result_path.open() as f:
                final_result = json.load(f)
            final_result["config_rank_source"] = entry
            final_candidates.append(final_result)

    ok_final = [x for x in final_candidates if x.get("status") == "ok"]
    if not ok_final:
        raise RuntimeError("No final candidate completed successfully within timeout")

    best_final = sorted(
        ok_final,
        key=lambda x: (
            x["best_val_mae"],
            x["outlier_abs_gt_10_count"],
            x["max_abs_diff"],
        ),
    )[0]

    chosen_model = Path(best_final["best_model"])
    chosen_dir = chosen_model.parent
    final_best_dir = root_out / "final_best"
    if final_best_dir.exists():
        shutil.rmtree(final_best_dir)
    shutil.copytree(chosen_dir, final_best_dir)

    final_csv = final_best_dir / "test_results.csv"
    if final_csv.exists():
        write_csv_with_header_and_diff(final_csv)

    with (root_out / "final_selection.json").open("w") as f:
        json.dump(
            {
                "selection_strategy": {
                    "primary": "best_val_mae",
                    "secondary": [
                        "outlier_abs_gt_10_count",
                        "max_abs_diff",
                    ],
                    "timeout_sec_per_run": timeout_sec,
                },
                "final_candidates": final_candidates,
                "best_final": best_final,
                "final_best_dir": str(final_best_dir.resolve()),
            },
            f,
            indent=2,
        )

    print("Final best directory:", final_best_dir.resolve())
    print("Best final val MAE:", best_final["best_val_mae"])
    print("Best final outlier count |diff|>10:", best_final["outlier_abs_gt_10_count"])


def run_mode(args):
    if len(args) != 6:
        print("Usage: run <data_dir> <out_dir> <seed> <epochs> <config_json>")
        return 2
    data_dir = Path(args[1]).expanduser().resolve()
    out_dir = Path(args[2]).expanduser().resolve()
    seed = int(args[3])
    epochs = int(args[4])
    cfg_dict = json.loads(args[5])
    cfg = make_cfg(**cfg_dict)

    result_path = out_dir / "result.json"
    try:
        result = run_single_trial(
            data_dir=data_dir,
            out_dir=out_dir,
            cfg=cfg,
            seed=seed,
            epochs=epochs,
        )
    except Exception as exc:
        result = {"status": "failed", "seed": seed, "error": str(exc), **cfg_dict}
        out_dir.mkdir(parents=True, exist_ok=True)
        with result_path.open("w") as f:
            json.dump(result, f, indent=2)
        return 1

    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        raise SystemExit(run_mode(sys.argv[1:]))
    orchestrate()
