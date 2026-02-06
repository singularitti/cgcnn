import csv
import os
import random
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from cgcnn.data import CIFData
from cgcnn.training import train_model

DEFAULT_SEARCH_SPACE = {
    "optim_name": ["Adam"],
    "lr": [0.03, 0.01, 0.005, 0.003, 0.001],
    "batch_size": [32, 64, 128],
    "n_conv": [2, 3, 4, 5],
    "atom_fea_len": [64, 96, 128],
    "h_fea_len": [64, 128, 256, 384],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "n_h": [2, 3],
}


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()


def write_tune_status(root_out, state, note=""):
    root_out.mkdir(parents=True, exist_ok=True)
    status_path = root_out / "STATUS.txt"
    with status_path.open("w") as f:
        f.write(f"state={state}\n")
        f.write(f"time_utc={now_utc_iso()}\n")
        if note:
            f.write(f"note={note}\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cfg_to_string(cfg):
    return ";".join([f"{k}={cfg[k]}" for k in sorted(cfg.keys())])


def parse_cfg_string(cfg_str):
    cfg = {}
    for part in cfg_str.split(";"):
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k in {"batch_size", "n_conv", "atom_fea_len", "h_fea_len", "n_h"}:
            cfg[k] = int(v)
        elif k in {"lr", "weight_decay"}:
            cfg[k] = float(v)
        else:
            cfg[k] = v
    return cfg


def write_rows_csv(path, headers, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_candidates(sample_size, seed):
    rng = random.Random(seed)
    all_cfgs = []

    for lr in DEFAULT_SEARCH_SPACE["lr"]:
        for bsz in DEFAULT_SEARCH_SPACE["batch_size"]:
            for nc in DEFAULT_SEARCH_SPACE["n_conv"]:
                for al in DEFAULT_SEARCH_SPACE["atom_fea_len"]:
                    for hl in DEFAULT_SEARCH_SPACE["h_fea_len"]:
                        for wd in DEFAULT_SEARCH_SPACE["weight_decay"]:
                            for nh in DEFAULT_SEARCH_SPACE["n_h"]:
                                all_cfgs.append({
                                    "optim_name": "Adam",
                                    "lr": lr,
                                    "batch_size": bsz,
                                    "n_conv": nc,
                                    "atom_fea_len": al,
                                    "h_fea_len": hl,
                                    "weight_decay": wd,
                                    "n_h": nh,
                                })

    if sample_size >= len(all_cfgs):
        return all_cfgs
    return rng.sample(all_cfgs, sample_size)


def build_split_lists(data_dir, seed, train_ratio, val_ratio, test_ratio):
    set_seed(seed)
    dataset = CIFData(str(data_dir))
    ids_and_targets = [
        (row[0], row[1] if len(row) > 1 else "") for row in dataset.id_prop_data
    ]
    total_size = len(ids_and_targets)

    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)

    train = ids_and_targets[:train_size]
    valid = ids_and_targets[-(valid_size + test_size) : -test_size]
    test = ids_and_targets[-test_size:]
    return train, valid, test


def write_split_csvs(out_dir, train_rows, valid_rows, test_rows):
    def to_rows(name, pairs):
        return [
            {
                "split": name,
                "id": p[0],
                "target": p[1],
            }
            for p in pairs
        ]

    headers = ["split", "id", "target"]
    write_rows_csv(
        out_dir / "train_examples.csv", headers, to_rows("train", train_rows)
    )
    write_rows_csv(
        out_dir / "validation_examples.csv", headers, to_rows("validation", valid_rows)
    )
    write_rows_csv(out_dir / "test_examples.csv", headers, to_rows("test", test_rows))


def parse_and_rewrite_test_results(test_results_csv):
    rows = []
    abs_diffs = []
    if not test_results_csv.exists():
        return rows, {
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
                cif_id = row[0].strip().replace("\r", "")
                target = float(row[1].strip().replace("\r", ""))
                pred = float(row[2].strip().replace("\r", ""))
            except ValueError:
                continue
            diff = target - pred
            mae_each = abs(diff)
            abs_diffs.append(mae_each)
            rows.append({
                "id": cif_id,
                "target": f"{target:.10f}",
                "prediction": f"{pred:.10f}",
                "diff": f"{diff:.10f}",
                "mae": f"{mae_each:.10f}",
            })

    if rows:
        write_rows_csv(
            test_results_csv,
            ["id", "target", "prediction", "diff", "mae"],
            rows,
        )

    if not abs_diffs:
        return rows, {
            "test_mae": None,
            "outlier_abs_gt_10_count": None,
            "max_abs_diff": None,
            "p95_abs_diff": None,
        }

    arr = np.array(abs_diffs, dtype=float)
    return rows, {
        "test_mae": float(arr.mean()),
        "outlier_abs_gt_10_count": int((arr > 10.0).sum()),
        "max_abs_diff": float(arr.max()),
        "p95_abs_diff": float(np.percentile(arr, 95)),
    }


def write_run_result_csv(path, result):
    headers = [
        "status",
        "seed",
        "epochs",
        "best_val_mae",
        "best_model",
        "test_mae",
        "outlier_abs_gt_10_count",
        "max_abs_diff",
        "p95_abs_diff",
        "optim_name",
        "lr",
        "batch_size",
        "n_conv",
        "atom_fea_len",
        "h_fea_len",
        "weight_decay",
        "n_h",
        "error",
    ]
    for h in headers:
        if h not in result:
            result[h] = ""
    write_rows_csv(path, headers, [result])


def read_single_row_csv(path):
    if not path.exists():
        return None
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None


def run_single_trial(data_dir, out_dir, cfg, seed, epochs):
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(out_dir)
    try:
        train_rows, valid_rows, test_rows = build_split_lists(
            data_dir=data_dir,
            seed=seed,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        write_split_csvs(out_dir, train_rows, valid_rows, test_rows)

        set_seed(seed)
        best_path = train_model(
            root_dir=str(data_dir),
            task="regression",
            epochs=epochs,
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            optim_name="Adam",
            atom_fea_len=cfg["atom_fea_len"],
            h_fea_len=cfg["h_fea_len"],
            n_conv=cfg["n_conv"],
            n_h=cfg["n_h"],
            cuda=False,
            workers=0,
            weight_decay=cfg["weight_decay"],
            momentum=0.9,
            print_freq=20,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        if not best_path or not Path(best_path).exists():
            raise RuntimeError("No best checkpoint produced")

        ckpt = torch.load(best_path, map_location="cpu")
        best_val_mae = float(ckpt["best_mae_error"])
        _, test_metrics = parse_and_rewrite_test_results(Path("test_results.csv"))

        return {
            "status": "ok",
            "seed": seed,
            "epochs": epochs,
            "best_val_mae": best_val_mae,
            "best_model": str(Path(best_path).resolve()),
            **test_metrics,
            **cfg,
            "error": "",
        }
    finally:
        os.chdir(cwd)


def run_trial_with_timeout(
    script_path, data_dir, run_dir, seed, epochs, cfg, timeout_sec
):
    cmd = [
        sys.executable,
        str(script_path),
        "run",
        str(seed),
        str(epochs),
        cfg_to_string(cfg),
        str(data_dir),
        str(run_dir),
    ]
    timed_out = False
    try:
        subprocess.run(cmd, check=False, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True

    result_path = run_dir / "run_result.csv"
    if timed_out:
        write_run_result_csv(
            result_path,
            {
                "status": "timeout",
                "seed": seed,
                "epochs": epochs,
                "optim_name": cfg["optim_name"],
                "lr": cfg["lr"],
                "batch_size": cfg["batch_size"],
                "n_conv": cfg["n_conv"],
                "atom_fea_len": cfg["atom_fea_len"],
                "h_fea_len": cfg["h_fea_len"],
                "weight_decay": cfg["weight_decay"],
                "n_h": cfg["n_h"],
                "error": f"timeout_{timeout_sec}s",
            },
        )
        return
    if not result_path.exists():
        write_run_result_csv(
            result_path,
            {
                "status": "failed",
                "seed": seed,
                "epochs": epochs,
                "optim_name": cfg["optim_name"],
                "lr": cfg["lr"],
                "batch_size": cfg["batch_size"],
                "n_conv": cfg["n_conv"],
                "atom_fea_len": cfg["atom_fea_len"],
                "h_fea_len": cfg["h_fea_len"],
                "weight_decay": cfg["weight_decay"],
                "n_h": cfg["n_h"],
                "error": "missing_run_result_csv",
            },
        )


def aggregate_config_results(config_dir):
    run_results = []
    for p in sorted(config_dir.glob("seed_*/run_result.csv")):
        row = read_single_row_csv(p)
        if row:
            run_results.append(row)

    ok_runs = [r for r in run_results if r.get("status") == "ok"]
    if not ok_runs:
        return None

    maes = np.array([float(r["best_val_mae"]) for r in ok_runs], dtype=float)
    outlier10 = np.array(
        [float(r["outlier_abs_gt_10_count"]) for r in ok_runs], dtype=float
    )
    maxabs = np.array([float(r["max_abs_diff"]) for r in ok_runs], dtype=float)
    p95 = np.array([float(r["p95_abs_diff"]) for r in ok_runs], dtype=float)
    test_mae = np.array([float(r["test_mae"]) for r in ok_runs], dtype=float)

    base = ok_runs[0]
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
        "optim_name": base["optim_name"],
        "lr": float(base["lr"]),
        "batch_size": int(float(base["batch_size"])),
        "n_conv": int(float(base["n_conv"])),
        "atom_fea_len": int(float(base["atom_fea_len"])),
        "h_fea_len": int(float(base["h_fea_len"])),
        "weight_decay": float(base["weight_decay"]),
        "n_h": int(float(base["n_h"])),
    }


def write_search_space_csv(path):
    rows = []
    for k, values in DEFAULT_SEARCH_SPACE.items():
        for v in values:
            rows.append({"parameter": k, "value": v})
    write_rows_csv(path, ["parameter", "value"], rows)


def write_sampled_hparams_csv(path, candidates):
    headers = [
        "config_id",
        "optim_name",
        "lr",
        "batch_size",
        "n_conv",
        "atom_fea_len",
        "h_fea_len",
        "weight_decay",
        "n_h",
    ]
    rows = []
    for i, cfg in enumerate(candidates, start=1):
        rows.append({
            "config_id": f"cfg_{i:02d}",
            "optim_name": cfg["optim_name"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "n_conv": cfg["n_conv"],
            "atom_fea_len": cfg["atom_fea_len"],
            "h_fea_len": cfg["h_fea_len"],
            "weight_decay": cfg["weight_decay"],
            "n_h": cfg["n_h"],
        })
    write_rows_csv(path, headers, rows)


def orchestrate(data_dir, root_out):
    root_out.mkdir(parents=True, exist_ok=True)

    search_epochs = 12
    final_epochs = 30
    seeds = [2026, 2027, 2028]
    timeout_sec = 15 * 60

    write_search_space_csv(root_out / "search_space.csv")
    candidates = build_candidates(sample_size=12, seed=42)
    write_sampled_hparams_csv(root_out / "sampled_hyperparameters.csv", candidates)

    script_path = Path(__file__).resolve()
    for idx, cfg in enumerate(candidates, start=1):
        cfg_dir = root_out / f"cfg_{idx:02d}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        write_rows_csv(
            cfg_dir / "hyperparameters.csv",
            [
                "optim_name",
                "lr",
                "batch_size",
                "n_conv",
                "atom_fea_len",
                "h_fea_len",
                "weight_decay",
                "n_h",
            ],
            [cfg],
        )
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

    ranked = sorted(
        aggregated,
        key=lambda x: (
            x["mean_val_mae"],
            x["mean_outlier_abs_gt_10_count"],
            x["mean_max_abs_diff"],
            x["std_val_mae"],
        ),
    )
    write_rows_csv(
        root_out / "ranked_search_results.csv",
        [
            "config_id",
            "num_success",
            "num_total",
            "mean_val_mae",
            "std_val_mae",
            "mean_test_mae",
            "mean_outlier_abs_gt_10_count",
            "mean_max_abs_diff",
            "mean_p95_abs_diff",
            "optim_name",
            "lr",
            "batch_size",
            "n_conv",
            "atom_fea_len",
            "h_fea_len",
            "weight_decay",
            "n_h",
        ],
        ranked,
    )

    top2 = ranked[:2]
    final_candidates = []
    for i, entry in enumerate(top2, start=1):
        cfg = {
            "optim_name": entry["optim_name"],
            "lr": entry["lr"],
            "batch_size": entry["batch_size"],
            "n_conv": entry["n_conv"],
            "atom_fea_len": entry["atom_fea_len"],
            "h_fea_len": entry["h_fea_len"],
            "weight_decay": entry["weight_decay"],
            "n_h": entry["n_h"],
        }
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
        row = read_single_row_csv(final_dir / "run_result.csv")
        if row:
            row["config_rank_source"] = entry["config_id"]
            final_candidates.append(row)

    ok_final = [x for x in final_candidates if x.get("status") == "ok"]
    if not ok_final:
        raise RuntimeError("No final candidate completed successfully within timeout")

    best_final = sorted(
        ok_final,
        key=lambda x: (
            float(x["best_val_mae"]),
            float(x["outlier_abs_gt_10_count"]),
            float(x["max_abs_diff"]),
        ),
    )[0]

    chosen_model = Path(best_final["best_model"])
    chosen_dir = chosen_model.parent
    final_best_dir = root_out / "final_best"
    if final_best_dir.exists():
        shutil.rmtree(final_best_dir)
    shutil.copytree(chosen_dir, final_best_dir)

    write_rows_csv(
        root_out / "final_candidates.csv",
        [
            "status",
            "seed",
            "epochs",
            "best_val_mae",
            "best_model",
            "test_mae",
            "outlier_abs_gt_10_count",
            "max_abs_diff",
            "p95_abs_diff",
            "optim_name",
            "lr",
            "batch_size",
            "n_conv",
            "atom_fea_len",
            "h_fea_len",
            "weight_decay",
            "n_h",
            "config_rank_source",
            "error",
        ],
        final_candidates,
    )

    write_rows_csv(
        root_out / "final_selection.csv",
        [
            "selection_primary",
            "selection_secondary_1",
            "selection_secondary_2",
            "timeout_sec_per_run",
            "status",
            "seed",
            "epochs",
            "best_val_mae",
            "best_model",
            "test_mae",
            "outlier_abs_gt_10_count",
            "max_abs_diff",
            "p95_abs_diff",
            "optim_name",
            "lr",
            "batch_size",
            "n_conv",
            "atom_fea_len",
            "h_fea_len",
            "weight_decay",
            "n_h",
        ],
        [
            {
                "selection_primary": "best_val_mae",
                "selection_secondary_1": "outlier_abs_gt_10_count",
                "selection_secondary_2": "max_abs_diff",
                "timeout_sec_per_run": timeout_sec,
                **best_final,
            }
        ],
    )

    print("Final best directory:", final_best_dir.resolve())
    print("Best final val MAE:", best_final["best_val_mae"])
    print(
        "Best final outlier count |diff|>10:",
        best_final["outlier_abs_gt_10_count"],
    )


def run_mode(args):
    if len(args) != 6:
        print("Usage: run <seed> <epochs> <cfg_string> <data_dir> <out_dir>")
        return 2

    seed = int(args[1])
    epochs = int(args[2])
    cfg = parse_cfg_string(args[3])
    data_dir = Path(args[4]).expanduser().resolve()
    out_dir = Path(args[5]).expanduser().resolve()

    result_path = out_dir / "run_result.csv"
    try:
        result = run_single_trial(
            data_dir=data_dir,
            out_dir=out_dir,
            cfg=cfg,
            seed=seed,
            epochs=epochs,
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "seed": seed,
            "epochs": epochs,
            "best_val_mae": "",
            "best_model": "",
            "test_mae": "",
            "outlier_abs_gt_10_count": "",
            "max_abs_diff": "",
            "p95_abs_diff": "",
            "optim_name": cfg.get("optim_name", "Adam"),
            "lr": cfg.get("lr", ""),
            "batch_size": cfg.get("batch_size", ""),
            "n_conv": cfg.get("n_conv", ""),
            "atom_fea_len": cfg.get("atom_fea_len", ""),
            "h_fea_len": cfg.get("h_fea_len", ""),
            "weight_decay": cfg.get("weight_decay", ""),
            "n_h": cfg.get("n_h", ""),
            "error": str(exc),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        write_run_result_csv(result_path, result)
        return 1

    write_run_result_csv(result_path, result)
    return 0


def orchestrate_mode(args):
    if len(args) != 3:
        print("Usage: tune <data_dir> <output_dir>")
        return 2
    data_dir = Path(args[1]).expanduser().resolve()
    root_out = Path(args[2]).expanduser().resolve()
    write_tune_status(root_out, "RUNNING", "tuning_started")
    error_path = root_out / "ERROR.txt"
    if error_path.exists():
        error_path.unlink()
    try:
        orchestrate(data_dir, root_out)
    except Exception as exc:
        write_tune_status(root_out, "FAILED", str(exc))
        with error_path.open("w") as f:
            f.write(f"time_utc={now_utc_iso()}\n")
            f.write(f"error={exc}\n\n")
            f.write(traceback.format_exc())
        return 1
    except KeyboardInterrupt:
        write_tune_status(root_out, "INTERRUPTED", "keyboard_interrupt")
        return 130
    else:
        write_tune_status(root_out, "SUCCESS", "tuning_finished")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        raise SystemExit(run_mode(sys.argv[1:]))
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        raise SystemExit(orchestrate_mode(sys.argv[1:]))
    print("Usage:")
    print("  tune <data_dir> <output_dir>")
    print("  run <seed> <epochs> <cfg_string> <data_dir> <out_dir>")
    raise SystemExit(2)
