from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run input-variable ablation for ConvLSTM")
    parser.add_argument("--era5-path", type=str, default="data_raw/era5_georgia_temp.nc")
    parser.add_argument("--prism-path", type=str, default="data_raw/prism")
    parser.add_argument("--input-sets", nargs="+", choices=["t2m", "core4", "extended"], default=["t2m", "core4", "extended"])
    parser.add_argument("--model", type=str, choices=["cnn", "convlstm"], default="convlstm")
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--results-dir", type=str, default="results/ablation")
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for input_set in args.input_sets:
        run_name = f"ablation_{args.model}_{input_set}"
        ckpt_path = work_dir / f"{run_name}.pt"
        eval_dir = work_dir / f"eval_{run_name}"

        print(f"Ablation run: model={args.model} input_set={input_set}")
        train_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "training" / "train_downscaler.py"),
            "--era5-path",
            args.era5_path,
            "--prism-path",
            args.prism_path,
            "--input-set",
            input_set,
            "--model",
            args.model,
            "--history-length",
            str(args.history_length),
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--grad-clip",
            str(args.grad_clip),
            "--split-seed",
            str(args.split_seed),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--checkpoint-out",
            str(ckpt_path),
            "--training-results-dir",
            "/tmp/robust-earth-forecast-ablation-logs",
            "--run-name",
            run_name,
        ]
        run_cmd(train_cmd)

        eval_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "evaluation" / "evaluate_model.py"),
            "--era5-path",
            args.era5_path,
            "--prism-path",
            args.prism_path,
            "--input-set",
            input_set,
            "--models",
            args.model,
            "--history-length",
            str(args.history_length),
            "--split-seed",
            str(args.split_seed),
            "--device",
            args.device,
            "--results-dir",
            str(eval_dir),
        ]
        if args.model == "cnn":
            eval_cmd.extend(["--cnn-checkpoint", str(ckpt_path)])
        else:
            eval_cmd.extend(["--convlstm-checkpoint", str(ckpt_path)])

        run_cmd(eval_cmd)

        metrics_path = eval_dir / args.model / "metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(f"Missing metrics file: {metrics_path}")
        metrics = json.loads(metrics_path.read_text())
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        rows.append(
            {
                "model": args.model,
                "input_set": input_set,
                "history_length": args.history_length,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
                "rmse": float(metrics.get("rmse", float("nan"))),
                "mae": float(metrics.get("mae", float("nan"))),
                "bias": float(metrics.get("bias", float("nan"))),
                "correlation": float(metrics.get("correlation", float("nan"))),
            }
        )

        shutil.rmtree(eval_dir, ignore_errors=True)
        ckpt_path.unlink(missing_ok=True)

    summary_path = out_dir / "ablation_summary.csv"
    fieldnames = [
        "model",
        "input_set",
        "history_length",
        "epochs",
        "learning_rate",
        "weight_decay",
        "best_val_loss",
        "rmse",
        "mae",
        "bias",
        "correlation",
    ]
    with summary_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Saved ablation summary: {summary_path}")


if __name__ == "__main__":
    main()
