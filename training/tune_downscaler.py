from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

try:
    import torch
except ModuleNotFoundError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight hyperparameter sweep for ERA5->PRISM models")
    parser.add_argument(
        "--dataset-version",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="ERA5/PRISM defaults from datasets/<version>/paths.json when paths omitted",
    )
    parser.add_argument("--era5-path", type=str, default=None)
    parser.add_argument("--prism-path", type=str, default=None)
    parser.add_argument("--input-set", type=str, choices=["t2m", "core4", "extended"], default="extended")
    parser.add_argument("--models", nargs="+", choices=["cnn", "convlstm"], default=["cnn", "convlstm"])
    parser.add_argument("--history-lengths", nargs="+", type=int, default=[3, 6])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3, 5e-4, 1e-4])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 1e-5])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default="results/tuning")
    parser.add_argument("--keep-checkpoints", action="store_true")
    return parser.parse_args()


def format_float(value: float) -> str:
    return f"{value:.0e}" if value < 1e-2 else str(value).replace(".", "p")


def build_train_command(args: argparse.Namespace, model: str, history: int, lr: float, wd: float, checkpoint_path: Path, run_name: str) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "train_downscaler.py"),
        "--era5-path",
        args.era5_path,
        "--prism-path",
        args.prism_path,
        "--input-set",
        args.input_set,
        "--model",
        model,
        "--history-length",
        str(history),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(lr),
        "--weight-decay",
        str(wd),
        "--l1-weight",
        str(args.l1_weight),
        "--grad-clip",
        str(args.grad_clip),
        "--scheduler",
        args.scheduler,
        "--split-seed",
        str(args.split_seed),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--checkpoint-out",
        str(checkpoint_path),
        "--training-results-dir",
        "/tmp/robust-earth-forecast-tuning-logs",
        "--run-name",
        run_name,
    ]

    if model == "convlstm":
        cmd.extend(["--hidden-channels", str(args.hidden_channels)])

    return cmd


def main() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for tuning. Install dependencies with: pip install -r requirements.txt")

    args = parse_args()

    from datasets.dataset_paths import apply_dataset_version_to_args

    apply_dataset_version_to_args(args)

    results_dir = Path(args.results_dir)
    ckpt_dir = results_dir / "checkpoints_tmp"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict] = []
    search_space = list(itertools.product(args.models, args.history_lengths, args.learning_rates, args.weight_decays))
    if args.max_runs is not None:
        search_space = search_space[: max(0, int(args.max_runs))]

    print(f"Starting tuning sweep: {len(search_space)} runs")
    for run_idx, (model, history, lr, wd) in enumerate(search_space, start=1):
        run_name = f"{model}_h{history}_lr{format_float(lr)}_wd{format_float(wd)}"
        checkpoint_path = ckpt_dir / f"{run_name}.pt"
        cmd = build_train_command(args, model, history, lr, wd, checkpoint_path, run_name)

        print(f"[{run_idx}/{len(search_space)}] {run_name}")
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

        row = {
            "run_name": run_name,
            "model": model,
            "input_set": args.input_set,
            "history_length": int(history),
            "learning_rate": float(lr),
            "weight_decay": float(wd),
            "l1_weight": float(args.l1_weight),
            "epochs": int(args.epochs),
            "status": "ok" if proc.returncode == 0 else "failed",
            "best_val_loss": None,
            "best_epoch": None,
            "checkpoint": str(checkpoint_path),
        }

        if proc.returncode == 0 and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            row["best_val_loss"] = float(checkpoint.get("best_val_loss", float("inf")))
            row["best_epoch"] = int(checkpoint.get("epoch", -1))
        else:
            row["error_tail"] = "\n".join((proc.stderr or "").splitlines()[-20:])

        runs.append(row)

        if checkpoint_path.exists() and not args.keep_checkpoints:
            checkpoint_path.unlink()

    csv_path = results_dir / "tuning_summary.csv"
    fieldnames = [
        "run_name",
        "model",
        "input_set",
        "history_length",
        "learning_rate",
        "weight_decay",
        "l1_weight",
        "epochs",
        "status",
        "best_val_loss",
        "best_epoch",
        "checkpoint",
        "error_tail",
    ]
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in runs:
            writer.writerow(row)

    ok_runs = [row for row in runs if row["status"] == "ok" and row["best_val_loss"] is not None]
    best_overall = min(ok_runs, key=lambda r: r["best_val_loss"]) if ok_runs else None

    best_by_model: Dict[str, Dict] = {}
    for model_name in args.models:
        model_runs = [row for row in ok_runs if row["model"] == model_name]
        if model_runs:
            best_by_model[model_name] = min(model_runs, key=lambda r: r["best_val_loss"])

    best_path = results_dir / "best_config.json"
    best_path.write_text(
        json.dumps(
            {
                "search_space_size": len(search_space),
                "best_overall": best_overall,
                "best_by_model": best_by_model,
            },
            indent=2,
        )
    )

    print(f"Saved tuning summary: {csv_path}")
    print(f"Saved best config: {best_path}")


if __name__ == "__main__":
    main()
