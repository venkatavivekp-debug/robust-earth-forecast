from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run core ERA5→PRISM experiments with fixed splits and unified evaluation.")
    p.add_argument("--era5-path", type=str, default="data_raw/era5_georgia_temp.nc")
    p.add_argument("--prism-path", type=str, default="data_raw/prism")
    p.add_argument("--input-sets", type=str, nargs="+", default=["t2m", "core4"], choices=["t2m", "core4"])
    p.add_argument("--histories", type=int, nargs="+", default=[1, 3, 6])
    p.add_argument("--epochs-cnn", type=int, default=80)
    p.add_argument("--epochs-convlstm", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--early-stopping-patience", type=int, default=12)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--overwrite", action="store_true", help="Overwrite per-experiment outputs under results/experiments/")
    return p.parse_args()


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def sanity_forward_pass(era5_path: str, prism_path: str, input_set: str, history: int) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the current Python environment.\n"
            "Run experiments from the repo virtual environment, for example:\n"
            "  source .venv/bin/activate\n"
            "  python scripts/run_core_experiments.py --input-sets t2m --histories 1 3 6 --overwrite\n"
            "Or:\n"
            "  .venv/bin/python scripts/run_core_experiments.py --input-sets t2m --histories 1 3 6 --overwrite"
        ) from exc

    from datasets.prism_dataset import ERA5_PRISM_Dataset
    from models.cnn_downscaler import CNNDownscaler
    from models.convlstm_downscaler import ConvLSTMDownscaler

    ds = ERA5_PRISM_Dataset(
        era5_path=era5_path,
        prism_path=prism_path,
        history_length=int(history),
        input_set=input_set,
        verbose=False,
    )
    x, y = ds[0]
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise RuntimeError("Non-finite values found during forward-pass sanity check")

    xb = x.unsqueeze(0)  # [B,T,C,H,W]
    target_hw = (int(y.shape[-2]), int(y.shape[-1]))

    cnn = CNNDownscaler(in_channels=int(x.shape[0] * x.shape[1]), out_channels=int(y.shape[0]))
    convlstm = ConvLSTMDownscaler(input_channels=int(x.shape[1]), hidden_channels=32, out_channels=int(y.shape[0]))

    with torch.no_grad():
        p1 = cnn(xb, target_size=target_hw)
        p2 = convlstm(xb, target_size=target_hw)

    if p1.shape != y.unsqueeze(0).shape:
        raise RuntimeError(f"CNN forward shape mismatch: pred={tuple(p1.shape)} y={tuple(y.unsqueeze(0).shape)}")
    if p2.shape != y.unsqueeze(0).shape:
        raise RuntimeError(f"ConvLSTM forward shape mismatch: pred={tuple(p2.shape)} y={tuple(y.unsqueeze(0).shape)}")


def read_baselines_summary(summary_csv: Path) -> List[Dict[str, str]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {summary_csv}")
    with summary_csv.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Empty evaluation summary: {summary_csv}")
    return rows


def require_finite(value: float, label: str) -> None:
    if not np.isfinite(value):
        raise RuntimeError(f"Non-finite {label}: {value}")


def write_experiment_rows(
    *,
    summary_out: Path,
    experiment_name: str,
    input_set: str,
    history: int,
    rows: List[Dict[str, str]],
) -> None:
    required_cols = ["model", "rmse", "mae", "bias", "correlation"]
    for col in required_cols:
        if col not in rows[0]:
            raise RuntimeError(f"Missing '{col}' in baselines_summary.csv for {experiment_name}")

    rmse_by_model: Dict[str, float] = {}
    for r in rows:
        m = r["model"]
        rmse_by_model[m] = float(r["rmse"])
        require_finite(rmse_by_model[m], f"rmse[{m}]")

    if "persistence" not in rmse_by_model:
        raise RuntimeError(f"Missing persistence baseline in {experiment_name}")
    persistence_rmse = rmse_by_model["persistence"]

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_out.exists()
    with summary_out.open("a", newline="") as fp:
        fieldnames = [
            "experiment",
            "model",
            "input_set",
            "history",
            "rmse",
            "mae",
            "bias",
            "correlation",
            "delta_vs_persistence",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()

        for r in rows:
            model = r["model"]
            rmse = float(r["rmse"])
            mae = float(r["mae"])
            bias = float(r["bias"])
            corr = float(r["correlation"])
            for val, lbl in [(rmse, "rmse"), (mae, "mae"), (bias, "bias"), (corr, "correlation")]:
                require_finite(val, f"{lbl}[{experiment_name}:{model}]")

            delta = rmse - persistence_rmse
            writer.writerow(
                {
                    "experiment": experiment_name,
                    "model": model,
                    "input_set": input_set,
                    "history": int(history),
                    "rmse": rmse,
                    "mae": mae,
                    "bias": bias,
                    "correlation": corr,
                    "delta_vs_persistence": delta,
                }
            )

    for model_name in ("cnn", "convlstm"):
        if model_name in rmse_by_model and rmse_by_model[model_name] >= persistence_rmse:
            print(
                f"Warning: {experiment_name} {model_name} RMSE ({rmse_by_model[model_name]:.4f}) "
                f">= persistence ({persistence_rmse:.4f})"
            )


def main() -> None:
    args = parse_args()
    results_root = PROJECT_ROOT / "results" / "experiments"
    summary_out = results_root / "summary.csv"
    if args.overwrite and summary_out.exists():
        summary_out.unlink()

    for input_set in args.input_sets:
        for history in args.histories:
            experiment_name = f"{input_set}_h{history}"
            exp_dir = results_root / experiment_name
            ckpt_dir = exp_dir / "checkpoints"
            train_dir = exp_dir / "training_logs"
            eval_dir = exp_dir / "evaluation"

            if args.overwrite and exp_dir.exists():
                shutil.rmtree(exp_dir)
            exp_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            sanity_forward_pass(args.era5_path, args.prism_path, input_set=input_set, history=int(history))

            cnn_ckpt = ckpt_dir / "cnn_best.pt"
            cl_ckpt = ckpt_dir / "convlstm_best.pt"

            # Train CNN
            run_cmd(
                [
                    PYTHON,
                    "training/train_downscaler.py",
                    "--era5-path",
                    args.era5_path,
                    "--prism-path",
                    args.prism_path,
                    "--model",
                    "cnn",
                    "--input-set",
                    input_set,
                    "--history-length",
                    str(history),
                    "--epochs",
                    str(args.epochs_cnn),
                    "--batch-size",
                    str(args.batch_size),
                    "--learning-rate",
                    str(args.lr),
                    "--weight-decay",
                    "0",
                    "--l1-weight",
                    "0.1",
                    "--grad-clip",
                    "1.0",
                    "--scheduler",
                    "plateau",
                    "--scheduler-patience",
                    "7",
                    "--scheduler-factor",
                    "0.5",
                    "--early-stopping-patience",
                    str(args.early_stopping_patience),
                    "--seed",
                    str(args.seed),
                    "--split-seed",
                    str(args.split_seed),
                    "--device",
                    args.device,
                    "--checkpoint-out",
                    str(cnn_ckpt),
                    "--training-results-dir",
                    str(train_dir),
                    "--run-name",
                    f"cnn_{experiment_name}",
                ]
            )

            # Train ConvLSTM
            run_cmd(
                [
                    PYTHON,
                    "training/train_downscaler.py",
                    "--era5-path",
                    args.era5_path,
                    "--prism-path",
                    args.prism_path,
                    "--model",
                    "convlstm",
                    "--input-set",
                    input_set,
                    "--history-length",
                    str(history),
                    "--epochs",
                    str(args.epochs_convlstm),
                    "--batch-size",
                    str(args.batch_size),
                    "--learning-rate",
                    str(args.lr),
                    "--weight-decay",
                    "1e-6",
                    "--l1-weight",
                    "0.1",
                    "--grad-clip",
                    "1.0",
                    "--scheduler",
                    "plateau",
                    "--scheduler-patience",
                    "10",
                    "--scheduler-factor",
                    "0.5",
                    "--early-stopping-patience",
                    str(args.early_stopping_patience),
                    "--seed",
                    str(args.seed),
                    "--split-seed",
                    str(args.split_seed),
                    "--device",
                    args.device,
                    "--checkpoint-out",
                    str(cl_ckpt),
                    "--training-results-dir",
                    str(train_dir),
                    "--run-name",
                    f"convlstm_{experiment_name}",
                ]
            )

            # Evaluate once for this (input_set, history) with consistent split from checkpoint metadata.
            run_cmd(
                [
                    PYTHON,
                    "evaluation/evaluate_model.py",
                    "--era5-path",
                    args.era5_path,
                    "--prism-path",
                    args.prism_path,
                    "--input-set",
                    input_set,
                    "--history-length",
                    str(history),
                    "--split-seed",
                    str(args.split_seed),
                    "--num-samples",
                    str(args.num_samples),
                    "--models",
                    "persistence",
                    "era5_upsampled",
                    "cnn",
                    "convlstm",
                    "--cnn-checkpoint",
                    str(cnn_ckpt),
                    "--convlstm-checkpoint",
                    str(cl_ckpt),
                    "--device",
                    args.device,
                    "--results-dir",
                    str(eval_dir),
                ]
            )
            rows = read_baselines_summary(eval_dir / "baselines_summary.csv")
            write_experiment_rows(
                summary_out=summary_out,
                experiment_name=experiment_name,
                input_set=input_set,
                history=int(history),
                rows=rows,
            )

    print(f"Wrote experiment outputs under: {results_root}")
    print(f"Wrote summary CSV: {summary_out}")

if __name__ == "__main__":
    main()

