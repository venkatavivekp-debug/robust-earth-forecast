from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ERA5 variable ablation study")
    p.add_argument("--era5-path", type=str, default="data_raw/era5_georgia_temp.nc")
    p.add_argument("--prism-path", type=str, default="data_raw/prism")
    p.add_argument("--model", type=str, default="convlstm", choices=["cnn", "convlstm"])
    p.add_argument("--era5-variables", nargs="+", default=["t2m"])
    p.add_argument("--history-length", type=int, default=3)
    p.add_argument("--forecast-horizon", type=int, default=1)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--results-dir", type=str, default="results/ablation")
    return p.parse_args()


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def read_metrics(path: Path) -> Dict[str, str]:
    import json

    data = json.loads(path.read_text())
    return {
        "rmse": str(data.get("rmse", "")),
        "mae": str(data.get("mae", "")),
        "bias": str(data.get("bias", "")),
        "temporal_diff_mae": str(data.get("temporal_diff_mae", "")),
    }


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []

    for var in args.era5_variables:
        ckpt = Path("checkpoints") / f"{args.model}_{var}.pt"
        eval_dir = results_root / f"{args.model}_{var}"

        status = "ok"
        metrics: Dict[str, str] = {}

        try:
            # TRAIN
            run_cmd([
                "python3",
                "training/train_downscaler.py",
                "--model", args.model,
                "--era5-path", args.era5_path,
                "--prism-path", args.prism_path,
                "--era5-variable", var,
                "--history-length", str(args.history_length),
                "--forecast-horizon", str(args.forecast_horizon),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--seed", str(args.seed),
                "--device", args.device,
                "--checkpoint-out", str(ckpt),
                "--training-results-dir", "results/training",
            ])

            # EVALUATE
            run_cmd([
                "python3",
                "evaluation/evaluate_model.py",
                "--models", args.model,
                "--era5-path", args.era5_path,
                "--prism-path", args.prism_path,
                "--era5-variable", var,
                "--history-length", str(args.history_length),
                "--forecast-horizon", str(args.forecast_horizon),
                "--seed", str(args.seed),
                "--device", args.device,
                "--results-dir", str(eval_dir),
                "--cnn-checkpoint", str(ckpt),
                "--convlstm-checkpoint", str(ckpt),
            ])

            metrics_path = eval_dir / args.model / "metrics.json"
            if metrics_path.exists():
                metrics = read_metrics(metrics_path)

        except Exception as e:
            status = f"failed: {type(e).__name__}"

        rows.append({
            "model": args.model,
            "era5_variable": var,
            "history_length": str(args.history_length),
            "forecast_horizon": str(args.forecast_horizon),
            "status": status,
            **metrics,
        })

    out_csv = results_root / "ablation_summary.csv"

    fieldnames = [
        "model",
        "era5_variable",
        "history_length",
        "forecast_horizon",
        "status",
        "rmse",
        "mae",
        "bias",
        "temporal_diff_mae",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()