from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small LR/capacity sanity check for terrain residual U-Net.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--output-dir", default="results/training_sanity/lr_capacity_diagnosis")
    parser.add_argument("--subset-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3, 3e-4, 1e-4])
    parser.add_argument("--base-channels", nargs="+", type=int, default=[24, 48])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_name(base_channels: int, learning_rate: float) -> str:
    lr_text = f"{learning_rate:.0e}".replace("-", "m")
    return f"base{base_channels}_lr{lr_text}"


def run_config(args: argparse.Namespace, base_channels: int, learning_rate: float, output_dir: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "overfit_small_subset.py"),
        "--dataset-version",
        str(args.dataset_version),
        "--static-covariate-path",
        str(repo_path(args.static_covariate_path)),
        "--output-dir",
        str(output_dir),
        "--subset-sizes",
        str(args.subset_size),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(learning_rate),
        "--unet-base-channels",
        str(base_channels),
        "--seed",
        str(args.seed),
        "--split-seed",
        str(args.split_seed),
        "--device",
        str(args.device),
        "--overwrite",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    metrics_path = output_dir / f"subset_{int(args.subset_size)}" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.update(
        {
            "config": output_dir.name,
            "base_channels": int(base_channels),
            "learning_rate": float(learning_rate),
            "epochs": int(args.epochs),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    output_root = repo_path(args.output_dir)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for base_channels in args.base_channels:
        for learning_rate in args.learning_rates:
            name = config_name(int(base_channels), float(learning_rate))
            metrics = run_config(args, int(base_channels), float(learning_rate), output_root / name)
            rows.append(metrics)
            print(
                f"{name:>15} | train_rmse={metrics['train_rmse']:.4f} "
                f"val_rmse={metrics['val_rmse']:.4f} train_hf={metrics['train_high_frequency_ratio']:.3f}"
            )

    fields = [
        "config",
        "base_channels",
        "learning_rate",
        "epochs",
        "train_rmse",
        "train_mae",
        "train_gradient_ratio",
        "train_high_frequency_ratio",
        "val_rmse",
        "val_mae",
        "val_gradient_ratio",
        "val_high_frequency_ratio",
    ]
    with (output_root / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
