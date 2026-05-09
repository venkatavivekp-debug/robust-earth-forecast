from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOSS_MODES = ("mse", "mse_l1", "mse_grad", "mse_l1_grad")
METRIC_COLUMNS = (
    "rmse",
    "mae",
    "bias",
    "correlation",
    "border_rmse",
    "center_rmse",
    "border_center_rmse_ratio",
    "variance_ratio",
    "gradient_ratio",
    "high_frequency_ratio",
    "local_contrast_ratio",
    "gradient_magnitude_rmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare detail-preserving loss variants for residual topo U-Net.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--output-dir", default="results/detail_preserving_loss")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--loss-modes", nargs="+", choices=LOSS_MODES, default=list(LOSS_MODES))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--grad-weight", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_cmd(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    stdout = proc.stdout.strip()
    if stdout:
        lines = stdout.splitlines()
        if len(lines) > 10:
            print(f"... {len(lines) - 10} log lines omitted")
        print("\n".join(lines[-10:]))


def run_one(args: argparse.Namespace, seed: int, loss_mode: str, output_dir: Path) -> None:
    if output_dir.exists() and (output_dir / "summary.csv").exists() and not args.overwrite:
        print(f"using existing {output_dir}")
        return

    cmd = [
        sys.executable,
        "scripts/run_topography_experiment.py",
        "--dataset-version",
        str(args.dataset_version),
        "--static-covariate-path",
        str(args.static_covariate_path),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(seed),
        "--split-seed",
        str(seed),
        "--device",
        str(args.device),
        "--target-mode",
        "residual",
        "--loss-mode",
        loss_mode,
        "--l1-weight",
        str(args.l1_weight),
        "--grad-weight",
        str(args.grad_weight),
        "--variants",
        "unet_core4_topo_h3",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    print(f"running detail-loss check seed={seed} loss_mode={loss_mode}")
    run_cmd(cmd)


def read_topo_row(path: Path, *, run_dir: str, loss_mode: str) -> Dict[str, Any]:
    with path.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    matches = [row for row in rows if row["model"] == "unet_core4_topo_h3"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one topo U-Net row in {path}, found {len(matches)}")
    row: Dict[str, Any] = matches[0]
    row["run_dir"] = run_dir
    row["loss_mode"] = loss_mode
    row["target_mode"] = "residual"
    for key in METRIC_COLUMNS:
        row[key] = float(row[key])
    row["seed"] = int(row["seed"])
    row["split_seed"] = int(row["split_seed"])
    row["num_samples"] = int(row["num_samples"])
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def aggregate_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["loss_mode"])].append(row)

    aggregate: List[Dict[str, Any]] = []
    for loss_mode, mode_rows in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "loss_mode": loss_mode,
            "target_mode": "residual",
            "model": "unet_core4_topo_h3",
            "input_set": "core4_topo",
            "seeds": ",".join(str(row["seed"]) for row in sorted(mode_rows, key=lambda item: item["seed"])),
            "n": len(mode_rows),
        }
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in mode_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregate.append(out)
    return aggregate


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    static_path = repo_path(args.static_covariate_path)
    if not static_path.exists():
        raise FileNotFoundError(f"Static covariate file not found: {static_path}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for loss_mode in args.loss_modes:
            run_dir = output_dir / f"seed_{seed}_{loss_mode}"
            run_one(args, seed, loss_mode, run_dir)
            rows.append(read_topo_row(run_dir / "summary.csv", run_dir=run_dir.name, loss_mode=loss_mode))

    summary_fields = [
        "run_dir",
        "model",
        "input_set",
        "target_mode",
        "loss_mode",
        "rmse",
        "mae",
        "bias",
        "correlation",
        "border_rmse",
        "center_rmse",
        "border_center_rmse_ratio",
        "variance_ratio",
        "gradient_ratio",
        "high_frequency_ratio",
        "local_contrast_ratio",
        "gradient_magnitude_rmse",
        "num_samples",
        "seed",
        "split_seed",
        "padding",
        "upsampling",
    ]
    write_csv(output_dir / "summary.csv", rows, summary_fields)

    aggregate = aggregate_rows(rows)
    aggregate_fields = ["loss_mode", "target_mode", "model", "input_set", "seeds", "n"]
    for metric in METRIC_COLUMNS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(output_dir / "aggregate_summary.csv", aggregate, aggregate_fields)

    payload = {
        "dataset_version": args.dataset_version,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "l1_weight": args.l1_weight,
        "grad_weight": args.grad_weight,
        "static_covariate_path": str(args.static_covariate_path),
        "summary": rows,
        "aggregate": aggregate,
    }
    (output_dir / "detail_preserving_loss_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("detail-preserving loss comparison")
    for row in aggregate:
        print(
            f"{row['loss_mode']:>11} | rmse={row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f} "
            f"mae={row['mae_mean']:.4f} +/- {row['mae_std']:.4f} "
            f"grad={row['gradient_ratio_mean']:.3f} hf={row['high_frequency_ratio_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
