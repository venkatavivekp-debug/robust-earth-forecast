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
    parser = argparse.ArgumentParser(description="Run topography-context U-Net stability across seeds.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--output-dir", default="results/topography_seed_stability")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--run-residual-check", action=argparse.BooleanOptionalAction, default=True)
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


def run_topography_experiment(args: argparse.Namespace, seed: int, output_dir: Path, target_mode: str, variants: List[str]) -> None:
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
        target_mode,
        "--variants",
        *variants,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    print(f"running topography experiment seed={seed} target_mode={target_mode} variants={','.join(variants)}")
    run_cmd(cmd)


def read_rows(path: Path, *, run_dir: str, target_mode: str) -> List[Dict[str, Any]]:
    with path.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    for row in rows:
        row["run_dir"] = run_dir
        row["target_mode"] = target_mode
        for key in METRIC_COLUMNS:
            row[key] = float(row[key])
        row["seed"] = int(row["seed"])
        row["split_seed"] = int(row["split_seed"])
        row["num_samples"] = int(row["num_samples"])
    return rows


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
        grouped[str(row["model"])].append(row)

    aggregate: List[Dict[str, Any]] = []
    for model, model_rows in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "model": model,
            "input_set": model_rows[0].get("input_set"),
            "target_mode": model_rows[0].get("target_mode"),
            "seeds": ",".join(str(row["seed"]) for row in sorted(model_rows, key=lambda item: item["seed"])),
            "n": len(model_rows),
        }
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in model_rows]
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

    variants = ["unet_core4_h3", "unet_core4_elev_h3", "unet_core4_topo_h3"]
    for seed in args.seeds:
        run_topography_experiment(args, seed, output_dir / f"seed_{seed}", "direct", variants)

    direct_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        direct_rows.extend(read_rows(output_dir / f"seed_{seed}" / "summary.csv", run_dir=f"seed_{seed}", target_mode="direct"))

    summary_fields = [
        "run_dir",
        "model",
        "input_set",
        "target_mode",
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
    write_csv(output_dir / "summary.csv", direct_rows, summary_fields)

    aggregate = aggregate_rows(direct_rows)
    aggregate_fields = ["model", "input_set", "target_mode", "seeds", "n"]
    for metric in METRIC_COLUMNS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(output_dir / "aggregate_summary.csv", aggregate, aggregate_fields)

    residual_rows: List[Dict[str, Any]] = []
    if args.run_residual_check:
        residual_dir = output_dir / f"residual_seed_{args.seeds[0]}"
        run_topography_experiment(args, args.seeds[0], residual_dir, "residual", ["unet_core4_topo_h3"])
        direct_topo = [
            row for row in direct_rows if row["model"] == "unet_core4_topo_h3" and row["seed"] == args.seeds[0]
        ]
        residual_topo = [
            row
            for row in read_rows(residual_dir / "summary.csv", run_dir=residual_dir.name, target_mode="residual")
            if row["model"] == "unet_core4_topo_h3"
        ]
        residual_rows = direct_topo + residual_topo
        write_csv(output_dir / "residual_comparison.csv", residual_rows, summary_fields)

    summary = {
        "dataset_version": args.dataset_version,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "static_covariate_path": str(args.static_covariate_path),
        "variants": variants,
        "aggregate": aggregate,
        "residual_comparison": residual_rows,
    }
    (output_dir / "topography_seed_stability_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("topography seed stability")
    for row in aggregate:
        print(
            f"{row['model']:>21} | rmse={row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f} "
            f"grad={row['gradient_ratio_mean']:.3f} +/- {row['gradient_ratio_std']:.3f} "
            f"hf={row['high_frequency_ratio_mean']:.4f} +/- {row['high_frequency_ratio_std']:.4f}"
        )
    if residual_rows:
        print("residual comparison")
        for row in residual_rows:
            print(
                f"{row['target_mode']:>8} | rmse={row['rmse']:.4f} mae={row['mae']:.4f} "
                f"border={row['border_rmse']:.4f} center={row['center_rmse']:.4f}"
            )


if __name__ == "__main__":
    main()
