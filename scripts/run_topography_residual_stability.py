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
    parser = argparse.ArgumentParser(description="Compare direct and residual terrain-conditioned U-Net across seeds.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument("--output-dir", default="results/topography_residual_stability")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
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


def run_one(args: argparse.Namespace, seed: int, target_mode: str, output_dir: Path) -> None:
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
        "unet_core4_topo_h3",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    print(f"running terrain residual check seed={seed} target_mode={target_mode}")
    run_cmd(cmd)


def read_topo_row(path: Path, *, run_dir: str, target_mode: str) -> Dict[str, Any]:
    with path.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    matches = [row for row in rows if row["model"] == "unet_core4_topo_h3"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one topo U-Net row in {path}, found {len(matches)}")
    row: Dict[str, Any] = matches[0]
    row["run_dir"] = run_dir
    row["target_mode"] = target_mode
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
        grouped[str(row["target_mode"])].append(row)

    aggregate: List[Dict[str, Any]] = []
    for mode, mode_rows in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "target_mode": mode,
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


def seed_deltas(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][str(row["target_mode"])] = row

    deltas: List[Dict[str, Any]] = []
    for seed, modes in sorted(by_seed.items()):
        if "direct" not in modes or "residual" not in modes:
            raise RuntimeError(f"Missing direct/residual row for seed {seed}")
        direct = modes["direct"]
        residual = modes["residual"]
        out: Dict[str, Any] = {"seed": seed}
        for metric in METRIC_COLUMNS:
            out[f"{metric}_direct"] = direct[metric]
            out[f"{metric}_residual"] = residual[metric]
            out[f"{metric}_delta_residual_minus_direct"] = residual[metric] - direct[metric]
        deltas.append(out)
    return deltas


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
        for target_mode in ("direct", "residual"):
            run_dir = output_dir / f"seed_{seed}_{target_mode}"
            run_one(args, seed, target_mode, run_dir)
            rows.append(read_topo_row(run_dir / "summary.csv", run_dir=run_dir.name, target_mode=target_mode))

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
    write_csv(output_dir / "summary.csv", rows, summary_fields)

    aggregate = aggregate_rows(rows)
    aggregate_fields = ["target_mode", "model", "input_set", "seeds", "n"]
    for metric in METRIC_COLUMNS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(output_dir / "aggregate_summary.csv", aggregate, aggregate_fields)

    deltas = seed_deltas(rows)
    delta_fields = ["seed"]
    for metric in METRIC_COLUMNS:
        delta_fields.extend(
            [f"{metric}_direct", f"{metric}_residual", f"{metric}_delta_residual_minus_direct"]
        )
    write_csv(output_dir / "seed_deltas.csv", deltas, delta_fields)

    payload = {
        "dataset_version": args.dataset_version,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "static_covariate_path": str(args.static_covariate_path),
        "summary": rows,
        "aggregate": aggregate,
        "seed_deltas": deltas,
    }
    (output_dir / "topography_residual_stability_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("terrain residual stability")
    for row in aggregate:
        print(
            f"{row['target_mode']:>8} | rmse={row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f} "
            f"mae={row['mae_mean']:.4f} +/- {row['mae_std']:.4f} "
            f"grad={row['gradient_ratio_mean']:.3f} hf={row['high_frequency_ratio_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
