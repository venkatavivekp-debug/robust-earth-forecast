from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled spatial benchmark across multiple seeds."
    )
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--output-dir", type=str, default="results/spatial_benchmark_seed_stability")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--input-set", choices=["t2m", "core4"], default="core4")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--target-mode", choices=["direct", "residual"], default="direct")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l1-weight", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When running seeds: overwrite each seed_* directory. When set without "
        "--no-clear-root, also deletes the output-dir root before running.",
    )
    parser.add_argument(
        "--no-clear-root",
        action="store_true",
        help="Do not delete output-dir root before running (keeps existing seed_* runs). "
        "Use after partial failures so completed seeds are preserved.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only scan seed_*/summary.json under output-dir and write summary.csv, "
        "aggregate_summary.csv, and summary.json (no training).",
    )
    return parser.parse_args()


def repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_cmd(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        lines = proc.stdout.strip().splitlines()
        if len(lines) > 12:
            print(f"... {len(lines) - 12} lines omitted")
        print("\n".join(lines[-12:]))


def run_seed(args: argparse.Namespace, root: Path, seed: int) -> List[Dict[str, Any]]:
    seed_dir = root / f"seed_{seed}"
    cmd = [
        sys.executable,
        "scripts/run_spatial_benchmark.py",
        "--dataset-version",
        str(args.dataset_version),
        "--input-set",
        str(args.input_set),
        "--history-length",
        str(args.history_length),
        "--target-mode",
        str(args.target_mode),
        "--split-seed",
        str(seed),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--l1-weight",
        str(args.l1_weight),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--device",
        str(args.device),
        "--num-samples",
        "0",
        "--output-dir",
        str(seed_dir),
        "--overwrite",
    ]
    print(f"seed {seed}")
    run_cmd(cmd)
    rows = json.loads((seed_dir / "summary.json").read_text(encoding="utf-8"))
    for row in rows:
        row["seed"] = int(seed)
        row["split_seed"] = int(seed)
        row["seed_output_dir"] = str(seed_dir.relative_to(PROJECT_ROOT))
    return rows


def write_summary(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    fields = [
        "seed",
        "model",
        "rmse",
        "mae",
        "bias",
        "correlation",
        "variance_ratio",
        "border_rmse",
        "center_rmse",
        "border_center_rmse_ratio",
        "gradient_error_correlation",
        "num_samples",
        "target_mode",
        "input_set",
        "history_length",
        "seed_output_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_aggregate(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    metrics = [
        "rmse",
        "mae",
        "bias",
        "correlation",
        "border_rmse",
        "center_rmse",
        "border_center_rmse_ratio",
        "gradient_error_correlation",
    ]
    models = sorted({str(row["model"]) for row in rows})
    fields = ["model", "n_seeds"]
    for metric in metrics:
        fields.extend([f"{metric}_mean", f"{metric}_std"])

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for model in models:
            model_rows = [row for row in rows if str(row["model"]) == model]
            out: Dict[str, Any] = {"model": model, "n_seeds": len(model_rows)}
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in model_rows], dtype=np.float64)
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            writer.writerow(out)


def collect_rows_from_disk(root: Path, expected_seeds: Optional[Sequence[int]]) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    found_seeds: List[int] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = SEED_DIR_RE.match(child.name)
        if not m:
            continue
        seed = int(m.group(1))
        summary_path = child / "summary.json"
        if not summary_path.is_file():
            continue
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in rows:
            row = dict(row)
            row["seed"] = int(seed)
            row["split_seed"] = int(seed)
            row["seed_output_dir"] = str(child.relative_to(PROJECT_ROOT))
            all_rows.append(row)
        found_seeds.append(seed)

    if expected_seeds is not None:
        missing = sorted(set(int(s) for s in expected_seeds) - set(found_seeds))
        if missing:
            raise RuntimeError(
                f"Missing completed seed directories under {root}: {missing}. "
                f"Found seeds: {sorted(found_seeds)}"
            )

    if not all_rows:
        raise RuntimeError(f"No seed_*/summary.json found under {root}")
    return all_rows


def main() -> None:
    args = parse_args()
    root = repo_path(args.output_dir)

    if args.aggregate_only:
        root.mkdir(parents=True, exist_ok=True)
        all_rows = collect_rows_from_disk(root, expected_seeds=args.seeds)
        write_summary(all_rows, root / "summary.csv")
        write_aggregate(all_rows, root / "aggregate_summary.csv")
        (root / "summary.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {root / 'summary.csv'}")
        print(f"wrote {root / 'aggregate_summary.csv'}")
        print(f"wrote {root / 'summary.json'}")
        return

    if root.exists() and args.overwrite and not args.no_clear_root:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        all_rows.extend(run_seed(args, root, int(seed)))

    write_summary(all_rows, root / "summary.csv")
    write_aggregate(all_rows, root / "aggregate_summary.csv")
    (root / "summary.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {root / 'summary.csv'}")
    print(f"wrote {root / 'aggregate_summary.csv'}")


if __name__ == "__main__":
    main()
