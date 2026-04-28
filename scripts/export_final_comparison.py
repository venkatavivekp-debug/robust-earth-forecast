#!/usr/bin/env python3
"""Build docs/experiments/final_comparison[_medium].json from a run_core_experiments output tree."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_rmse_by_model(csv_path: Path) -> Dict[str, float]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    out: Dict[str, float] = {}
    for row in rows:
        m = str(row.get("model", "")).strip().lower()
        if not m:
            continue
        out[m] = float(row["rmse"])
    return out


def collect_grid(
    experiments_root: Path,
    *,
    input_sets: List[str],
    histories: List[int],
) -> Tuple[List[Dict[str, Any]], float]:
    convlstm_rows: List[Dict[str, Any]] = []
    ref_persist: float | None = None

    for inp in input_sets:
        for h in histories:
            exp = experiments_root / f"{inp}_h{h}" / "evaluation" / "baselines_summary.csv"
            rm = read_rmse_by_model(exp)
            if "persistence" not in rm or "convlstm" not in rm:
                raise RuntimeError(f"{exp}: need persistence and convlstm rows")
            if "cnn" not in rm:
                raise RuntimeError(f"{exp}: need cnn row")
            for k, v in rm.items():
                if not math.isfinite(v):
                    raise RuntimeError(f"{exp}: non-finite rmse for {k}")
            convlstm_rows.append({"input_set": inp, "history": int(h), "rmse": float(rm["convlstm"])})
            if inp == "core4" and int(h) == 3:
                ref_persist = float(rm["persistence"])

    if ref_persist is None:
        raise RuntimeError("Need core4 history 3 experiment to define canonical persistence_rmse in JSON")

    return convlstm_rows, ref_persist


def best_convlstm(rows: List[Dict[str, Any]], persistence: float) -> Dict[str, Any]:
    best = min(rows, key=lambda r: float(r["rmse"]))
    rmse = float(best["rmse"])
    delta = rmse - persistence
    pct = 100.0 * (persistence - rmse) / persistence if persistence > 0 else float("nan")
    return {
        "model": "convlstm",
        "input_set": str(best["input_set"]),
        "history": int(best["history"]),
        "rmse": rmse,
        "persistence_rmse": persistence,
        "delta_vs_persistence": delta,
        "percent_improvement": float(pct),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final_comparison JSON from results/experiments* tree.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=None,
        help="Root containing <input>_h<h>/evaluation/baselines_summary.csv (default: results/experiments or medium)",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="Used for output filename default and dataset_version field",
    )
    parser.add_argument(
        "--input-sets",
        type=str,
        nargs="+",
        default=["t2m", "core4"],
    )
    parser.add_argument("--histories", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--era5-path", type=str, default=None, help="Optional: for usable sample counts")
    parser.add_argument("--prism-path", type=str, default=None)
    args = parser.parse_args()

    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    exp_root = args.experiments_root
    if exp_root is None:
        exp_root = root / "results" / ("experiments_medium" if args.dataset_version == "medium" else "experiments")

    out_path = args.out
    if out_path is None:
        name = "final_comparison_medium.json" if args.dataset_version == "medium" else "final_comparison.json"
        out_path = root / "docs" / "experiments" / name

    conv_rows, persistence = collect_grid(exp_root, input_sets=list(args.input_sets), histories=list(args.histories))

    best_o = best_convlstm(conv_rows, persistence)
    best_t2m = min((r for r in conv_rows if r["input_set"] == "t2m"), key=lambda r: float(r["rmse"]))
    best_c4 = min((r for r in conv_rows if r["input_set"] == "core4"), key=lambda r: float(r["rmse"]))

    def pack_best(row: Dict[str, Any]) -> Dict[str, Any]:
        rmse = float(row["rmse"])
        d = rmse - persistence
        pct = 100.0 * (persistence - rmse) / persistence if persistence > 0 else float("nan")
        return {
            "model": "convlstm",
            "input_set": str(row["input_set"]),
            "history": int(row["history"]),
            "rmse": rmse,
            "persistence_rmse": persistence,
            "delta_vs_persistence": d,
            "percent_improvement": float(pct),
        }

    usable: Dict[str, Any] | None = None
    if args.era5_path and args.prism_path:
        from datasets.prism_dataset import ERA5_PRISM_Dataset

        usable = {}
        for h in args.histories:
            ds = ERA5_PRISM_Dataset(
                era5_path=args.era5_path,
                prism_path=args.prism_path,
                history_length=int(h),
                input_set="core4",
                verbose=False,
            )
            usable[f"history_{h}"] = int(len(ds))

    lines = [
        f"{best_c4['input_set']} best ConvLSTM RMSE={float(best_c4['rmse']):.4f} vs best t2m={float(best_t2m['rmse']):.4f}.",
        f"Canonical persistence RMSE (core4, history=3 eval)={persistence:.6f}.",
        f"Best overall ConvLSTM: {best_o['input_set']} history={int(best_o['history'])} RMSE={float(best_o['rmse']):.6f}.",
    ]
    payload: Dict[str, Any] = {
        "dataset_version": str(args.dataset_version),
        "best_config": {
            "model": "convlstm",
            "input_set": str(best_o["input_set"]),
            "history": int(best_o["history"]),
        },
        "best_rmse": float(best_o["rmse"]),
        "improvement_vs_persistence": {
            "persistence_rmse": float(persistence),
            "delta_vs_persistence": float(best_o["delta_vs_persistence"]),
            "percent_improvement": float(best_o["percent_improvement"]),
        },
        "best_overall": best_o,
        "best_t2m": pack_best(best_t2m),
        "best_core4": pack_best(best_c4),
        "core4_outperforms_t2m": float(best_c4["rmse"]) < float(best_t2m["rmse"]),
        "convlstm_rmse_by_history": conv_rows,
        "clear_conclusion": " ".join(lines),
    }
    if usable is not None:
        payload["usable_samples"] = usable

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
