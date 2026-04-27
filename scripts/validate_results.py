#!/usr/bin/env python3
"""Sanity-check committed experiment JSON (no training)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate docs/experiments JSON structure and RMSE fields.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help="Directory containing JSON (default: <repo>/docs/experiments)",
    )
    args = parser.parse_args()
    root = project_root()
    exp_dir = args.experiments_dir or (root / "docs" / "experiments")
    final_path = exp_dir / "final_comparison.json"
    errors: List[str] = []
    warns: List[str] = []

    if not final_path.exists():
        errors.append(f"Missing {final_path}")
        _print_report(errors, warns)
        return 1

    data: Dict[str, Any] = json.loads(final_path.read_text(encoding="utf-8"))

    imp = data.get("improvement_vs_persistence")
    if not isinstance(imp, dict) or "persistence_rmse" not in imp:
        errors.append("final_comparison.json: missing improvement_vs_persistence.persistence_rmse")
    else:
        p = imp["persistence_rmse"]
        if not isinstance(p, (int, float)) or not math.isfinite(float(p)):
            errors.append("persistence_rmse is missing or non-finite")

    rows = data.get("convlstm_rmse_by_history")
    if not isinstance(rows, list) or not rows:
        errors.append("final_comparison.json: missing or empty convlstm_rmse_by_history")
    else:
        for i, item in enumerate(rows):
            if not isinstance(item, dict):
                errors.append(f"convlstm_rmse_by_history[{i}] is not an object")
                continue
            for key in ("input_set", "history", "rmse"):
                if key not in item:
                    errors.append(f"convlstm_rmse_by_history[{i}] missing '{key}'")
            rm = item.get("rmse")
            if isinstance(rm, (int, float)) and not math.isfinite(float(rm)):
                errors.append(f"convlstm_rmse_by_history[{i}] has non-finite rmse")

    if errors:
        _print_report(errors, warns)
        return 1

    p_rmse = float(data["improvement_vs_persistence"]["persistence_rmse"])
    beats_flags: List[bool] = []
    by_input: Dict[str, List[bool]] = {}
    for item in data["convlstm_rmse_by_history"]:
        inp = str(item["input_set"])
        rm = float(item["rmse"])
        b = rm < p_rmse
        beats_flags.append(b)
        by_input.setdefault(inp, []).append(b)

    if not any(beats_flags):
        warns.append("No ConvLSTM configuration beats persistence in final_comparison.json")
    if any(beats_flags) and not all(beats_flags):
        warns.append(
            "ConvLSTM beats persistence for some history/input rows but not others — "
            "expect instability when N is small (see docs/experiments/results_summary.md)."
        )

    for inp, flags in sorted(by_input.items()):
        if any(flags) and not all(flags):
            warns.append(f"Input set {inp!r}: mixed beat/non-beat across history lengths.")

    # Optional: error_analysis.json shape
    err_path = exp_dir / "error_analysis.json"
    if err_path.exists():
        err = json.loads(err_path.read_text(encoding="utf-8"))
        if "correlation_gradient_error" in err:
            c = err["correlation_gradient_error"]
            if not isinstance(c, (int, float)) or not math.isfinite(float(c)):
                warns.append("error_analysis.json: correlation_gradient_error non-finite")

    _print_report(errors, warns)
    return 0 if not errors else 1


def _print_report(errors: List[str], warns: List[str]) -> None:
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
    if warns:
        print("WARNINGS:")
        for w in warns:
            print(f"  - {w}")
    if not errors and not warns:
        print("validate_results: OK (final_comparison.json checks passed)")


if __name__ == "__main__":
    raise SystemExit(main())
