#!/usr/bin/env python3
"""Print archived experiment metrics from docs/experiments/*.json (no training or recompute)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_final_comparison_table(data: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    p = float(data["improvement_vs_persistence"]["persistence_rmse"])
    header = ["Model", "Variables", "History", "RMSE", "Beats persistence"]
    rows: List[List[str]] = [
        ["Persistence", "—", "—", repr(p), "baseline"],
    ]
    for item in data["convlstm_rmse_by_history"]:
        rmse = float(item["rmse"])
        beats = "Yes" if rmse < p else "No"
        rows.append(
            [
                "ConvLSTM",
                str(item["input_set"]),
                str(int(item["history"])),
                repr(rmse),
                beats,
            ]
        )
    return header, rows


def format_table_terminal(header: List[str], rows: List[List[str]]) -> str:
    def fmt_cell(s: str, w: int) -> str:
        s = str(s)
        return s[: w - 1] + "…" if len(s) > w else s.ljust(w)

    w = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))]
    lines = [" | ".join(header[i].ljust(w[i]) for i in range(len(header)))]
    lines.append("-+-".join("-" * w[i] for i in range(len(header))))
    for r in rows:
        lines.append(" | ".join(r[i].ljust(w[i]) for i in range(len(header))))
    return "\n".join(lines)


def format_table_markdown(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize committed docs/experiments JSON metrics.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help="Directory containing JSON summaries (default: <repo>/docs/experiments)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print markdown table (terminal-friendly)",
    )
    parser.add_argument(
        "--write-md",
        type=Path,
        default=None,
        help="Write markdown table to this path (only final_comparison grid)",
    )
    args = parser.parse_args()

    root = project_root()
    exp_dir = args.experiments_dir or (root / "docs" / "experiments")
    final_path = exp_dir / "final_comparison.json"
    if not final_path.exists():
        raise SystemExit(f"Missing {final_path}")

    data = load_json(final_path)
    header, rows = build_final_comparison_table(data)

    if args.markdown or args.write_md:
        md = format_table_markdown(header, rows)
        if args.write_md:
            args.write_md.parent.mkdir(parents=True, exist_ok=True)
            args.write_md.write_text(md + "\n", encoding="utf-8")
            print(f"Wrote {args.write_md}")
        if args.markdown:
            print(md)
    else:
        print(f"Source: {final_path.relative_to(root)}")
        print(format_table_terminal(header, rows))

    # Optional: list other JSON files without parsing deeply
    others = sorted(p for p in exp_dir.glob("*.json") if p.name != "final_comparison.json")
    if others:
        print("\nOther JSON in experiments dir:")
        for p in others:
            print(f"  {p.relative_to(root)}")


if __name__ == "__main__":
    main()
