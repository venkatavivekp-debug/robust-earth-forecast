"""Dataset version presets (paths relative to repository root unless absolute)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

VALID_VERSIONS = ("small", "medium")


def paths_for_dataset_version(version: str, *, project_root: Path | None = None) -> Tuple[str, str]:
    if version not in VALID_VERSIONS:
        raise ValueError(f"dataset_version must be one of {VALID_VERSIONS}, got {version!r}")
    root = project_root or Path(__file__).resolve().parents[1]
    cfg_path = root / "datasets" / version / "paths.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing dataset config: {cfg_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    era5 = data.get("era5_path")
    prism = data.get("prism_path")
    if not era5 or not prism:
        raise ValueError(f"{cfg_path} must define era5_path and prism_path")
    return str(era5), str(prism)


def apply_dataset_version_to_args(args: object) -> None:
    """Mutate args.era5_path / args.prism_path when unset (None)."""
    ver = getattr(args, "dataset_version", None)
    if ver is None:
        return
    era = getattr(args, "era5_path", None)
    pr = getattr(args, "prism_path", None)
    if era is not None and pr is not None:
        return
    e_path, p_path = paths_for_dataset_version(str(ver))
    if era is None:
        setattr(args, "era5_path", e_path)
    if pr is None:
        setattr(args, "prism_path", p_path)
