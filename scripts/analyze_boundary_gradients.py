from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare boundary gradient preservation across decoder outputs.")
    parser.add_argument("--bilinear-dir", default="results/pixelshuffle_overfit/bilinear/residual")
    parser.add_argument("--pixelshuffle-dir", default="results/pixelshuffle_overfit/pixelshuffle/residual")
    parser.add_argument("--border-fraction", type=float, default=0.10)
    parser.add_argument("--output-dir", default="results/boundary_gradient_comparison")
    parser.add_argument("--image-out", default="docs/images/boundary_gradient_comparison.png")
    parser.add_argument("--doc-out", default="docs/research/boundary_gradient_findings.md")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_plot_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def conv2d_reflect(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    padded = np.pad(data, ((1, 1), (1, 1)), mode="reflect")
    out = np.zeros_like(data, dtype=np.float64)
    h, w = data.shape
    for iy in range(3):
        for ix in range(3):
            out += kernel[iy, ix] * padded[iy : iy + h, ix : ix + w]
    return out


def sobel_gradient(arr: np.ndarray) -> np.ndarray:
    kx = np.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    ky = np.asarray([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
    gx = conv2d_reflect(arr, kx)
    gy = conv2d_reflect(arr, ky)
    return np.sqrt(gx**2 + gy**2)


def border_mask(shape: Tuple[int, int], fraction: float) -> np.ndarray:
    h, w = shape
    width = max(1, int(round(min(h, w) * fraction)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return 0.0
    r = float(np.corrcoef(aa, bb)[0, 1])
    return r if np.isfinite(r) else 0.0


def load_arrays(bilinear_dir: Path, pixelshuffle_dir: Path) -> Dict[str, np.ndarray]:
    target = np.load(bilinear_dir / "target.npy")
    era5 = np.load(bilinear_dir / "era5_bilinear.npy")
    bilinear = np.load(bilinear_dir / "prediction.npy")
    pixelshuffle = np.load(pixelshuffle_dir / "prediction.npy")
    if target.shape != era5.shape or target.shape != bilinear.shape or target.shape != pixelshuffle.shape:
        raise RuntimeError("Gradient comparison arrays do not share the same shape")
    return {
        "PRISM target": target,
        "ERA5 bilinear": era5,
        "U-Net bilinear": bilinear,
        "U-Net PixelShuffle": pixelshuffle,
    }


def metric_rows(gradients: Dict[str, np.ndarray], mask: np.ndarray) -> List[Dict[str, Any]]:
    target_grad = gradients["PRISM target"]
    rows: List[Dict[str, Any]] = []
    for name, grad in gradients.items():
        border_mean = float(np.mean(grad[mask]))
        interior_mean = float(np.mean(grad[~mask]))
        rows.append(
            {
                "method": name,
                "border_gradient_mean": border_mean,
                "interior_gradient_mean": interior_mean,
                "border_interior_gradient_ratio": border_mean / max(interior_mean, 1e-8),
                "gradient_pearson_r_vs_prism": 1.0 if name == "PRISM target" else pearson_r(grad, target_grad),
            }
        )
    return rows


def save_plot(
    *,
    output_path: Path,
    gradients: Dict[str, np.ndarray],
) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    methods = list(gradients.keys())
    target_grad = gradients["PRISM target"]
    vmax = float(max(np.percentile(grad, 99) for grad in gradients.values()))
    diff_vmax = float(max(np.percentile(np.abs(target_grad - grad), 99) for name, grad in gradients.items() if name != "PRISM target"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 7), constrained_layout=True)
    for col, name in enumerate(methods):
        im = axes[0, col].imshow(gradients[name], cmap="magma", vmin=0.0, vmax=vmax)
        axes[0, col].set_title(name)
        axes[0, col].axis("off")
        fig.colorbar(im, ax=axes[0, col], shrink=0.72)
        if name == "PRISM target":
            diff = np.zeros_like(target_grad)
        else:
            diff = target_grad - gradients[name]
        im2 = axes[1, col].imshow(diff, cmap="coolwarm", vmin=-diff_vmax, vmax=diff_vmax)
        axes[1, col].set_title("PRISM grad - method grad")
        axes[1, col].axis("off")
        fig.colorbar(im2, ax=axes[1, col], shrink=0.72)
    fig.suptitle("Boundary gradient comparison")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_doc(output_path: Path, rows: List[Dict[str, Any]], image_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_by_method = {row["method"]: row for row in rows}
    bilinear_r = row_by_method["U-Net bilinear"]["gradient_pearson_r_vs_prism"]
    pixel_r = row_by_method["U-Net PixelShuffle"]["gradient_pearson_r_vs_prism"]
    read = (
        "PixelShuffle improves gradient-map alignment over the bilinear decoder."
        if pixel_r > bilinear_r
        else "PixelShuffle does not improve gradient-map alignment over the bilinear decoder in this overfit comparison."
    )
    lines = [
        "# Boundary Gradient Findings",
        "",
        "This diagnostic compares Sobel gradient magnitude maps for the fixed overfit sample.",
        "It uses saved prediction arrays from the bilinear and PixelShuffle single-sample runs.",
        "",
        "| Method | Border grad mean | Interior grad mean | Border/interior | Pearson r vs PRISM grad |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['border_gradient_mean']:.4f} | {row['interior_gradient_mean']:.4f} | "
            f"{row['border_interior_gradient_ratio']:.4f} | {row['gradient_pearson_r_vs_prism']:.4f} |"
        )
    lines.extend(
        [
            "",
            read,
            "",
            f"Figure: `{image_path}`.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bilinear_dir = repo_path(args.bilinear_dir)
    pixelshuffle_dir = repo_path(args.pixelshuffle_dir)
    arrays = load_arrays(bilinear_dir, pixelshuffle_dir)
    gradients = {name: sobel_gradient(arr) for name, arr in arrays.items()}
    mask = border_mask(arrays["PRISM target"].shape, float(args.border_fraction))
    rows = metric_rows(gradients, mask)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "boundary_gradient_metrics.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    image_path = repo_path(args.image_out)
    save_plot(output_path=image_path, gradients=gradients)
    write_doc(repo_path(args.doc_out), rows, image_path.relative_to(PROJECT_ROOT))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
