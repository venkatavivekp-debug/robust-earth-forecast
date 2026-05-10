from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from datasets.dataset_paths import apply_dataset_version_to_args
from datasets.prism_dataset import ERA5_PRISM_Dataset
from evaluation.evaluate_model import load_checkpoint_model, normalize_input_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize intermediate U-Net feature maps for one sample.")
    parser.add_argument("--dataset-version", choices=["small", "medium"], default="medium")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--prism-path", default=None)
    parser.add_argument("--input-set", default="core4_topo", choices=["core4", "core4_elev", "core4_topo"])
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--static-covariate-path", default="data_processed/static/georgia_prism_topography.nc")
    parser.add_argument(
        "--checkpoint",
        default="results/topography_residual_stability/seed_42_residual/checkpoints/unet_core4_topo_h3_best.pt",
    )
    parser.add_argument("--sample-index", type=int, default=-1, help="Validation sample index; -1 uses first checkpoint val index.")
    parser.add_argument("--output-dir", default="results/feature_map_visualization")
    parser.add_argument("--image-out", default="docs/images/feature_map_visualization.png")
    parser.add_argument("--doc-out", default="docs/research/feature_map_findings.md")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
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


def activation_stats(tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    arr = tensor.detach().cpu().numpy()[0]
    mean_map = np.mean(arr, axis=0)
    std_map = np.std(arr, axis=0)
    metrics = {
        "mean_map_spatial_std": float(np.std(mean_map)),
        "mean_map_range": float(np.max(mean_map) - np.min(mean_map)),
        "channel_std_mean": float(np.mean(std_map)),
        "channel_std_spatial_std": float(np.std(std_map)),
    }
    return mean_map, std_map, metrics


def save_feature_plot(
    *,
    maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str,
) -> None:
    configure_plot_cache()
    import matplotlib.pyplot as plt

    names = list(maps.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(names), figsize=(3.1 * len(names), 6), constrained_layout=True)
    for col, name in enumerate(names):
        mean_map, std_map = maps[name]
        im0 = axes[0, col].imshow(mean_map, cmap="viridis")
        axes[0, col].set_title(f"{name}\nmean")
        axes[0, col].axis("off")
        fig.colorbar(im0, ax=axes[0, col], shrink=0.70)
        im1 = axes[1, col].imshow(std_map, cmap="magma")
        axes[1, col].set_title("channel std")
        axes[1, col].axis("off")
        fig.colorbar(im1, ax=axes[1, col], shrink=0.70)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_doc(
    *,
    output_path: Path,
    summary: Dict[str, Any],
    image_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enc_structured = bool(summary["encoder_spatial_structure"])
    conclusion = (
        "The encoder activations are spatially structured, so the network is not ignoring spatial inputs."
        if enc_structured
        else "The encoder activations are nearly uniform, which would point to an input/feature-learning problem."
    )
    lines = [
        "# Feature Map Findings",
        "",
        "This diagnostic visualizes mean and channel-spread activation maps from a trained",
        "skip-connected U-Net on one validation sample.",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Sample index: `{summary['sample_index']}`",
        f"- Encoder spatial structure: `{'yes' if enc_structured else 'weak/no'}`",
        "",
        conclusion,
        "",
        "Layer summaries:",
        "",
        "| Layer | mean-map spatial std | mean-map range | mean channel std |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["layers"]:
        lines.append(
            f"| {row['layer']} | {row['mean_map_spatial_std']:.6f} | "
            f"{row['mean_map_range']:.6f} | {row['channel_std_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Panel: `{image_path}`.",
            "",
            "Spatially structured activations do not mean the final residual is predictable.",
            "They only rule out a simple failure where the encoder is producing uniform maps.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_dataset_version_to_args(args)
    checkpoint_path = repo_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    static_path = repo_path(args.static_covariate_path)
    if args.input_set in {"core4_elev", "core4_topo"} and not static_path.exists():
        raise FileNotFoundError(f"Static covariate file not found: {static_path}")

    dataset = ERA5_PRISM_Dataset(
        era5_path=str(repo_path(args.era5_path)),
        prism_path=str(repo_path(args.prism_path)),
        history_length=int(args.history_length),
        input_set=str(args.input_set),
        static_covariate_path=str(static_path) if args.input_set in {"core4_elev", "core4_topo"} else None,
        verbose=False,
    )
    device = resolve_device(str(args.device))
    model, ckpt_history, input_norm, ckpt_input_set, _train_indices, val_indices = load_checkpoint_model(
        "unet", checkpoint_path, device
    )
    if int(ckpt_history) != int(args.history_length):
        raise RuntimeError(f"Checkpoint history_length={ckpt_history} does not match {args.history_length}")
    if ckpt_input_set is not None and ckpt_input_set != args.input_set:
        raise RuntimeError(f"Checkpoint input_set={ckpt_input_set} does not match {args.input_set}")
    if int(args.sample_index) >= 0:
        sample_index = int(args.sample_index)
    else:
        if not val_indices:
            raise RuntimeError("Checkpoint does not contain validation indices")
        sample_index = int(val_indices[0])

    activations: Dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            if torch.is_tensor(output):
                activations[name] = output.detach()

        return hook

    hooks = [
        model.enc1.register_forward_hook(capture("enc1")),
        model.enc2.register_forward_hook(capture("enc2")),
        model.bottleneck.register_forward_hook(capture("bottleneck")),
        model.dec2.register_forward_hook(capture("decoder_first_up")),
        model.dec1.register_forward_hook(capture("decoder_second_up")),
    ]
    try:
        x, y = dataset[sample_index]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        with torch.no_grad():
            model(normalize_input_batch(x, input_norm), target_size=(y.shape[-2], y.shape[-1]))
    finally:
        for hook in hooks:
            hook.remove()

    # Skip tensors are the encoder outputs passed to the decoder.
    activations["skip_enc1"] = activations["enc1"]
    activations["skip_enc2"] = activations["enc2"]

    ordered = [
        "enc1",
        "enc2",
        "bottleneck",
        "decoder_first_up",
        "decoder_second_up",
        "skip_enc1",
        "skip_enc2",
    ]
    plot_maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    layer_rows: List[Dict[str, Any]] = []
    for name in ordered:
        mean_map, std_map, metrics = activation_stats(activations[name])
        plot_maps[name] = (mean_map, std_map)
        row = {"layer": name}
        row.update(metrics)
        layer_rows.append(row)

    enc_std = float(np.mean([row["mean_map_spatial_std"] for row in layer_rows if row["layer"] in {"enc1", "enc2"}]))
    summary = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "dataset_version": str(args.dataset_version),
        "input_set": str(args.input_set),
        "history_length": int(args.history_length),
        "sample_index": sample_index,
        "encoder_mean_map_spatial_std_mean": enc_std,
        "encoder_spatial_structure": bool(enc_std > 1e-4),
        "layers": layer_rows,
    }

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    image_path = repo_path(args.image_out)
    save_feature_plot(maps=plot_maps, output_path=image_path, title="U-Net feature maps")
    write_doc(
        output_path=repo_path(args.doc_out),
        summary=summary,
        image_path=image_path.relative_to(PROJECT_ROOT),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
