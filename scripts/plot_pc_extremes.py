"""Visualize what neural PC1 / PC2 code by showing the scenes at their extremes.

For each principal component (PC1, PC2 of the neural-activity PCA), this samples
N scenes from the extreme-negative end, the middle, and the extreme-positive end
of that PC's score and lays their rendered frames out in a 3-row grid. The goal
is an intuitive, qualitative picture: scanning a row left-to-right and comparing
the negative/middle/positive bands shows what visual structure the PC tracks.

Off-pipeline diagnostic; not wired into the Snakefile. Reuses the PC scores the
`pca` rule already computed (`neural_pca_2d` in the pca plot-data .npz), so it
stays consistent with the canonical pipeline PCA.

Usage:
    uv run python scripts/plot_pc_extremes.py
    uv run python scripts/plot_pc_extremes.py --n 10 --pcs 1 2
    uv run python scripts/plot_pc_extremes.py \
        --plot-data data/zscore/pca_plot_data.npz --output-dir figures/zscore
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import IMAGE_SIZE
from analyses.plot_style import paper_style
from scripts.load_config import load_config

# Bands sampled along each PC axis, top row → bottom row.
BANDS = ["extreme −", "middle", "extreme +"]


def _band_indices(order, n):
    """Given scene indices sorted ascending by PC score, return (neg, mid, pos)
    index groups of size n: the lowest n, the n centred on the median, the
    highest n. Each group is itself ordered by ascending PC score.
    """
    m = len(order)
    n = min(n, m // 3)  # keep the three bands disjoint
    neg = order[:n]
    pos = order[m - n :]
    start = (m - n) // 2
    mid = order[start : start + n]
    return neg, mid, pos


def _frame_images(renders, idx):
    """Decode flat RGBA render bytes at the given scene indices to RGB images.

    Render rows in ``data/scenes.npz`` can hold more than one frame's bytes
    (e.g. 49152 = 3 × 64×64×4); we take the leading RGBA frame, exactly as the
    canonical ``plot_figures.plot_sample_scenes`` does with its ``:rgba_bytes``
    slice.
    """
    rgba_bytes = IMAGE_SIZE * IMAGE_SIZE * 4
    imgs = (
        renders[idx][:, :rgba_bytes]
        .astype(np.uint8)
        .reshape(len(idx), IMAGE_SIZE, IMAGE_SIZE, 4)[..., :3]
    )
    return imgs


def plot_pc_extremes(scores, renders, pc_number, n, output_path):
    """Render the 3×n extreme/middle grid for one PC and save it."""
    order = np.argsort(scores)  # ascending PC score
    bands = _band_indices(order, n)
    n_eff = len(bands[0])

    with paper_style():
        fig, axes = plt.subplots(
            3,
            n_eff,
            figsize=(1.05 * n_eff, 1.05 * 3 + 0.6),
            squeeze=False,
        )
        for row, (band_name, idx) in enumerate(zip(BANDS, bands)):
            imgs = _frame_images(renders, idx)
            for col in range(n_eff):
                ax = axes[row, col]
                ax.imshow(imgs[col])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{scores[idx[col]]:+.1f}", fontsize=6, pad=1.5)
            lo, hi = scores[idx[0]], scores[idx[-1]]
            axes[row, 0].set_ylabel(
                f"PC{pc_number} {band_name}\n[{lo:+.1f}, {hi:+.1f}]",
                fontsize=7,
                rotation=0,
                ha="right",
                va="center",
            )
        fig.suptitle(
            f"Scenes at the extremes and middle of neural PC{pc_number} "
            f"(N={n_eff} per band)",
            fontsize=9,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
    print(f"Saved → {output_path}")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--norm",
        default=cfg["block_norm"],
        help="block-norm variant (selects data/{norm}/pca_plot_data.npz and "
        "figures/{norm}/...). Default: config block_norm.",
    )
    ap.add_argument(
        "--plot-data",
        default=None,
        help="override path to the pca plot-data .npz holding neural_pca_2d "
        "(PC1/PC2 scores). Default: data/{norm}/pca_plot_data.npz.",
    )
    ap.add_argument(
        "--scenes",
        default=os.path.join("data", "scenes.npz"),
        help="scenes .npz holding the renders to display.",
    )
    ap.add_argument("--n", type=int, default=10, help="scenes per band (default 10).")
    ap.add_argument(
        "--pcs",
        type=int,
        nargs="+",
        default=[1, 2],
        help="which PCs to plot (1-indexed; default: 1 2).",
    )
    ap.add_argument(
        "--frame",
        choices=["initial", "early", "late"],
        default="initial",
        help="which rendered frame to display (default: initial).",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="where to write figures. Default: figures/{norm}.",
    )
    args = ap.parse_args()

    plot_data_path = args.plot_data or os.path.join(
        "data", args.norm, "pca_plot_data.npz"
    )
    output_dir = args.output_dir or os.path.join("figures", args.norm)

    pca = np.load(plot_data_path, allow_pickle=False)
    neural_pca_2d = pca["neural_pca_2d"]  # (n_scenes, >=2)

    scenes = np.load(args.scenes, allow_pickle=False)
    render_key = f"{args.frame}_renders"
    if render_key not in scenes.files:
        raise SystemExit(
            f"'{render_key}' not in {args.scenes} (have: {sorted(scenes.files)})"
        )
    renders = scenes[render_key]

    if neural_pca_2d.shape[0] != renders.shape[0]:
        raise SystemExit(
            f"row mismatch: {neural_pca_2d.shape[0]} PC scores in {plot_data_path} "
            f"vs {renders.shape[0]} scenes in {args.scenes} — out of sync."
        )
    print(
        f"  norm={args.norm}  scores={plot_data_path}  "
        f"scenes={args.scenes}  n_scenes={renders.shape[0]}"
    )

    for pc in args.pcs:
        if pc < 1 or pc > neural_pca_2d.shape[1]:
            print(f"  skipping PC{pc}: only {neural_pca_2d.shape[1]} PCs available")
            continue
        scores = neural_pca_2d[:, pc - 1]
        out = os.path.join(output_dir, f"pc{pc}_extremes_{args.frame}.pdf")
        plot_pc_extremes(scores, renders, pc, args.n, out)


if __name__ == "__main__":
    main()
