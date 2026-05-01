"""Quick sample-scenes plot for the 3-frame fixtures.

Renders n samples from a 3-frame .npz showing (t=0, t=mid, t=late) side-by-side.
Off-pipeline; not wired into the Snakefile.

Usage:
    uv run python scripts/plot_sample_scenes_3frame.py \
        --scenes data/scenes_expB.npz --output figures/expB_sample_scenes.pdf
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import IMAGE_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.scenes, allow_pickle=False)
    meta = json.loads(str(data['metadata_json']))
    t_mid = meta.get('t_mid', '?')
    t_late = meta.get('t_late', '?')

    rng = np.random.default_rng(args.seed)
    n_total = data['initial_renders'].shape[0]
    idx = rng.choice(n_total, size=args.n, replace=False)

    init = data['initial_renders'][idx].astype(np.uint8).reshape(args.n, IMAGE_SIZE, IMAGE_SIZE, 4)
    mid = data['mid_renders'][idx].astype(np.uint8).reshape(args.n, IMAGE_SIZE, IMAGE_SIZE, 4)
    late = data['late_renders'][idx].astype(np.uint8).reshape(args.n, IMAGE_SIZE, IMAGE_SIZE, 4)
    ip = data['initial_physics_labels'][idx]

    fig, axes = plt.subplots(args.n, 3, figsize=(6, 2 * args.n))
    titles = [f't = 0', f't = {t_mid}', f't = {t_late}']
    for r in range(args.n):
        for c, im in enumerate([init[r], mid[r], late[r]]):
            ax = axes[r, c]
            ax.imshow(im[..., :3])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        # row label: scene physics
        ax = axes[r, 0]
        v = ip[r, 7]   # linvel_x
        a = ip[r, 15]  # x_accel
        z = ip[r, 2]   # pos_z
        ax.set_ylabel(f"v={v:+.1f}\na={a:+.1f}\nz={z:.2f}",
                      fontsize=8, rotation=0, ha='right', va='center')

    fig.suptitle(os.path.basename(args.scenes), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {args.output}")


if __name__ == '__main__':
    main()
