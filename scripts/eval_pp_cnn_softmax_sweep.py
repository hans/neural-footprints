"""
Spatial-softmax sweep.

Builds on eval_pp_cnn_simple.py: the prior 32-filter softmax model hit
mean R² = 0.756 at 34k params (= 95% of gridpool's 0.796 at 2% the params).
This script pushes that operating point further with three knobs:

  * ``n_filters``    — keypoint count at the softmax layer (32 / 64 / 128).
  * ``learned_temp`` — per-channel learnable inverse temperature so each
    keypoint can pick its own softmax sharpness.
  * ``include_variance`` — append per-channel E[x²], E[y²] (i.e. spread of
    the attention map) as 2K extra features.
  * ``head_depth`` / ``hidden_dim`` — small post-readout MLP.

Conv tower is the same shape as the v1 softmax (kernel 5/3/3, stride 2/2/1)
but the middle layer widens with ``n_filters``: 4 → 32 → max(32, K//2) → K.
Softmax operates on a 16×16 grid (image_size // 4).

Defaults run a 5-config sweep on rotation-locked scenes and write per-config
JSON to outputs/pp_cnn_softmax_sweep/<config>.json plus a combined
outputs/pp_cnn_softmax_sweep.json with side-by-side numbers vs the v1 soft
max and gridpool baselines.

Usage
-----
    uv run python scripts/eval_pp_cnn_softmax_sweep.py
    uv run python scripts/eval_pp_cnn_softmax_sweep.py --configs v2_128_temp_mlp
    uv run python scripts/eval_pp_cnn_softmax_sweep.py --device mps --epochs 200
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import IMAGE_SIZE
from models import SpatialSoftmaxV2, build_frame_stack
from scripts.eval_pp import PHYSICS_LABELS, load_scenes_any
from scripts.eval_pp_cnn_simple import compute_valid_dims, train_one


# ---------------------------------------------------------------------------
# Sweep configs
# ---------------------------------------------------------------------------

# (name, kwargs). Names are stable so output JSONs are diffable across runs.
SWEEP_CONFIGS = {
    # Re-run of the v1 design for an apples-to-apples reference under this
    # script's training loop. Should reproduce ~0.756 mean R².
    'v2_32_baseline':       dict(n_filters=32,  learned_temp=False, include_variance=False, hidden_dim=64,  head_depth=2),
    # Knob 1: temperature only.
    'v2_32_temp':           dict(n_filters=32,  learned_temp=True,  include_variance=False, hidden_dim=64,  head_depth=2),
    # Knob 2: more filters.
    'v2_64_temp':           dict(n_filters=64,  learned_temp=True,  include_variance=False, hidden_dim=128, head_depth=2),
    'v2_128_temp':          dict(n_filters=128, learned_temp=True,  include_variance=False, hidden_dim=128, head_depth=2),
    # Knob 3: wider/deeper head.
    'v2_128_temp_mlp':      dict(n_filters=128, learned_temp=True,  include_variance=False, hidden_dim=256, head_depth=3),
    # Knob 4: keypoint variance features.
    'v2_128_temp_var':      dict(n_filters=128, learned_temp=True,  include_variance=True,  hidden_dim=256, head_depth=3),
}

V1_BASELINES = {
    'pca50_mlp_3frame': 0.6553,
    'prior_cnn_gap_v1': 0.499,
    'gridpool_simple':  0.7963,         # 1.70M params, simple-CNN sweep
    'spatial_softmax_v1': 0.7564,       # 34k  params, simple-CNN sweep
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='data/scenes.npz')
    ap.add_argument('--configs', nargs='+', default=list(SWEEP_CONFIGS),
                    help="config names to run (default: all in SWEEP_CONFIGS)")
    ap.add_argument('--device', default='cpu',
                    help="cpu | cuda | mps")
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--min-epochs', type=int, default=60,
                    help="don't early-stop before this many epochs; "
                         "guards against the loss-plateau-flatlining-on-MPS pathology")
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log-every', type=int, default=25)
    ap.add_argument('--per-channel-temp', action='store_true', default=True,
                    help="learn one β per filter (default) vs one global β")
    ap.add_argument('--global-temp', dest='per_channel_temp',
                    action='store_false',
                    help="single global β instead of per-channel")
    ap.add_argument('--out-dir', default='outputs/pp_cnn_softmax_sweep')
    ap.add_argument('--summary', default='outputs/pp_cnn_softmax_sweep.json')
    args = ap.parse_args()

    unknown = set(args.configs) - set(SWEEP_CONFIGS)
    if unknown:
        ap.error(f"unknown configs: {sorted(unknown)}; "
                 f"available: {sorted(SWEEP_CONFIGS)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"=== eval_pp_cnn_softmax_sweep ===  device={device}", flush=True)
    print(f"args: {vars(args)}", flush=True)

    print(f"\nLoading scenes from {args.scenes} ...", flush=True)
    scenes = load_scenes_any(args.scenes)
    n = scenes['initial_renders'].shape[0]
    print(f"  n_scenes={n}", flush=True)

    print("Building frame stack ...", flush=True)
    frames = build_frame_stack(scenes)
    frames_f32 = frames.astype(np.float32) / 255.0
    print(f"  shape={frames.shape}  ({frames.nbytes/1e6:.0f} MB uint8)", flush=True)

    initial_physics = scenes['initial_physics_labels']
    valid_dims, full_physics_dim = compute_valid_dims(initial_physics)
    n_observable = int(valid_dims.sum())
    print(f"  observable physics dims: {n_observable}", flush=True)

    phys_scaler = StandardScaler()
    y = phys_scaler.fit_transform(initial_physics[:, valid_dims])

    idx = np.arange(n)
    idx_tr, idx_val = train_test_split(idx, test_size=args.val_frac, random_state=42)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    for name in args.configs:
        cfg = dict(SWEEP_CONFIGS[name])  # copy
        cfg.setdefault('temp_per_channel', args.per_channel_temp)

        print(f"\n=== config: {name} ===", flush=True)
        print(f"  cfg: {cfg}", flush=True)

        torch.manual_seed(args.seed)        # reset per-config so init is comparable
        net = SpatialSoftmaxV2(n_frames=3, n_channels=4, image_size=IMAGE_SIZE,
                               output_dim=n_observable, **cfg)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  params: {n_params:,}", flush=True)

        fit = train_one(net, frames_f32, y, idx_tr, idx_val,
                        device=device, n_epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr,
                        patience=args.patience, log_every=args.log_every,
                        label=name, min_epochs=args.min_epochs)

        per_dim_r2 = fit['per_dim_r2']
        full_per_dim_r2 = np.zeros(full_physics_dim)
        full_per_dim_r2[valid_dims] = per_dim_r2

        print(f"\n  per-dim R² (val):", flush=True)
        for i, lbl in enumerate(PHYSICS_LABELS):
            if valid_dims[i]:
                print(f"     {i:2d} {lbl:10s}  R²={full_per_dim_r2[i]:+.4f}", flush=True)
        mean_r2 = float(per_dim_r2.mean())
        min_r2 = float(per_dim_r2.min())
        print(f"  mean valid R² = {mean_r2:+.4f}  min = {min_r2:+.4f}", flush=True)

        rec = {
            'config_name': name,
            'config': cfg,
            'n_params': int(n_params),
            'val_mse': float(fit['best_val_mse']),
            'best_epoch': int(fit['best_epoch']),
            'train_time_s': float(fit['train_time_s']),
            'mean_valid_dim_r2': mean_r2,
            'min_valid_dim_r2': min_r2,
            'per_dim_r2': {PHYSICS_LABELS[i]: float(full_per_dim_r2[i])
                           for i in range(full_physics_dim)},
        }
        results[name] = rec

        single_path = os.path.join(args.out_dir, f"{name}.json")
        with open(single_path, 'w') as f:
            json.dump({'args': vars(args), 'device': str(device),
                       'n_scenes': int(n), 'n_observable_dims': int(n_observable),
                       'baselines_for_reference': V1_BASELINES,
                       'result': rec}, f, indent=2)
        print(f"  saved → {single_path}", flush=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===", flush=True)
    print(f"  {'config':24s}  {'params':>10s}  {'val_mse':>8s}  {'mean R²':>8s}  "
          f"{'min R²':>8s}  {'epoch':>6s}  {'time(s)':>8s}", flush=True)
    for name, r in results.items():
        print(f"  {name:24s}  {r['n_params']:>10,}  {r['val_mse']:>8.4f}  "
              f"{r['mean_valid_dim_r2']:>+8.4f}  {r['min_valid_dim_r2']:>+8.4f}  "
              f"{r['best_epoch']:>6d}  {r['train_time_s']:>8.1f}", flush=True)

    print("\n  reference baselines (rotation-locked scenes):", flush=True)
    for k, v in V1_BASELINES.items():
        print(f"    {k:24s}  mean R² = {v:.4f}", flush=True)

    summary = {
        'args': vars(args),
        'device': str(device),
        'n_scenes': int(n),
        'n_observable_dims': int(n_observable),
        'baselines_for_reference': V1_BASELINES,
        'configs_run': args.configs,
        'results': results,
    }
    os.makedirs(os.path.dirname(args.summary) or '.', exist_ok=True)
    with open(args.summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {args.summary}", flush=True)


if __name__ == '__main__':
    main()
