"""
Inverse-model architecture sweep for multi-object diverse MuJoCo scenes.

Evaluates 4 configs of SpatialSoftmaxV2 (and variants) on the richer scenes
produced by the scene-diversity-mujoco branch, which adds background spheres,
a backdrop wall, a cylinder shape, per-object specular, and wider camera jitter.

Configs
-------
  baseline       — current best (v2_128_temp_mlp) on RGBA only
  big_filters    — 256 filters instead of 128, RGBA only
  depth_aug      — 128 filters + depth channel (RGBA + depth = 5 ch/frame)
  temporal_delta — 128 filters + keypoint-velocity deltas via
                   SpatialSoftmaxTemporalDelta

Usage
-----
    uv run python scripts/eval_inverse_diversity.py --device auto
    uv run python scripts/eval_inverse_diversity.py --device auto --n-scenes 500 --epochs 200
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
from models import (SpatialSoftmaxV2, SpatialSoftmaxTemporalDelta,
                    SpatialSoftmaxDepthGated, SpatialSoftmaxDepthGatedTemporalDelta,
                    build_frame_stack)
from scripts.eval_pp import PHYSICS_LABELS, load_scenes_any
from scripts.eval_pp_cnn_simple import compute_valid_dims, train_one
from scripts.load_config import load_config


def select_device(arg):
    """Auto-select best available device.

    NOTE: MPS (Apple Silicon) is intentionally skipped. SpatialSoftmaxV2 uses
    strided Conv2d layers whose backward pass has a confirmed MPS bug that
    produces ~1000x wrong gradients, silently preventing training convergence.
    CUDA is preferred over CPU when available; otherwise CPU is used.
    """
    if arg != 'auto':
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def ensure_scenes(scenes_path, n_scenes_override=None):
    """Load scenes from path, or generate fresh ones if missing."""
    if os.path.exists(scenes_path):
        print(f"Loading scenes from {scenes_path} ...", flush=True)
        return load_scenes_any(scenes_path)

    print(f"{scenes_path} not found — generating scenes ...", flush=True)
    cfg = load_config()
    n = n_scenes_override if n_scenes_override is not None else cfg['n_scenes']
    from scene_generator import generate_scenes
    from scripts.io_utils import save_scenes

    t0 = time.time()
    scenes = generate_scenes(n, cfg['random_seed'], n_timesteps=cfg['n_timesteps'])
    print(f"  generated {n} scenes in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(scenes_path) or '.', exist_ok=True)
    save_scenes(scenes, scenes_path)
    print(f"  saved → {scenes_path}", flush=True)
    return load_scenes_any(scenes_path)


# ---------------------------------------------------------------------------
# Frame stack helpers
# ---------------------------------------------------------------------------

def build_frame_stack_with_depth(scenes):
    """Stack 3 frames of RGBA + depth into (N, 3, 5, H, W) float32.

    Each per-frame render in initial_renders / early_renders / late_renders is
    stored as a flat uint8-cast float32 vector with layout:

        [RGBA_bytes (4*H*W uint8 values) | depth_bytes (H*W*4 uint8, raw float32) | seg_bytes]

    RGBA values are 0-255 (already uint8-like).  Depth bytes need to be
    interpreted as float32: take the [4*H*W : 8*H*W] slice, cast to uint8,
    then view as float32.

    Returns float32 array (N, 3, 5, H, W) where channel ordering is:
        channels 0-3: RGBA / 255.0
        channel 4:    depth (MuJoCo near-normalised [0, 1])
    """
    init = scenes['initial_renders']
    if 'mid_renders' in scenes and 'late_renders' in scenes:
        f1, f2 = scenes['mid_renders'], scenes['late_renders']
    elif 'early_renders' in scenes and 'late_renders' in scenes:
        f1, f2 = scenes['early_renders'], scenes['late_renders']
    else:
        raise ValueError("scenes lacks 3 frames")

    N = init.shape[0]
    H = W = IMAGE_SIZE
    rgba_n = H * W * 4   # uint8-valued floats per frame
    depth_n_bytes = H * W * 4  # raw bytes of H*W float32 depth values

    def extract_rgba_and_depth_raw(frame_data):
        """frame_data: (N, render_bytes_per_frame) float32.
        Returns (rgba (N,4,H,W) float32 in [0,1], depth_raw (N,H*W) float32 in meters).
        """
        rgba = frame_data[:, :rgba_n].astype(np.float32) / 255.0   # (N, rgba_n)
        rgba = rgba.reshape(N, H, W, 4).transpose(0, 3, 1, 2)      # (N, 4, H, W)

        # Depth: bytes [rgba_n : rgba_n + depth_n_bytes] need to be re-cast as float32.
        # The render vector stores raw byte buffers cast to uint8 then to float32,
        # so we reverse: cast back to uint8, then view as float32.
        depth_raw = frame_data[:, rgba_n:rgba_n + depth_n_bytes]    # (N, H*W*4)
        depth_uint8 = depth_raw.astype(np.uint8)                    # (N, H*W*4) uint8
        depth_f32 = depth_uint8.view(np.float32)                    # (N, H*W) float32 in meters
        return rgba, depth_f32

    rgba_init, d_init = extract_rgba_and_depth_raw(init)
    rgba_mid, d_mid = extract_rgba_and_depth_raw(f1)
    rgba_late, d_late = extract_rgba_and_depth_raw(f2)

    # Normalize depth globally across all scenes and frames so values are ~[0,1]
    all_depth = np.concatenate([d_init, d_mid, d_late], axis=0)
    d_min = all_depth.min()
    d_max = all_depth.max()
    d_range = max(d_max - d_min, 1e-6)

    def norm_depth(d_raw):
        return ((d_raw - d_min) / d_range).reshape(N, 1, H, W)

    def make_5ch(rgba, d_raw):
        return np.concatenate([rgba, norm_depth(d_raw)], axis=1)    # (N, 5, H, W)

    f_init = make_5ch(rgba_init, d_init)
    f_mid = make_5ch(rgba_mid, d_mid)
    f_late = make_5ch(rgba_late, d_late)

    # Stack: (N, 3, 5, H, W)
    return np.stack([f_init, f_mid, f_late], axis=1)


# ---------------------------------------------------------------------------
# Sweep configs
# ---------------------------------------------------------------------------

SWEEP_CONFIGS = {
    'baseline': {
        'model_cls': 'SpatialSoftmaxV2',
        'n_channels': 4,
        'kwargs': dict(n_filters=128, learned_temp=True, hidden_dim=256,
                       head_depth=3, dropout_rate=0.05),
    },
    'big_filters': {
        'model_cls': 'SpatialSoftmaxV2',
        'n_channels': 4,
        'kwargs': dict(n_filters=256, learned_temp=True, hidden_dim=256,
                       head_depth=3, dropout_rate=0.05),
    },
    'depth_aug': {
        'model_cls': 'SpatialSoftmaxV2',
        'n_channels': 5,
        'kwargs': dict(n_filters=128, learned_temp=True, hidden_dim=256,
                       head_depth=3, dropout_rate=0.05),
    },
    'temporal_delta': {
        'model_cls': 'SpatialSoftmaxTemporalDelta',
        'n_channels': 4,
        'kwargs': dict(n_filters=128, learned_temp=True, hidden_dim=256,
                       head_depth=3, dropout_rate=0.05),
    },
    'depth_gated': {
        'model_cls': 'SpatialSoftmaxDepthGated',
        'n_channels': 5,
        'kwargs': dict(n_filters=128, learned_temp=True, hidden_dim=256,
                       head_depth=3, dropout_rate=0.05, depth_gamma_init=2.0),
    },
    'depth_gated_strong': {
        'model_cls': 'SpatialSoftmaxDepthGated',
        'n_channels': 5,
        'kwargs': dict(n_filters=256, learned_temp=True, hidden_dim=256,
                       head_depth=3, dropout_rate=0.05, depth_gamma_init=4.0),
    },
    'depth_gated_temporal': {
        'model_cls': 'SpatialSoftmaxDepthGatedTemporalDelta',
        'n_channels': 5,
        'kwargs': dict(n_filters=128, learned_temp=True, hidden_dim=256,
                       head_depth=3, dropout_rate=0.05, depth_gamma_init=2.0),
    },
    'depth_gated_edepth': {
        'model_cls': 'SpatialSoftmaxDepthGated',
        'n_channels': 5,
        'kwargs': dict(n_filters=128, learned_temp=True, hidden_dim=256,
                       head_depth=3, dropout_rate=0.05, depth_gamma_init=2.0,
                       include_variance=True, include_edepth=True),
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Inverse-model diversity sweep for multi-object MuJoCo scenes")
    ap.add_argument('--scenes', default='data/scenes.npz',
                    help="path to scenes.npz (generated if missing)")
    ap.add_argument('--n-scenes', type=int, default=None,
                    help="override n_scenes when generating a fresh fixture")
    ap.add_argument('--device', default='auto',
                    help="auto | cuda | mps | cpu")
    ap.add_argument('--configs', nargs='+', default=list(SWEEP_CONFIGS),
                    help="config names to run (default: all)")
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--min-epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log-every', type=int, default=25)
    ap.add_argument('--output', default='outputs/inverse_diversity_sweep.json')
    args = ap.parse_args()

    unknown = set(args.configs) - set(SWEEP_CONFIGS)
    if unknown:
        ap.error(f"unknown configs: {sorted(unknown)}; available: {sorted(SWEEP_CONFIGS)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    print(f"=== eval_inverse_diversity ===  device={device}", flush=True)
    print(f"args: {vars(args)}", flush=True)

    # ------------------------------------------------------------------
    # Load / generate scenes
    # ------------------------------------------------------------------
    scenes = ensure_scenes(args.scenes, n_scenes_override=args.n_scenes)
    n = scenes['initial_renders'].shape[0]
    print(f"  n_scenes={n}", flush=True)

    # Validate that scenes have the MuJoCo multi-frame format
    render_per_frame = scenes['initial_renders'].shape[1]
    expected_rgba_only = IMAGE_SIZE * IMAGE_SIZE * 4
    expected_full = IMAGE_SIZE * IMAGE_SIZE * 12  # RGBA + depth + seg
    if render_per_frame == expected_rgba_only:
        print(f"  WARNING: scenes appear to be old PyBullet RGBA-only format "
              f"(render_per_frame={render_per_frame}). Depth-aug will not work correctly.",
              flush=True)
    elif render_per_frame == expected_full:
        print(f"  scenes: MuJoCo format with depth+seg (render_per_frame={render_per_frame})",
              flush=True)
    else:
        print(f"  scenes: render_per_frame={render_per_frame} "
              f"(expected {expected_full} for full MuJoCo)", flush=True)

    has_depth = (render_per_frame == expected_full)

    # Build frame stacks
    print("Building RGBA frame stack ...", flush=True)
    frames_rgba = build_frame_stack(scenes)               # (N, 3, 4, H, W) uint8
    frames_rgba_f32 = frames_rgba.astype(np.float32) / 255.0

    if has_depth:
        print("Building RGBA+depth frame stack ...", flush=True)
        frames_depth = build_frame_stack_with_depth(scenes)  # (N, 3, 5, H, W) float32
        # depth is already float32 in correct range
        print(f"  depth shape={frames_depth.shape}  dtype={frames_depth.dtype}",
              flush=True)
        # Quick sanity: depth channel should be in [0, 1]
        depth_vals = frames_depth[:, :, 4, :, :]
        print(f"  depth stats: min={depth_vals.min():.4f}  max={depth_vals.max():.4f}  "
              f"mean={depth_vals.mean():.4f}", flush=True)
    else:
        frames_depth = None
        print("  (skipping depth stack — old format)", flush=True)

    # Physics labels
    initial_physics = scenes['initial_physics_labels']
    valid_dims, full_physics_dim = compute_valid_dims(initial_physics)
    n_observable = int(valid_dims.sum())
    print(f"  observable physics dims: {n_observable}", flush=True)
    print(f"  valid dims: {[PHYSICS_LABELS[i] for i in range(full_physics_dim) if valid_dims[i]]}",
          flush=True)

    phys_scaler = StandardScaler()
    y = phys_scaler.fit_transform(initial_physics[:, valid_dims])

    idx = np.arange(n)
    if args.val_frac <= 0.0:
        idx_tr, idx_val = idx, idx  # overfit mode: validate on train set
    else:
        idx_tr, idx_val = train_test_split(idx, test_size=args.val_frac, random_state=42)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    results = {}

    for name in args.configs:
        cfg = SWEEP_CONFIGS[name]
        model_cls_name = cfg['model_cls']
        n_channels = cfg['n_channels']
        kwargs = dict(cfg['kwargs'])

        if n_channels == 5 and frames_depth is None:
            print(f"\n=== config: {name} SKIPPED (no depth data) ===", flush=True)
            results[name] = {'skipped': True, 'reason': 'no_depth_data'}
            continue

        print(f"\n=== config: {name} ===", flush=True)
        print(f"  model_cls={model_cls_name}  n_channels={n_channels}  kwargs={kwargs}",
              flush=True)

        # Select frame data
        frames_f32 = frames_depth if n_channels == 5 else frames_rgba_f32

        torch.manual_seed(args.seed)

        _MODEL_REGISTRY = {
            'SpatialSoftmaxV2': SpatialSoftmaxV2,
            'SpatialSoftmaxTemporalDelta': SpatialSoftmaxTemporalDelta,
            'SpatialSoftmaxDepthGated': SpatialSoftmaxDepthGated,
            'SpatialSoftmaxDepthGatedTemporalDelta': SpatialSoftmaxDepthGatedTemporalDelta,
        }
        if model_cls_name not in _MODEL_REGISTRY:
            raise ValueError(f"unknown model_cls: {model_cls_name}; "
                             f"available: {sorted(_MODEL_REGISTRY)}")
        model_cls = _MODEL_REGISTRY[model_cls_name]

        net = model_cls(
            n_frames=3, n_channels=n_channels, image_size=IMAGE_SIZE,
            output_dim=n_observable, **kwargs
        )
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
            'model_cls': model_cls_name,
            'n_channels': n_channels,
            'config': kwargs,
            'n_params': int(n_params),
            'val_mse': float(fit['best_val_mse']),
            'best_epoch': int(fit['best_epoch']),
            'train_time_s': float(fit.get('train_time_s', 0.0)),
            'mean_valid_dim_r2': mean_r2,
            'min_valid_dim_r2': min_r2,
            'per_dim_r2': {PHYSICS_LABELS[i]: float(full_per_dim_r2[i])
                           for i in range(full_physics_dim)},
        }
        results[name] = rec

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n=== Summary ===", flush=True)
    print(f"  {'config':20s}  {'model_cls':26s}  {'params':>10s}  "
          f"{'mean R²':>8s}  {'min R²':>8s}  {'epoch':>6s}", flush=True)
    for name, r in results.items():
        if r.get('skipped'):
            print(f"  {name:20s}  SKIPPED", flush=True)
            continue
        print(f"  {name:20s}  {r['model_cls']:26s}  {r['n_params']:>10,}  "
              f"{r['mean_valid_dim_r2']:>+8.4f}  {r['min_valid_dim_r2']:>+8.4f}  "
              f"{r['best_epoch']:>6d}", flush=True)

    print(f"\n  Reference: PyBullet single-obj best = 0.828 (v2_128_temp_mlp)", flush=True)

    # Check if any variant beats baseline by >= 0.05
    baseline_r2 = results.get('baseline', {}).get('mean_valid_dim_r2', None)
    if baseline_r2 is not None:
        for name, r in results.items():
            if name == 'baseline' or r.get('skipped'):
                continue
            delta = r.get('mean_valid_dim_r2', -9) - baseline_r2
            if delta >= 0.05:
                print(f"\n  *** {name} beats baseline by {delta:+.4f} — "
                      f"RECOMMEND for deployment ***", flush=True)

    summary = {
        'args': vars(args),
        'device': str(device),
        'n_scenes': int(n),
        'n_observable_dims': int(n_observable),
        'has_depth_channel': bool(has_depth),
        'valid_dim_labels': [PHYSICS_LABELS[i] for i in range(full_physics_dim)
                             if valid_dims[i]],
        'configs_run': args.configs,
        'results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {args.output}", flush=True)


if __name__ == '__main__':
    main()
