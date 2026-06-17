"""
Super-simplified CNN diagnostic.

The current ``InverseCNNNet`` (3 stride-2 conv layers → AdaptiveAvgPool2d(1) →
MLP head, ~291k params) sits below the PCA(50)+MLP baseline (mean R² 0.50 vs
0.66 on rotation-locked scenes). The diagnosed bottleneck: ``AdaptiveAvgPool2d(1)``
collapses spatial position, but pos_x / pos_y are exactly what we need to
recover. This script tries four progressively-more-CNN-like architectures and
reports per-dim R² for each on the same val split.

Variants
--------
1. ``linear`` — flatten 3 frames → Linear → 7. True floor; no nonlinearity, no
   conv. If this matches PCA+MLP we don't need a CNN at all.

2. ``downsample_mlp`` — AvgPool2d(8) on each frame (64→8) → flatten → MLP →
   7. No learned features, just spatial pooling + MLP. Tests "is PCA+MLP just
   coarse spatial averaging?".

3. ``gridpool`` — same conv tower as ``InverseCNNNet`` but
   ``AdaptiveAvgPool2d(grid_size)`` instead of (1). Default grid_size=4 keeps a
   4x4 spatial layout. Single-line variant of the original CNN.

4. ``spatial_softmax`` — 2 conv layers → per-channel 2D softmax → expected
   (x, y) per channel. Strong inductive bias: "tell me *where* features fire,
   not how much they fire on average." Tiny.

Usage
-----
    uv run python scripts/eval_pp_cnn_simple.py --variant all

    # Just one
    uv run python scripts/eval_pp_cnn_simple.py --variant gridpool --grid-size 4

Defaults assume rotation-locked scenes already exist at data/scenes.npz.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import IMAGE_SIZE, N_OBJECTS
from models import build_frame_stack
from scripts.eval_pp import PHYSICS_LABELS, load_scenes_any


def compute_valid_dims(physics_labels):
    full = physics_labels.shape[1]
    observable_offsets = list(range(0, 3)) + list(range(7, 10)) + [15]
    obs_idx = [i * 16 + j for i in range(N_OBJECTS) for j in observable_offsets]
    has_var = physics_labels.std(axis=0) > 1e-4
    obs_mask = np.zeros(full, dtype=bool)
    obs_mask[obs_idx] = True
    return obs_mask & has_var, full


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Flatten 3-frame stack → single Linear → output. No nonlinearity."""
    def __init__(self, n_frames, n_channels, image_size, output_dim):
        super().__init__()
        in_dim = n_frames * n_channels * image_size * image_size
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class DownsampleMLP(nn.Module):
    """AvgPool2d(8) per frame → flatten → MLP. No learned conv features."""
    def __init__(self, n_frames, n_channels, image_size, output_dim,
                 pool=8, hidden_dim=256):
        super().__init__()
        ds = image_size // pool
        feat_dim = n_frames * n_channels * ds * ds
        self.pool = nn.AvgPool2d(kernel_size=pool)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        B, F_, C, H, W = x.shape
        x = self.pool(x.reshape(B * F_, C, H, W))
        x = x.reshape(B, F_, C, x.shape[-2], x.shape[-1])
        return self.head(x)


class GridPoolCNN(nn.Module):
    """3 stride-2 conv layers → AdaptiveAvgPool2d(grid_size) → MLP head.

    grid_size=1 reproduces the original InverseCNNNet (without BN). grid_size>1
    preserves spatial layout at the cost of a wider head input.

    batch_norm=True inserts BatchNorm2d after each Conv2d and BatchNorm1d after
    each head Linear (matching InverseCNNNet's BN placement) so the
    training-regime change can be isolated independently of the GAP/grid choice.
    """
    def __init__(self, n_frames, n_channels, output_dim,
                 grid_size=4, hidden_dim=256, batch_norm=False):
        super().__init__()
        self.n_frames = n_frames
        self.conv_feat_dim = 128
        self.grid_size = grid_size
        self.batch_norm = batch_norm

        bn2 = (lambda c: nn.BatchNorm2d(c)) if batch_norm else (lambda c: nn.Identity())
        bn1 = (lambda c: nn.BatchNorm1d(c)) if batch_norm else (lambda c: nn.Identity())

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1),
            bn2(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            bn2(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, self.conv_feat_dim, kernel_size=3, stride=2, padding=1),
            bn2(self.conv_feat_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(grid_size),
            nn.Flatten(),
        )
        per_frame = self.conv_feat_dim * grid_size * grid_size
        feat_dim = per_frame * n_frames
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), bn1(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2), bn1(hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        B, F_, C, H, W = x.shape
        feats = self.conv(x.reshape(B * F_, C, H, W))
        feats = feats.reshape(B, F_ * feats.shape[-1])
        return self.head(feats)


class SpatialSoftmaxCNN(nn.Module):
    """Per-frame: stride-2 conv → conv → spatial softmax → expected (x, y) per channel.

    The spatial softmax produces 2*K features per frame (x, y of expected
    activation per channel). Convs are stride-2 then stride-1 so the softmax
    operates on a 16×16 grid — sharp enough for localization, ~16× cheaper
    on CPU than the no-stride variant.
    """
    def __init__(self, n_frames, n_channels, image_size, output_dim,
                 n_filters=32, hidden_dim=64):
        super().__init__()
        self.n_filters = n_filters

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_filters, kernel_size=3, padding=1),
        )
        # Softmax operates on 16×16 = 256 spatial positions (64 → 32 → 16).
        sm_size = image_size // 4
        ys = torch.linspace(-1.0, 1.0, sm_size)
        xs = torch.linspace(-1.0, 1.0, sm_size)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', gx.reshape(-1), persistent=False)
        self.register_buffer('grid_y', gy.reshape(-1), persistent=False)

        feat_dim = n_frames * n_filters * 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        B, F_, C, H, W = x.shape
        feats = self.conv(x.reshape(B * F_, C, H, W))  # (B*F, K, H', W')
        K, Hs, Ws = feats.shape[1], feats.shape[2], feats.shape[3]
        flat = feats.reshape(B * F_, K, Hs * Ws)
        attn = F.softmax(flat, dim=-1)  # (B*F, K, H'*W')
        ex = (attn * self.grid_x).sum(dim=-1)  # (B*F, K)
        ey = (attn * self.grid_y).sum(dim=-1)
        coords = torch.stack([ex, ey], dim=-1).reshape(B, F_ * K * 2)
        return self.head(coords)


VARIANTS = ['linear', 'downsample_mlp', 'gridpool', 'spatial_softmax']


def make_net(variant, n_frames, n_channels, image_size, output_dim, *, args):
    if variant == 'linear':
        return LinearProbe(n_frames, n_channels, image_size, output_dim)
    if variant == 'downsample_mlp':
        return DownsampleMLP(n_frames, n_channels, image_size, output_dim,
                             pool=args.downsample_pool,
                             hidden_dim=args.hidden_dim)
    if variant == 'gridpool':
        return GridPoolCNN(n_frames, n_channels, output_dim,
                           grid_size=args.grid_size,
                           hidden_dim=args.hidden_dim,
                           batch_norm=args.bn)
    if variant == 'spatial_softmax':
        return SpatialSoftmaxCNN(n_frames, n_channels, image_size, output_dim,
                                 n_filters=args.softmax_filters,
                                 hidden_dim=args.hidden_dim_small)
    raise ValueError(f"unknown variant {variant!r}")


# ---------------------------------------------------------------------------
# Training (mirrors eval_pp_cnn_gpu.train_cnn but slimmer, no BN, no curves)
# ---------------------------------------------------------------------------

def train_one(net, frames_f32, y, idx_tr, idx_val, *, device,
              n_epochs, batch_size, lr, patience, log_every, label,
              min_epochs=0):
    X_tr = torch.tensor(frames_f32[idx_tr], dtype=torch.float32, device=device)
    y_tr = torch.tensor(y[idx_tr], dtype=torch.float32, device=device)
    X_val = torch.tensor(frames_f32[idx_val], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[idx_val], dtype=torch.float32, device=device)

    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=batch_size, shuffle=True)

    best_val = float('inf')
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(f"  [{label}] training: device={device}  n_train={X_tr.shape[0]}  n_val={X_val.shape[0]}", flush=True)

    t0 = time.time()
    for epoch in range(n_epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            tl = loss_fn(net(X_tr), y_tr).item()
            vl = loss_fn(net(X_val), y_val).item()
        cur_lr = opt.param_groups[0]['lr']
        history.append({'epoch': epoch + 1, 'train_loss': tl,
                        'val_loss': vl, 'lr': cur_lr})

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  [{label}]  ep {epoch+1:>3d}  train={tl:.4f}  val={vl:.4f}  lr={cur_lr:.2e}", flush=True)
        sch.step(vl)

        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience and (epoch + 1) >= min_epochs:
                print(f"  [{label}] early stop @ epoch {epoch+1}  best val={best_val:.4f} @ {best_epoch}", flush=True)
                break

    elapsed = time.time() - t0
    print(f"  [{label}] done in {elapsed:.1f}s  best val={best_val:.4f} @ epoch {best_epoch}", flush=True)

    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    net.eval()
    with torch.no_grad():
        y_pred_val = net(X_val).detach().cpu().numpy()
    per_dim_r2 = r2_score(y[idx_val], y_pred_val, multioutput='raw_values')

    return {
        'per_dim_r2': per_dim_r2,
        'best_val_mse': best_val,
        'best_epoch': best_epoch,
        'train_time_s': elapsed,
        'history': history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='data/scenes.npz')
    ap.add_argument('--variant', default='all',
                    choices=['all'] + VARIANTS,
                    help="which architecture(s) to evaluate")
    ap.add_argument('--device', default='cpu',
                    help="cpu | cuda | mps   (BN-free models train fast on CPU)")
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--hidden-dim', type=int, default=256,
                    help="hidden dim for downsample_mlp / gridpool heads")
    ap.add_argument('--hidden-dim-small', type=int, default=64,
                    help="hidden dim for spatial_softmax head")
    ap.add_argument('--grid-size', type=int, default=4,
                    help="AdaptiveAvgPool2d output side for gridpool")
    ap.add_argument('--downsample-pool', type=int, default=8,
                    help="AvgPool2d kernel for downsample_mlp")
    ap.add_argument('--softmax-filters', type=int, default=32,
                    help="number of channels at the softmax layer")
    ap.add_argument('--bn', action='store_true',
                    help="enable BatchNorm in gridpool conv + head (off by default)")
    ap.add_argument('--frame-diffs', action='store_true',
                    help="concat (t1-t0) and (t2-t1) as extra channels per frame "
                         "(diff channels: t0=zeros, t1=t1-t0, t2=t2-t1, all in [-1,1])")
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--output', default='outputs/pp_cnn_simple.json')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"=== eval_pp_cnn_simple ===  device={device}", flush=True)
    print(f"args: {vars(args)}", flush=True)

    print(f"\nLoading scenes from {args.scenes} ...", flush=True)
    scenes = load_scenes_any(args.scenes)
    n = scenes['initial_renders'].shape[0]
    print(f"  n_scenes={n}", flush=True)

    print("Building frame stack ...", flush=True)
    frames = build_frame_stack(scenes)
    frames_f32 = frames.astype(np.float32) / 255.0
    print(f"  shape={frames.shape}  ({frames.nbytes/1e6:.0f} MB uint8)", flush=True)

    if args.frame_diffs:
        d1 = frames_f32[:, 1] - frames_f32[:, 0]
        d2 = frames_f32[:, 2] - frames_f32[:, 1]
        zeros = np.zeros_like(frames_f32[:, 0])
        diffs = np.stack([zeros, d1, d2], axis=1)
        frames_f32 = np.concatenate([frames_f32, diffs], axis=2)
        print(f"  + diff channels  shape={frames_f32.shape}  "
              f"({frames_f32.nbytes/1e6:.0f} MB float32)", flush=True)
    n_channels = frames_f32.shape[2]

    initial_physics = scenes['initial_physics_labels']
    valid_dims, full_physics_dim = compute_valid_dims(initial_physics)
    n_observable = int(valid_dims.sum())
    print(f"  observable physics dims: {n_observable}", flush=True)

    phys_scaler = StandardScaler()
    y = phys_scaler.fit_transform(initial_physics[:, valid_dims])

    idx = np.arange(n)
    idx_tr, idx_val = train_test_split(idx, test_size=args.val_frac, random_state=42)

    variants = VARIANTS if args.variant == 'all' else [args.variant]
    results = {}

    for variant in variants:
        print(f"\n=== variant: {variant} ===", flush=True)
        net = make_net(variant, n_frames=3, n_channels=n_channels,
                       image_size=IMAGE_SIZE, output_dim=n_observable,
                       args=args)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  params: {n_params:,}", flush=True)

        fit = train_one(net, frames_f32, y, idx_tr, idx_val,
                        device=device, n_epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr,
                        patience=args.patience, log_every=args.log_every,
                        label=variant)

        per_dim_r2 = fit['per_dim_r2']
        full_per_dim_r2 = np.zeros(full_physics_dim)
        full_per_dim_r2[valid_dims] = per_dim_r2

        print(f"\n  per-dim R² (val):", flush=True)
        for i, name in enumerate(PHYSICS_LABELS):
            if valid_dims[i]:
                print(f"     {i:2d} {name:10s}  R²={full_per_dim_r2[i]:+.4f}", flush=True)
        mean_r2 = float(per_dim_r2.mean())
        min_r2 = float(per_dim_r2.min())
        print(f"  mean valid R² = {mean_r2:+.4f}  min = {min_r2:+.4f}", flush=True)

        results[variant] = {
            'n_params': int(n_params),
            'val_mse': float(fit['best_val_mse']),
            'best_epoch': int(fit['best_epoch']),
            'train_time_s': float(fit['train_time_s']),
            'mean_valid_dim_r2': mean_r2,
            'min_valid_dim_r2': min_r2,
            'per_dim_r2': {PHYSICS_LABELS[i]: float(full_per_dim_r2[i])
                           for i in range(full_physics_dim)},
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===", flush=True)
    print(f"  {'variant':20s}  {'params':>10s}  {'val_mse':>8s}  {'mean R²':>8s}  {'min R²':>8s}  {'epoch':>6s}  {'time(s)':>8s}", flush=True)
    for v, r in results.items():
        print(f"  {v:20s}  {r['n_params']:>10,}  {r['val_mse']:>8.4f}  "
              f"{r['mean_valid_dim_r2']:>+8.4f}  {r['min_valid_dim_r2']:>+8.4f}  "
              f"{r['best_epoch']:>6d}  {r['train_time_s']:>8.1f}", flush=True)
    print(f"\n  PCA(50)+MLP baseline: mean R² = 0.6553 (val_mse=0.347)", flush=True)
    print(f"  prior CNN (291k params, GAP): mean R² = 0.499", flush=True)

    out = {
        'args': vars(args),
        'device': str(device),
        'n_scenes': int(n),
        'n_observable_dims': int(n_observable),
        'baselines_for_reference': {
            'pca50_mlp_3frame': 0.6553,
            'prior_cnn_gap_v1': 0.499,
        },
        'variants': results,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.output}", flush=True)


if __name__ == '__main__':
    main()
