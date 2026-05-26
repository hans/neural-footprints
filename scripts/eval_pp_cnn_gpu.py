"""
Phase-2 CNN follow-up: BatchNorm + higher learning rate + GPU + learning-curve plot.

Why this exists
---------------
The first diagnostic (``scripts/eval_pp_cnn.py``) ran 78 min on CPU, never
early-stopped, and plateaued at mean R² = +0.499 — same as the PCA50 baseline.
The "best" epoch was the *last* one (300), so val loss was still drifting down
at the wall: the result was likely under-trained, which makes the spec's
row-3 conclusion ("perceiver isn't the bottleneck") softer than it should be.

This script tightens that:

  - BatchNorm after every conv and Linear-in-head (via the new ``batch_norm``
    flag on :class:`analyses.predictive_processing.InverseCNNNet`).
  - lr default bumped 1e-3 → 3e-3 (BN should support it).
  - GPU-aware: ``--device auto`` picks ``cuda`` > ``mps`` > ``cpu``.
  - Per-epoch train + val MSE printed unbuffered, recorded in JSON, and
    rendered to ``outputs/pp_cnn_gpu_curves.pdf`` (linear + log y).
  - Same downstream metrics as v1 (``r2_inferred``, ``r2_resid_inferred``).

Designed to be ``scp``'d to a GPU box and run as-is — only depends on the rest
of this repo (config, scene_generator, neural_model, analyses/).

Usage
-----

    uv run python scripts/eval_pp_cnn_gpu.py
    # Default: 300 epochs, lr=3e-3, BN on, auto device.

    uv run python scripts/eval_pp_cnn_gpu.py --epochs 600 --lr 5e-3 --batch-size 128
    # If the v1 run is anything to go by, you may need 500+ epochs to truly
    # plateau. Bump ``--patience`` along with ``--epochs`` if you want a real
    # early stop instead of running to the wall.
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import IMAGE_SIZE, N_OBJECTS
from analyses.encoding import pca_reduce_render, ridge_r2_per_neuron
from analyses.predictive_processing import InverseCNNNet
from neural_model import generate_neural_activity
from scripts.eval_pp import PHYSICS_LABELS, load_scenes_any
from scripts.load_config import load_config

# ---------------------------------------------------------------------------
# Device, fixture, frame stack
# ---------------------------------------------------------------------------


def select_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_scenes(scenes_path, n_scenes_override=None):
    if os.path.exists(scenes_path):
        print(f"Loading scenes from {scenes_path} ...", flush=True)
        return load_scenes_any(scenes_path)

    print(
        f"{scenes_path} not found — generating fixture (no Snakemake) ...", flush=True
    )
    cfg = load_config()
    n = n_scenes_override if n_scenes_override is not None else cfg["n_scenes"]
    from scene_generator import generate_scenes
    from scripts.io_utils import save_scenes

    t0 = time.time()
    scenes = generate_scenes(n, cfg["random_seed"], n_timesteps=cfg["n_timesteps"])
    print(f"  generated {n} scenes in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(scenes_path) or ".", exist_ok=True)
    save_scenes(scenes, scenes_path)
    print(f"  saved → {scenes_path}", flush=True)
    return load_scenes_any(scenes_path)


def build_frame_stack(scenes):
    """Stack the three frames into (N, 3, 4, H, W) uint8 — same as v1."""
    init = scenes["initial_renders"]
    if "mid_renders" in scenes and "late_renders" in scenes:
        f1, f2 = scenes["mid_renders"], scenes["late_renders"]
    elif "early_renders" in scenes and "late_renders" in scenes:
        f1, f2 = scenes["early_renders"], scenes["late_renders"]
    else:
        raise ValueError("scenes lacks 3 frames")
    n = init.shape[0]
    H = W = IMAGE_SIZE
    frames = np.stack([init, f1, f2], axis=1).astype(np.uint8)
    return frames.reshape(n, 3, H, W, 4).transpose(0, 1, 4, 2, 3)


def compute_valid_dims(physics_labels):
    """Same observable-dim mask logic as InverseModel.fit / InverseCNN.fit."""
    full = physics_labels.shape[1]
    observable_offsets = list(range(0, 3)) + list(range(7, 10)) + [15]
    obs_idx = [i * 16 + j for i in range(N_OBJECTS) for j in observable_offsets]
    has_var = physics_labels.std(axis=0) > 1e-4
    obs_mask = np.zeros(full, dtype=bool)
    obs_mask[obs_idx] = True
    return obs_mask & has_var, full


# ---------------------------------------------------------------------------
# Training loop with per-epoch logging
# ---------------------------------------------------------------------------


def train_cnn(
    net,
    frames_f32,
    y,
    idx_tr,
    idx_val,
    *,
    device,
    n_epochs,
    batch_size,
    lr,
    patience,
    log_every,
):
    """Train net; return dict with per-dim R², best state, full per-epoch history.

    Tensors are pushed to ``device`` once. On a small GPU this is fine for
    n≈2000 RGBA-64 frames (~330 MB float32 across the full dataset).
    """
    X_tr = torch.tensor(frames_f32[idx_tr], dtype=torch.float32, device=device)
    y_tr = torch.tensor(y[idx_tr], dtype=torch.float32, device=device)
    X_val = torch.tensor(frames_f32[idx_val], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[idx_val], dtype=torch.float32, device=device)

    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(
        f"  training: device={device}  n_train={X_tr.shape[0]}  n_val={X_val.shape[0]}",
        flush=True,
    )
    print(f"  {'epoch':>5}  {'train':>10}  {'val':>10}  {'lr':>8}", flush=True)

    t_start = time.time()
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
        cur_lr = opt.param_groups[0]["lr"]
        history.append(
            {"epoch": epoch + 1, "train_loss": tl, "val_loss": vl, "lr": cur_lr}
        )

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"  {epoch+1:>5d}  {tl:>10.4f}  {vl:>10.4f}  {cur_lr:>8.2e}", flush=True
            )
        sch.step(vl)

        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {
                k: v.detach().clone().cpu() for k, v in net.state_dict().items()
            }
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(
                    f"  early stop @ epoch {epoch+1}, "
                    f"best val={best_val:.4f} @ epoch {best_epoch}",
                    flush=True,
                )
                break

    print(f"  total train time: {time.time()-t_start:.1f}s", flush=True)

    # Restore best checkpoint and compute per-dim R² on val.
    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    net.eval()
    with torch.no_grad():
        y_pred_val = net(X_val).detach().cpu().numpy()
    per_dim_r2 = r2_score(y[idx_val], y_pred_val, multioutput="raw_values")

    return {
        "per_dim_r2": per_dim_r2,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Inference helpers (predict + h2 extraction) without instantiating a wrapper
# ---------------------------------------------------------------------------


def _batched(arr, batch_size):
    for i in range(0, arr.shape[0], batch_size):
        yield i, arr[i : i + batch_size]


def predict_full(net, frames_f32, *, device, batch_size=256):
    net.eval()
    outs = []
    with torch.no_grad():
        for _, xb in _batched(frames_f32, batch_size):
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            outs.append(net(xb_t).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def extract_h2(net, frames_f32, *, device, batch_size=256):
    net.eval()
    acts = []
    with torch.no_grad():
        for _, xb in _batched(frames_f32, batch_size):
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            _, a = net.forward_with_activations(xb_t)
            acts.append(a["h2"].detach().cpu().numpy())
    return np.concatenate(acts, axis=0)


# ---------------------------------------------------------------------------
# r2_inferred / r2_resid_inferred  (mirrors v1)
# ---------------------------------------------------------------------------


def encoding_metrics(
    neural, scenes, inferred_physics, render_pca_dim, n_splits=5, random_state=42
):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    alphas = np.logspace(-2, 6, 20)

    render_data = scenes["program_states"][:, scenes["metadata"]["render_indices"]]
    eff_dim = min(render_pca_dim, render_data.shape[0] - 1, render_data.shape[1])
    render_pca, _, _ = pca_reduce_render(
        render_data, eff_dim, random_state=random_state
    )
    inf_scaled = StandardScaler().fit_transform(inferred_physics)

    r2_inferred = ridge_r2_per_neuron(inf_scaled, neural, alphas=alphas, cv=cv)

    ridge_r = RidgeCV(alphas=alphas, alpha_per_target=True)
    y_pred_render = cross_val_predict(ridge_r, render_pca, neural, cv=cv)
    y_resid = neural - y_pred_render
    var_kept = float(y_resid.var(axis=0).mean() / neural.var(axis=0).mean())

    ridge_res = RidgeCV(alphas=alphas, alpha_per_target=True)
    y_hat = cross_val_predict(ridge_res, inf_scaled, y_resid, cv=cv)
    ss_res = ((y_resid - y_hat) ** 2).sum(axis=0)
    ss_tot = ((y_resid - y_resid.mean(axis=0)) ** 2).sum(axis=0)
    r2_resid_inferred = 1 - ss_res / ss_tot

    return {
        "r2_inferred_mean": float(r2_inferred.mean()),
        "r2_inferred_max": float(r2_inferred.max()),
        "r2_resid_inferred_mean": float(r2_resid_inferred.mean()),
        "residual_variance_fraction": var_kept,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_curves(history, out_path, best_epoch=None, lr=None):
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]
    lrs = [h["lr"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train, lw=1, label="train")
    axes[0].plot(epochs, val, lw=1, label="val")
    if best_epoch is not None:
        axes[0].axvline(
            best_epoch, color="k", ls=":", lw=0.8, label=f"best val @ {best_epoch}"
        )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE (scaled physics)")
    axes[0].set_title("Learning curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(epochs, train, lw=1, label="train")
    axes[1].semilogy(epochs, val, lw=1, label="val")
    if best_epoch is not None:
        axes[1].axvline(best_epoch, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE (log y)")
    axes[1].set_title("Learning curves (log y)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    axes[2].semilogy(epochs, lrs, lw=1, color="C2")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("learning rate (log y)")
    axes[2].set_title("LR schedule (ReduceLROnPlateau)")
    axes[2].grid(alpha=0.3, which="both")

    suptitle = f"InverseCNN-BN  (lr={lr})" if lr is not None else "InverseCNN-BN"
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes.npz")
    ap.add_argument("--n-scenes", type=int, default=None)
    ap.add_argument(
        "--device", default="auto", help="auto (cuda > mps > cpu) | cuda | mps | cpu"
    )
    ap.add_argument("--hidden-dim", type=int, default=cfg["pp_hidden_dim"])
    ap.add_argument("--dropout", type=float, default=cfg["pp_dropout_rate"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="bumped from v1's 1e-3 (BN should support it)",
    )
    ap.add_argument(
        "--no-bn",
        action="store_true",
        help="Disable BatchNorm (replicates v1's net architecture).",
    )
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-encoding",
        action="store_true",
        help="Skip neural projection + encoding/residual analysis.",
    )
    ap.add_argument("--output", default="outputs/pp_cnn_gpu.json")
    ap.add_argument("--curves", default="outputs/pp_cnn_gpu_curves.pdf")
    ap.add_argument("--log-every", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    print(f"=== eval_pp_cnn_gpu ===  device={device}", flush=True)
    print(f"args: {vars(args)}", flush=True)

    scenes = ensure_scenes(args.scenes, n_scenes_override=args.n_scenes)
    n = scenes["initial_renders"].shape[0]
    print(f"  n_scenes={n}", flush=True)

    print("\nBuilding frame stack ...", flush=True)
    t0 = time.time()
    frames = build_frame_stack(scenes)
    print(
        f"  shape={frames.shape}  ({frames.nbytes/1e6:.0f} MB uint8)  "
        f"built in {time.time()-t0:.1f}s",
        flush=True,
    )
    frames_f32 = frames.astype(np.float32) / 255.0

    initial_physics = scenes["initial_physics_labels"]
    valid_dims, full_physics_dim = compute_valid_dims(initial_physics)
    n_observable = int(valid_dims.sum())

    phys_scaler = StandardScaler()
    y = phys_scaler.fit_transform(initial_physics[:, valid_dims])

    idx = np.arange(n)
    idx_tr, idx_val = train_test_split(idx, test_size=args.val_frac, random_state=42)

    use_bn = not args.no_bn
    print(
        f"\nBuilding net (BN={use_bn}, hidden_dim={args.hidden_dim}, "
        f"output_dim={n_observable}) ...",
        flush=True,
    )
    net = InverseCNNNet(
        n_frames=3,
        n_channels=4,
        output_dim=n_observable,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout,
        batch_norm=use_bn,
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  params: {n_params:,}", flush=True)

    fit_out = train_cnn(
        net,
        frames_f32,
        y,
        idx_tr,
        idx_val,
        device=device,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        log_every=args.log_every,
    )

    per_dim_r2 = fit_out["per_dim_r2"]
    full_per_dim_r2 = np.zeros(full_physics_dim)
    full_per_dim_r2[valid_dims] = per_dim_r2

    print("\nPer-dim physics R² (val split):", flush=True)
    for i, name in enumerate(PHYSICS_LABELS):
        if valid_dims[i]:
            print(f"   {i:2d} {name:10s}  R²={full_per_dim_r2[i]:+.4f}", flush=True)
    mean_r2 = float(per_dim_r2.mean())
    min_r2 = float(per_dim_r2.min())
    print(
        f"  mean over valid dims: {mean_r2:+.4f}    min dim: {min_r2:+.4f}", flush=True
    )

    metrics = {
        "tag": "cnn_bn_gpu",
        "args": vars(args),
        "device": str(device),
        "n_scenes": int(n),
        "n_params": int(n_params),
        "val_mse": float(fit_out["best_val_mse"]),
        "early_stop_epoch": int(fit_out["best_epoch"]),
        "per_dim_r2": {
            PHYSICS_LABELS[i]: float(full_per_dim_r2[i])
            for i in range(full_physics_dim)
        },
        "mean_valid_dim_r2": mean_r2,
        "min_valid_dim_r2": min_r2,
        "valid_dims": [
            PHYSICS_LABELS[i] for i in range(full_physics_dim) if valid_dims[i]
        ],
        "history": fit_out["history"],
    }

    if not args.no_encoding:
        print("\nProjecting CNN outputs into neural activity ...", flush=True)
        t0 = time.time()
        pred_scaled = predict_full(net, frames_f32, device=device)
        valid_preds = phys_scaler.inverse_transform(pred_scaled)
        const_values = initial_physics.mean(axis=0)
        inferred_physics = np.tile(const_values, (n, 1))
        inferred_physics[:, valid_dims] = valid_preds

        h2 = extract_h2(net, frames_f32, device=device)
        print(
            f"  inferred_physics={inferred_physics.shape}  h2={h2.shape}  "
            f"in {time.time()-t0:.1f}s",
            flush=True,
        )

        render = scenes["program_states"][:, scenes["metadata"]["render_indices"]]
        neural_input = np.concatenate([render, h2, inferred_physics], axis=1).astype(
            np.float32
        )
        neural, _ = generate_neural_activity(
            neural_input,
            cfg["random_seed"],
            n_neurons=cfg["n_neurons"],
            noise_level=cfg["noise_level"],
        )
        print(f"  neural_activity={neural.shape}", flush=True)

        print("\nComputing r2_inferred and r2_resid_inferred ...", flush=True)
        t0 = time.time()
        enc = encoding_metrics(neural, scenes, inferred_physics, cfg["render_pca_dim"])
        print(f"  done in {time.time()-t0:.1f}s", flush=True)
        for k, v in enc.items():
            print(f"    {k} = {v:+.4f}", flush=True)
        metrics.update(enc)

    target_mean, target_min = 0.65, 0.40
    decision = "PASS" if (mean_r2 >= target_mean and min_r2 >= target_min) else "FAIL"
    metrics["decision"] = decision
    metrics["decision_thresholds"] = {"mean_r2": target_mean, "min_dim_r2": target_min}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved → {args.output}", flush=True)

    if args.curves:
        os.makedirs(os.path.dirname(args.curves) or ".", exist_ok=True)
        plot_curves(
            fit_out["history"],
            args.curves,
            best_epoch=fit_out["best_epoch"],
            lr=args.lr,
        )
        print(f"Saved learning curves → {args.curves}", flush=True)

    print(
        "\n=== Decision (specs/inverse_model_input_repr.md success criterion) ===",
        flush=True,
    )
    print(
        f"  mean R²    = {mean_r2:+.4f}  vs target {target_mean:+.2f}  "
        f"{'PASS' if mean_r2 >= target_mean else 'FAIL'}",
        flush=True,
    )
    print(
        f"  min dim R² = {min_r2:+.4f}  vs target {target_min:+.2f}  "
        f"{'PASS' if min_r2 >= target_min else 'FAIL'}",
        flush=True,
    )
    print(f"  → {decision}", flush=True)


if __name__ == "__main__":
    main()
