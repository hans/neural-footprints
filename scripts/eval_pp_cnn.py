"""
Off-pipeline CNN diagnostic for the InverseModel input representation.

Phase 2 of ``specs/inverse_model_input_repr.md``: train an ``InverseCNN`` on the
raw three-frame stack (skipping pixel PCA), then measure whether per-dim physics
R² clears the spec's pass thresholds (mean ≥ 0.65, every observable dim ≥ 0.4).

Standalone — no Snakemake coupling. Reads ``data/scenes.npz`` if present,
otherwise generates a fresh fixture via ``scene_generator.generate_scenes`` so
the script runs on a clean checkout. Single JSON output at
``outputs/pp_cnn.json``.

Usage::

    uv run python scripts/eval_pp_cnn.py
    uv run python scripts/eval_pp_cnn.py --epochs 20 --n-scenes 200   # quick smoke
    uv run python scripts/eval_pp_cnn.py --no-encoding                # skip neural projection
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from config import IMAGE_SIZE
from analyses.encoding import pca_reduce_render, ridge_r2_per_neuron
from analyses.predictive_processing import InverseCNN
from neural_model import generate_neural_activity
from scripts.eval_pp import PHYSICS_LABELS, load_scenes_any
from scripts.load_config import load_config


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def ensure_scenes(scenes_path, n_scenes_override=None):
    """Return scenes dict; build a fresh fixture at scenes_path if missing."""
    if os.path.exists(scenes_path):
        print(f"Loading scenes from {scenes_path} ...")
        return load_scenes_any(scenes_path)

    print(f"{scenes_path} not found — generating a fresh fixture (no Snakemake) ...")
    cfg = load_config()
    n = n_scenes_override if n_scenes_override is not None else cfg['n_scenes']

    from scene_generator import generate_scenes
    from scripts.io_utils import save_scenes

    t0 = time.time()
    scenes = generate_scenes(n, cfg['random_seed'], n_timesteps=cfg['n_timesteps'])
    print(f"  generated {n} scenes in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(scenes_path) or '.', exist_ok=True)
    save_scenes(scenes, scenes_path)
    print(f"  saved → {scenes_path}")
    return load_scenes_any(scenes_path)


def build_frame_stack(scenes):
    """Stack the three rendered frames into a (N, 3, 4, H, W) uint8 tensor.

    Uses ``mid_renders``/``late_renders`` if present (3-frame fixture from
    gen_scenes_3frame.py), otherwise falls back to ``early_renders``/
    ``late_renders`` (the canonical scene_generator.generate_scenes output).
    """
    init = scenes['initial_renders']
    if 'mid_renders' in scenes and 'late_renders' in scenes:
        f1, f2 = scenes['mid_renders'], scenes['late_renders']
    elif 'early_renders' in scenes and 'late_renders' in scenes:
        f1, f2 = scenes['early_renders'], scenes['late_renders']
    else:
        raise ValueError(
            "scenes lacks 3 frames; need (initial, mid|early, late). Re-generate with "
            "config.yaml's 3-frame settings (pp_early_frame, pp_late_frame)."
        )
    n = init.shape[0]
    H = W = IMAGE_SIZE
    frames = np.stack([init, f1, f2], axis=1).astype(np.uint8)
    # (N, 3, H*W*4) → (N, 3, H, W, 4) → (N, 3, 4, H, W)
    return frames.reshape(n, 3, H, W, 4).transpose(0, 1, 4, 2, 3)


# ---------------------------------------------------------------------------
# Encoding / residual metrics
# ---------------------------------------------------------------------------

def _encoding_metrics(neural, scenes, inferred_physics, render_pca_dim,
                      n_splits=5, random_state=42):
    """Compute r2_inferred and r2_resid_inferred without the full pipeline runs.

    r2_inferred       = ridge(neural ~ inferred_physics), 5-fold CV.
    r2_resid_inferred = ridge(y_resid ~ inferred_physics) where y_resid is
                        neural minus its 5-fold OOF render-PCA prediction.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    alphas = np.logspace(-2, 6, 20)

    render_data = scenes['program_states'][:, scenes['metadata']['render_indices']]
    # PCA n_components is bounded by min(n_samples, n_features); on tiny smoke
    # fixtures n_samples is the binding constraint.
    effective_render_pca_dim = min(render_pca_dim, render_data.shape[0] - 1, render_data.shape[1])
    render_pca, _, _ = pca_reduce_render(render_data, effective_render_pca_dim,
                                         random_state=random_state)
    inf_scaled = StandardScaler().fit_transform(inferred_physics)

    r2_inferred = ridge_r2_per_neuron(inf_scaled, neural, alphas=alphas, cv=cv)

    ridge_render = RidgeCV(alphas=alphas, alpha_per_target=True)
    y_pred_render = cross_val_predict(ridge_render, render_pca, neural, cv=cv)
    y_resid = neural - y_pred_render
    var_kept = float(y_resid.var(axis=0).mean() / neural.var(axis=0).mean())

    ridge_resid = RidgeCV(alphas=alphas, alpha_per_target=True)
    y_hat = cross_val_predict(ridge_resid, inf_scaled, y_resid, cv=cv)
    ss_res = ((y_resid - y_hat) ** 2).sum(axis=0)
    ss_tot = ((y_resid - y_resid.mean(axis=0)) ** 2).sum(axis=0)
    r2_resid_inferred = 1 - ss_res / ss_tot

    return {
        'r2_inferred_mean': float(r2_inferred.mean()),
        'r2_inferred_max': float(r2_inferred.max()),
        'r2_resid_inferred_mean': float(r2_resid_inferred.mean()),
        'residual_variance_fraction': var_kept,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='data/scenes.npz')
    ap.add_argument('--n-scenes', type=int, default=None,
                    help="Override n_scenes when generating fixture (default: config n_scenes).")
    ap.add_argument('--hidden-dim', type=int, default=cfg['pp_hidden_dim'])
    ap.add_argument('--dropout', type=float, default=cfg['pp_dropout_rate'])
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-encoding', action='store_true',
                    help="Skip neural projection + encoding/residual analysis.")
    ap.add_argument('--output', default='outputs/pp_cnn.json')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== eval_pp_cnn ===")
    print(f"args: {vars(args)}")

    scenes = ensure_scenes(args.scenes, n_scenes_override=args.n_scenes)
    n = scenes['initial_renders'].shape[0]
    print(f"  n_scenes={n}")

    print("\nBuilding frame stack ...")
    t0 = time.time()
    frames = build_frame_stack(scenes)
    print(f"  shape={frames.shape}  ({frames.nbytes / 1e6:.0f} MB uint8) "
          f"built in {time.time()-t0:.1f}s")

    initial_physics = scenes['initial_physics_labels']

    print("\nFitting InverseCNN ...")
    t0 = time.time()
    cnn = InverseCNN(hidden_dim=args.hidden_dim, dropout_rate=args.dropout)
    cnn.fit(frames, initial_physics,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            val_frac=args.val_frac, patience=args.patience)
    train_time = time.time() - t0

    best_val_mse = min(h['val_loss'] for h in cnn.history_)
    n_params = sum(p.numel() for p in cnn.net_.parameters())
    print(f"  trained in {train_time:.1f}s  best val MSE={best_val_mse:.4f} "
          f"@ epoch {cnn.best_epoch_}  params={n_params:,}")

    full_per_dim_r2 = np.zeros(cnn.full_physics_dim_)
    full_per_dim_r2[cnn.valid_dims_] = cnn.per_dim_r2_

    print("\nPer-dim physics R² (val split):")
    for i, name in enumerate(PHYSICS_LABELS):
        if cnn.valid_dims_[i]:
            print(f"   {i:2d} {name:10s}  R²={full_per_dim_r2[i]:+.4f}")
    mean_r2 = float(cnn.per_dim_r2_.mean())
    min_r2 = float(cnn.per_dim_r2_.min())
    print(f"  mean over valid dims: {mean_r2:+.4f}    min dim: {min_r2:+.4f}")

    metrics = {
        'tag': 'cnn',
        'args': vars(args),
        'n_scenes': int(n),
        'n_params': int(n_params),
        'train_time_s': float(train_time),
        'val_mse': float(best_val_mse),
        'early_stop_epoch': int(cnn.best_epoch_),
        'per_dim_r2': {PHYSICS_LABELS[i]: float(full_per_dim_r2[i])
                       for i in range(cnn.full_physics_dim_)},
        'mean_valid_dim_r2': mean_r2,
        'min_valid_dim_r2': min_r2,
        'valid_dims': [PHYSICS_LABELS[i] for i in range(cnn.full_physics_dim_)
                       if cnn.valid_dims_[i]],
    }

    if not args.no_encoding:
        print("\nProjecting CNN outputs into neural activity ...")
        t0 = time.time()
        inferred_physics = cnn.predict(frames)
        layer = cfg.get('pp_neural_layer', 'h2')
        hidden_acts = cnn.extract_activations(frames, layer=layer)
        print(f"  inferred_physics={inferred_physics.shape}  "
              f"hidden({layer})={hidden_acts.shape}  in {time.time()-t0:.1f}s")

        render = scenes['program_states'][:, scenes['metadata']['render_indices']]
        neural_input = np.concatenate(
            [render, hidden_acts, inferred_physics], axis=1
        ).astype(np.float32)
        neural, _ = generate_neural_activity(
            neural_input, cfg['random_seed'],
            n_neurons=cfg['n_neurons'], noise_level=cfg['noise_level'],
        )
        print(f"  neural_activity shape={neural.shape}")

        print("\nComputing r2_inferred and r2_resid_inferred ...")
        t0 = time.time()
        enc = _encoding_metrics(neural, scenes, inferred_physics,
                                render_pca_dim=cfg['render_pca_dim'])
        print(f"  done in {time.time()-t0:.1f}s")
        print(f"  r2_inferred mean       = {enc['r2_inferred_mean']:.4f}")
        print(f"  r2_resid_inferred mean = {enc['r2_resid_inferred_mean']:.4f}")
        print(f"  residual var fraction  = {enc['residual_variance_fraction']:.4f}")
        metrics.update(enc)

    target_mean = 0.65
    target_per_dim = 0.40
    decision = 'PASS' if (mean_r2 >= target_mean and min_r2 >= target_per_dim) else 'FAIL'
    metrics['decision'] = decision
    metrics['decision_thresholds'] = {'mean_r2': target_mean, 'min_dim_r2': target_per_dim}

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved → {args.output}")

    print("\n=== Decision (specs/inverse_model_input_repr.md success criterion) ===")
    print(f"  mean R²    = {mean_r2:+.4f}  "
          f"{'≥' if mean_r2 >= target_mean else '<'} target {target_mean:+.2f}  "
          f"{'PASS' if mean_r2 >= target_mean else 'FAIL'}")
    print(f"  min dim R² = {min_r2:+.4f}  "
          f"{'≥' if min_r2 >= target_per_dim else '<'} target {target_per_dim:+.2f}  "
          f"{'PASS' if min_r2 >= target_per_dim else 'FAIL'}")
    print(f"  → {decision}")


if __name__ == '__main__':
    main()
