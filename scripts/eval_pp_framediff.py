"""Phase 1 of specs/inverse_model_input_repr.md: frame-difference features.

Replaces the three concatenated whitened PCAs of raw frames with
``concat(pca(t0), pca(t_early) − pca(t0), pca(t_late) − pca(t_early))`` —
PCA fit on t=0 only, applied to all three frames so subtraction lives in a
shared basis. Same input width as the baseline (3 × pixel_pca_dim), same
InverseMLPNet, same training schedule, same split. The only thing that
changes is the linear pre-mix.

Reports per-dim val R², mean R², inferred-physics encoding R²,
residualized inferred-physics R², val MSE, early-stop epoch, wall-clock.

Success criterion (from spec): mean R² ≥ ~0.65 with no individual dim
below ~0.4. If hit → wire `build_pp_diff_features` into config and skip
Phase 2 (CNN). Otherwise the per-dim profile points to the next move.

Standalone: builds its own scenes fixture if `data/scenes.npz` is absent.

    uv run python scripts/eval_pp_framediff.py
    uv run python scripts/eval_pp_framediff.py --skip-residual  # faster smoke

Output: outputs/pp_framediff.json + summary table to stdout.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from scripts.load_config import load_config
from scripts.io_utils import load_scenes, save_scenes
from analyses.predictive_processing import (
    InverseModel,
    build_pp_features,
    build_pp_diff_features,
)
from analyses.pp_io import extract_activations
from analyses.encoding import run_encoding_analysis
from analyses.residual import run_residual_analysis
from neural_model import generate_neural_activity


# Same observable-dim mapping as scripts/sweep_pixel_pca_dim.py.
SPEC_DIM_NAMES = {
    0:  'pos_x',
    1:  'pos_y',
    2:  'pos_z',
    7:  'vel_x',
    15: 'x_accel',
}


def _ensure_scenes(path, cfg):
    """Return loaded scenes; build a fresh fixture if `path` is missing."""
    if os.path.exists(path):
        print(f"  loading scenes from {path}")
        return load_scenes(path)
    print(f"  {path} not found — generating a fresh fixture (this takes a minute or two)...")
    from scene_generator import generate_scenes
    n = min(cfg.get('n_scenes', 2000), 2000)
    scenes = generate_scenes(n, cfg['random_seed'], n_timesteps=cfg['n_timesteps'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_scenes(scenes, path)
    print(f"  saved fresh fixture → {path}")
    return scenes


def _gen_neural(scenes, cfg):
    """Generate the fixed neural target used for encoding/residual analyses.

    Mirrors scripts/sweep_pixel_pca_dim.py: prefer the cached PP activations
    written by the main pipeline so the neural population is the same one the
    paper analyses run against. Otherwise, train a default-config InverseModel
    on the *baseline* (non-diff) features so the neural target reflects the
    pre-fix representation — that's the apples-to-apples comparator for asking
    whether the diff basis improves inferred-physics quality.
    """
    render_indices = scenes['metadata']['render_indices']
    render = scenes['program_states'][:, render_indices]

    pp_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'pp_activations.npz')
    if os.path.exists(pp_path):
        print(f"  using cached PP activations from {pp_path}")
        pp = np.load(pp_path)
        hidden_acts = pp['hidden_acts']
        inferred_physics = pp['inferred_physics']
    else:
        print(f"  no cached PP activations — training default-config InverseModel "
              f"(baseline features) for neural projection...")
        feats = build_pp_features(scenes, pixel_pca_dim=cfg['pp_pixel_pca_dim'])
        inv_default = InverseModel()
        inv_default.fit(feats['pixel_pca_concat'], scenes['initial_physics_labels'])
        layer = cfg.get('pp_neural_layer', 'h2')
        hidden_acts = extract_activations(inv_default, feats['pixel_pca_concat'], layer=layer)
        inferred_physics = inv_default.predict(feats['pixel_pca_concat'])

    neural_input = np.concatenate(
        [render, hidden_acts, inferred_physics], axis=1
    ).astype(np.float32)

    neural, neural_meta = generate_neural_activity(
        neural_input, cfg['random_seed'],
        n_neurons=cfg['n_neurons'], noise_level=cfg['noise_level'],
    )
    return neural, neural_meta


def _eval_one(scenes, neural, neural_meta, *,
              feature_builder, label, cfg, run_residual=True,
              pixel_pca_dim=50):
    """Train one InverseModel on `feature_builder(scenes)` and collect metrics."""
    print(f"\n{'=' * 60}")
    print(f"variant = {label}")
    print('=' * 60)

    initial_physics = scenes['initial_physics_labels']

    print(f"  building features (pixel_pca_dim={pixel_pca_dim})...")
    feats = feature_builder(scenes, pixel_pca_dim=pixel_pca_dim)
    X = feats['pixel_pca_concat']
    print(f"  feature shape = {X.shape}")

    print(f"  training InverseModel (300 epochs, lr 1e-3, dropout 0.05, patience 50)...")
    t0 = time.time()
    inv = InverseModel()
    inv.fit(
        X, initial_physics,
        n_epochs=300, batch_size=64, lr=1e-3, val_frac=0.15, patience=50,
    )
    train_time_s = time.time() - t0
    print(f"  trained in {train_time_s:.1f}s "
          f"(stopped at epoch {inv.stopped_epoch_}, best val MSE={inv.best_val_loss_:.4f})")

    full_per_dim_r2 = np.zeros(inv.full_physics_dim_)
    full_per_dim_r2[inv.valid_dims_] = inv.per_dim_r2_
    per_dim_r2 = {name: float(full_per_dim_r2[idx]) for idx, name in SPEC_DIM_NAMES.items()}
    mean_r2 = float(inv.per_dim_r2_.mean())

    print(f"\n  per-dim R² (val):")
    for name, r2 in per_dim_r2.items():
        print(f"    {name:8s}  {r2:+.4f}")
    print(f"    mean      {mean_r2:+.4f}")

    inferred_physics = inv.predict(X)

    print(f"\n  encoding analysis: neural ~ inferred_physics ...")
    enc = run_encoding_analysis(
        neural, scenes, neural_meta,
        render_pca_dim=cfg['render_pca_dim'],
        inferred_physics=inferred_physics,
    )
    r2_inferred = float(np.mean(enc['r2_inferred']))
    r2_inferred_combined = float(np.mean(enc['r2_inferred_combined']))
    delta_r2_inferred = float(np.mean(enc['delta_r2_inferred']))
    print(f"    r2_inferred           = {r2_inferred:.4f}")
    print(f"    r2_inferred_combined  = {r2_inferred_combined:.4f}")
    print(f"    delta_r2_inferred     = {delta_r2_inferred:+.6f}")

    r2_resid_inferred = None
    if run_residual:
        print(f"\n  residual analysis: render-residualized neural ~ inferred_physics ...")
        resid = run_residual_analysis(
            neural, scenes, neural_meta,
            render_pca_dim=cfg['render_pca_dim'],
            inferred_physics=inferred_physics,
        )
        r2_resid_inferred = float(np.mean(resid['r2_resid_inferred']))
        print(f"    r2_resid_inferred     = {r2_resid_inferred:.4f}  "
              "(should be ~0 — residualization collapses inferred signal)")

    return {
        'variant':            label,
        'pixel_pca_dim':      int(pixel_pca_dim),
        'train_time_s':       float(train_time_s),
        'best_val_loss':      float(inv.best_val_loss_),
        'stopped_epoch':      int(inv.stopped_epoch_),
        'per_dim_r2':         per_dim_r2,
        'mean_r2':            mean_r2,
        'r2_inferred':        r2_inferred,
        'r2_inferred_combined': r2_inferred_combined,
        'delta_r2_inferred':  delta_r2_inferred,
        'r2_resid_inferred':  r2_resid_inferred,
        'feature_shape':      list(X.shape),
        'n_observable_dims':  int(inv.valid_dims_.sum()),
    }


def _print_summary(results):
    print("\n" + "=" * 100)
    print("FRAME-DIFF SUMMARY")
    print("=" * 100)
    cols = ['variant', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'x_accel',
            'mean', 'r2_inf', 'r2_inf_resid', 'val_MSE', 'epoch', 'sec']
    widths = [11, 8, 8, 8, 8, 9, 8, 8, 13, 9, 7, 7]
    header = "".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in results:
        row_vals = [
            f"{r['variant']:>10s}",
            f"{r['per_dim_r2']['pos_x']:+.3f}",
            f"{r['per_dim_r2']['pos_y']:+.3f}",
            f"{r['per_dim_r2']['pos_z']:+.3f}",
            f"{r['per_dim_r2']['vel_x']:+.3f}",
            f"{r['per_dim_r2']['x_accel']:+.3f}",
            f"{r['mean_r2']:+.3f}",
            f"{r['r2_inferred']:+.3f}",
            (f"{r['r2_resid_inferred']:+.3f}" if r['r2_resid_inferred'] is not None else "  --"),
            f"{r['best_val_loss']:.3f}",
            f"{r['stopped_epoch']:d}",
            f"{r['train_time_s']:.0f}",
        ]
        print("".join(f"{v:>{w}}" for v, w in zip(row_vals, widths)))
    print("=" * 100)


def _interpret(results):
    """Decision matrix from specs/inverse_model_input_repr.md."""
    diff = next((r for r in results if r['variant'] == 'frame_diff'), None)
    base = next((r for r in results if r['variant'] == 'baseline'), None)
    if diff is None:
        print("\nNo frame_diff result — nothing to interpret.")
        return

    pdr = diff['per_dim_r2']
    mean = diff['mean_r2']
    min_dim = min(pdr.values())
    min_name = min(pdr, key=pdr.get)

    print("\nInterpretation (decision matrix from specs/inverse_model_input_repr.md):")
    print(f"  frame_diff: mean R² = {mean:+.3f}, "
          f"min dim = {min_name} @ {min_dim:+.3f}")
    if base is not None:
        print(f"  baseline:   mean R² = {base['mean_r2']:+.3f}, "
              f"x_accel = {base['per_dim_r2']['x_accel']:+.3f}, "
              f"pos_z = {base['per_dim_r2']['pos_z']:+.3f}")
        print(f"  lift:       mean R² = {mean - base['mean_r2']:+.3f}, "
              f"x_accel = {pdr['x_accel'] - base['per_dim_r2']['x_accel']:+.3f}, "
              f"pos_z   = {pdr['pos_z']   - base['per_dim_r2']['pos_z']:+.3f}")

    if mean >= 0.65 and min_dim >= 0.4:
        verdict = ("ROW 1 — frame-diff exposes the temporal signal cleanly. "
                   "Recommend: wire build_pp_diff_features into gen_features / "
                   "config.yaml; SKIP Phase 2.")
    elif mean >= 0.55 and pdr['pos_z'] > (base['per_dim_r2']['pos_z'] if base else 0.0) \
            and pdr['x_accel'] <= 0.2:
        verdict = ("ROW 2 — velocity-scale info is in the diff basis but second-difference "
                   "still gets washed. Recommend: run Phase 2 (CNN keeps spatial detail "
                   "PCA throws away).")
    elif base is not None and mean <= base['mean_r2'] + 0.02:
        verdict = ("ROW 3 — diff basis isn't carrying the right invariances either; "
                   "linear pre-mix is fundamentally insufficient. Recommend: run Phase 2.")
    else:
        verdict = ("MIXED — partial improvement that doesn't match a clean spec row. "
                   "Inspect per-dim slopes; default to running Phase 2 if x_accel "
                   "remains the weakest dim.")
    print(f"  → {verdict}")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default=os.path.join('data', 'scenes.npz'))
    ap.add_argument('--out', default=os.path.join('outputs', 'pp_framediff.json'))
    ap.add_argument('--pixel-pca-dim', type=int, default=50,
                    help="Per-frame PCA dim. Spec default: 50.")
    ap.add_argument('--skip-baseline', action='store_true',
                    help="Skip the baseline (non-diff) variant; only run frame_diff.")
    ap.add_argument('--skip-residual', action='store_true',
                    help="Skip residual analysis (faster smoke runs).")
    args = ap.parse_args()

    print("=" * 60)
    print("Phase 1: frame-difference features")
    print("=" * 60)
    print(f"  pixel_pca_dim : {args.pixel_pca_dim}")
    print(f"  output JSON   : {args.out}")
    print(f"  baseline      : {'skipped' if args.skip_baseline else 'on'}")
    print(f"  residual      : {'skipped' if args.skip_residual else 'on'}")

    print(f"\nLoading scenes...")
    scenes = _ensure_scenes(args.scenes, cfg)
    print(f"  n = {len(scenes['initial_renders'])}")

    print(f"\nGenerating neural activity (fixed across variants)...")
    neural, neural_meta = _gen_neural(scenes, cfg)
    print(f"  neural shape = {neural.shape}")

    results = []
    if not args.skip_baseline:
        results.append(_eval_one(
            scenes, neural, neural_meta,
            feature_builder=build_pp_features,
            label='baseline',
            cfg=cfg, run_residual=not args.skip_residual,
            pixel_pca_dim=args.pixel_pca_dim,
        ))

    results.append(_eval_one(
        scenes, neural, neural_meta,
        feature_builder=build_pp_diff_features,
        label='frame_diff',
        cfg=cfg, run_residual=not args.skip_residual,
        pixel_pca_dim=args.pixel_pca_dim,
    ))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        'results': results,
        'config': {
            'n_scenes':         len(scenes['initial_renders']),
            'pixel_pca_dim':    args.pixel_pca_dim,
            'pp_hidden_dim':    cfg['pp_hidden_dim'],
            'pp_dropout_rate':  cfg['pp_dropout_rate'],
            'pp_early_frame':   cfg['pp_early_frame'],
            'pp_late_frame':    cfg['pp_late_frame'],
            'render_pca_dim':   cfg['render_pca_dim'],
            'n_neurons':        cfg['n_neurons'],
            'noise_level':      cfg['noise_level'],
            'random_seed':      cfg['random_seed'],
        },
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {args.out}")

    _print_summary(results)
    _interpret(results)


if __name__ == '__main__':
    main()
