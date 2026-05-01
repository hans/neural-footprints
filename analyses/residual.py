"""
Residual encoding analysis.

Tests the canonical two-step residualization procedure used in neuroscience:
regress sensory predictors out of neural activity, then ask whether the
residual variance is still predicted by abstract features.

Procedure (single-pass, matching published practice):

  1. y_resid = y - cross_val_predict(RidgeCV, X_render_pca, y, cv=5)
  2. For each predictor set X (render, gt physics, inferred physics, combined):
        ŷ = cross_val_predict(RidgeCV, X, y_resid, cv=5)
        R² per neuron between y_resid and ŷ.

The same KFold layout is reused for both stages — this is what `cv=5` does in
sklearn by default and matches what residualization analyses report in the
literature. It introduces mild fold-leakage (training-fold residuals are built
by a stage-1 model fit on data that includes the held-out fold) but the spec's
purpose is to evaluate the procedure as published, not engineer it away.
A nested-CV variant could be added later as a robustness check.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from analyses.encoding import pca_reduce_render, ridge_r2_per_neuron


def _r2_per_neuron(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot


def _ridge_cv_predict(X, y, cv, alphas):
    ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
    return cross_val_predict(ridge, X, y, cv=cv)


def run_residual_analysis(neural_activity, scenes, neural_meta,
                          *, render_pca_dim, inferred_physics,
                          n_splits=5, random_state=42):
    """
    Run residual encoding analysis on pp-port neural activity.

    Returns dict with per-neuron R² arrays for raw and residualized neural,
    across {render, gt physics, inferred physics, render+inferred} predictor
    sets, plus the matched raw-neural baselines for paired comparison.
    """
    print("\n" + "=" * 60)
    print("RESIDUAL ANALYSIS: encoding on render-residualized neural")
    print("=" * 60)

    program_states = scenes['program_states']
    physics_labels = scenes['physics_labels']
    render_indices = scenes['metadata']['render_indices']

    n_scenes, n_neurons = neural_activity.shape
    print(f"  n_scenes={n_scenes}, n_neurons={n_neurons}")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    alphas = np.logspace(-2, 6, 20)

    # --- Predictor matrices ---
    print(f"\nReducing render slice to {render_pca_dim} PCA components...")
    render_data = program_states[:, render_indices]
    render_pca, pca, _ = pca_reduce_render(render_data, render_pca_dim,
                                           random_state=random_state)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    physics_scaled = StandardScaler().fit_transform(physics_labels)
    inferred_scaled = StandardScaler().fit_transform(inferred_physics)
    combined_inferred = np.hstack([render_pca, inferred_scaled])

    predictor_sets = {
        'render': render_pca,
        'physics_gt': physics_scaled,
        'inferred': inferred_scaled,
        'combined': combined_inferred,
    }

    # --- Raw-neural baselines (matches encoding.py's reporting) ---
    print("\nRaw-neural R² baselines:")
    r2_raw = {}
    for name, X in predictor_sets.items():
        r2_raw[name] = ridge_r2_per_neuron(X, neural_activity, alphas=alphas, cv=cv)
        print(f"  R² (raw, {name:>10}): mean={r2_raw[name].mean():.4f}")

    # --- Stage 1: out-of-fold render residuals ---
    print("\nStage 1: cross-validated render residuals...")
    y_pred_render = _ridge_cv_predict(render_pca, neural_activity, cv=cv,
                                      alphas=alphas)
    y_resid = neural_activity - y_pred_render
    var_kept = y_resid.var(axis=0).mean() / neural_activity.var(axis=0).mean()
    print(f"  mean residual variance / raw variance = {var_kept:.4f}")

    # --- Stage 2: encode residuals from each predictor set ---
    print("\nStage 2: encoding on residual neural...")
    r2_resid = {}
    for name, X in predictor_sets.items():
        y_hat = _ridge_cv_predict(X, y_resid, cv=cv, alphas=alphas)
        r2_resid[name] = _r2_per_neuron(y_resid, y_hat)
        print(f"  R² (resid, {name:>10}): mean={r2_resid[name].mean():.4f}")

    # Sanity: render-on-residual should be ~0
    if r2_resid['render'].mean() > 0.05:
        print(f"  WARNING: r2_render_resid mean is {r2_resid['render'].mean():.4f}, "
              "expected ~0 — stage-1 ridge may be underfitting render.")

    return {
        'r2_raw_render': r2_raw['render'],
        'r2_raw_physics_gt': r2_raw['physics_gt'],
        'r2_raw_inferred': r2_raw['inferred'],
        'r2_raw_combined': r2_raw['combined'],
        'r2_resid_render': r2_resid['render'],
        'r2_resid_physics_gt': r2_resid['physics_gt'],
        'r2_resid_inferred': r2_resid['inferred'],
        'r2_resid_combined': r2_resid['combined'],
        'residual_variance_fraction': float(var_kept),
        'render_pca_dim': int(render_pca_dim),
        'n_splits': int(n_splits),
        'random_state': int(random_state),
    }
