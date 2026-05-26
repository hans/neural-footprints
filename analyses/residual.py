"""
Residual encoding analysis.

Tests the canonical two-step residualization procedure used in neuroscience:
regress sensory predictors out of neural activity, then ask whether the
residual variance is still predicted by abstract features. Sensory
predictors are the 3-frame brain pixels (RGBA only) — same input the
encoding/RSA/dynamics analyses consume.

Procedure (single-pass, matching published practice):

  1. y_resid = y - cross_val_predict(RidgeCV, X_pixel_pca, y, cv=5)
  2. For each predictor set X (pixels, gt physics):
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

from analyses.encoding import pca_reduce_pixels
from scene_generator import extract_brain_pixels


def _r2_per_neuron(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot


def _ridge_cv_predict(X, y, cv, alphas):
    ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
    return cross_val_predict(ridge, X, y, cv=cv)


def run_residual_analysis(
    neural_activity,
    scenes,
    neural_meta,
    *,
    pixel_pca_dim,
    r2_raw_pixel,
    r2_raw_physics_gt,
    predicted_pixel_pca=None,
    n_splits=5,
    random_state=42,
):
    """
    Run residual encoding analysis.

    Returns dict with per-neuron R² arrays for raw and residualized neural,
    across {pixel, gt physics} predictor sets. Raw-neural baselines are
    passed in (from the encoding analysis) so this stage and dissociation
    report identical numbers.
    """
    print("\n" + "=" * 60)
    print("RESIDUAL ANALYSIS: encoding on pixel-residualized neural")
    print("=" * 60)

    program_states = scenes["program_states"]
    physics_labels = scenes["physics_labels"]
    metadata = scenes["metadata"]

    n_scenes, n_neurons = neural_activity.shape
    print(f"  n_scenes={n_scenes}, n_neurons={n_neurons}")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    alphas = np.logspace(-2, 6, 20)

    # --- Predictor matrices ---
    print(f"\nReducing brain pixels to {pixel_pca_dim} PCA components...")
    pixel_data = extract_brain_pixels(program_states, metadata)
    pixel_pca, pca, _ = pca_reduce_pixels(
        pixel_data, pixel_pca_dim, random_state=random_state
    )
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    physics_scaled = StandardScaler().fit_transform(physics_labels)

    predictor_sets = {
        "pixel": pixel_pca,
        "physics_gt": physics_scaled,
    }

    # --- Raw-neural baselines (reused from encoding analysis) ---
    r2_raw = {
        "pixel": np.asarray(r2_raw_pixel),
        "physics_gt": np.asarray(r2_raw_physics_gt),
    }
    print("\nRaw-neural R² baselines (from encoding analysis):")
    for name, arr in r2_raw.items():
        print(f"  R² (raw, {name:>10}): mean={arr.mean():.4f}")

    # --- Stage 1: out-of-fold pixel residuals ---
    print("\nStage 1: cross-validated pixel residuals...")
    y_pred_pixel = _ridge_cv_predict(pixel_pca, neural_activity, cv=cv, alphas=alphas)
    y_resid = neural_activity - y_pred_pixel
    var_kept = y_resid.var(axis=0).mean() / neural_activity.var(axis=0).mean()
    print(f"  mean residual variance / raw variance = {var_kept:.4f}")

    # --- Stage 2: encode residuals from each predictor set ---
    print("\nStage 2: encoding on residual neural...")
    r2_resid = {}
    for name, X in predictor_sets.items():
        y_hat = _ridge_cv_predict(X, y_resid, cv=cv, alphas=alphas)
        r2_resid[name] = _r2_per_neuron(y_resid, y_hat)
        print(f"  R² (resid, {name:>10}): mean={r2_resid[name].mean():.4f}")

    # Sanity: pixel-on-residual should be ~0
    if r2_resid["pixel"].mean() > 0.05:
        print(
            f"  WARNING: r2_pixel_resid mean is {r2_resid['pixel'].mean():.4f}, "
            "expected ~0 — stage-1 ridge may be underfitting pixels."
        )

    result = {
        "r2_raw_pixel": r2_raw["pixel"],
        "r2_raw_physics_gt": r2_raw["physics_gt"],
        "r2_resid_pixel": r2_resid["pixel"],
        "r2_resid_physics_gt": r2_resid["physics_gt"],
        "residual_variance_fraction": float(var_kept),
        "pixel_pca_dim": int(pixel_pca_dim),
        "n_splits": int(n_splits),
        "random_state": int(random_state),
    }

    # --- Predicted-S Stage-1 residualization ---
    if predicted_pixel_pca is not None:
        print("\nStage 1 (predicted-S): cross-validated predicted-pixel residuals...")
        y_pred_pred = _ridge_cv_predict(
            predicted_pixel_pca, neural_activity, cv=cv, alphas=alphas
        )
        y_resid_pred = neural_activity - y_pred_pred
        var_kept_pred = (
            y_resid_pred.var(axis=0).mean() / neural_activity.var(axis=0).mean()
        )
        print(f"  mean residual variance / raw variance = {var_kept_pred:.4f}")

        print("\nStage 2 (predicted-S): encoding on predicted-S residual neural...")
        r2_resid_pred_self = _r2_per_neuron(
            y_resid_pred,
            _ridge_cv_predict(predicted_pixel_pca, y_resid_pred, cv=cv, alphas=alphas),
        )
        r2_resid_physics_via_pred = _r2_per_neuron(
            y_resid_pred,
            _ridge_cv_predict(physics_scaled, y_resid_pred, cv=cv, alphas=alphas),
        )
        print(
            f"  R² (resid_pred, predicted_S):  mean={r2_resid_pred_self.mean():.4f}  "
            "(sanity — expect ~0)"
        )
        print(
            f"  R² (resid_pred, physics_gt):   mean={r2_resid_physics_via_pred.mean():.4f}  "
            "(expect ~0)"
        )

        if r2_resid_pred_self.mean() > 0.05:
            print(
                f"  WARNING: predicted-S self-residual mean is "
                f"{r2_resid_pred_self.mean():.4f}, expected ~0."
            )

        result["r2_resid_predicted_pixel"] = r2_resid_pred_self
        result["r2_resid_physics_gt_via_predicted_pixel"] = r2_resid_physics_via_pred
        result["residual_variance_fraction_predicted"] = float(var_kept_pred)

    return result
