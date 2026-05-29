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
    raw_pixel_pca,
    predicted_pixel_pca=None,
    inferred_physics_labels=None,
    n_splits=5,
    random_state=42,
):
    """
    Run residual encoding analysis (two-condition variance partitioning).

    Condition (a): residualize neural on X only → r2_P_given_X (expect high,
      physics still visible after removing raw-frame variance).
    Condition (b): residualize neural on [X, S] → r2_P_given_XS (expect ~0,
      KEY — physics collapses once predicted-frame variance is also removed).

    When inferred_physics_labels is provided, the same residualization stages
    are repeated substituting P_inf for P, yielding r2_P_inf_given_X and
    r2_P_inf_given_XS.
    """
    print("\n" + "=" * 60)
    print("RESIDUAL ANALYSIS: variance partitioning with X and S")
    print("=" * 60)

    physics_labels = scenes["physics_labels"]

    n_scenes, n_neurons = neural_activity.shape
    print(f"  n_scenes={n_scenes}, n_neurons={n_neurons}")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    alphas = np.logspace(-2, 6, 20)

    physics_scaled = StandardScaler().fit_transform(physics_labels)

    # Standardize inferred physics (if provided)
    physics_inf_scaled = None
    if inferred_physics_labels is not None:
        physics_inf_scaled = StandardScaler().fit_transform(inferred_physics_labels)

    # --- Condition (a): residualize on X ---
    print("\nStage 1a: cross-validated residuals after removing X...")
    y_pred_X = _ridge_cv_predict(raw_pixel_pca, neural_activity, cv=cv, alphas=alphas)
    y_resid_X = neural_activity - y_pred_X
    var_kept_X = y_resid_X.var(axis=0).mean() / neural_activity.var(axis=0).mean()
    print(f"  residual variance fraction = {var_kept_X:.4f}")

    print("Stage 2a: encode GT physics from X-residual neural...")
    r2_P_given_X = _r2_per_neuron(
        y_resid_X,
        _ridge_cv_predict(physics_scaled, y_resid_X, cv=cv, alphas=alphas),
    )
    print(f"  r2_P | X: mean={r2_P_given_X.mean():.4f}  (expect > 0.01)")

    result = {
        "r2_P_given_X": r2_P_given_X,
        "residual_variance_fraction_X": float(var_kept_X),
        "n_splits": int(n_splits),
        "random_state": int(random_state),
    }

    # --- Condition (a) with inferred physics ---
    if physics_inf_scaled is not None:
        print("Stage 2a (inf): encode inferred physics from X-residual neural...")
        r2_P_inf_given_X = _r2_per_neuron(
            y_resid_X,
            _ridge_cv_predict(physics_inf_scaled, y_resid_X, cv=cv, alphas=alphas),
        )
        print(f"  r2_P_inf | X: mean={r2_P_inf_given_X.mean():.4f}")
        result["r2_P_inf_given_X"] = r2_P_inf_given_X

    # --- Condition (b): residualize on [X, S] (KEY) ---
    if predicted_pixel_pca is not None:
        print("\nStage 1b: cross-validated residuals after removing [X, S]...")
        XS = np.hstack([raw_pixel_pca, predicted_pixel_pca])
        y_pred_XS = _ridge_cv_predict(XS, neural_activity, cv=cv, alphas=alphas)
        y_resid_XS = neural_activity - y_pred_XS
        var_kept_XS = y_resid_XS.var(axis=0).mean() / neural_activity.var(axis=0).mean()
        print(f"  residual variance fraction = {var_kept_XS:.4f}")

        print("Stage 2b: encode GT physics from [X,S]-residual neural (KEY)...")
        r2_P_given_XS = _r2_per_neuron(
            y_resid_XS,
            _ridge_cv_predict(physics_scaled, y_resid_XS, cv=cv, alphas=alphas),
        )
        print(f"  r2_P | X,S: mean={r2_P_given_XS.mean():.4f}  (expect < 0.01 KEY)")

        result["r2_P_given_XS"] = r2_P_given_XS
        result["residual_variance_fraction_XS"] = float(var_kept_XS)

        # --- Condition (b) with inferred physics ---
        if physics_inf_scaled is not None:
            print("Stage 2b (inf): encode inferred physics from [X,S]-residual neural...")
            r2_P_inf_given_XS = _r2_per_neuron(
                y_resid_XS,
                _ridge_cv_predict(physics_inf_scaled, y_resid_XS, cv=cv, alphas=alphas),
            )
            print(f"  r2_P_inf | X,S: mean={r2_P_inf_given_XS.mean():.4f}")
            result["r2_P_inf_given_XS"] = r2_P_inf_given_XS

    return result
