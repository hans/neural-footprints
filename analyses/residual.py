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

from analyses.encoding import _null_summary


def _r2_per_neuron(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot


def _ridge_cv_predict(X, y, cv, alphas):
    ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
    return cross_val_predict(ridge, X, y, cv=cv)


def _compute_residual_null(
    residual_targets,
    physics_variants,
    *,
    n_permutations,
    seed,
    cv,
    alphas,
):
    """Matched permutation null for the residualization R²s.

    Stage-1 residuals are physics-independent, so the caller passes them in
    pre-computed (``residual_targets``) and we only re-run the stage-2 decoder
    per shuffle — the exact estimator that produced the observed R²s.

    A single row-permutation is drawn per iteration and applied to *every*
    physics variant (GT and inferred), so GT/inferred nulls are paired — the
    same apples-to-apples convention used in ``encoding._compute_null_distribution``.

    Parameters
    ----------
    residual_targets : dict[str, np.ndarray]
        Maps a residual key (``"X"``, ``"XS"``) to its fixed residual-neural
        target ``[n_scenes, n_neurons]``.
    physics_variants : dict[str, np.ndarray]
        Maps a physics key (``"P"``, ``"P_inf"``) to its standardized labels.

    Returns
    -------
    dict
        ``_null_summary`` outputs for every (physics, residual) combination,
        prefixed ``r2_<P>_given_<R>`` (e.g. ``r2_P_given_X``), plus the observed
        mean R² for each combination keyed ``<prefix>_observed_array``.
    """
    any_target = next(iter(residual_targets.values()))
    n_scenes, n_neurons = any_target.shape
    rng = np.random.default_rng(seed)

    combos = [
        (f"r2_{pkey}_given_{rkey}", phys, resid)
        for pkey, phys in physics_variants.items()
        for rkey, resid in residual_targets.items()
    ]
    null_arrays = {prefix: np.empty((n_permutations, n_neurons)) for prefix, _, _ in combos}

    print(
        f"\nComputing matched residualization null ({n_permutations} shuffles, "
        f"{len(combos)} conditions)..."
    )
    for p in range(n_permutations):
        perm = rng.permutation(n_scenes)
        for prefix, phys, resid in combos:
            pred = _ridge_cv_predict(phys[perm], resid, cv=cv, alphas=alphas)
            null_arrays[prefix][p] = _r2_per_neuron(resid, pred)
        if (p + 1) % 10 == 0 or p == 0:
            head = combos[0][0]
            print(f"  perm {p + 1}/{n_permutations}: {head} null mean R² = {null_arrays[head][p].mean():.4f}")

    return null_arrays


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
    compute_null=True,
    n_null_permutations=300,
    null_seed=0,
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

    When compute_null is set, a matched permutation null is added for every
    residual R²: the fixed stage-1 residuals are reused while physics rows are
    shuffled through the stage-2 decoder, giving each R² its own chance band
    (keys null_<prefix>_pvalue / _ci_lo / _ci_hi / _mean / _observed). This is
    the correct reference — the encoding-physics null uses a different target
    (raw neural) and cannot bound these residual R²s.
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

    # --- Matched permutation null (shuffle physics, refit stage 2 only) ---
    # Stage-1 residuals are physics-independent, so they're reused as fixed
    # targets — the null isolates the stage-2 decoder, the same estimator that
    # produced the observed R²s. Gives each residualization R² its OWN chance
    # band (the encoding-physics null is the wrong reference: different target).
    if compute_null and n_null_permutations > 0:
        residual_targets = {"X": y_resid_X}
        if predicted_pixel_pca is not None:
            residual_targets["XS"] = y_resid_XS
        physics_variants = {"P": physics_scaled}
        if physics_inf_scaled is not None:
            physics_variants["P_inf"] = physics_inf_scaled

        null_arrays = _compute_residual_null(
            residual_targets,
            physics_variants,
            n_permutations=n_null_permutations,
            seed=null_seed,
            cv=cv,
            alphas=alphas,
        )
        for prefix, null_array in null_arrays.items():
            observed_mean = float(np.asarray(result[prefix]).mean())
            result.update(_null_summary(prefix, null_array, observed=observed_mean))
            p = result[f"null_{prefix}_pvalue"]
            lo, hi = result[f"null_{prefix}_ci_lo"], result[f"null_{prefix}_ci_hi"]
            print(
                f"  {prefix}: observed={observed_mean:+.4f} "
                f"null95%=[{lo:+.4f},{hi:+.4f}] p(obs>=null)={p:.3f}"
            )

    return result
