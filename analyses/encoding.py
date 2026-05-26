"""
Simulation 1: Encoding model false negatives.

Demonstrates that adding physics labels to a pixel-based encoding model
produces negligible improvement in R², despite physics being causally operative.
The sensory regressor is the 3-frame RGBA pixel concatenation — what a
scientist would actually have access to from a camera. Depth and
segmentation buffers exist in the program state (and so leak into neural
activity through the random projection) but are deliberately excluded from
the analysis side, matching the real-world observability constraint.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from config import PIXEL_PCA_DIM as _CFG_PIXEL_PCA_DIM
from scene_generator import extract_brain_pixels


def pca_reduce_pixels(pixel_data, n_components, random_state=42):
    """StandardScaler + PCA on 3-frame pixel data. Returns (pixel_pca, pca, scaler)."""
    scaler = StandardScaler()
    pixel_scaled = scaler.fit_transform(pixel_data)
    pca = PCA(n_components=n_components, random_state=random_state)
    pixel_pca = pca.fit_transform(pixel_scaled)
    return pixel_pca, pca, scaler


def ridge_r2_per_neuron(X, neural_activity, alphas=None, cv=5):
    """Cross-validated ridge regression, returns R² array of shape [n_neurons]."""
    if alphas is None:
        alphas = np.logspace(-2, 6, 20)
    ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
    predictions = cross_val_predict(ridge, X, neural_activity, cv=cv)
    ss_res = ((neural_activity - predictions) ** 2).sum(axis=0)
    ss_tot = ((neural_activity - neural_activity.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot


def ridge_r2_per_neuron_fast(X, Y, alphas=None, cv=5):
    """
    Closed-form K-fold ridge with cached per-fold SVD; α picked per target via
    LOO inside each train fold (Hat-matrix diagonal trick). Vectorized across
    α and Y columns. Mirrors RidgeCV(alpha_per_target=True) wrapped in
    cross_val_predict(cv=K), but ~50–100× faster by lifting sklearn's per-target
    Python loop out of the inner ridge fit.

    Returns per-neuron OOF R² of shape [n_targets].
    """
    if alphas is None:
        alphas = np.logspace(-2, 6, 20)
    alphas = np.asarray(alphas, dtype=float)
    n, p = X.shape
    n_targets = Y.shape[1]

    Y_oof = np.empty_like(Y, dtype=float)
    for train_idx, test_idx in KFold(n_splits=cv, shuffle=False).split(X):
        X_tr, X_te, Y_tr = X[train_idx], X[test_idx], Y[train_idx]
        U, S, Vt = np.linalg.svd(X_tr, full_matrices=False)
        S2 = S**2
        UtY = U.T @ Y_tr  # [k, n_targets]
        U2 = U**2  # [n_tr, k]

        # LOO MSE per (alpha, target). Loop over α to keep memory in MB-range.
        loo_mse = np.empty((alphas.size, n_targets))
        for a, alpha in enumerate(alphas):
            gain = S2 / (S2 + alpha)  # [k]
            Yhat = U @ (gain[:, None] * UtY)  # [n_tr, n_targets]
            H_diag = U2 @ gain  # [n_tr]
            resid = (Y_tr - Yhat) / (1.0 - H_diag)[:, None]
            loo_mse[a] = (resid**2).mean(axis=0)

        best_alpha = alphas[np.argmin(loo_mse, axis=0)]  # [n_targets]
        factor = S[:, None] / (S2[:, None] + best_alpha[None, :])  # [k, n_targets]
        beta = Vt.T @ (factor * UtY)  # [p, n_targets]
        Y_oof[test_idx] = X_te @ beta

    ss_res = ((Y - Y_oof) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot


def run_encoding_analysis(
    neural_activity,
    scenes,
    neural_meta,
    *,
    pixel_pca_dim=None,
    predicted_pixel_pca=None,
    compute_null=True,
    n_null_permutations=50,
    null_seed=0,
):
    """
    Run encoding model analysis.

    1. PCA-reduce 3-frame brain pixels to PIXEL_PCA_DIM components
    2. Ridge regression: neural ~ pixel_PCA -> R² per neuron
    3. Ridge regression: neural ~ pixel_PCA + physics_labels -> R²
    4. DeltaR² should be tiny
    5. Subsampling curve: vary neurons sampled, plot DeltaR² + significance
    6. Control: MLP physics_labels -> behavior_label
    7. Permutation null: shuffle scene→physics association n_null_permutations
       times, refit physics-only and combined encoders. Yields a null
       distribution for r2_physics_only and r2_combined; ΔR² null is derived
       (combined null minus fixed pixel-only baseline).
    """
    if pixel_pca_dim is None:
        pixel_pca_dim = _CFG_PIXEL_PCA_DIM

    print("\n" + "=" * 60)
    print("SIMULATION 1: Encoding Model False Negatives")
    print("=" * 60)

    program_states = scenes["program_states"]
    physics_labels = scenes["physics_labels"]
    behavior_labels = scenes["behavior_labels"]
    metadata = scenes["metadata"]

    n_scenes, n_neurons = neural_activity.shape

    # --- Extract and PCA-reduce 3-frame brain pixels (RGBA only) ---
    print(
        f"\nExtracting brain pixels and reducing to {pixel_pca_dim} PCA components..."
    )
    pixel_data = extract_brain_pixels(program_states, metadata)
    pixel_pca, pca, pixel_scaler = pca_reduce_pixels(pixel_data, pixel_pca_dim)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    del pixel_data  # free ~393 MB — PCA-reduced result is all we need

    # --- Standardize physics labels ---
    scaler_phys = StandardScaler()
    physics_scaled = scaler_phys.fit_transform(physics_labels)

    # --- Encoding model: pixels only ---
    # Use the SVD-cached fast helper for all fits — same implementation for
    # observed and null lets the permutation null directly bracket the real
    # ΔR². Validated to match sklearn's RidgeCV(alpha_per_target=True) wrapped
    # in cross_val_predict to ~1e-2 absolute (sklearn's gcv_mode picks slightly
    # different alphas than our LOO-via-SVD on near-noise targets).
    print("\nFitting encoding model: neural ~ pixel_PCA ...")
    r2_pixel_only = ridge_r2_per_neuron_fast(pixel_pca, neural_activity)

    # --- Encoding model: physics only ---
    print("Fitting encoding model: neural ~ physics_labels ...")
    r2_physics_only = ridge_r2_per_neuron_fast(physics_scaled, neural_activity)

    # --- Encoding model: pixels + physics labels ---
    print("Fitting encoding model: neural ~ pixel_PCA + physics_labels ...")
    combined = np.hstack([pixel_pca, physics_scaled])
    r2_combined = ridge_r2_per_neuron_fast(combined, neural_activity)

    del combined
    delta_r2 = r2_combined - r2_pixel_only
    mean_r2_pixel = r2_pixel_only.mean()
    mean_r2_phys = r2_physics_only.mean()
    mean_r2_comb = r2_combined.mean()
    mean_delta = delta_r2.mean()

    print(f"\n  Mean R² (pixel only):         {mean_r2_pixel:.4f}")
    print(f"  Mean R² (physics only):       {mean_r2_phys:.4f}")
    print(f"  Mean R² (pixel+physics):      {mean_r2_comb:.4f}")
    print(f"  Mean ΔR²:                     {mean_delta:.6f}")

    # --- Encoding model: predicted pixels (forward-model render, Predicted S) ---
    r2_predicted_pixel = None
    mean_r2_predicted_pixel = None
    r2_combined_pred = None
    delta_r2_pred = None
    if predicted_pixel_pca is not None:
        print("Fitting encoding model: neural ~ predicted_pixel_PCA (Predicted S) ...")
        r2_predicted_pixel = ridge_r2_per_neuron_fast(
            predicted_pixel_pca, neural_activity
        )
        mean_r2_predicted_pixel = r2_predicted_pixel.mean()
        print(f"  Mean R² (predicted S):        {mean_r2_predicted_pixel:.4f}")
        print(
            "Fitting encoding model: neural ~ predicted_pixel_PCA + physics_labels ..."
        )
        combined_pred = np.hstack([predicted_pixel_pca, physics_scaled])
        r2_combined_pred = ridge_r2_per_neuron_fast(combined_pred, neural_activity)
        delta_r2_pred = r2_combined_pred - r2_predicted_pixel
        print(f"  Mean R² (predicted S + physics): {r2_combined_pred.mean():.4f}")
        print(f"  Mean ΔR² (physics | predicted S): {delta_r2_pred.mean():.6f}")

    # --- Control: physics_labels -> behavior_label ---
    # MLP because KE = 0.5*m*v² is nonlinear in the physics label features.
    print("\nControl: MLP physics_labels -> behavior_label ...")
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(64,), max_iter=500, random_state=42, early_stopping=True
    )
    log_scores = cross_val_score(
        mlp_clf, physics_scaled, behavior_labels, cv=5, scoring="accuracy"
    )
    control_acc = log_scores.mean()
    print(
        f"  Behavior prediction accuracy: {control_acc:.2%} (±{log_scores.std():.2%})"
    )
    print(
        "  (High accuracy expected: KE label is a deterministic function of physics labels)"
    )

    # --- Subsampling curve ---
    print("\nComputing subsampling curve...")
    neuron_counts = [10, 25, 50, 100, 200, 300, 400, n_neurons]
    neuron_counts = [n for n in neuron_counts if n <= n_neurons]
    rng = np.random.default_rng(42)

    subsample_means = []
    subsample_sems = []
    subsample_sig_fracs = []

    for n_sub in neuron_counts:
        # Random subsample of neurons, repeated 20 times
        deltas = []
        for _ in range(20):
            idx = rng.choice(n_neurons, size=n_sub, replace=False)
            sub_delta = delta_r2[idx]
            deltas.append(sub_delta.mean())
        deltas = np.array(deltas)
        subsample_means.append(deltas.mean())
        subsample_sems.append(deltas.std() / np.sqrt(len(deltas)))
        # Fraction of subsamples where mean ΔR² > 0
        subsample_sig_fracs.append((deltas > 0).mean())

    # --- Permutation null: shuffle scene→physics association ---
    if compute_null:
        null_results = _compute_null_distribution(
            pixel_pca,
            physics_scaled,
            neural_activity,
            r2_pixel_only,
            r2_physics_only,
            r2_combined,
            delta_r2,
            n_permutations=n_null_permutations,
            seed=null_seed,
        )
    else:
        null_results = _empty_null_results(neural_activity.shape[1])

    # --- Fit full encoder on all data (for downstream dynamics analysis) ---
    print("\nFitting full encoder for downstream use...")
    alphas = np.logspace(-2, 6, 20)
    encoder_ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
    encoder_ridge.fit(pixel_pca, neural_activity)

    result = {
        "r2_pixel_only": r2_pixel_only,
        "r2_physics_only": r2_physics_only,
        "r2_combined": r2_combined,
        "delta_r2": delta_r2,
        "control_accuracy": control_acc,
        "control_accuracy_std": log_scores.std(),
        "subsample_means": subsample_means,
        "subsample_sems": subsample_sems,
        "subsample_neuron_counts": neuron_counts,
        **null_results,
        "encoder": {
            "scaler": pixel_scaler,
            "pca": pca,
            "ridge": encoder_ridge,
            "scaler_phys": scaler_phys,
        },
    }
    if r2_predicted_pixel is not None:
        result["r2_predicted_pixel"] = r2_predicted_pixel
    if r2_combined_pred is not None:
        result["r2_combined_pred"] = r2_combined_pred
        result["delta_r2_pred"] = delta_r2_pred
    return result


def _compute_null_distribution(
    pixel_pca,
    physics_scaled,
    neural_activity,
    r2_pixel_only,
    r2_physics_only,
    r2_combined,
    delta_r2,
    *,
    n_permutations,
    seed,
):
    """
    Run n_permutations row-shuffles of physics_scaled. For each shuffle, fit
    physics-only and combined ridge encoders. Returns null arrays + summary
    CIs and one-sided p-values.

    Pixel-only baseline is fixed across permutations (only physics columns
    move), so ΔR² null is r2_combined_null minus the observed r2_pixel_only.
    """
    n_scenes = neural_activity.shape[0]
    n_neurons = neural_activity.shape[1]
    print(f"\nComputing permutation null ({n_permutations} shuffles)...")
    rng = np.random.default_rng(seed)
    r2_phys_null = np.empty((n_permutations, n_neurons))
    r2_comb_null = np.empty((n_permutations, n_neurons))
    for p in range(n_permutations):
        perm = rng.permutation(n_scenes)
        physics_perm = physics_scaled[perm]
        r2_phys_null[p] = ridge_r2_per_neuron_fast(physics_perm, neural_activity)
        combined_perm = np.hstack([pixel_pca, physics_perm])
        r2_comb_null[p] = ridge_r2_per_neuron_fast(combined_perm, neural_activity)
        if (p + 1) % 10 == 0 or p == 0:
            print(
                f"  perm {p + 1}/{n_permutations}: "
                f"physics_null mean R² = {r2_phys_null[p].mean():.4f}, "
                f"combined_null mean R² = {r2_comb_null[p].mean():.4f}"
            )
    delta_null = r2_comb_null - r2_pixel_only[None, :]
    return {
        "r2_physics_only_null": r2_phys_null,
        "r2_combined_null": r2_comb_null,
        "delta_r2_null": delta_null,
        **_null_summary("physics", r2_phys_null, observed=r2_physics_only.mean()),
        **_null_summary("combined", r2_comb_null, observed=r2_combined.mean()),
        **_null_summary("delta", delta_null, observed=delta_r2.mean()),
    }


def _null_summary(prefix, null_array, observed):
    """Per-perm mean across neurons → 95% CI + one-sided p-value vs observed mean."""
    perm_means = null_array.mean(axis=1)  # [n_permutations]
    lo, hi = np.percentile(perm_means, [2.5, 97.5])
    pvalue = float((perm_means >= observed).mean())
    return {
        f"null_{prefix}_perm_means": perm_means,
        f"null_{prefix}_ci_lo": float(lo),
        f"null_{prefix}_ci_hi": float(hi),
        f"null_{prefix}_mean": float(perm_means.mean()),
        f"null_{prefix}_pvalue": pvalue,
        f"null_{prefix}_observed": float(observed),
    }


def _empty_null_results(n_neurons):
    empty = np.empty((0, n_neurons))
    empty_means = np.empty(0)
    nan = float("nan")
    base = {
        "r2_physics_only_null": empty,
        "r2_combined_null": empty,
        "delta_r2_null": empty,
    }
    for prefix in ("physics", "combined", "delta"):
        base.update(
            {
                f"null_{prefix}_perm_means": empty_means,
                f"null_{prefix}_ci_lo": nan,
                f"null_{prefix}_ci_hi": nan,
                f"null_{prefix}_mean": nan,
                f"null_{prefix}_pvalue": nan,
                f"null_{prefix}_observed": nan,
            }
        )
    return base
