"""
Simulation 2: RSA dominated by pixel structure.

Shows that neural RDM tracks the raw-frame RDM (X) and forward-model RDM (S)
but not the physics RDM, and partial correlation controlling for both X and S
removes any residual physics signal.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from config import RSA_SUBSAMPLE as _CFG_RSA_SUBSAMPLE


def _compute_rdm(data):
    """Compute representational dissimilarity matrix using correlation distance."""
    return pdist(data, metric="correlation")


def _partial_spearman(x, y, z):
    """
    Partial Spearman correlation between x and y, controlling for z.
    Uses rank-based residualization via OLS on ranks.
    """
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)

    n = len(rx)
    design = np.column_stack([np.ones(n), rz])

    def residualize(a):
        coef, _, _, _ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    corr, pval = spearmanr(residualize(rx), residualize(ry))
    return corr, pval


def _partial_spearman_2(x, y, z1, z2):
    """
    Partial Spearman correlation between x and y, controlling for z1 and z2.
    Ranks all inputs, residualizes x and y on [1, ranked_z1, ranked_z2] via lstsq.
    """
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rz1 = rankdata(z1)
    rz2 = rankdata(z2)

    n = len(rx)
    design = np.column_stack([np.ones(n), rz1, rz2])

    def residualize(a):
        coef, _, _, _ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    corr, pval = spearmanr(residualize(rx), residualize(ry))
    return corr, pval


def run_rsa_analysis(
    neural_activity,
    scenes,
    neural_meta,
    *,
    raw_pixel_pca,
    rsa_subsample=None,
    predicted_pixel_pca=None,
    inferred_physics_labels=None,
):
    """
    Run RSA analysis on a subsample of scenes.

    1. Build RDMs from neural, X (raw frames), physics, S (predicted frames)
    2. Spearman correlations: neural<->X (high), neural<->physics
    3. partial_P_given_X  = partial Spearman(neural, physics | X)
    4. partial_P_given_XS = partial Spearman(neural, physics | X, S)  [KEY ≈ 0]

    When inferred_physics_labels is provided, the same RSA metrics are computed
    with P_inf (inferred physics) substituted for GT physics, yielding
    corr_neural_P_inf, partial_P_inf_given_X, partial_P_inf_given_XS.
    """
    if rsa_subsample is None:
        rsa_subsample = _CFG_RSA_SUBSAMPLE

    print("\n" + "=" * 60)
    print("SIMULATION 2: RSA — Variance Partitioning")
    print("=" * 60)

    physics_labels = scenes["physics_labels"]

    n_scenes = neural_activity.shape[0]
    n_sub = min(rsa_subsample, n_scenes)

    # Subsample scenes for tractability
    rng = np.random.default_rng(123)
    sub_idx = rng.choice(n_scenes, size=n_sub, replace=False)
    sub_idx.sort()

    neural_sub = neural_activity[sub_idx]
    X_sub = raw_pixel_pca[sub_idx]
    physics_sub = physics_labels[sub_idx]
    S_sub = predicted_pixel_pca[sub_idx] if predicted_pixel_pca is not None else None
    physics_inf_sub = (
        inferred_physics_labels[sub_idx] if inferred_physics_labels is not None else None
    )

    # Standardize physics for RDM
    scaler_phys = StandardScaler()
    physics_scaled = scaler_phys.fit_transform(physics_sub)

    # Standardize inferred physics for RDM
    physics_inf_scaled = None
    if physics_inf_sub is not None:
        scaler_inf = StandardScaler()
        physics_inf_scaled = scaler_inf.fit_transform(physics_inf_sub)

    # Compute RDMs
    print(f"\nSubsampled {n_sub} scenes for RSA.")
    print("Computing RDMs...")
    rdm_neural = _compute_rdm(neural_sub)
    rdm_X = _compute_rdm(X_sub)
    rdm_physics = _compute_rdm(physics_scaled)

    for rdm in [rdm_neural, rdm_X, rdm_physics]:
        rdm[np.isnan(rdm)] = 0.0

    # Spearman correlations (GT physics)
    corr_neural_X, p_nX = spearmanr(rdm_neural, rdm_X)
    corr_neural_P, p_nP = spearmanr(rdm_neural, rdm_physics)
    corr_X_P, _ = spearmanr(rdm_X, rdm_physics)

    print(f"\n  Spearman neural<->X:           r={corr_neural_X:.4f}  (p={p_nX:.2e})")
    print(f"  Spearman neural<->physics(GT): r={corr_neural_P:.4f}  (p={p_nP:.2e})")
    print(f"  Spearman X<->physics(GT):      r={corr_X_P:.4f}")

    # Partial correlation: neural<->physics | X
    partial_P_given_X, partial_p_X = _partial_spearman(rdm_neural, rdm_physics, rdm_X)
    print(
        f"  Partial neural<->physics(GT) | X:    r={partial_P_given_X:.4f}  (p={partial_p_X:.2e})"
    )

    result = {
        "corr_neural_X": corr_neural_X,
        "corr_neural_P": corr_neural_P,
        "corr_X_P": corr_X_P,
        "partial_P_given_X": partial_P_given_X,
        "rdm_neural": rdm_neural,
        "rdm_X": rdm_X,
        "rdm_physics": rdm_physics,
        "n_sub": n_sub,
    }

    # Inferred-physics RDM and correlations
    rdm_physics_inf = None
    if physics_inf_scaled is not None:
        rdm_physics_inf = _compute_rdm(physics_inf_scaled)
        rdm_physics_inf[np.isnan(rdm_physics_inf)] = 0.0

        corr_neural_P_inf, p_nPinf = spearmanr(rdm_neural, rdm_physics_inf)
        corr_X_P_inf, _ = spearmanr(rdm_X, rdm_physics_inf)
        print(
            f"  Spearman neural<->physics(inf): r={corr_neural_P_inf:.4f}  (p={p_nPinf:.2e})"
        )

        partial_P_inf_given_X, _ = _partial_spearman(
            rdm_neural, rdm_physics_inf, rdm_X
        )
        print(
            f"  Partial neural<->physics(inf) | X:   r={partial_P_inf_given_X:.4f}"
        )

        result["rdm_physics_inf"] = rdm_physics_inf
        result["corr_neural_P_inf"] = corr_neural_P_inf
        result["corr_X_P_inf"] = corr_X_P_inf
        result["partial_P_inf_given_X"] = partial_P_inf_given_X

    # Predicted-S RDM and partial controlling for both X and S
    if S_sub is not None:
        rdm_S = _compute_rdm(S_sub)
        rdm_S[np.isnan(rdm_S)] = 0.0
        corr_neural_S, _ = spearmanr(rdm_neural, rdm_S)
        print(f"  Spearman neural<->S:           r={corr_neural_S:.4f}")

        partial_P_given_XS, partial_p_XS = _partial_spearman_2(
            rdm_neural, rdm_physics, rdm_X, rdm_S
        )
        print(
            f"  Partial neural<->physics(GT) | X,S:  r={partial_P_given_XS:.4f}  "
            f"(p={partial_p_XS:.2e})  [KEY]"
        )

        result["rdm_S"] = rdm_S
        result["corr_neural_S"] = corr_neural_S
        result["partial_P_given_XS"] = partial_P_given_XS

        # Inferred-physics partial controlling for both X and S
        if rdm_physics_inf is not None:
            partial_P_inf_given_XS, _ = _partial_spearman_2(
                rdm_neural, rdm_physics_inf, rdm_X, rdm_S
            )
            print(
                f"  Partial neural<->physics(inf) | X,S: r={partial_P_inf_given_XS:.4f}"
            )
            result["partial_P_inf_given_XS"] = partial_P_inf_given_XS

    return result
