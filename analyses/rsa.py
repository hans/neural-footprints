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


def _rsa_null_summary(prefix, perm_values, observed, two_sided=False):
    """Per-perm scalar null → 95% CI + p-value.

    two_sided: use |perm| >= |observed| (for KEY ≈-0 tests);
               else one-sided (perm >= observed, for expected-positive tests).
    """
    lo, hi = np.percentile(perm_values, [2.5, 97.5])
    if two_sided:
        pvalue = float((np.abs(perm_values) >= abs(observed)).mean())
    else:
        pvalue = float((perm_values >= observed).mean())
    return {
        f"null_{prefix}_perm_values": perm_values,
        f"null_{prefix}_ci_lo": float(lo),
        f"null_{prefix}_ci_hi": float(hi),
        f"null_{prefix}_mean": float(perm_values.mean()),
        f"null_{prefix}_pvalue": pvalue,
        f"null_{prefix}_observed": float(observed),
    }


def _compute_rsa_null_distribution(
    physics_scaled_sub,
    rdm_neural,
    rdm_X,
    *,
    rdm_S=None,
    physics_inf_scaled_sub=None,
    observed_corr_P,
    observed_partial_P_given_X,
    observed_partial_P_given_XS=None,
    observed_corr_P_inf=None,
    observed_partial_P_inf_given_X=None,
    observed_partial_P_inf_given_XS=None,
    n_permutations=300,
    seed=0,
):
    """Mantel permutation null for RSA partial correlations.

    Permutes scene rows of physics_scaled_sub (not RDM cells), re-derives the
    physics RDM, then recomputes partial Spearman correlations against the fixed
    rdm_neural / rdm_X / rdm_S. The same row permutation is applied to
    physics_inf_scaled_sub for apples-to-apples inferred-physics comparison.
    """
    n_sub = physics_scaled_sub.shape[0]
    has_S = rdm_S is not None
    has_inf = physics_inf_scaled_sub is not None

    rng = np.random.default_rng(seed)

    corr_P_null = np.empty(n_permutations)
    partial_X_null = np.empty(n_permutations)
    partial_XS_null = np.empty(n_permutations) if has_S else None
    corr_P_inf_null = np.empty(n_permutations) if has_inf else None
    partial_inf_X_null = np.empty(n_permutations) if has_inf else None
    partial_inf_XS_null = np.empty(n_permutations) if (has_inf and has_S) else None

    print(f"\nComputing RSA permutation null ({n_permutations} shuffles)...")
    for p in range(n_permutations):
        perm = rng.permutation(n_sub)
        physics_perm = physics_scaled_sub[perm]
        rdm_physics_perm = _compute_rdm(physics_perm)
        rdm_physics_perm[np.isnan(rdm_physics_perm)] = 0.0

        corr_P_null[p] = spearmanr(rdm_neural, rdm_physics_perm)[0]
        partial_X_null[p] = _partial_spearman(rdm_neural, rdm_physics_perm, rdm_X)[0]
        if has_S:
            partial_XS_null[p] = _partial_spearman_2(
                rdm_neural, rdm_physics_perm, rdm_X, rdm_S
            )[0]

        if has_inf:
            inf_perm = physics_inf_scaled_sub[perm]
            rdm_physics_inf_perm = _compute_rdm(inf_perm)
            rdm_physics_inf_perm[np.isnan(rdm_physics_inf_perm)] = 0.0
            corr_P_inf_null[p] = spearmanr(rdm_neural, rdm_physics_inf_perm)[0]
            partial_inf_X_null[p] = _partial_spearman(
                rdm_neural, rdm_physics_inf_perm, rdm_X
            )[0]
            if has_S:
                partial_inf_XS_null[p] = _partial_spearman_2(
                    rdm_neural, rdm_physics_inf_perm, rdm_X, rdm_S
                )[0]

        if (p + 1) % 50 == 0 or p == 0:
            print(f"  perm {p + 1}/{n_permutations}: corr_P_null = {corr_P_null[p]:.4f}")

    out = {}
    out.update(_rsa_null_summary("corr_neural_P", corr_P_null, observed_corr_P, two_sided=False))
    out.update(_rsa_null_summary("partial_P_given_X", partial_X_null, observed_partial_P_given_X, two_sided=False))
    if has_S and observed_partial_P_given_XS is not None:
        out.update(_rsa_null_summary("partial_P_given_XS", partial_XS_null, observed_partial_P_given_XS, two_sided=True))
    if has_inf:
        out.update(_rsa_null_summary("corr_neural_P_inf", corr_P_inf_null, observed_corr_P_inf, two_sided=False))
        out.update(_rsa_null_summary("partial_P_inf_given_X", partial_inf_X_null, observed_partial_P_inf_given_X, two_sided=False))
        if has_S and observed_partial_P_inf_given_XS is not None:
            out.update(_rsa_null_summary("partial_P_inf_given_XS", partial_inf_XS_null, observed_partial_P_inf_given_XS, two_sided=True))
    return out


def _empty_rsa_null_results():
    nan = float("nan")
    out = {}
    for prefix in (
        "corr_neural_P",
        "partial_P_given_X",
        "partial_P_given_XS",
        "corr_neural_P_inf",
        "partial_P_inf_given_X",
        "partial_P_inf_given_XS",
    ):
        out.update({
            f"null_{prefix}_perm_values": np.empty(0),
            f"null_{prefix}_ci_lo": nan,
            f"null_{prefix}_ci_hi": nan,
            f"null_{prefix}_mean": nan,
            f"null_{prefix}_pvalue": nan,
            f"null_{prefix}_observed": nan,
        })
    return out


def _build_rsa_rdms(
    neural_activity,
    scenes,
    *,
    raw_pixel_pca,
    rsa_subsample,
    predicted_pixel_pca=None,
    inferred_physics_labels=None,
    subsample_seed=123,
):
    """Subsample scenes and build the fixed RDMs used by RSA.

    Shared by run_rsa_analysis and scripts/run_rsa_null_intervention.py so both
    analyze the identical subsample (seed 123) with identical scaling/RDMs. Only
    the neural RDM changes under the intervention null; X / S / physics(inf) RDMs
    are computed here once and reused across draws.

    Returns a dict with sub_idx, neural_sub, the scaled physics regressors, and
    the correlation-distance RDMs (rdm_S / rdm_physics_inf are None when their
    inputs are absent). NaN cells are zeroed exactly as the legacy inline code did.
    """
    physics_labels = scenes["physics_labels"]
    n_scenes = neural_activity.shape[0]
    n_sub = min(rsa_subsample, n_scenes)

    rng = np.random.default_rng(subsample_seed)
    sub_idx = rng.choice(n_scenes, size=n_sub, replace=False)
    sub_idx.sort()

    neural_sub = neural_activity[sub_idx]
    X_sub = raw_pixel_pca[sub_idx]
    physics_sub = physics_labels[sub_idx]
    S_sub = predicted_pixel_pca[sub_idx] if predicted_pixel_pca is not None else None
    physics_inf_sub = (
        inferred_physics_labels[sub_idx] if inferred_physics_labels is not None else None
    )

    physics_scaled = StandardScaler().fit_transform(physics_sub)
    physics_inf_scaled = (
        StandardScaler().fit_transform(physics_inf_sub)
        if physics_inf_sub is not None
        else None
    )

    def _rdm(data):
        r = _compute_rdm(data)
        r[np.isnan(r)] = 0.0
        return r

    rdm_neural = _rdm(neural_sub)
    rdm_X = _rdm(X_sub)
    rdm_physics = _rdm(physics_scaled)
    rdm_S = _rdm(S_sub) if S_sub is not None else None
    rdm_physics_inf = _rdm(physics_inf_scaled) if physics_inf_scaled is not None else None

    return {
        "n_sub": n_sub,
        "sub_idx": sub_idx,
        "neural_sub": neural_sub,
        "S_sub": S_sub,
        "physics_scaled": physics_scaled,
        "physics_inf_scaled": physics_inf_scaled,
        "rdm_neural": rdm_neural,
        "rdm_X": rdm_X,
        "rdm_physics": rdm_physics,
        "rdm_S": rdm_S,
        "rdm_physics_inf": rdm_physics_inf,
    }


def run_rsa_analysis(
    neural_activity,
    scenes,
    neural_meta,
    *,
    raw_pixel_pca,
    rsa_subsample=None,
    predicted_pixel_pca=None,
    inferred_physics_labels=None,
    compute_null=True,
    n_null_permutations=300,
    null_seed=0,
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

    rdms = _build_rsa_rdms(
        neural_activity,
        scenes,
        raw_pixel_pca=raw_pixel_pca,
        rsa_subsample=rsa_subsample,
        predicted_pixel_pca=predicted_pixel_pca,
        inferred_physics_labels=inferred_physics_labels,
    )
    n_sub = rdms["n_sub"]
    S_sub = rdms["S_sub"]
    physics_scaled = rdms["physics_scaled"]
    physics_inf_scaled = rdms["physics_inf_scaled"]
    rdm_neural = rdms["rdm_neural"]
    rdm_X = rdms["rdm_X"]
    rdm_physics = rdms["rdm_physics"]

    print(f"\nSubsampled {n_sub} scenes for RSA.")
    print("Computing RDMs...")

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
    rdm_physics_inf = rdms["rdm_physics_inf"]
    if physics_inf_scaled is not None:
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
        rdm_S = rdms["rdm_S"]
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

    # --- Permutation null ---
    if compute_null:
        null_results = _compute_rsa_null_distribution(
            physics_scaled,
            rdm_neural,
            rdm_X,
            rdm_S=result.get("rdm_S"),
            physics_inf_scaled_sub=physics_inf_scaled,
            observed_corr_P=corr_neural_P,
            observed_partial_P_given_X=partial_P_given_X,
            observed_partial_P_given_XS=result.get("partial_P_given_XS"),
            observed_corr_P_inf=result.get("corr_neural_P_inf"),
            observed_partial_P_inf_given_X=result.get("partial_P_inf_given_X"),
            observed_partial_P_inf_given_XS=result.get("partial_P_inf_given_XS"),
            n_permutations=n_null_permutations,
            seed=null_seed,
        )
    else:
        null_results = _empty_rsa_null_results()
    result.update(null_results)

    return result
