"""
Physics-Block Intervention Null for the two partial RSA statistics.

Replaces the free Mantel permutation null (which destroys the regressor<->pixel
correlation and therefore contains neither buried physics NOR leakage) with a
closed-form intervention on the neural generative model. For a derangement pi of
the analyzed scenes, the null world removes each scene's own non-pixel physics
content and injects another scene's:

    neural_null(i) = neural(i)
                     - W_hid @ u_hid(i)  + W_hid @ u_hid(pi(i))
                     - W_inf @ u_inf(i)  + W_inf @ u_inf(pi(i))

raw_frames (X) and fwd_render (S) stay matched to scene i, so the regressor's
pixel pathway — and hence RDM-space leakage — is preserved, while the explicit,
dimensionally-separate physics block is destroyed. See the spec for the full
derivation; key design decisions realized here:

  * Both norms run, output namespaced outputs/{norm}/rsa_null_intervention.json.
  * Blocks are recomputed in-process (no gen_neural schema change); a bit-level
    reproduction gate confirms the recompute matches the persisted neural.npz
    before any draw runs. (pi=identity would NOT test the W-split, so we don't
    rely on it.)
  * Derangement ranges over the 500-scene RSA subsample (seed 123), reusing the
    exact sub_idx so rdm_X / rdm_S / rdm_physics_inf are byte-identical to the
    observed run — only rdm_neural moves.
  * Both-blocks null + inferred-only variant (localizes hidden-layer vs readout).
  * Validation checks V1 (leakage reproduced + hidden pathway removed), V2
    (participation ratio matched), V3 (old null's inadequacy).
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gc
import json
import numpy as np
from scipy.stats import spearmanr

from load_config import load_config
from io_utils import load_scenes, load_neural
from neural_model import normalize_block_truncated_svd, normalize_block_zscore
from analyses.rsa import (
    _build_rsa_rdms,
    _compute_rdm,
    _partial_spearman,
    _partial_spearman_2,
)
from analyses.encoding import pca_reduce_pixels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_brain_pixels(states, metadata):
    """RGBA bytes from the three brain-input frames, concatenated.

    Pure-numpy mirror of scene_generator.extract_brain_pixels, inlined so this
    script does not import scene_generator (which pulls in mujoco/OpenGL and
    cannot load in a headless container). Kept byte-identical to the original.
    """
    fri = metadata["frame_render_indices"]
    rgba_bytes = (
        metadata["target_pixel_indices"].stop - metadata["target_pixel_indices"].start
    )
    return np.concatenate(
        [
            states[:, s.start : s.start + rgba_bytes]
            for s in (fri["initial"], fri["early"], fri["late"])
        ],
        axis=1,
    )

def _reduce_block(block_f32, norm):
    """Per-block post-normalization representation, identical to gen_neural.

    Centers by per-dimension mean (float32, as gen_neural does) then applies the
    block normalization: truncated-SVD stable-rank U_k, or per-dim z-scoring.
    Returns the reduced block (n_scenes x width) that W projects.
    """
    centered = block_f32 - block_f32.mean(axis=0)
    if norm == "truncated_svd":
        U_k, _k, _sr, _bn = normalize_block_truncated_svd(centered)
        return U_k
    if norm == "zscore":
        normed, _stds, _bn = normalize_block_zscore(centered)
        return normed
    raise ValueError(f"unsupported norm {norm!r}")


def _derangement(n, rng):
    """Draw a uniform permutation of range(n) with no fixed points."""
    idx = np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == idx):
            return perm


def _participation_ratio(data):
    """(sum lambda)^2 / sum(lambda^2) of the feature covariance eigenspectrum."""
    x = data - data.mean(axis=0, keepdims=True)
    # Gram on the smaller axis; nonzero eigenvalues match the covariance spectrum.
    n, d = x.shape
    gram = (x @ x.T if n <= d else x.T @ x).astype(np.float64)
    lam = np.linalg.eigvalsh(gram)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def _rdm0(data):
    """Correlation-distance RDM with NaN cells zeroed (as the RSA path does)."""
    r = _compute_rdm(data)
    r[np.isnan(r)] = 0.0
    return r


def _fold_gap(fold, neural, XS_all, Pinf_all, u_hid, u_inf, W_hid, W_inf, n_draws):
    """Observed statistic and its leakage floor on one disjoint scene fold.

    `fold` is a set of scene indices; `XS_all` is (X_all, S_all), the full-scene
    pixel regressors. Builds the fold's four RDMs, computes observed
    partial_P_inf_given_XS, then runs n_draws intervention swaps on the same fold
    for the leakage-floor median. Returns (observed, floor_median, observed-floor).
    """
    from sklearn.preprocessing import StandardScaler

    X_all, S_all = XS_all
    neural_f = neural[fold]
    rdm_n = _rdm0(neural_f)
    rdm_X = _rdm0(X_all[fold])
    rdm_S = _rdm0(S_all[fold])
    rdm_P = _rdm0(StandardScaler().fit_transform(Pinf_all[fold]))

    observed = float(_partial_spearman_2(rdm_n, rdm_P, rdm_X, rdm_S)[0])

    base = neural_f.astype(np.float64)
    C_hid = u_hid[fold] @ W_hid.T
    C_inf = u_inf[fold] @ W_inf.T
    m = len(fold)
    floors = np.empty(n_draws)
    for d in range(n_draws):
        perm = _derangement(m, np.random.default_rng(d))
        nb = base - C_hid - C_inf + C_hid[perm] + C_inf[perm]
        floors[d] = _partial_spearman_2(_rdm0(nb), rdm_P, rdm_X, rdm_S)[0]
    floor_med = float(np.median(floors))
    return observed, floor_med, observed - floor_med


def _summ(values):
    """Distribution summary for a per-draw statistic."""
    values = np.asarray(values, dtype=float)
    return {
        "values": values.tolist(),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p2.5": float(np.percentile(values, 2.5)),
        "p97.5": float(np.percentile(values, 97.5)),
    }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_intervention_null(
    *,
    scenes_path,
    neural_path,
    forward_renders_path,
    pp_activations_path,
    inferred_path,
    out_path,
    norm,
    cfg,
    n_draws=100,
    rsa_results_path=None,
):
    scenes = load_scenes(scenes_path)
    pp = np.load(pp_activations_path)
    fwd = np.load(forward_renders_path)
    neural_saved, meta = load_neural(neural_path)
    inferred_all = np.load(inferred_path)["inferred_physics_all"]

    n_scenes, n_neurons = neural_saved.shape
    noise_level = cfg["noise_level"]
    W_saved = np.asarray(meta["W"])
    D_proj = W_saved.shape[1]
    render_indices = scenes["metadata"]["render_indices"]

    # ---- Reproduction gate ---------------------------------------------------
    # Rebuild the projection block-by-block (the raw/fwd blocks are ~1.2 GB each,
    # so they are reduced and freed one at a time — the full input is never
    # materialized). We reconstruct the full signal Sigma_b u_b @ W_b.T and the
    # noise, then assert the result matches the persisted neural.npz. This is the
    # only proof the k-column split is correct AND that the recomputed reduced
    # blocks match what W actually projected (pi=identity would not test either).
    rng = np.random.default_rng(cfg["random_seed"])
    # generate_neural_activity draws W first, then noise, from this same stream.
    W_draw = rng.normal(0, 1.0 / np.sqrt(D_proj), size=(n_neurons, D_proj)).astype(
        np.float32
    )
    assert np.allclose(W_draw, W_saved, rtol=1e-5, atol=1e-6), (
        f"[{norm}] W is not reproducible from seed {cfg['random_seed']} "
        f"(max|dW|={np.abs(W_draw - W_saved).max():.3e})"
    )
    del W_draw

    saved_widths = list(
        meta["block_k_values"] if norm == "truncated_svd" else meta["block_sizes"]
    )
    block_order = ["raw_frames", "fwd_render", "hidden_acts", "inferred_physics"]

    # Build the ~1.2 GB raw-frame block once (reused for the gate and the X
    # regressor), then free the scene render buffers so peak memory stays bounded.
    raw_block = np.concatenate(
        [scenes["initial_renders"], scenes["early_renders"], scenes["late_renders"]],
        axis=1,
    ).astype(np.float32)
    for k in ("initial_renders", "early_renders", "late_renders"):
        scenes.pop(k, None)
    gc.collect()

    raw_pixel_pca, _, _ = pca_reduce_pixels(raw_block, cfg["pixel_pca_dim"])

    signal = np.zeros((n_scenes, n_neurons), dtype=np.float64)
    kept = {}  # hidden/inferred reduced blocks + their W columns, for the swap
    col = 0
    widths = []
    for name in block_order:
        if name == "raw_frames":
            block = raw_block
        elif name == "fwd_render":
            block = fwd["forward_program_states"][:, render_indices].astype(np.float32)
        elif name == "hidden_acts":
            block = pp["hidden_acts"].astype(np.float32)
        else:
            block = pp["inferred_physics"].astype(np.float32)

        u = _reduce_block(block, norm)  # native (float32); big blocks stay f32
        if name != "raw_frames":
            del block
        w = u.shape[1]
        W_b = W_saved[:, col : col + w]
        signal += u @ W_b.T  # f32 matmul -> tiny (n_scenes x n_neurons), upcast on add
        if name in ("hidden_acts", "inferred_physics"):
            # Tiny blocks — promote to f64 for a well-conditioned swap.
            kept[name] = (u.astype(np.float64), W_b.astype(np.float64))
        del u
        widths.append(w)
        col += w
        gc.collect()
    del raw_block
    gc.collect()
    assert col == D_proj, (col, D_proj)
    assert widths == saved_widths, (widths, saved_widths)

    signal_std = float(signal.std())
    noise = noise_level * signal_std * rng.normal(0, 1, size=signal.shape)
    neural_regen = (signal + noise).astype(np.float32)
    del signal, noise

    max_abs = float(np.abs(neural_regen - neural_saved).max())
    scale = float(np.abs(neural_saved).mean())
    max_rel = max_abs / scale if scale > 0 else float("inf")
    repro_ok = bool(np.allclose(neural_regen, neural_saved, rtol=1e-3, atol=1e-4))
    if not repro_ok:
        raise AssertionError(
            f"[{norm}] reproduction gate FAILED: max_abs={max_abs:.3e} "
            f"max_rel={max_rel:.3e}. The recomputed projection diverges from the "
            f"persisted neural.npz (likely a cross-environment SVD/BLAS mismatch); "
            f"the closed-form null is invalid — regenerate neural per draw instead."
        )
    print(f"[{norm}] reproduction gate OK  (max_abs={max_abs:.3e}, max_rel={max_rel:.3e})")

    u_hid, W_hid = kept["hidden_acts"]
    u_inf, W_inf = kept["inferred_physics"]

    # ---- Fixed RDMs on the identical subsample (X, S, P_inf regressor) --------
    # raw_pixel_pca (X) already computed above from raw_block.
    predicted_brain_pixels = _extract_brain_pixels(
        fwd["forward_program_states"], scenes["metadata"]
    )
    predicted_pixel_pca, _, _ = pca_reduce_pixels(
        predicted_brain_pixels, cfg["pixel_pca_dim"]
    )
    del predicted_brain_pixels, fwd, pp

    rdms = _build_rsa_rdms(
        neural_regen,
        scenes,
        raw_pixel_pca=raw_pixel_pca,
        rsa_subsample=cfg["rsa_subsample"],
        predicted_pixel_pca=predicted_pixel_pca,
        inferred_physics_labels=inferred_all,
    )
    sub_idx = rdms["sub_idx"]
    n_sub = rdms["n_sub"]
    rdm_X = rdms["rdm_X"]
    rdm_S = rdms["rdm_S"]
    rdm_physics_inf = rdms["rdm_physics_inf"]
    rdm_neural_obs = rdms["rdm_neural"]

    # ---- Observed statistics (apples-to-apples with the draws) ---------------
    obs_partial_XS = float(
        _partial_spearman_2(rdm_neural_obs, rdm_physics_inf, rdm_X, rdm_S)[0]
    )
    obs_partial_X = float(_partial_spearman(rdm_neural_obs, rdm_physics_inf, rdm_X)[0])
    obs_corr = float(spearmanr(rdm_neural_obs, rdm_physics_inf)[0])

    base_sub = neural_regen[sub_idx].astype(np.float64)
    obs_pr = _participation_ratio(base_sub)

    # Precompute the two physics-block contributions on the subsample rows.
    C_hid = u_hid[sub_idx] @ W_hid.T  # [n_sub x n_neurons]
    C_inf = u_inf[sub_idx] @ W_inf.T

    # Consistency vs the persisted RSA result (should match within Spearman/PCA
    # determinism; recompute is the source of truth for the null comparison).
    consistency = {}
    if rsa_results_path and os.path.exists(rsa_results_path):
        with open(rsa_results_path) as fh:
            saved = json.load(fh)
        consistency = {
            "rsa_results_partial_P_inf_given_XS": saved.get("partial_P_inf_given_XS"),
            "recomputed_partial_P_inf_given_XS": obs_partial_XS,
            "abs_diff": (
                abs(saved.get("partial_P_inf_given_XS", float("nan")) - obs_partial_XS)
            ),
        }

    # ---- Draws ---------------------------------------------------------------
    both = {"XS": [], "X": [], "corr": [], "pr": []}
    info = {"XS": [], "X": [], "corr": [], "pr": []}
    # V3: does free permutation of the regressor destroy its pixel correlation?
    freeperm_corr_X_Pinf = []
    physics_inf_scaled = rdms["physics_inf_scaled"]

    for d in range(n_draws):
        rng = np.random.default_rng(d)
        perm = _derangement(n_sub, rng)

        # both-blocks: swap hidden + inferred coherently
        nb = base_sub - C_hid - C_inf + C_hid[perm] + C_inf[perm]
        rb = _compute_rdm(nb)
        rb[np.isnan(rb)] = 0.0
        both["XS"].append(_partial_spearman_2(rb, rdm_physics_inf, rdm_X, rdm_S)[0])
        both["X"].append(_partial_spearman(rb, rdm_physics_inf, rdm_X)[0])
        both["corr"].append(spearmanr(rb, rdm_physics_inf)[0])
        both["pr"].append(_participation_ratio(nb))

        # inferred-only: swap inferred readout alone (hidden stays matched)
        ni = base_sub - C_inf + C_inf[perm]
        ri = _compute_rdm(ni)
        ri[np.isnan(ri)] = 0.0
        info["XS"].append(_partial_spearman_2(ri, rdm_physics_inf, rdm_X, rdm_S)[0])
        info["X"].append(_partial_spearman(ri, rdm_physics_inf, rdm_X)[0])
        info["corr"].append(spearmanr(ri, rdm_physics_inf)[0])
        info["pr"].append(_participation_ratio(ni))

        # V3: free-permute the regressor rows, recompute its pixel-RDM correlation
        phys_perm = physics_inf_scaled[perm]
        rdm_pinf_perm = _compute_rdm(phys_perm)
        rdm_pinf_perm[np.isnan(rdm_pinf_perm)] = 0.0
        freeperm_corr_X_Pinf.append(spearmanr(rdm_X, rdm_pinf_perm)[0])

        if (d + 1) % 20 == 0 or d == 0:
            print(
                f"[{norm}] draw {d + 1}/{n_draws}: "
                f"both XS={both['XS'][-1]:.4f}  inf XS={info['XS'][-1]:.4f}"
            )

    both_XS = _summ(both["XS"])
    info_XS = _summ(info["XS"])
    obs_corr_X_Pinf = float(spearmanr(rdm_X, rdm_physics_inf)[0])
    freeperm_mean = float(np.mean(freeperm_corr_X_Pinf))

    # ---- Disjoint-fold cross-validation of the gap ---------------------------
    # Scene-sampling uncertainty via independent size-n_sub folds partitioned
    # (without replacement) from the full scene set — no with-replacement
    # duplicate-pair hack. Each fold yields its own observed and leakage floor,
    # so gap = observed - floor is self-contained; the spread across folds is a
    # clean t-CI. Headline point estimates stay on the seed-123 subsample.
    from scipy.stats import t as _t_dist

    K = n_scenes // n_sub
    fold_order = np.random.default_rng(12345).permutation(n_scenes)[: K * n_sub]
    folds = fold_order.reshape(K, n_sub)
    print(f"[{norm}] disjoint-fold CV: {K} folds x {n_sub} scenes...")
    fold_rows = []
    for k in range(K):
        obs_k, floor_k, gap_k = _fold_gap(
            folds[k], neural_regen, (raw_pixel_pca, predicted_pixel_pca),
            inferred_all, u_hid, u_inf, W_hid, W_inf, n_draws=n_draws,
        )
        fold_rows.append({"observed": obs_k, "leakage_floor_median": floor_k, "gap": gap_k})
        print(f"[{norm}]   fold {k}: obs={obs_k:.4f} floor={floor_k:.4f} gap={gap_k:+.4f}")

    gaps = np.array([r["gap"] for r in fold_rows])
    gap_mean = float(gaps.mean())
    gap_sd = float(gaps.std(ddof=1))
    tcrit = float(_t_dist.ppf(0.975, K - 1))
    half = tcrit * gap_sd / np.sqrt(K)
    gci_lo, gci_hi = gap_mean - half, gap_mean + half
    fold_cv = {
        "n_folds": K,
        "fold_size": int(n_sub),
        "fold_seed": 12345,
        "folds": fold_rows,
        "gap_mean": gap_mean,
        "gap_sd": gap_sd,
        "gap_t_ci_lo": float(gci_lo),
        "gap_t_ci_hi": float(gci_hi),
        "resolved_above_leakage": bool(gci_lo > 0.0),
        "note": (
            f"gap = observed - leakage-floor median, per disjoint {n_sub}-scene fold. "
            f"95% t-CI on the mean gap ({K - 1} df). resolved_above_leakage: CI lower "
            "bound > 0. False => consistent with all-leakage."
        ),
    }

    # ---- Validation checks ---------------------------------------------------
    v1_above_zero = both_XS["p2.5"] > 0.0
    v1_le_inferred = both_XS["median"] <= info_XS["median"] + 1e-6
    v1 = {
        "both_median": both_XS["median"],
        "both_p2.5": both_XS["p2.5"],
        "inferred_median": info_XS["median"],
        "both_above_zero": bool(v1_above_zero),
        "both_le_inferred": bool(v1_le_inferred),
        "passed": bool(v1_above_zero and v1_le_inferred),
        "note": (
            "V1 needs BOTH: null centered above zero (leakage reproduced) AND "
            "both-blocks <= inferred-only (hidden pathway actually removed)."
        ),
    }
    both_pr = _summ(both["pr"])
    info_pr = _summ(info["pr"])
    v2 = {
        "observed_pr": obs_pr,
        "both_pr_median": both_pr["median"],
        "inferred_pr_median": info_pr["median"],
        "ratio_both_over_observed": both_pr["median"] / obs_pr if obs_pr else float("nan"),
        "note": "Expect ~1: physics dims are a negligible fraction of total structure.",
    }
    v3 = {
        "observed_corr_X_Pinf": obs_corr_X_Pinf,
        "freeperm_corr_X_Pinf_mean": freeperm_mean,
        "note": (
            "Free permutation drives the regressor<->pixel RDM correlation toward "
            "zero, so residualization has nothing to remove — why the old null "
            "contained neither leakage nor buried physics."
        ),
    }

    result = {
        "norm": norm,
        "n_draws": n_draws,
        "n_sub": int(n_sub),
        "reproduction": {
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "passed": repro_ok,
        },
        "observed": {
            "partial_P_inf_given_XS": obs_partial_XS,
            "partial_P_inf_given_X": obs_partial_X,
            "corr_neural_P_inf": obs_corr,
            "participation_ratio": obs_pr,
        },
        "both_blocks": {
            "partial_P_inf_given_XS": both_XS,
            "partial_P_inf_given_X": _summ(both["X"]),
            "corr_neural_P_inf": _summ(both["corr"]),
            "participation_ratio": both_pr,
        },
        "inferred_only": {
            "partial_P_inf_given_XS": info_XS,
            "partial_P_inf_given_X": _summ(info["X"]),
            "corr_neural_P_inf": _summ(info["corr"]),
            "participation_ratio": info_pr,
        },
        "observed_minus_null_median": {
            "both_blocks_partial_P_inf_given_XS": obs_partial_XS - both_XS["median"],
            "inferred_only_partial_P_inf_given_XS": obs_partial_XS - info_XS["median"],
        },
        "disjoint_fold_cv": fold_cv,
        "V1_leakage_and_removal": v1,
        "V2_participation_ratio": v2,
        "V3_old_null_inadequacy": v3,
        "consistency_vs_rsa_results": consistency,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{norm}] wrote {out_path}")
    print(
        f"[{norm}] observed XS={obs_partial_XS:.4f}  "
        f"both-null median={both_XS['median']:.4f} "
        f"[{both_XS['p2.5']:.4f},{both_XS['p97.5']:.4f}]  "
        f"inf-null median={info_XS['median']:.4f}  "
        f"obs-both_median={obs_partial_XS - both_XS['median']:.4f}"
    )
    print(
        f"[{norm}] fold-CV gap mean={gap_mean:+.4f}  95% t-CI=[{gci_lo:.4f},{gci_hi:.4f}]  "
        f"resolved_above_leakage={fold_cv['resolved_above_leakage']}"
    )
    return result


def main():
    cfg = load_config()
    if "snakemake" in globals():
        smk = globals()["snakemake"]
        norm = smk.wildcards.norm
        run_intervention_null(
            scenes_path=smk.input.scenes,
            neural_path=smk.input.neural,
            forward_renders_path=smk.input.forward_renders,
            pp_activations_path=smk.input.pp_activations,
            inferred_path=smk.input.inferred,
            out_path=smk.output.results,
            norm=norm,
            cfg=cfg,
            n_draws=smk.params.get("n_draws", 100),
            rsa_results_path=smk.input.get("rsa_results", None),
        )
    else:
        # Standalone: `python scripts/run_rsa_null_intervention.py <norm> [n_draws]`
        norm = sys.argv[1] if len(sys.argv) > 1 else cfg.get("block_norm", "truncated_svd")
        n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        run_intervention_null(
            scenes_path="data/scenes.npz",
            neural_path=f"data/{norm}/neural.npz",
            forward_renders_path="data/forward_renders.npz",
            pp_activations_path="data/pp_activations.npz",
            inferred_path=f"data/{norm}/inferred_physics.npz",
            out_path=f"outputs/{norm}/rsa_null_intervention.json",
            norm=norm,
            cfg=cfg,
            n_draws=n_draws,
            rsa_results_path=f"outputs/{norm}/rsa_results.json",
        )


if __name__ == "__main__" or "snakemake" in globals():
    main()
