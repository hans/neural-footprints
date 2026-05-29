"""
Per-block P contribution diagnostic.

Decomposes neural signal variance into per-block contributions and quantifies,
per block, how much of that contribution is P-decodable under each block norm.

Entry point: run_p_block_contribution(...)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

# Normalization helpers are in the project root — importers must set up sys.path
from neural_model import normalize_block_zscore, normalize_block_truncated_svd
from analyses.encoding import ridge_r2_per_neuron_fast, pca_reduce_pixels


def _reconstruct_block_signals(neural_meta, blocks_raw):
    """
    Reconstruct each block's noiseless contribution to the neural signal.

    Parameters
    ----------
    neural_meta : dict
        From load_neural; must contain W, means, block_sizes, block_k_values,
        and normalization/block_norm info. block_k_values is None for zscore.
    blocks_raw : dict[str, ndarray]
        Raw (un-centered) block arrays, keyed by block name.

    Returns
    -------
    block_signals : dict[str, ndarray]  shape [n_scenes x n_neurons] each
    total_signal  : ndarray             shape [n_scenes x n_neurons]
    block_k_values : list[int]          post-normalization sizes used for W slicing
    """
    W = neural_meta["W"]                       # [n_neurons x D_proj]
    means = neural_meta["means"]               # [D_total]
    block_sizes = neural_meta["block_sizes"]   # list[int]
    block_names = neural_meta["block_names"]   # list[str]
    norm = neural_meta.get("block_norm") or neural_meta.get("normalization", "zscore")

    # Resolve norm string (block_norm is the canonical key saved by gen_neural)
    if norm in ("truncated_svd", "stable_rank_trunc"):
        use_tsvd = True
    else:
        use_tsvd = False

    # block_k_values from metadata under truncated_svd; under zscore = block_sizes
    meta_k = neural_meta.get("block_k_values")
    if meta_k is not None and not use_tsvd:
        # zscore: k values should equal block_sizes — use block_sizes for clarity
        meta_k = None

    block_signals = {}
    k_values = []
    means_start = 0
    W_col_start = 0

    for i, (name, size) in enumerate(zip(block_names, block_sizes)):
        block_raw = blocks_raw[name].astype(np.float32)
        block_means = means[means_start: means_start + size]
        centered = block_raw - block_means

        if use_tsvd:
            # Use the same helper as generate_neural_activity for bit-identical U_k
            U_k, k, _, _ = normalize_block_truncated_svd(centered)
            normalized_block = U_k
        else:
            # zscore: recompute stds from centered block (same algorithm as gen)
            normalized_block, _, _ = normalize_block_zscore(centered)
            k = size

        k_values.append(k)
        W_b = W[:, W_col_start: W_col_start + k]  # [n_neurons x k]
        signal_b = normalized_block @ W_b.T         # [n_scenes x n_neurons]
        block_signals[name] = signal_b

        means_start += size
        W_col_start += k

    total_signal = sum(block_signals[name] for name in block_names)
    return block_signals, total_signal, k_values


def _var_share(signal_b, total_signal):
    """
    Attribution-form variance share: cov(signal_b, total) / var(total),
    summed over neurons.  Sums to 1 across blocks.
    """
    n = signal_b.shape[0]
    # Centre each column separately (mean over scenes)
    sb = signal_b - signal_b.mean(axis=0)
    st = total_signal - total_signal.mean(axis=0)
    cov_sum = (sb * st).sum()       # Σ_n cov(signal_b[:,n], total[:,n]) * (n-1)
    var_sum = (st * st).sum()       # Σ_n var(total[:,n]) * (n-1)
    if var_sum == 0.0:
        return 0.0
    return float(cov_sum / var_sum)


def _var_share_independent(signal_b, total_signal):
    """
    Independent-form variance share: var(signal_b) / var(total),
    summed over neurons.  Does NOT sum to 1 if blocks are correlated.
    """
    st = total_signal - total_signal.mean(axis=0)
    sb = signal_b - signal_b.mean(axis=0)
    var_b = (sb * sb).sum()
    var_t = (st * st).sum()
    if var_t == 0.0:
        return 0.0
    return float(var_b / var_t)


def run_p_block_contribution(
    *,
    neural_meta,
    blocks_raw,
    physics_labels,
    pixel_pca_dim,
    high_dim_blocks=("raw_frames", "fwd_render"),
    seed=42,
):
    """
    Per-block P contribution diagnostic.

    Parameters
    ----------
    neural_meta : dict
        From load_neural; must include W, means, block_sizes, block_names,
        block_k_values (None for zscore), and block_norm.
    blocks_raw : dict[str, ndarray]
        Raw (un-centered) float32 arrays for each block.
    physics_labels : ndarray [n_scenes x n_phys]
        Raw physics labels — standardized internally.
    pixel_pca_dim : int
        PCA dimensionality for high-dim block feature-space decoding.
    high_dim_blocks : tuple[str]
        Blocks that receive PCA reduction before feature-space P decoding.
    seed : int
        Random state passed to pca_reduce_pixels.

    Returns
    -------
    dict with keys: block_names, block_sizes, block_k_values, var_share,
        var_share_independent, r2_P_from_block_raw, r2_P_from_block_signal,
        effective_P_contribution, r2_P_from_total_signal,
        r2_P_per_physics_dim, norm.
    """
    block_names = neural_meta["block_names"]
    block_sizes = neural_meta["block_sizes"]

    # Standardize physics labels (matches encoding analysis)
    scaler = StandardScaler()
    physics_scaled = scaler.fit_transform(physics_labels.astype(np.float64))

    # --- Step 1: Feature-space P decodability (norm-agnostic) ---
    r2_P_from_block_raw = []
    r2_P_per_physics_dim = {}

    for name in block_names:
        block = blocks_raw[name].astype(np.float32)
        if name in high_dim_blocks:
            block_feat, _, _ = pca_reduce_pixels(block, pixel_pca_dim, random_state=seed)
        else:
            block_feat = block.astype(np.float64)

        # ridge_r2_per_neuron_fast(X, Y): predicts each col of Y from X
        # Here X=block_feat, Y=physics_scaled → returns [n_phys] R²s
        # Some physics dims have zero variance (constant) — exclude their nan R²s.
        r2_per_phys = ridge_r2_per_neuron_fast(block_feat, physics_scaled)
        r2_P_per_physics_dim[name] = r2_per_phys.tolist()
        valid = np.isfinite(r2_per_phys)
        r2_mean = float(r2_per_phys[valid].mean()) if valid.any() else float("nan")
        r2_P_from_block_raw.append(r2_mean)

    # --- Step 2: Block-wise neural signal contributions ---
    block_signals, total_signal, k_values = _reconstruct_block_signals(
        neural_meta, blocks_raw
    )

    var_share_list = []
    var_share_indep_list = []
    r2_P_from_block_signal = []
    effective_P_contribution = []

    for name in block_names:
        sig_b = block_signals[name]

        vs = _var_share(sig_b, total_signal)
        vs_indep = _var_share_independent(sig_b, total_signal)
        var_share_list.append(vs)
        var_share_indep_list.append(vs_indep)

        # R²(P → signal_b): predict physics from block signal
        # Exclude constant physics dims (nan R²) from the mean.
        r2_per_phys = ridge_r2_per_neuron_fast(
            sig_b.astype(np.float64), physics_scaled
        )
        valid = np.isfinite(r2_per_phys)
        r2_b = float(r2_per_phys[valid].mean()) if valid.any() else float("nan")
        r2_P_from_block_signal.append(r2_b)
        effective_P_contribution.append(r2_b * vs)

    # --- Step 3: Sanity check — R²(P → total_signal) ---
    r2_per_phys_total = ridge_r2_per_neuron_fast(
        total_signal.astype(np.float64), physics_scaled
    )
    valid_total = np.isfinite(r2_per_phys_total)
    r2_P_from_total_signal = float(
        r2_per_phys_total[valid_total].mean()
    ) if valid_total.any() else float("nan")

    # Resolve norm string for output
    norm = neural_meta.get("block_norm") or neural_meta.get("normalization", "unknown")

    return {
        "block_names": block_names,
        "block_sizes": block_sizes,
        "block_k_values": k_values,
        "var_share": var_share_list,
        "var_share_independent": var_share_indep_list,
        "r2_P_from_block_raw": r2_P_from_block_raw,
        "r2_P_from_block_signal": r2_P_from_block_signal,
        "effective_P_contribution": effective_P_contribution,
        "r2_P_from_total_signal": r2_P_from_total_signal,
        "r2_P_per_physics_dim": r2_P_per_physics_dim,
        "norm": norm,
    }
