"""
Neural activity generation via random linear projection of raw program state.

A single random matrix W projects the entire program state (render + physics bytes)
into simulated neural activity. The render/physics variance ratio is NOT a parameter —
it emerges from the structure of the program state.
"""

import numpy as np

from config import N_NEURONS as _CFG_N_NEURONS, NOISE_LEVEL as _CFG_NOISE_LEVEL


def generate_neural_activity(
    program_states,
    seed,
    *,
    n_neurons=None,
    noise_level=None,
    block_sizes=None,
    normalization="operator_norm",
):
    """
    Generate neural activity from raw program states via random linear projection.

    Parameters
    ----------
    program_states : ndarray [n_scenes x D]
        Raw bytes cast to float32.
    seed : int
        Random seed for reproducibility.
    block_sizes : list[int] or None
        Sizes of consecutive blocks in program_states. If None, treats entire D
        as one block.
    normalization : {"operator_norm", "stable_rank_trunc"}
        Per-block normalization scheme.

        "operator_norm" (default): divide each block by its largest singular
        value (operator norm). Preserves the original D-dimensional input to W.

        "stable_rank_trunc": truncate each block to its stable rank k =
        max(1, round(||X||_F² / ||X||_op²)) via truncated SVD (Gram-matrix
        eigendecomposition), then whiten within that subspace. W is sized
        (n_neurons, sum_of_ks) instead of (n_neurons, D_total). Unit-norm
        columns from eigendecomposition serve as the whitened scores.

    Returns
    -------
    neural_activity : ndarray [n_scenes x N_NEURONS]
    metadata : dict
        Contains W matrix and variance diagnostics.
    """
    if n_neurons is None:
        n_neurons = _CFG_N_NEURONS
    if noise_level is None:
        noise_level = _CFG_NOISE_LEVEL

    rng = np.random.default_rng(seed)
    n_scenes, D = program_states.shape

    # Step 1: Center per dimension
    means = program_states.mean(axis=0)
    centered = program_states - means

    if block_sizes is None:
        block_sizes = [D]

    if normalization == "operator_norm":
        # Divide each block by its largest singular value in-place.
        # In-place avoids allocating another full copy of the (potentially large) matrix.
        block_norms = []
        start = 0
        for size in block_sizes:
            block = centered[:, start : start + size]
            if size == 0:
                sigma = 1.0
            elif min(block.shape) <= 1:
                sigma = float(np.linalg.norm(block, ord=2))
                if sigma == 0.0:
                    sigma = 1.0
            else:
                # Gram matrix on the smaller axis — stays float64 for numerical safety.
                n, d = block.shape
                gram = (block @ block.T if n <= d else block.T @ block).astype(
                    np.float64
                )
                sigma = float(np.sqrt(np.linalg.eigvalsh(gram).max()))
                if sigma == 0.0:
                    sigma = 1.0
            block_norms.append(sigma)
            centered[:, start : start + size] /= sigma
            start += size

        normalized = centered
        D_proj = D
        block_stable_ranks = None
        block_k_values = None

    elif normalization == "stable_rank_trunc":
        # Truncate each block to its stable rank via Gram-matrix eigendecomposition.
        # Uses U[:, :k] (unit-norm columns = whitened scores within subspace).
        # W is then sized (n_neurons, sum_of_ks).
        blocks_reduced = []
        block_stable_ranks = []
        block_k_values = []
        block_norms = []
        start = 0
        for size in block_sizes:
            block = centered[:, start : start + size].astype(np.float64)
            if size == 0:
                block_stable_ranks.append(0.0)
                block_k_values.append(0)
                block_norms.append(1.0)
                start += size
                continue

            n, d = block.shape
            if min(n, d) <= 1:
                # Trivially rank-1: normalise to unit norm and keep.
                norm = float(np.linalg.norm(block))
                if norm == 0.0:
                    norm = 1.0
                blocks_reduced.append((block / norm).astype(np.float32))
                block_stable_ranks.append(1.0)
                block_k_values.append(1)
                block_norms.append(norm)
                start += size
                continue

            if n <= d:
                # n×n Gram matrix — eigvecs are left singular vectors U.
                gram = (block @ block.T).astype(np.float64)
                eigvals, eigvecs = np.linalg.eigh(gram)  # ascending order
                sr = float(eigvals.sum() / eigvals.max()) if eigvals.max() > 0 else 1.0
                k = max(1, round(sr))
                # Top-k eigvecs (last k columns in ascending-order output).
                U_k = eigvecs[:, -k:].astype(np.float32)
                blocks_reduced.append(U_k)
            else:
                # d×d Gram matrix — eigvecs are right singular vectors V.
                # Recover U = X V / S for unit-norm columns.
                gram = (block.T @ block).astype(np.float64)
                eigvals, eigvecs = np.linalg.eigh(gram)  # ascending order
                sr = float(eigvals.sum() / eigvals.max()) if eigvals.max() > 0 else 1.0
                k = max(1, round(sr))
                V_k = eigvecs[:, -k:]  # (d, k) — top-k right singular vectors
                S_k = np.sqrt(np.maximum(eigvals[-k:], 0.0))  # (k,)
                # Guard divide-by-zero for near-zero singular values
                S_k = np.where(S_k > 0, S_k, 1.0)
                U_k = (block @ V_k / S_k[None, :]).astype(np.float32)  # (n, k)
                blocks_reduced.append(U_k)

            block_stable_ranks.append(sr)
            block_k_values.append(k)
            block_norms.append(
                float(np.sqrt(eigvals.max())) if eigvals.max() > 0 else 1.0
            )
            start += size

        normalized = np.concatenate(blocks_reduced, axis=1)  # (n_scenes, sum_of_ks)
        D_proj = normalized.shape[1]

    else:
        raise ValueError(f"Unknown normalization: {normalization!r}")

    # Random projection matrix — scale by 1/sqrt(D_proj) where D_proj is the
    # actual input dimensionality after normalization.
    W = rng.normal(0, 1.0 / np.sqrt(D_proj), size=(n_neurons, D_proj)).astype(
        np.float32
    )

    signal = normalized @ W.T  # [n_scenes x n_neurons]

    signal_std = signal.std()
    noise = noise_level * signal_std * rng.normal(0, 1, size=signal.shape)

    neural_activity = signal + noise

    var_per_dim = normalized.var(axis=0)
    total_var = float(var_per_dim.sum())

    metadata = {
        "W": W,
        "means": means,
        "block_norms": np.array(block_norms),
        "signal_std": signal_std,
        "var_per_dim": var_per_dim,
        "total_var": total_var,
        "normalization": normalization,
        "block_stable_ranks": (
            np.array(block_stable_ranks) if block_stable_ranks is not None else None
        ),
        "block_k_values": (
            np.array(block_k_values, dtype=int) if block_k_values is not None else None
        ),
    }

    return neural_activity, metadata


def print_variance_diagnostic(
    scene_metadata, neural_metadata, block_sizes, block_names=None
):
    """
    Print the key diagnostic: how much variance comes from render vs physics slices.

    This ratio is NOT set by a parameter — it is printed as a finding.
    Only meaningful for the "operator_norm" normalization where var_per_dim
    indices still correspond to original input dimensions.
    """
    D_render = scene_metadata["D_render_bytes"]
    D_physics = (
        scene_metadata["D_physics_labels"]
        + scene_metadata["D_scene_config"]
        + scene_metadata.get("D_scene_lighting", 0)
    )

    var_per_dim = neural_metadata["var_per_dim"]
    block_norms = neural_metadata["block_norms"]

    render_var = var_per_dim[:D_render].sum()
    physics_var = var_per_dim[D_render:].sum()
    total_var = neural_metadata["total_var"]

    render_frac = render_var / total_var * 100
    physics_frac = physics_var / total_var * 100
    ratio = D_render / D_physics

    if block_names is None:
        block_names = [f"block_{i}" for i in range(len(block_sizes))]
    norm_strs = "  ".join(
        f"{name}={norm:.3g}" for name, norm in zip(block_names, block_norms)
    )

    print("\n" + "=" * 60)
    print("VARIANCE DIAGNOSTIC (key result)")
    print("=" * 60)
    print(
        f"Program state: D_render={D_render}, D_physics={D_physics}, ratio={ratio:.1f}x"
    )
    print(f"Block operator norms: {norm_strs}")
    print(f"Variance fraction from render slice:  {render_frac:.1f}%")
    print(f"Variance fraction from physics slice: {physics_frac:.1f}%")
    print(f"Total normalized variance: {total_var:.1f}")
    print(f"Signal std: {neural_metadata['signal_std']:.4f}")
    print("=" * 60 + "\n")
