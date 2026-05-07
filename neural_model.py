"""
Neural activity generation via random linear projection of raw program state.

A single random matrix W projects the entire program state (render + physics bytes)
into simulated neural activity. The render/physics variance ratio is NOT a parameter —
it emerges from the structure of the program state.
"""

import numpy as np
from scipy.sparse.linalg import svds

from config import N_NEURONS as _CFG_N_NEURONS, NOISE_LEVEL as _CFG_NOISE_LEVEL


def generate_neural_activity(program_states, seed, *, n_neurons=None, noise_level=None,
                              block_sizes=None):
    """
    Generate neural activity from raw program states via random linear projection.

    Parameters
    ----------
    program_states : ndarray [n_scenes x D]
        Raw bytes cast to float32.
    seed : int
        Random seed for reproducibility.
    block_sizes : list[int] or None
        Sizes of consecutive blocks in program_states for per-block operator-norm
        normalization (render, physics, config, lighting). If None, treats the
        entire D as one block.

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

    # Step 2: Per-block operator-norm normalization
    # Each block is divided by its largest singular value so the trace ratio
    # across blocks reflects intrinsic-dim (participation ratio) asymmetry,
    # not raw amplitude differences.
    if block_sizes is None:
        block_sizes = [D]

    blocks = []
    block_norms = []
    start = 0
    for size in block_sizes:
        block = centered[:, start:start + size]
        if size == 0:
            sigma = 1.0
        elif min(block.shape) <= 1:
            # svds requires k < min(shape); use full norm for tiny blocks.
            sigma = float(np.linalg.norm(block, ord=2))
            if sigma == 0.0:
                sigma = 1.0
        else:
            sigma = float(svds(block.astype(np.float64), k=1,
                               return_singular_vectors=False, solver='arpack')[0])
            if sigma == 0.0:
                sigma = 1.0
        block_norms.append(sigma)
        blocks.append((block / sigma).astype(program_states.dtype))
        start += size

    normalized = np.concatenate(blocks, axis=1)  # [n_scenes x D]

    # Step 3: Random projection matrix
    W = rng.normal(0, 1.0 / np.sqrt(D), size=(n_neurons, D))

    # Step 4: Signal
    signal = normalized @ W.T  # [n_scenes x n_neurons]

    # Step 5: Noise
    signal_std = signal.std()
    noise = noise_level * signal_std * rng.normal(0, 1, size=signal.shape)

    # Step 6: Neural activity
    neural_activity = signal + noise

    # --- Variance diagnostics ---
    var_per_dim = normalized.var(axis=0)
    total_var = var_per_dim.sum()

    metadata = {
        'W': W,
        'means': means,
        'block_norms': np.array(block_norms),
        'signal_std': signal_std,
        'var_per_dim': var_per_dim,
        'total_var': total_var,
    }

    return neural_activity, metadata


def print_variance_diagnostic(scene_metadata, neural_metadata, block_sizes):
    """
    Print the key diagnostic: how much variance comes from render vs physics slices.

    This ratio is NOT set by a parameter — it is printed as a finding.
    """
    D_render = scene_metadata['D_render_bytes']
    D_physics = (scene_metadata['D_physics_labels'] + scene_metadata['D_scene_config']
                 + scene_metadata.get('D_scene_lighting', 0))
    D_total = scene_metadata['D_total']

    var_per_dim = neural_metadata['var_per_dim']
    block_norms = neural_metadata['block_norms']

    render_var = var_per_dim[:D_render].sum()
    physics_var = var_per_dim[D_render:].sum()
    total_var = neural_metadata['total_var']

    render_frac = render_var / total_var * 100
    physics_frac = physics_var / total_var * 100
    ratio = D_render / D_physics

    # Label norms by block name
    block_names = ['render', 'physics', 'config', 'lighting']
    norm_strs = '  '.join(
        f'{name}={norm:.3g}'
        for name, norm in zip(block_names, block_norms)
    )

    print("\n" + "=" * 60)
    print("VARIANCE DIAGNOSTIC (key result)")
    print("=" * 60)
    print(f"Program state: D_render={D_render}, D_physics={D_physics}, ratio={ratio:.1f}x")
    print(f"Block operator norms: {norm_strs}")
    print(f"Variance fraction from render slice:  {render_frac:.1f}%")
    print(f"Variance fraction from physics slice: {physics_frac:.1f}%")
    print(f"Total normalized variance: {total_var:.1f}")
    print(f"Signal std: {neural_metadata['signal_std']:.4f}")
    print("=" * 60 + "\n")
