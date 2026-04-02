"""
Neural activity generation via random linear projection of raw program state.

A single random matrix W projects the entire program state (render + physics bytes)
into simulated neural activity. The render/physics variance ratio is NOT a parameter —
it emerges from the structure of the program state.
"""

import numpy as np

from config import N_NEURONS as _CFG_N_NEURONS, NOISE_LEVEL as _CFG_NOISE_LEVEL


def generate_neural_activity(program_states, seed, *, n_neurons=None, noise_level=None):
    """
    Generate neural activity from raw program states via random linear projection.

    Parameters
    ----------
    program_states : ndarray [n_scenes x D]
        Raw bytes cast to float32.
    seed : int
        Random seed for reproducibility.

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

    # Step 1: Standardize per dimension (z-score across scenes)
    means = program_states.mean(axis=0)
    stds = program_states.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant dimensions
    standardized = (program_states - means) / stds

    # Step 2: Random projection matrix
    W = rng.normal(0, 1.0 / np.sqrt(D), size=(n_neurons, D))

    # Step 3: Signal
    signal = standardized @ W.T  # [n_scenes x n_neurons]

    # Step 4: Noise
    signal_std = signal.std()
    noise = noise_level * signal_std * rng.normal(0, 1, size=signal.shape)

    # Step 5: Neural activity
    neural_activity = signal + noise

    # --- Variance diagnostics ---
    # We need to know where render vs physics dimensions are to report variance fractions.
    # This is computed from the standardized state, not the neural activity.
    var_per_dim = standardized.var(axis=0)  # variance per dimension after z-scoring
    total_var = var_per_dim.sum()

    metadata = {
        'W': W,
        'means': means,
        'stds': stds,
        'signal_std': signal_std,
        'var_per_dim': var_per_dim,
        'total_var': total_var,
    }

    return neural_activity, metadata


def print_variance_diagnostic(scene_metadata, neural_metadata):
    """
    Print the key diagnostic: how much variance comes from render vs physics slices.

    This ratio is NOT set by a parameter — it is printed as a finding.
    """
    D_render = scene_metadata['D_render_bytes']
    D_physics = (scene_metadata['D_physics_labels'] + scene_metadata['D_scene_config']
                 + scene_metadata.get('D_scene_lighting', 0))
    D_total = scene_metadata['D_total']

    var_per_dim = neural_metadata['var_per_dim']

    render_var = var_per_dim[:D_render].sum()
    physics_var = var_per_dim[D_render:].sum()
    total_var = neural_metadata['total_var']

    render_frac = render_var / total_var * 100
    physics_frac = physics_var / total_var * 100
    ratio = D_render / D_physics

    print("\n" + "=" * 60)
    print("VARIANCE DIAGNOSTIC (key result)")
    print("=" * 60)
    print(f"Program state: D_render={D_render}, D_physics={D_physics}, ratio={ratio:.1f}x")
    print(f"Variance fraction from render slice:  {render_frac:.1f}%")
    print(f"Variance fraction from physics slice: {physics_frac:.1f}%")
    print(f"Total standardized variance: {total_var:.1f}")
    print(f"Signal std: {neural_metadata['signal_std']:.4f}")
    print("=" * 60 + "\n")
