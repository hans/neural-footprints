"""
Neural activity generation via random linear projection of raw program state.

A single random matrix W projects the entire program state (render + physics bytes)
into simulated neural activity. The render/physics variance ratio is NOT a parameter —
it emerges from the structure of the program state.
"""

import numpy as np

from config import N_NEURONS, NOISE_LEVEL


def generate_neural_activity(program_states, seed):
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
    rng = np.random.default_rng(seed)
    n_scenes, D = program_states.shape

    # Step 1: Standardize per dimension (z-score across scenes)
    means = program_states.mean(axis=0)
    stds = program_states.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant dimensions
    standardized = (program_states - means) / stds

    # Step 2: Random projection matrix
    W = rng.normal(0, 1.0 / np.sqrt(D), size=(N_NEURONS, D))

    # Step 3: Signal
    signal = standardized @ W.T  # [n_scenes x N_NEURONS]

    # Step 4: Noise
    signal_std = signal.std()
    noise = NOISE_LEVEL * signal_std * rng.normal(0, 1, size=signal.shape)

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


