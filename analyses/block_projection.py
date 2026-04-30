"""Block-structured random projection for the subtractive-analysis pipeline.

Generates simulated neural activity from a *list* of input blocks rather than
a single concatenated state. Each block:

  * Has its own input array (per scene, D_b dimensions).
  * Owns a contiguous group of neurons in the output population.
  * Is z-scored independently before projection.
  * Has its own weight scale (sigma_b) sampled from N(0, sigma_b / sqrt(D_b)).
  * Maps to a contiguous rectangular region of a 2-D grid layout, so the
    resulting "brain map" reads as a literal blob structure when reshaped
    to grid_shape.

This intentionally diverges from neural_model.generate_neural_activity (which
uses a single random W onto a flat concatenated state); both modules coexist
and serve different pipelines.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Block:
    """One sub-population in the block-structured projection.

    Attributes
    ----------
    name : str
        Label used in metadata and diagnostics ("sensory", "abstract", ...).
    inputs : np.ndarray  [n_scenes, D_b]
        Per-scene input features for this block. Will be z-scored across
        scenes before projection.
    n_neurons : int
        How many output neurons this block contributes to the population.
    weight_std : float
        Variance scale; weights are sampled from N(0, weight_std / sqrt(D_b)).
    grid_region : tuple[tuple[int, int], tuple[int, int]]
        ((row_lo, row_hi), (col_lo, col_hi)) defining the contiguous
        rectangle on the 2-D grid this block occupies. The number of grid
        cells must equal n_neurons.
    """
    name: str
    inputs: np.ndarray
    n_neurons: int
    weight_std: float
    grid_region: tuple


def _zscore(X):
    """Per-column z-score across scenes (rows). Constant columns get std=1."""
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return (X - means) / stds, means, stds


def _grid_indices(grid_shape, region):
    """Return row-major flat indices for the cells inside a rectangular region."""
    (r_lo, r_hi), (c_lo, c_hi) = region
    rows = np.arange(r_lo, r_hi)
    cols = np.arange(c_lo, c_hi)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    flat = (rr * grid_shape[1] + cc).reshape(-1)
    return flat, rr.reshape(-1), cc.reshape(-1)


def generate_block_structured_neural(blocks, *, grid_shape, seed,
                                     noise_level=0.3):
    """Project a list of input blocks into block-structured neural activity.

    Parameters
    ----------
    blocks : list[Block]
        Sub-populations to project. Block.inputs.shape[0] must agree across
        blocks (number of scenes). Block.n_neurons across all blocks must
        sum to grid_shape[0] * grid_shape[1].
    grid_shape : tuple[int, int]
        (n_rows, n_cols). Total cells = sum of block n_neurons.
    seed : int
        RNG seed.
    noise_level : float
        Gaussian noise scale, multiplied by the population-wide signal std.

    Returns
    -------
    neural_activity : float32 [n_scenes, n_total_neurons]
        Row-major flattened over the 2-D grid (so reshape to grid_shape gives
        the spatial map directly).
    meta : dict with
        'blocks':           list of dicts (per-block W, name, indices, ...).
        'block_assignment': int8 [n_total_neurons] block index per neuron.
        'block_names':      list[str] indexed by block_assignment.
        'grid_shape':       tuple
        'grid_positions':   int [n_total_neurons, 2]  (row, col) per neuron.
        'signal_std':       float
        'var_per_dim':      float [sum_b D_b]  per-input-dim variance after z-scoring
        'block_var_offsets':list[(start, end)] where each block's z-scored
                            input dims start/end in var_per_dim.
    """
    rng = np.random.default_rng(seed)

    # -- Validation ----------------------------------------------------------
    n_scenes_per_block = {b.inputs.shape[0] for b in blocks}
    if len(n_scenes_per_block) != 1:
        raise ValueError(f"All block inputs must share n_scenes; got {n_scenes_per_block}")
    n_scenes = n_scenes_per_block.pop()

    n_total_neurons = sum(b.n_neurons for b in blocks)
    expected = grid_shape[0] * grid_shape[1]
    if n_total_neurons != expected:
        raise ValueError(
            f"Block neuron counts sum to {n_total_neurons} but grid_shape "
            f"{grid_shape} has {expected} cells."
        )

    # -- Project each block --------------------------------------------------
    signal = np.zeros((n_scenes, n_total_neurons), dtype=np.float32)
    block_assignment = np.zeros(n_total_neurons, dtype=np.int8)
    grid_positions = np.zeros((n_total_neurons, 2), dtype=np.int32)
    var_per_dim_chunks = []
    block_var_offsets = []
    block_meta = []

    cursor = 0  # tracks where in var_per_dim each block sits

    for b_idx, b in enumerate(blocks):
        flat_idx, rows, cols = _grid_indices(grid_shape, b.grid_region)
        if flat_idx.size != b.n_neurons:
            raise ValueError(
                f"Block {b.name!r}: grid_region has {flat_idx.size} cells, "
                f"but n_neurons={b.n_neurons}."
            )

        Z, means, stds = _zscore(b.inputs.astype(np.float32))
        D_b = Z.shape[1]
        scale = b.weight_std / np.sqrt(D_b)
        W_b = rng.normal(0, scale, size=(b.n_neurons, D_b)).astype(np.float32)
        block_signal = Z @ W_b.T  # [n_scenes, n_neurons]

        signal[:, flat_idx] = block_signal
        block_assignment[flat_idx] = b_idx
        grid_positions[flat_idx, 0] = rows
        grid_positions[flat_idx, 1] = cols

        var_per_dim_chunks.append(Z.var(axis=0))
        block_var_offsets.append((cursor, cursor + D_b))
        cursor += D_b

        block_meta.append({
            'name': b.name,
            'n_neurons': b.n_neurons,
            'D_b': D_b,
            'weight_std': float(b.weight_std),
            'grid_region': b.grid_region,
            'flat_idx': flat_idx,
            'W': W_b,
            'input_means': means.astype(np.float32),
            'input_stds':  stds.astype(np.float32),
        })

    # -- Add noise ----------------------------------------------------------
    signal_std = float(signal.std())
    noise = noise_level * signal_std * rng.normal(0, 1, size=signal.shape).astype(np.float32)
    neural_activity = signal + noise

    var_per_dim = np.concatenate(var_per_dim_chunks)

    meta = {
        'blocks': block_meta,
        'block_assignment': block_assignment,
        'block_names': [b.name for b in blocks],
        'grid_shape': tuple(grid_shape),
        'grid_positions': grid_positions,
        'signal_std': signal_std,
        'noise_level': float(noise_level),
        'var_per_dim': var_per_dim.astype(np.float32),
        'block_var_offsets': block_var_offsets,
    }
    return neural_activity, meta


def print_block_variance_diagnostic(meta):
    """Per-block variance fraction (after z-scoring), like neural_model's diagnostic."""
    var_per_dim = meta['var_per_dim']
    total = float(var_per_dim.sum())
    pct = lambda v: 100.0 * v / total if total > 0 else 0.0
    print("\n" + "=" * 60)
    print("BLOCK VARIANCE DIAGNOSTIC (post z-score, pre-projection)")
    print("=" * 60)
    for b, (lo, hi) in zip(meta['blocks'], meta['block_var_offsets']):
        v = float(var_per_dim[lo:hi].sum())
        print(f"  {b['name']:>12}: D_b={b['D_b']:4d}  weight_std={b['weight_std']:.3f}  "
              f"n_neurons={b['n_neurons']:4d}  var_frac={pct(v):.1f}%")
    print(f"  total z-scored variance: {total:.1f}")
    print(f"  population signal std:   {meta['signal_std']:.4f}")
    print("=" * 60 + "\n")
