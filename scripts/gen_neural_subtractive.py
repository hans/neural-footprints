"""Generate block-structured neural activity for the subtractive pipeline.

Sensory Block reads render-PCA features (whitened PCA of program_state's
render slice). The Abstract Block has two modes:

  * mode="ground_truth": abstract input is just the scene's true N as a
    1-D scalar. Cleaner upper-bound variant; oracle val_r2 = 1.0.
  * mode="inferred":     abstract input is [h2 | inferred_N] from a trained
    CardinalityModel that learned numerosity from the same pixels.

The two blocks are projected with different weight variances and assigned
contiguous regions on a 2-D grid.

Output: data/neural_subtractive_{regime}_{mode}.npz
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from load_config import load_config
from scene_generator_numerosity import load_numerosity_scenes
from analyses.block_projection import (
    Block,
    generate_block_structured_neural,
    print_block_variance_diagnostic,
)


cfg = load_config()
sub_cfg = cfg['subtractive']
regime = snakemake.wildcards.regime  # noqa: F821
mode   = snakemake.wildcards.mode    # noqa: F821

print(f"\nGenerating block-structured neural activity (regime={regime}, mode={mode})")
print("=" * 60)

scenes = load_numerosity_scenes(snakemake.input.scenes)  # noqa: F821

render = scenes['program_states']  # already render-only
print(f"  render shape: {render.shape}")

# Whitened render PCA — drives the Sensory Block
render_scaler = StandardScaler()
render_pca = PCA(n_components=sub_cfg['render_pca_dim_for_projection'],
                 whiten=True, random_state=42)
render_features = render_pca.fit_transform(render_scaler.fit_transform(render))
print(f"  render PCA features (sensory): {render_features.shape}")

if mode == "ground_truth":
    # Oracle: drive the abstract block from true N. Block-projection z-scores
    # the input across scenes, so binary N → ±1 endpoints. abstract_weight_std
    # was tuned for the 257-D inferential input; with D_b=1 it controls the
    # full per-neuron condition shift (see plan §2). Don't retune yet.
    inferred_N = scenes['N'].astype(np.float32)
    abstract_input = inferred_N.reshape(-1, 1)
    cardinality_val_r2 = 1.0
    print(f"  abstract input [ground-truth N]: {abstract_input.shape}")
elif mode == "inferred":
    acts = np.load(snakemake.input.cardinality_acts)  # noqa: F821
    hidden_acts = acts['hidden_acts']
    inferred_N = acts['inferred_N']
    abstract_input = np.concatenate(
        [hidden_acts, inferred_N.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    cardinality_val_r2 = float(acts['val_r2'])
    print(f"  abstract input [h2 | inferred_N]: {abstract_input.shape}")
else:
    raise ValueError(f"Unknown subtractive mode: {mode!r}")

sensory_region  = tuple(tuple(x) for x in sub_cfg['sensory_grid_region'])
abstract_region = tuple(tuple(x) for x in sub_cfg['abstract_grid_region'])
grid_shape = tuple(sub_cfg['grid_shape'])

blocks = [
    Block(
        name='sensory',
        inputs=render_features,
        n_neurons=sub_cfg['sensory_n_neurons'],
        weight_std=sub_cfg['sensory_weight_std'],
        grid_region=sensory_region,
    ),
    Block(
        name='abstract',
        inputs=abstract_input,
        n_neurons=sub_cfg['abstract_n_neurons'],
        weight_std=sub_cfg['abstract_weight_std'],
        grid_region=abstract_region,
    ),
]

neural_activity, meta = generate_block_structured_neural(
    blocks,
    grid_shape=grid_shape,
    seed=sub_cfg['random_seed'],
    noise_level=sub_cfg['noise_level'],
)
print_block_variance_diagnostic(meta)

# Pack to npz. Block-level objects (W matrices, masks) are flattened into
# arrays so we can serialize without pickling.
np.savez_compressed(
    snakemake.output.neural,  # noqa: F821
    neural_activity=neural_activity.astype(np.float32),
    block_assignment=meta['block_assignment'],
    grid_shape=np.array(meta['grid_shape'], dtype=np.int32),
    grid_positions=meta['grid_positions'],
    block_names=np.array(meta['block_names']),
    signal_std=np.array(meta['signal_std'], dtype=np.float32),
    noise_level=np.array(meta['noise_level'], dtype=np.float32),
    var_per_dim=meta['var_per_dim'],
    block_var_offsets=np.array(meta['block_var_offsets'], dtype=np.int64),
    condition=scenes['condition'],
    N=scenes['N'],
    inferred_N=inferred_N.astype(np.float32).ravel(),
    cardinality_val_r2=np.array(cardinality_val_r2, dtype=np.float32),
    regime=np.array(regime),
    mode=np.array(mode),
)
print(f"  Saved -> {snakemake.output.neural}")  # noqa: F821
