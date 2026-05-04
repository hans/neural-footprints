"""Train the CardinalityModel for one numerosity regime.

Outputs:
  data/cardinality_model_{regime}.pt        — checkpoint with model + pixel-PCA pipeline
  data/cardinality_activations_{regime}.npz — h2 activations + inferred N per scene

Mirrors scripts/train_pp_for_neural.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from load_config import load_config
from scene_generator_numerosity import load_numerosity_scenes
from analyses.cardinality import CardinalityModel, whitened_pca_features
from analyses.cardinality_io import save_cardinality_model


cfg = load_config()
sub_cfg = cfg['subtractive']
regime = snakemake.wildcards.regime  # noqa: F821

print(f"\nTraining CardinalityModel (regime={regime})")
print("=" * 60)

scenes = load_numerosity_scenes(snakemake.input.scenes)  # noqa: F821
N = scenes['N']
rgba_initial = scenes['rgba_initial']
n = len(N)
print(f"  Loaded {n} scenes  (low N={(scenes['condition'] == 0).sum()}, "
      f"high N={(scenes['condition'] == 1).sum()})")

feats, pixel_scaler, pixel_pca = whitened_pca_features(
    rgba_initial, pixel_pca_dim=sub_cfg['cardinality_pixel_pca_dim'],
)
print(f"  Pixel-PCA features: {feats.shape}")

model = CardinalityModel(
    hidden_dim=sub_cfg['cardinality_hidden_dim'],
    dropout_rate=sub_cfg['cardinality_dropout_rate'],
)
model.fit(feats, N)
val_r2 = float(model.per_dim_r2_.mean())

# Sanity check — the demonstration is meaningless if the cognitive model
# isn't actually learning N. With binary N (3 vs 12) this should be easy.
if val_r2 < 0.5:
    raise RuntimeError(
        f"CardinalityModel val R² = {val_r2:.3f} is too low (< 0.5). "
        f"The abstract block input would not faithfully encode N. "
        f"Increase cardinality_pixel_pca_dim, increase n_scenes_per_condition, "
        f"or check the pixel renders."
    )

layer = sub_cfg['cardinality_neural_layer']
hidden_acts = model.extract_activations(feats, layer=layer)
inferred_N = model.predict(feats)
print(f"  hidden_acts ({layer}): {hidden_acts.shape}")
print(f"  inferred_N:           {inferred_N.shape}")
print(f"  inferred_N stats: low_mean={inferred_N[scenes['condition'] == 0].mean():.2f}  "
      f"high_mean={inferred_N[scenes['condition'] == 1].mean():.2f}")

save_cardinality_model(model, pixel_scaler, pixel_pca, snakemake.output.model)  # noqa: F821
np.savez_compressed(
    snakemake.output.acts,  # noqa: F821
    hidden_acts=hidden_acts.astype(np.float32),
    inferred_N=inferred_N.astype(np.float32),
    layer=np.array(layer),
    val_r2=np.array(val_r2),
)
print(f"\n  Saved checkpoint   -> {snakemake.output.model}")  # noqa: F821
print(f"  Saved activations  -> {snakemake.output.acts}")  # noqa: F821
