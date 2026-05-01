"""Train the InverseModel that defines the cognitive layer of neural generation.

Outputs:
  data/inverse_model.pt   — checkpoint (state_dict + scalers + dim metadata)
  data/pp_activations.npz — per-scene hidden activations + inferred physics

The model is fit on ALL scenes (with an internal 15% held-out val split for early
stopping). Predictions for every scene are then dumped so gen_neural.py can stitch
them into the projection input. run_pp.py loads the same checkpoint to keep the
neural projection and the reported PP analysis numerically consistent.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from load_config import load_config
from io_utils import load_scenes
from analyses.predictive_processing import InverseModel, build_pp_features
from analyses.pp_io import save_inverse_model, extract_activations


cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)

initial_physics = scenes['initial_physics_labels']
n = len(scenes['initial_renders'])

print(f"\nTraining InverseModel for neural generation on {n} scenes")
print("=" * 60)

# Shared with run_predictive_processing_analysis (analyses/predictive_processing.py).
feats = build_pp_features(scenes, pixel_pca_dim=cfg['pp_pixel_pca_dim'])
pixel_pca_two_frame = feats['pixel_pca_concat']
print(f"  pixel PCA features: {pixel_pca_two_frame.shape}")

inv = InverseModel()
inv.fit(pixel_pca_two_frame, initial_physics)

layer = cfg.get('pp_neural_layer', 'h2')
hidden_acts = extract_activations(inv, pixel_pca_two_frame, layer=layer)
inferred_physics = inv.predict(pixel_pca_two_frame)

print(f"\n  hidden_acts ({layer}): {hidden_acts.shape}")
print(f"  inferred_physics:       {inferred_physics.shape}")

save_inverse_model(inv, snakemake.output.model)
np.savez_compressed(
    snakemake.output.pp_acts,
    hidden_acts=hidden_acts,
    inferred_physics=inferred_physics,
    layer=np.array(layer),
)
print(f"\n  Saved checkpoint → {snakemake.output.model}")
print(f"  Saved activations → {snakemake.output.pp_acts}")
