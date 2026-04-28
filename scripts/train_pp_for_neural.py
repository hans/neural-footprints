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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from load_config import load_config
from io_utils import load_scenes
from analyses.predictive_processing import InverseModel
from analyses.pp_io import save_inverse_model, extract_activations


cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)

initial_renders = scenes['initial_renders']
early_renders   = scenes['early_renders']
initial_physics = scenes['initial_physics_labels']
n = len(initial_renders)

print(f"\nTraining InverseModel for neural generation on {n} scenes")
print("=" * 60)

pixel_pca_dim = cfg['pp_pixel_pca_dim']

# Match run_predictive_processing_analysis exactly: t0 PCA + early PCA + frame-diff PCA.
# The frame-diff channel is what gives the inverse model a usable motion signal.
pca_t0 = PCA(n_components=pixel_pca_dim, whiten=True, random_state=42)
pixel_pca_t0 = pca_t0.fit_transform(StandardScaler().fit_transform(initial_renders))

pca_early = PCA(n_components=pixel_pca_dim, whiten=True, random_state=42)
pixel_pca_early = pca_early.fit_transform(StandardScaler().fit_transform(early_renders))

pca_diff = PCA(n_components=pixel_pca_dim, whiten=True, random_state=42)
pixel_pca_diff = pca_diff.fit_transform(
    StandardScaler().fit_transform(early_renders - initial_renders)
)

pixel_pca_two_frame = np.concatenate([pixel_pca_t0, pixel_pca_early, pixel_pca_diff], axis=1)
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
