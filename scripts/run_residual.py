import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.residual import run_residual_analysis
from analyses.encoding import pca_reduce_pixels
from scene_generator import extract_brain_pixels

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)

# X: raw observed frames (initial + early + late), PCA-reduced
raw_frames = np.concatenate(
    [scenes["initial_renders"], scenes["early_renders"], scenes["late_renders"]], axis=1
)
raw_pixel_pca, _, _ = pca_reduce_pixels(raw_frames, cfg["pixel_pca_dim"])
del raw_frames

# S: brain pixels from forward-model program states, PCA-reduced
fwd = np.load(snakemake.input.forward_renders)
fwd_states = fwd["forward_program_states"]
predicted_brain_pixels = extract_brain_pixels(fwd_states, scenes["metadata"])
predicted_pixel_pca, _, _ = pca_reduce_pixels(
    predicted_brain_pixels, cfg["pixel_pca_dim"]
)
del fwd_states, predicted_brain_pixels, fwd

# P_inf: inferred physics from the inverse model run on per-norm neural activity
inferred_data = np.load(snakemake.input.inferred)
inferred_physics_labels = inferred_data["inferred_physics_all"]

results = run_residual_analysis(
    neural,
    scenes,
    neural_meta,
    raw_pixel_pca=raw_pixel_pca,
    predicted_pixel_pca=predicted_pixel_pca,
    inferred_physics_labels=inferred_physics_labels,
    compute_null=cfg.get("residual_compute_null", True),
    n_null_permutations=cfg.get("residual_n_null_permutations", 300),
)
save_results(results, snakemake.output.results)

# Load pre-residualization R²(P → neural) from encoding results.
# Note: encoding uses KFold(shuffle=False) while residual uses KFold(shuffle=True,
# random_state=42) — same number of folds (5) but different fold assignment.
with open(snakemake.input.encoding) as f:
    enc = json.load(f)
r2_P_neural = np.array(enc["r2_P"])

plot_arrays = {"r2_P_neural": r2_P_neural}
for key in ("r2_P_given_X", "r2_P_given_XS",
            "r2_P_inf_given_X", "r2_P_inf_given_XS"):
    if key in results:
        val = results[key]
        plot_arrays[key] = np.array(val) if not isinstance(val, np.ndarray) else val
for key in ("residual_variance_fraction_X", "residual_variance_fraction_XS"):
    if key in results:
        plot_arrays[key] = np.array(results[key])
np.savez_compressed(snakemake.output.plot_data, **plot_arrays)
