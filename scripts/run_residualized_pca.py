import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.residualized_pca import run_residualized_pca_analysis
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

results = run_residualized_pca_analysis(
    neural,
    scenes,
    neural_meta,
    raw_pixel_pca=raw_pixel_pca,
    predicted_pixel_pca=predicted_pixel_pca,
)
save_results(results, snakemake.output.results)

# Flatten per-condition decode curves / chance bands into the plot NPZ.
conditions = [c for c in ("raw", "resid_X", "resid_XS", "pixel") if c in results]
plot_arrays = {"conditions": np.array(conditions)}
for cond in conditions:
    block = results[cond]
    plot_arrays[f"pc_counts__{cond}"] = np.array(block["pc_counts"])
    plot_arrays[f"cumvar__{cond}"] = np.array(block["cumulative_variance"])
    for name, accs in block["decode_accs_per_target"].items():
        plot_arrays[f"decode__{cond}__{name}"] = np.array(accs)
        chance = block["chance_per_target"][name]
        plot_arrays[f"chance_lo__{cond}__{name}"] = np.array(chance["lo"])
        plot_arrays[f"chance_hi__{cond}__{name}"] = np.array(chance["hi"])
for key in ("residual_variance_fraction_X", "residual_variance_fraction_XS"):
    if key in results:
        plot_arrays[key] = np.array(results[key])

np.savez_compressed(snakemake.output.plot_data, **plot_arrays)
