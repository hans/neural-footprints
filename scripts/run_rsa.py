import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.rsa import run_rsa_analysis
from analyses.encoding import pca_reduce_pixels
from scene_generator import extract_brain_pixels

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)
fwd = np.load(snakemake.input.forward_renders)

# X: raw observed frames (initial + early + late), PCA-reduced
raw_frames = np.concatenate(
    [scenes["initial_renders"], scenes["early_renders"], scenes["late_renders"]], axis=1
)
raw_pixel_pca, _, _ = pca_reduce_pixels(raw_frames, cfg["pixel_pca_dim"])
del raw_frames

# S: brain pixels from forward-model program states, PCA-reduced
fwd_states = fwd["forward_program_states"]
predicted_brain_pixels = extract_brain_pixels(fwd_states, scenes["metadata"])
predicted_pixel_pca, _, _ = pca_reduce_pixels(
    predicted_brain_pixels, cfg["pixel_pca_dim"]
)
del fwd_states, predicted_brain_pixels, fwd

results = run_rsa_analysis(
    neural,
    scenes,
    neural_meta,
    raw_pixel_pca=raw_pixel_pca,
    rsa_subsample=cfg["rsa_subsample"],
    predicted_pixel_pca=predicted_pixel_pca,
)

# Separate large arrays from JSON results
rdm_neural = results.pop("rdm_neural")
rdm_X = results.pop("rdm_X")
rdm_physics = results.pop("rdm_physics")
rdm_S = results.pop("rdm_S", None)
n_sub = results.pop("n_sub")

save_results(results, snakemake.output.results)

# Save plot data
plot_kwargs = dict(
    rdm_neural=rdm_neural,
    rdm_X=rdm_X,
    rdm_physics=rdm_physics,
    n_sub=np.array(n_sub),
    corr_neural_X=np.array(results["corr_neural_X"]),
    corr_neural_P=np.array(results["corr_neural_P"]),
    partial_P_given_X=np.array(results["partial_P_given_X"]),
)
if rdm_S is not None:
    plot_kwargs["rdm_S"] = rdm_S
    plot_kwargs["corr_neural_S"] = np.array(results["corr_neural_S"])
if "partial_P_given_XS" in results:
    plot_kwargs["partial_P_given_XS"] = np.array(results["partial_P_given_XS"])
np.savez_compressed(snakemake.output.plot_data, **plot_kwargs)
