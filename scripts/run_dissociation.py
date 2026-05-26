import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, load_encoder, save_results
from analyses.dissociation import run_dissociation_analysis
from analyses.encoding import pca_reduce_pixels
from scene_generator import extract_brain_pixels

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)
encoder = load_encoder(snakemake.input.encoder)
fwd = np.load(snakemake.input.forward_renders)

# Predicted-S features for dissociation
fwd_states = fwd["forward_program_states"]
predicted_brain_pixels = extract_brain_pixels(fwd_states, scenes["metadata"])
predicted_pixel_pca, _, _ = pca_reduce_pixels(
    predicted_brain_pixels, cfg["pixel_pca_dim"]
)

encoding_results = {
    "r2_pixel_only": encoder.pop("r2_pixel_only"),
    "r2_physics_only": encoder.pop("r2_physics_only"),
    "r2_combined": encoder.pop("r2_combined"),
    "r2_physics_only_null": encoder.pop("r2_physics_only_null"),
    "r2_combined_null": encoder.pop("r2_combined_null"),
    "delta_r2_null": encoder.pop("delta_r2_null"),
}

results = run_dissociation_analysis(
    neural,
    scenes,
    neural_meta,
    encoder,
    encoding_results,
    objective=cfg["behavioral_objective"],
    behavioral_pca_dim=cfg["behavioral_pca_dim"],
    predicted_pixel_pca=predicted_pixel_pca,
    predicted_brain_pixels=predicted_brain_pixels,
    forward_program_states=fwd_states,
)

# Separate large arrays / plot-only data from JSON results
plot_arrays = {}
for key in [
    "r2_pixel",
    "r2_physics",
    "r2_combined",
    "r2_physics_null",
    "r2_combined_null",
    "delta_r2_null",
    "chance",
    "predicted_init_imgs",
    "predicted_pixel_imgs",
    "predicted_physics_imgs",
    "predicted_final_imgs",
]:
    plot_arrays[key] = results.pop(key)

# Optional predicted-S arrays
for key in ["r2_predicted_pixel", "predicted_fwd_imgs"]:
    if key in results:
        plot_arrays[key] = results.pop(key)

# Add scalar plot data
plot_arrays["pixel_score"] = np.array(results["pixel_behavioral_score"])
plot_arrays["physics_score"] = np.array(results["physics_behavioral_score"])
plot_arrays["combined_score"] = np.array(results["combined_behavioral_score"])
plot_arrays["metric_label"] = np.array(results["metric_label"])
plot_arrays["fg_pixel_score"] = np.array(results["fg_pixel_behavioral_score"])
plot_arrays["fg_physics_score"] = np.array(results["fg_physics_behavioral_score"])
plot_arrays["delta_pixel_score"] = np.array(results["delta_pixel_behavioral_score"])
plot_arrays["delta_physics_score"] = np.array(results["delta_physics_behavioral_score"])
if "predicted_pixel_behavioral_score" in results:
    plot_arrays["predicted_pixel_score"] = np.array(
        results["predicted_pixel_behavioral_score"]
    )

save_results(results, snakemake.output.results)

# Save plot data
np.savez_compressed(snakemake.output.plot_data, **plot_arrays)
