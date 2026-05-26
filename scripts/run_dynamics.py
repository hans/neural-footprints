import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, load_results, load_encoder, save_results
from analyses.dynamics import run_dynamics_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)
encoding = load_results(snakemake.input.encoding)
encoder = load_encoder(snakemake.input.encoder)
inferred_physics = np.load(snakemake.input.inferred)["inferred_physics_all"]

encoding_delta_r2 = np.mean(encoding["delta_r2"])

results = run_dynamics_analysis(
    neural,
    scenes,
    neural_meta,
    encoding_delta_r2,
    encoder,
    inferred_physics=inferred_physics,
    dynamics_pca_dim=cfg["dynamics_pca_dim"],
)
save_results(results, snakemake.output.results)

# Save plot data
plot_arrays = {
    "r2_physics_forward": results["r2_physics_forward"],
    "r2_pixel_forward": results["r2_pixel_forward"],
    "encoding_delta_r2": np.array(results["encoding_delta_r2"]),
}
if results.get("r2_inferred_forward") is not None:
    plot_arrays["r2_inferred_forward"] = results["r2_inferred_forward"]
np.savez_compressed(snakemake.output.plot_data, **plot_arrays)
