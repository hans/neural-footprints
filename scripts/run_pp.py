import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.predictive_processing import run_predictive_processing_analysis
from analyses.pp_io import load_inverse_model

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, _ = load_neural(snakemake.input.neural)

# Reuse the InverseModel checkpoint that produced the activations baked into
# neural_input (see scripts/train_pp_for_neural.py). Passing it through
# inv_model= keeps the inferred-physics array consistent with what was
# projected into neural activity, with no retraining inside the analysis.
inv_model = load_inverse_model(snakemake.input.model)

results = run_predictive_processing_analysis(
    neural,
    scenes,
    pixel_pca_dim=cfg["pp_pixel_pca_dim"],
    inv_model=inv_model,
)

inferred_physics_all = results.pop("inferred_physics_all")
plot_data = results.pop("plot_data")

save_results(results, snakemake.output.results)

np.savez_compressed(
    snakemake.output.inferred, inferred_physics_all=inferred_physics_all
)

np.savez_compressed(snakemake.output.plot_data, **plot_data)
