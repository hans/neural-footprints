import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.predictive_processing import run_predictive_processing_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, _ = load_neural(snakemake.input.neural)

results = run_predictive_processing_analysis(
    neural, scenes,
    pixel_pca_dim=cfg['pp_pixel_pca_dim'],
)

inferred_physics_all = results.pop('inferred_physics_all')
plot_data = results.pop('plot_data')

save_results(results, snakemake.output.results)

np.savez_compressed(snakemake.output.inferred,
                    inferred_physics_all=inferred_physics_all)

np.savez_compressed(snakemake.output.plot_data, **plot_data)
