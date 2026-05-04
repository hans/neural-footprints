import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.residual import run_residual_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)

results = run_residual_analysis(
    neural, scenes, neural_meta,
    render_pca_dim=cfg['render_pca_dim'],
)
save_results(results, snakemake.output.results)

np.savez_compressed(
    snakemake.output.plot_data,
    r2_raw_render=results['r2_raw_render'],
    r2_raw_physics_gt=results['r2_raw_physics_gt'],
    r2_resid_render=results['r2_resid_render'],
    r2_resid_physics_gt=results['r2_resid_physics_gt'],
    residual_variance_fraction=np.array(results['residual_variance_fraction']),
)
