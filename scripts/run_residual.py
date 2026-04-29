import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.residual import run_residual_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)
inferred_physics = np.load(snakemake.input.inferred)['inferred_physics_all']

results = run_residual_analysis(
    neural, scenes, neural_meta,
    render_pca_dim=cfg['render_pca_dim'],
    inferred_physics=inferred_physics,
)

save_results(results, snakemake.output.results)

plot_arrays = {k: np.asarray(v) for k, v in results.items()
               if isinstance(v, np.ndarray) or k.startswith('r2_')}
plot_arrays['residual_variance_fraction'] = np.array(results['residual_variance_fraction'])
np.savez_compressed(snakemake.output.plot_data, **plot_arrays)
