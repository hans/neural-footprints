import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.rsa import run_rsa_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)

inferred_physics = None
if hasattr(snakemake.input, 'inferred'):
    inferred_physics = np.load(snakemake.input.inferred)['inferred_physics_all']

results = run_rsa_analysis(
    neural, scenes, neural_meta,
    rsa_subsample=cfg['rsa_subsample'],
    render_pca_dim=cfg['render_pca_dim'],
    inferred_physics=inferred_physics,
)

# Separate large arrays from JSON results
rdm_neural = results.pop('rdm_neural')
rdm_render = results.pop('rdm_render')
rdm_physics = results.pop('rdm_physics')
rdm_inferred = results.pop('rdm_inferred')
n_sub = results.pop('n_sub')

save_results(results, snakemake.output.results)

# Save plot data
plot_arrays = dict(
    rdm_neural=rdm_neural,
    rdm_render=rdm_render,
    rdm_physics=rdm_physics,
    n_sub=np.array(n_sub),
    corr_neural_render=np.array(results['corr_neural_render']),
    corr_neural_physics=np.array(results['corr_neural_physics']),
    partial_corr=np.array(results['partial_neural_physics']),
)
if rdm_inferred is not None:
    plot_arrays['rdm_inferred'] = rdm_inferred
    plot_arrays['corr_neural_inferred'] = np.array(results['corr_neural_inferred'])
    plot_arrays['partial_neural_inferred'] = np.array(results['partial_neural_inferred'])
np.savez_compressed(snakemake.output.plot_data, **plot_arrays)
