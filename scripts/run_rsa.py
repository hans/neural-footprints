import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.rsa import run_rsa_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)

results = run_rsa_analysis(
    neural, scenes, neural_meta,
    rsa_subsample=cfg['rsa_subsample'],
    pixel_pca_dim=cfg['pixel_pca_dim'],
)

# Separate large arrays from JSON results
rdm_neural = results.pop('rdm_neural')
rdm_pixel = results.pop('rdm_pixel')
rdm_physics = results.pop('rdm_physics')
n_sub = results.pop('n_sub')

save_results(results, snakemake.output.results)

# Save plot data
np.savez_compressed(snakemake.output.plot_data,
    rdm_neural=rdm_neural,
    rdm_pixel=rdm_pixel,
    rdm_physics=rdm_physics,
    n_sub=np.array(n_sub),
    corr_neural_pixel=np.array(results['corr_neural_pixel']),
    corr_neural_physics=np.array(results['corr_neural_physics']),
    partial_corr=np.array(results['partial_neural_physics']),
)
