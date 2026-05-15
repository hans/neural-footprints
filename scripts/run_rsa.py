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

# Predicted-S features for RSA
fwd_states = fwd['forward_program_states']
predicted_brain_pixels = extract_brain_pixels(fwd_states, scenes['metadata'])
predicted_pixel_pca, _, _ = pca_reduce_pixels(predicted_brain_pixels,
                                               cfg['pixel_pca_dim'])

results = run_rsa_analysis(
    neural, scenes, neural_meta,
    rsa_subsample=cfg['rsa_subsample'],
    pixel_pca_dim=cfg['pixel_pca_dim'],
    predicted_pixel_pca=predicted_pixel_pca,
)

# Separate large arrays from JSON results
rdm_neural = results.pop('rdm_neural')
rdm_pixel = results.pop('rdm_pixel')
rdm_physics = results.pop('rdm_physics')
rdm_predicted = results.pop('rdm_predicted', None)
n_sub = results.pop('n_sub')

save_results(results, snakemake.output.results)

# Save plot data
plot_kwargs = dict(
    rdm_neural=rdm_neural,
    rdm_pixel=rdm_pixel,
    rdm_physics=rdm_physics,
    n_sub=np.array(n_sub),
    corr_neural_pixel=np.array(results['corr_neural_pixel']),
    corr_neural_physics=np.array(results['corr_neural_physics']),
    partial_corr=np.array(results['partial_neural_physics']),
)
if rdm_predicted is not None:
    plot_kwargs['rdm_predicted'] = rdm_predicted
    plot_kwargs['corr_neural_predicted'] = np.array(results['corr_neural_predicted'])
np.savez_compressed(snakemake.output.plot_data, **plot_kwargs)
