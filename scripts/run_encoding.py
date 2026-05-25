import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results, save_encoder
from analyses.encoding import run_encoding_analysis, pca_reduce_pixels
from scene_generator import extract_brain_pixels

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)
fwd = np.load(snakemake.input.forward_renders)

# Predicted-S: brain pixels extracted from the forward-model program states
fwd_states = fwd['forward_program_states']
predicted_brain_pixels = extract_brain_pixels(fwd_states, scenes['metadata'])
predicted_pixel_pca, _, _ = pca_reduce_pixels(predicted_brain_pixels,
                                               cfg['pixel_pca_dim'])
# Free the 1.18 GB fwd_states and ~393 MB pixel buffer — only the PCA result
# is needed downstream.
del fwd_states, predicted_brain_pixels, fwd
# W is the projection matrix used only during neural generation; drop it here
# to reclaim ~590 MB before the ridge fits.
del neural_meta['W']

results = run_encoding_analysis(
    neural, scenes, neural_meta,
    pixel_pca_dim=cfg['pixel_pca_dim'],
    predicted_pixel_pca=predicted_pixel_pca,
    compute_null=cfg.get('encoding_compute_null', True),
    n_null_permutations=cfg.get('encoding_n_null_permutations', 50),
)
encoder = results.pop('encoder')
# Forward arrays needed by downstream analyses (dissociation, etc.).
for key in ('r2_pixel_only', 'r2_physics_only', 'r2_combined',
            'r2_physics_only_null', 'r2_combined_null', 'delta_r2_null'):
    encoder[key] = results[key]
save_results(results, snakemake.output.results)
save_encoder(encoder, snakemake.output.encoder)

# Save plot data
plot_kwargs = dict(
    r2_pixel_only=results['r2_pixel_only'],
    r2_physics_only=results['r2_physics_only'],
    r2_combined=results['r2_combined'],
    r2_physics_only_null=results['r2_physics_only_null'],
    r2_combined_null=results['r2_combined_null'],
    delta_r2_null=results['delta_r2_null'],
    subsample_means=np.array(results['subsample_means']),
    subsample_sems=np.array(results['subsample_sems']),
    neuron_counts=np.array(results['subsample_neuron_counts']),
    control_accuracy=np.array(results['control_accuracy']),
    control_accuracy_std=np.array(results['control_accuracy_std']),
)
if 'r2_predicted_pixel' in results:
    plot_kwargs['r2_predicted_pixel'] = results['r2_predicted_pixel']
if 'r2_combined_pred' in results:
    plot_kwargs['r2_combined_pred'] = results['r2_combined_pred']
    plot_kwargs['delta_r2_pred'] = results['delta_r2_pred']
np.savez_compressed(snakemake.output.plot_data, **plot_kwargs)
