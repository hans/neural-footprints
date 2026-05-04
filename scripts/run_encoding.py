import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results, save_encoder
from analyses.encoding import run_encoding_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)

results = run_encoding_analysis(
    neural, scenes, neural_meta,
    render_pca_dim=cfg['render_pca_dim'],
)
encoder = results.pop('encoder')
encoder['r2_render_only'] = results['r2_render_only']
encoder['r2_physics_only'] = results['r2_physics_only']
encoder['r2_combined'] = results['r2_combined']
save_results(results, snakemake.output.results)
save_encoder(encoder, snakemake.output.encoder)

# Save plot data
np.savez_compressed(snakemake.output.plot_data,
    r2_render_only=results['r2_render_only'],
    r2_physics_only=results['r2_physics_only'],
    r2_combined=results['r2_combined'],
    subsample_means=np.array(results['subsample_means']),
    subsample_sems=np.array(results['subsample_sems']),
    neuron_counts=np.array(results['subsample_neuron_counts']),
    control_accuracy=np.array(results['control_accuracy']),
    control_accuracy_std=np.array(results['control_accuracy_std']),
)
