import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, load_encoder, load_results, save_results
from analyses.dissociation import run_dissociation_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)
encoder = load_encoder(snakemake.input.encoder)

encoding_results = {
    'r2_pixels_only': encoder.pop('r2_pixels_only'),
    'r2_physics_only': encoder.pop('r2_physics_only'),
    'r2_combined': encoder.pop('r2_combined'),
}

pp_results = None
if hasattr(snakemake.input, 'pp_results'):
    pp_results = load_results(snakemake.input.pp_results)

results = run_dissociation_analysis(
    neural, scenes, neural_meta,
    encoder, encoding_results,
    objective=cfg['behavioral_objective'],
    behavioral_pca_dim=cfg['behavioral_pca_dim'],
    pp_results=pp_results,
)

# Separate large arrays / plot-only data from JSON results
plot_arrays = {}
for key in ['r2_render', 'r2_physics', 'r2_combined', 'chance',
            'predicted_init_imgs', 'predicted_render_imgs',
            'predicted_physics_imgs', 'predicted_final_imgs']:
    plot_arrays[key] = results.pop(key)

# Override frame visualizations with PP-derived frames so all four panels come
# from the same held-out scenes and the "physics model" panel reflects the
# inferred-physics PP chain (matching the bar plot's pp_chain_score) rather
# than the GT-physics oracle.
pp_plot = np.load(snakemake.input.pp_plot_data, allow_pickle=False)
plot_arrays['predicted_init_imgs']    = pp_plot['init_frame_imgs']
plot_arrays['predicted_render_imgs']  = pp_plot['render_frame_imgs']
plot_arrays['predicted_physics_imgs'] = pp_plot['pp_frame_imgs']
plot_arrays['predicted_final_imgs']   = pp_plot['final_frame_imgs']

# Add scalar plot data
plot_arrays['render_score'] = np.array(results['render_behavioral_score'])
plot_arrays['physics_score'] = np.array(results['physics_behavioral_score'])
plot_arrays['combined_score'] = np.array(results['combined_behavioral_score'])
plot_arrays['metric_label'] = np.array(results['metric_label'])
if results.get('pp_chain_score') is not None:
    plot_arrays['pp_chain_score'] = np.array(results['pp_chain_score'])

save_results(results, snakemake.output.results)

# Save plot data
np.savez_compressed(snakemake.output.plot_data, **plot_arrays)
