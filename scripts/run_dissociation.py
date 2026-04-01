import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.dissociation import run_dissociation_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)

results = run_dissociation_analysis(
    neural, scenes, neural_meta,
    objective=cfg['behavioral_objective'],
    render_pca_dim=cfg['render_pca_dim'],
    behavioral_pca_dim=cfg['behavioral_pca_dim'],
)

# Separate large arrays / plot-only data from JSON results
plot_arrays = {}
for key in ['r2_render', 'r2_physics', 'chance',
            'predicted_init_imgs', 'predicted_render_imgs',
            'predicted_physics_imgs', 'predicted_final_imgs']:
    plot_arrays[key] = results.pop(key)

# Add scalar plot data
plot_arrays['render_score'] = np.array(results['render_behavioral_score'])
plot_arrays['physics_score'] = np.array(results['physics_behavioral_score'])
plot_arrays['metric_label'] = np.array(results['metric_label'])

save_results(results, snakemake.output.results)

# Save plot data
np.savez_compressed(snakemake.output.plot_data, **plot_arrays)
