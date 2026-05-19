import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.residual import run_residual_analysis
from analyses.encoding import pca_reduce_pixels
from scene_generator import extract_brain_pixels

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)

with open(snakemake.input.encoding) as f:
    encoding_results = json.load(f)

fwd = np.load(snakemake.input.forward_renders)
fwd_states = fwd['forward_program_states']
predicted_brain_pixels = extract_brain_pixels(fwd_states, scenes['metadata'])
predicted_pixel_pca, _, _ = pca_reduce_pixels(predicted_brain_pixels,
                                               cfg['pixel_pca_dim'])
del fwd_states, predicted_brain_pixels, fwd

results = run_residual_analysis(
    neural, scenes, neural_meta,
    pixel_pca_dim=cfg['pixel_pca_dim'],
    r2_raw_pixel=np.asarray(encoding_results['r2_pixel_only']),
    r2_raw_physics_gt=np.asarray(encoding_results['r2_physics_only']),
    predicted_pixel_pca=predicted_pixel_pca,
)
save_results(results, snakemake.output.results)

plot_arrays = dict(
    r2_raw_pixel=results['r2_raw_pixel'],
    r2_raw_physics_gt=results['r2_raw_physics_gt'],
    r2_resid_pixel=results['r2_resid_pixel'],
    r2_resid_physics_gt=results['r2_resid_physics_gt'],
    residual_variance_fraction=np.array(results['residual_variance_fraction']),
)
for key in ('r2_resid_predicted_pixel', 'r2_resid_physics_gt_via_predicted_pixel',
            'residual_variance_fraction_predicted'):
    if key in results:
        val = results[key]
        plot_arrays[key] = np.array(val) if not isinstance(val, np.ndarray) else val
np.savez_compressed(snakemake.output.plot_data, **plot_arrays)
