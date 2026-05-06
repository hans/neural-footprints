import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.pca_analysis import run_pca_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)

results = run_pca_analysis(neural, scenes, neural_meta)

# Separate large arrays from JSON results
neural_pca_2d = results.pop('neural_pca_2d')
motion_dir = results.pop('motion_dir')
n_neurons = results.pop('n_neurons')

save_results(results, snakemake.output.results)

# Per-target decode curves and chance bands flattened into the .npz so the
# plotter can read each one without JSON.
decode_accs_per_target = results['decode_accs_per_target']
chance_per_target = results['chance_per_target']
target_names = results['target_names']
per_target_arrays = {}
for name in target_names:
    per_target_arrays[f'decode_accs__{name}'] = np.array(decode_accs_per_target[name])
    per_target_arrays[f'chance_lo__{name}']   = np.array(chance_per_target[name]['lo'])
    per_target_arrays[f'chance_hi__{name}']   = np.array(chance_per_target[name]['hi'])

np.savez_compressed(snakemake.output.plot_data,
    cumvar=np.array(results['cumulative_variance']),
    neural_pca_2d=neural_pca_2d,
    motion_dir=motion_dir,
    pc_counts=np.array(results['pc_counts']),
    target_names=np.array(target_names),
    n_neurons=np.array(n_neurons),
    **per_target_arrays,
)
