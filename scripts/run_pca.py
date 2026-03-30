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

# Save plot data
np.savez_compressed(snakemake.output.plot_data,
    cumvar=np.array(results['cumulative_variance']),
    neural_pca_2d=neural_pca_2d,
    motion_dir=motion_dir,
    pc_counts=np.array(results['pc_counts']),
    decode_accs=np.array(results['decode_accuracies']),
    n_neurons=np.array(n_neurons),
)
