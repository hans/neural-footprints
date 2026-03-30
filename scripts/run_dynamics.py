import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_config import load_config
from io_utils import load_scenes, load_neural, load_results, load_encoder, save_results
from analyses.dynamics import run_dynamics_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)
encoding = load_results(snakemake.input.encoding)
encoder = load_encoder(snakemake.input.encoder)

import numpy as np
encoding_delta_r2 = np.mean(encoding['delta_r2'])

fig_dir = os.path.dirname(snakemake.output.figure)
os.makedirs(fig_dir, exist_ok=True)

results = run_dynamics_analysis(
    neural, scenes, neural_meta, encoding_delta_r2, encoder,
    fig_dir=fig_dir,
    behavioral_pca_dim=cfg['behavioral_pca_dim'],
)
save_results(results, snakemake.output.results)
