import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
from analyses.encoding import run_encoding_analysis

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, neural_meta = load_neural(snakemake.input.neural)

fig_dir = os.path.dirname(snakemake.output.figure)
os.makedirs(fig_dir, exist_ok=True)

results = run_encoding_analysis(
    neural, scenes, neural_meta, fig_dir=fig_dir,
    pixel_pca_dim=cfg['pixel_pca_dim'],
)
save_results(results, snakemake.output.results)
