import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_config import load_config
from io_utils import load_scenes
from analyses.plot_figures import plot_sample_scenes

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)

fig_dir = os.path.dirname(snakemake.output.figure)
os.makedirs(fig_dir, exist_ok=True)

target_pi = scenes['metadata']['target_pixel_indices']
plot_sample_scenes(
    scenes['initial_renders'], scenes['target_renders'],
    rgba_bytes=target_pi.stop - target_pi.start,
    image_size=cfg['image_size'],
    n_timesteps=cfg['n_timesteps'],
    fig_dir=fig_dir,
)
