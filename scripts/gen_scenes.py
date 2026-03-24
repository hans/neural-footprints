import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_config import load_config
from io_utils import save_scenes
from scene_generator import generate_scenes, save_sample_renders

cfg = load_config()

with open(snakemake.input.bullet_k) as f:
    bullet_k = json.load(f)['bullet_k']

scenes = generate_scenes(
    cfg['n_scenes'], cfg['random_seed'], bullet_k,
    n_timesteps=cfg['n_timesteps'],
)

save_scenes(scenes, snakemake.output.scenes)

fig_dir = os.path.dirname(snakemake.output.figure)
os.makedirs(fig_dir, exist_ok=True)
save_sample_renders(scenes, fig_dir)
