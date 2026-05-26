import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_config import load_config
from io_utils import save_scenes
from scene_generator import generate_scenes

cfg = load_config()

scenes = generate_scenes(
    cfg["n_scenes"],
    cfg["random_seed"],
    n_timesteps=cfg["n_timesteps"],
    use_gui=True,
)

save_scenes(scenes, snakemake.output.scenes)
