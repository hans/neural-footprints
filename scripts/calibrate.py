import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_config import load_config
from scene_generator import calibrate_bullet_size

cfg = load_config()
bullet_k = calibrate_bullet_size(n_timesteps=cfg['n_timesteps'])

os.makedirs(os.path.dirname(snakemake.output[0]), exist_ok=True)
with open(snakemake.output[0], 'w') as f:
    json.dump({'bullet_k': bullet_k}, f)
