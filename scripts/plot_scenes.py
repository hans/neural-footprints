import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes
from analyses.plot_figures import plot_sample_scenes
from scene_generator import resimulate_scene

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)

fig_dir = os.path.dirname(snakemake.output.figure)
os.makedirs(fig_dir, exist_ok=True)

HIRES = 256
N_SAMPLES = 6

n = min(N_SAMPLES, len(scenes['scene_configs']))
hires_initial = []
hires_target = []
for i in range(n):
    cfg_i = scenes['scene_configs'][i]
    phys_i = scenes['initial_physics_labels'][i]
    gray_i = scenes['pillar_grays'][i]
    light_i = scenes['lightings'][i]
    hires_initial.append(resimulate_scene(
        cfg_i, phys_i, n_timesteps=0,
        pillar_gray=gray_i, lighting=light_i, render_size=HIRES, use_gui=True,
    ))
    hires_target.append(resimulate_scene(
        cfg_i, phys_i,
        pillar_gray=gray_i, lighting=light_i, render_size=HIRES, use_gui=True,
    ))

plot_sample_scenes(
    np.stack(hires_initial), np.stack(hires_target),
    rgba_bytes=HIRES * HIRES * 4,
    image_size=HIRES,
    n_timesteps=cfg['n_timesteps'],
    fig_dir=fig_dir,
)
