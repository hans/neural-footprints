import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes
from analyses.plot_figures import plot_sample_scenes
from scene_generator import resimulate_scene
from config import IMAGE_SIZE

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
fwd = np.load(snakemake.input.forward_renders)
fwd_states = fwd['forward_program_states']

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

# Extract forward model's initial-frame RGBA (native 64×64, upscaled by matplotlib)
metadata = scenes['metadata']
fri = metadata['frame_render_indices']
rgba_n = metadata['target_pixel_indices'].stop - metadata['target_pixel_indices'].start
fwd_init_rgba = fwd_states[:n, fri['initial'].start:fri['initial'].start + rgba_n]
fwd_renders = np.clip(fwd_init_rgba, 0, 255).astype(np.uint8).reshape(
    n, IMAGE_SIZE, IMAGE_SIZE, 4)

plot_sample_scenes(
    np.stack(hires_initial), np.stack(hires_target),
    rgba_bytes=HIRES * HIRES * 4,
    image_size=HIRES,
    n_timesteps=cfg['n_timesteps'],
    fig_dir=fig_dir,
    fwd_renders=fwd_renders,
)
