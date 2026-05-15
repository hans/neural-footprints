"""Generate forward-model renders from inferred physics.

Input: data/scenes.npz, data/pp_activations.npz
Output: data/forward_renders.npz

For each scene, patches shape_configs with the inverse model's inferred
x_accel, then resimulates from inferred initial physics to produce a
full program_state vector (same layout as scenes['program_states']).

The resulting forward_program_states replace actual camera renders in
gen_neural.py: the brain state becomes [fwd_render | inv_acts | P_hat],
so raw sensation enters only through the inverse model's input.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import numpy as np
import pybullet as p

from io_utils import load_scenes
from scene_generator import resimulate_scene, open_render_client

scenes = load_scenes(snakemake.input.scenes)
pp = np.load(snakemake.input.pp_activations)

inferred_physics = pp['inferred_physics']   # [n_scenes x 16*N_OBJECTS]
program_states = scenes['program_states']
scene_configs = scenes['scene_configs']
pillar_grays = scenes['pillar_grays']
lightings = scenes['lightings']

n_scenes = len(program_states)
D = program_states.shape[1]
n_objects = inferred_physics.shape[1] // 16

print(f"Generating forward renders for {n_scenes} scenes ({n_objects} object(s))...")
forward_program_states = np.zeros((n_scenes, D), dtype=np.float32)

pc = open_render_client(use_gui=True)
try:
    for i in range(n_scenes):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Scene {i+1}/{n_scenes}...")
        cfg_copy = copy.deepcopy(scene_configs[i])
        for j in range(n_objects):
            cfg_copy[j]['x_accel'] = float(inferred_physics[i, j * 16 + 15])
        forward_program_states[i] = resimulate_scene(
            cfg_copy, inferred_physics[i],
            return_program_state=True,
            pillar_gray=pillar_grays[i],
            lighting=lightings[i],
            use_gui=True,
            physics_client=pc,
        )
finally:
    p.disconnect(pc)

np.savez_compressed(snakemake.output.forward_renders,
                    forward_program_states=forward_program_states)
print(f"Saved {snakemake.output.forward_renders}: shape={forward_program_states.shape}")
