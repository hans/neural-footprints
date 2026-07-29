"""Render high-res frames for the animation-scene dissociation slide figure.

Run on Mac host (needs GLFW):
    cd <workdir> && .venv/bin/python scripts/render_slide_anim_frames.py

Uses the 4 scenes from the existing scene animations (indices from config
animation_seed=7 sampling) rendered with zero gravity, 480 timesteps, 256px.

Captures frames at t=0, 160, 320 (inputs) and t=480 (target).
Target rendered twice: oracle (true) and inferred initial physics.

Output: data/slide_anim_frames.npz
    input_frames   (4, 3, 256, 256, 3)  uint8  -- [scene, t_idx, H, W, RGB]
    oracle_target  (4, 256, 256, 3)     uint8  -- t=480 oracle
    inferred_target(4, 256, 256, 3)     uint8  -- t=480 inferred physics
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "glfw"

import json
import numpy as np

from scene_generator import render_scene_frames

RENDER_SIZE   = 256
N_STEPS       = 480          # total timesteps (4× the model's 120-step window)
STRIDE        = 160          # capture every 160th step → frames at t=0,160,320,480
GRAVITY       = [0, 0, 0]   # zero gravity: objects fly indefinitely
CAMERA_FOV    = 90
SCENE_INDICES = [1248, 1367, 1794, 1886]   # same as scene_animations/

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_PATH  = os.path.join(DATA_DIR, "slide_anim_frames.npz")

print("Loading scene data…")
raw = np.load(os.path.join(DATA_DIR, "scenes.npz"), allow_pickle=True)
scene_configs   = json.loads(raw["scene_configs_json"].item())
init_phys_all   = raw["initial_physics_labels"]   # (2000, 16)
pillar_grays    = raw["pillar_grays"]
lightings       = json.loads(raw["lightings_json"].item())

print("Loading inferred physics…")
inf_raw       = np.load(os.path.join(DATA_DIR, "inferred_physics.npz"), allow_pickle=True)
inferred_phys = inf_raw["inferred_physics_all"]   # (2000, 16)

input_frames_all   = []
oracle_target_all  = []
inferred_target_all = []

for scene_idx in SCENE_INDICES:
    cfg   = scene_configs[scene_idx]
    phys  = init_phys_all[scene_idx]
    inf   = inferred_phys[scene_idx]
    pg    = float(pillar_grays[scene_idx])
    light = lightings[scene_idx]

    kw = dict(
        pillar_gray=pg,
        lighting=light,
        render_size=RENDER_SIZE,
        n_timesteps=N_STEPS,
        stride=STRIDE,
        gravity=GRAVITY,
        camera_fov=CAMERA_FOV,
    )

    print(f"  Scene {scene_idx}: oracle…", end=" ", flush=True)
    # oracle: true initial physics, captures [t=0, t=160, t=320, t=480]
    oracle_frames = render_scene_frames(cfg, phys, **kw)
    # oracle_frames shape: (4, H, W, 3)  [stride=160, n_steps=480 → 1 + 480/160 = 4]
    input_frames_all.append(oracle_frames[:3])       # t=0, t=160, t=320
    oracle_target_all.append(oracle_frames[3])        # t=480

    print(f"inferred…", end=" ", flush=True)
    # inferred: inverse-model physics, only need the final target frame
    inferred_frames = render_scene_frames(cfg, inf, **kw)
    inferred_target_all.append(inferred_frames[3])    # t=480

    print("done")

np.savez_compressed(
    OUT_PATH,
    input_frames    = np.stack(input_frames_all),     # (4, 3, 256, 256, 3)
    oracle_target   = np.stack(oracle_target_all),    # (4, 256, 256, 3)
    inferred_target = np.stack(inferred_target_all),  # (4, 256, 256, 3)
)
print(f"\nSaved → {OUT_PATH}")
d = np.load(OUT_PATH)
for k in d.files:
    print(f"  {k}: {d[k].shape} {d[k].dtype}")
