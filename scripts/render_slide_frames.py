"""Render high-resolution (256px) frames for the behavioral dissociation slide figure.

Run this on the Mac host (needs GLFW/OpenGL):
    cd <workdir> && .venv/bin/python scripts/render_slide_frames.py

Output: data/slide_frames.npz with keys:
    init_imgs        (8, 256, 256, 4) uint8  -- t=0, true physics
    early_imgs       (8, 256, 256, 4) uint8  -- t=PP_EARLY_FRAME, true physics
    late_imgs        (8, 256, 256, 4) uint8  -- t=PP_LATE_FRAME, true physics
    oracle_imgs      (8, 256, 256, 4) uint8  -- t=N_TIMESTEPS, true physics
    inferred_imgs    (8, 256, 256, 4) uint8  -- t=N_TIMESTEPS, inferred physics

These 8 scenes are indices 0–7, matching the existing dissociation_plot_data.npz
(predicted_pixel_imgs, etc.) so columns align.
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GLFW works on macOS; override the container default of osmesa.
if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "glfw"

import json
import numpy as np
from scene_generator import resimulate_scene
from config import PP_EARLY_FRAME, PP_LATE_FRAME, N_TIMESTEPS

RENDER_SIZE = 256
N_SCENES = 8
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "slide_frames.npz")

print(f"Loading scene data…")
raw = np.load(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "scenes.npz"),
    allow_pickle=True,
)
scene_configs = json.loads(raw["scene_configs_json"].item())
initial_physics_labels = raw["initial_physics_labels"]   # (2000, 16)
pillar_grays = raw["pillar_grays"]
lightings = json.loads(raw["lightings_json"].item())

print(f"Loading inferred physics…")
inf_raw = np.load(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "inferred_physics.npz"),
    allow_pickle=True,
)
inferred_physics = inf_raw["inferred_physics_all"]  # (2000, 16)

print(f"Rendering {N_SCENES} scenes at {RENDER_SIZE}px "
      f"(t=0 / t={PP_EARLY_FRAME} / t={PP_LATE_FRAME} / t={N_TIMESTEPS} oracle+inferred)…")
print(f"  PP_EARLY_FRAME={PP_EARLY_FRAME}, PP_LATE_FRAME={PP_LATE_FRAME}, N_TIMESTEPS={N_TIMESTEPS}")

init_imgs    = []
early_imgs   = []
late_imgs    = []
oracle_imgs  = []
inferred_imgs = []

for j in range(N_SCENES):
    cfg   = scene_configs[j]
    phys  = initial_physics_labels[j]
    inf   = inferred_physics[j]
    pg    = float(pillar_grays[j])
    light = lightings[j]

    kw = dict(pillar_gray=pg, lighting=light, render_size=RENDER_SIZE)

    print(f"  Scene {j+1}/{N_SCENES}: t=0 …", end=" ", flush=True)
    init_imgs.append(resimulate_scene(cfg, phys, n_timesteps=0, **kw))

    print(f"early …", end=" ", flush=True)
    early_imgs.append(resimulate_scene(cfg, phys, n_timesteps=PP_EARLY_FRAME, **kw))

    print(f"late …", end=" ", flush=True)
    late_imgs.append(resimulate_scene(cfg, phys, n_timesteps=PP_LATE_FRAME, **kw))

    print(f"oracle target …", end=" ", flush=True)
    oracle_imgs.append(resimulate_scene(cfg, phys, n_timesteps=N_TIMESTEPS, **kw))

    print(f"inferred target …", end="", flush=True)
    inferred_imgs.append(resimulate_scene(cfg, inf, n_timesteps=N_TIMESTEPS, **kw))

    print(" done")

np.savez_compressed(
    OUT_PATH,
    init_imgs    = np.stack(init_imgs),
    early_imgs   = np.stack(early_imgs),
    late_imgs    = np.stack(late_imgs),
    oracle_imgs  = np.stack(oracle_imgs),
    inferred_imgs = np.stack(inferred_imgs),
)
print(f"\nSaved → {OUT_PATH}")
print("Shapes:")
for k in ("init_imgs", "early_imgs", "late_imgs", "oracle_imgs", "inferred_imgs"):
    arr = np.load(OUT_PATH)[k]
    print(f"  {k}: {arr.shape} {arr.dtype}")
