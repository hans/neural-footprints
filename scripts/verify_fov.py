"""
Analytic FOV check for the wider-frame-spacing config.

Samples 200 scenes from the same distributions as scene_generator.py,
propagates x analytically (constant x_accel, no y/z coupling in x),
and reports the fraction of objects that would be off-screen at t=n_timesteps.
"""

import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.load_config import load_config

cfg = load_config()

N_SAMPLES = 200
DT = 0.002  # MuJoCo default timestep (s)
T = cfg["n_timesteps"] * DT

LINVEL_X_MAX = cfg.get("linvel_x_max", 3.0)
X_ACCEL_MAX = cfg.get("x_accel_max", 6.0)
CAMERA_FOV = cfg["camera_fov"]  # vertical = horizontal for square image

half_angle_rad = math.radians(CAMERA_FOV / 2)
tan_half = math.tan(half_angle_rad)

rng = np.random.default_rng(cfg["random_seed"])

n_offscreen = 0
max_abs_pos_x = 0.0

for _ in range(N_SAMPLES):
    side = rng.choice([-1, 1])
    x0 = side * float(rng.uniform(0.6, 1.5))
    y0 = float(rng.uniform(-1.5, -0.5))

    v_x = float(rng.uniform(-LINVEL_X_MAX, LINVEL_X_MAX))
    x_accel = float(rng.uniform(-X_ACCEL_MAX, X_ACCEL_MAX))

    cam_jitter = [float(v) for v in rng.uniform(-0.5, 0.5, size=3)]
    cam_target_jitter_x = float(rng.uniform(-0.25, 0.25))

    # Final x position (analytic — y unchanged, no x-friction from floor at these timescales)
    x_final = x0 + v_x * T + 0.5 * x_accel * T**2

    # Camera eye position
    cam_x = cam_jitter[0]
    cam_y = -3.0 + cam_jitter[1]

    # Object depth from camera in y direction (y0 is constant — no y force)
    depth_y = y0 - cam_y  # positive (object is in front of camera)

    # Visible x half-width at object's depth (simple pinhole approximation)
    half_width_x = depth_y * tan_half

    # Camera optical axis x at object plane (based on look-at target)
    axis_x = cam_x + (cam_target_jitter_x - cam_x) * (depth_y / (-cam_y))

    off_screen = abs(x_final - axis_x) > half_width_x
    if off_screen:
        n_offscreen += 1
    max_abs_pos_x = max(max_abs_pos_x, abs(x_final))

frac = n_offscreen / N_SAMPLES
print(f"Config: n_timesteps={cfg['n_timesteps']}, T={T:.3f}s, fov={CAMERA_FOV}°")
print(f"  linvel_x_max={LINVEL_X_MAX}, x_accel_max={X_ACCEL_MAX}")
print(f"  Off-screen fraction: {n_offscreen}/{N_SAMPLES} = {frac:.1%}")
print(f"  Max |pos_x| at t={cfg['n_timesteps']}: {max_abs_pos_x:.3f} m")
print(
    f"  Visible half-width range: [{1.5*tan_half:.2f}, {2.5*tan_half:.2f}] m (at min/max depth)"
)
if frac > 0.05:
    print("  WARNING: >5% off-screen — consider reducing linvel_x_max/x_accel_max")
else:
    print("  OK: ≤5% off-screen — linvel_x_max/x_accel_max unchanged")
