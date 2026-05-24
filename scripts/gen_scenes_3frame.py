"""
Three-frame scene generator with configurable camera FOV.

Off-pipeline. Renders at t=0, t=t_mid, t=t_late (in addition to the final
frame) and lets the camera FOV be set freely. Saves a custom .npz format
that scripts/eval_pp.py auto-detects.

Usage:
    uv run python scripts/gen_scenes_3frame.py \
        --t-mid 5 --t-late 15 --fov 90 \
        --output data/scenes_3f_fov90.npz
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pybullet as p
import pybullet_data

from config import IMAGE_SIZE, N_OBJECTS
from scene_generator import (
    _create_scene, _collect_physics_labels, _sample_lighting,
    _encode_scene_config, _encode_scene_lighting,
    _compute_total_kinetic_energy, _build_program_state,
    _DEFAULT_LIGHTING, SCENE_CONFIG_DIM, SCENE_LIGHTING_DIM,
)
from scripts.load_config import load_config


def _render_with_fov(pc, lighting, fov):
    if lighting is None:
        lighting = _DEFAULT_LIGHTING
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0, -3, 2],
        cameraTargetPosition=[0, 0, 0.3],
        cameraUpVector=[0, 0, 1],
        physicsClientId=pc,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov, aspect=1.0, nearVal=0.1, farVal=10.0,
        physicsClientId=pc,
    )
    _, _, rgba, depth, seg = p.getCameraImage(
        width=IMAGE_SIZE, height=IMAGE_SIZE,
        viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        shadow=1,
        lightDirection=lighting['lightDirection'],
        lightColor=lighting['lightColor'],
        lightDistance=lighting['lightDistance'],
        physicsClientId=pc,
    )
    rgba_arr = np.array(rgba, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 4)
    depth_arr = np.array(depth, dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
    seg_arr = np.array(seg, dtype=np.int32).reshape(IMAGE_SIZE, IMAGE_SIZE)
    return rgba_arr.tobytes(), depth_arr.tobytes(), seg_arr.tobytes()


def generate_3frame_scenes(n_scenes, seed, n_timesteps, t_mid, t_late, fov):
    rng = np.random.default_rng(seed)
    rgba_count = IMAGE_SIZE * IMAGE_SIZE * 4
    depth_count = IMAGE_SIZE * IMAGE_SIZE * 4
    seg_count = IMAGE_SIZE * IMAGE_SIZE * 4
    render_total = rgba_count + depth_count + seg_count
    physics_dim = 16 * N_OBJECTS
    config_dim = SCENE_CONFIG_DIM * N_OBJECTS
    D = render_total + physics_dim + config_dim + SCENE_LIGHTING_DIM

    program_states = np.zeros((n_scenes, D), dtype=np.float32)
    physics_labels = np.zeros((n_scenes, physics_dim), dtype=np.float32)
    initial_physics_labels = np.zeros((n_scenes, physics_dim), dtype=np.float32)
    initial_renders = np.zeros((n_scenes, rgba_count), dtype=np.float32)
    mid_renders = np.zeros((n_scenes, rgba_count), dtype=np.float32)
    late_renders = np.zeros((n_scenes, rgba_count), dtype=np.float32)
    kinetic_energies = np.zeros(n_scenes, dtype=np.float32)
    all_scene_configs, all_pillar_grays, all_lightings = [], [], []

    for i in range(n_scenes):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Scene {i+1}/{n_scenes}...")

        pc = p.connect(p.DIRECT)
        scene_seed = rng.integers(0, 2**31)
        scene_rng = np.random.default_rng(scene_seed)

        lighting = _sample_lighting(scene_rng)
        all_lightings.append(lighting)

        (body_ids, masses, frictions, _, shape_configs,
         pillar_gray, _ground_body_id) = _create_scene(pc, scene_rng, lighting)
        all_scene_configs.append(shape_configs)
        all_pillar_grays.append(pillar_gray)

        applied_accels = [cfg.get('x_accel', 0.0) for cfg in shape_configs]
        initial_physics_labels[i] = _collect_physics_labels(
            body_ids, masses, frictions, pc, applied_accels=applied_accels,
        )
        init_rgba, _, _ = _render_with_fov(pc, lighting, fov)
        initial_renders[i] = np.frombuffer(init_rgba, dtype=np.uint8).astype(np.float32)

        for t in range(n_timesteps):
            for obj_idx, bid in enumerate(body_ids):
                x_accel = shape_configs[obj_idx].get('x_accel', 0.0)
                if x_accel != 0.0:
                    p.applyExternalForce(bid, -1,
                                         [x_accel * masses[obj_idx], 0, 0],
                                         [0, 0, 0], p.WORLD_FRAME,
                                         physicsClientId=pc)
            p.stepSimulation(physicsClientId=pc)
            if t + 1 == t_mid:
                mid_rgba, _, _ = _render_with_fov(pc, lighting, fov)
                mid_renders[i] = np.frombuffer(mid_rgba, dtype=np.uint8).astype(np.float32)
            if t + 1 == t_late:
                late_rgba, _, _ = _render_with_fov(pc, lighting, fov)
                late_renders[i] = np.frombuffer(late_rgba, dtype=np.uint8).astype(np.float32)

        physics_labels[i] = _collect_physics_labels(
            body_ids, masses, frictions, pc, applied_accels=applied_accels,
        )
        kinetic_energies[i] = _compute_total_kinetic_energy(body_ids, masses, pc)

        rgba_bytes, depth_bytes, seg_bytes = _render_with_fov(pc, lighting, fov)
        scene_config_vec = _encode_scene_config(shape_configs)
        lighting_vec = _encode_scene_lighting(pillar_gray, lighting)
        program_states[i] = _build_program_state(
            rgba_bytes, depth_bytes, seg_bytes, physics_labels[i],
            scene_config_vec, lighting_vec
        )
        p.disconnect(pc)

    median_ke = np.median(kinetic_energies)
    behavior_labels = (kinetic_energies > median_ke).astype(np.int32)

    metadata = {
        'D_render_bytes': render_total,
        'D_physics_labels': physics_dim,
        'D_scene_config': config_dim,
        'D_scene_lighting': SCENE_LIGHTING_DIM,
        'D_total': D,
        'pixel_indices': [0, rgba_count],
        'render_indices': [0, render_total],
        'fov': fov,
        't_mid': t_mid,
        't_late': t_late,
    }

    return {
        'program_states': program_states,
        'physics_labels': physics_labels,
        'initial_physics_labels': initial_physics_labels,
        'initial_renders': initial_renders,
        'mid_renders': mid_renders,
        'late_renders': late_renders,
        # Alias so 2-frame eval paths still work (uses late as the "early" frame)
        'early_renders': late_renders,
        'behavior_labels': behavior_labels,
        'kinetic_energies': kinetic_energies,
        'scene_configs': all_scene_configs,
        'pillar_grays': all_pillar_grays,
        'lightings': all_lightings,
        'metadata': metadata,
    }


def save_3frame_scenes(scenes, path):
    arrays = {}
    for key in ['program_states', 'physics_labels', 'initial_physics_labels',
                'initial_renders', 'mid_renders', 'late_renders', 'early_renders',
                'behavior_labels', 'kinetic_energies']:
        arrays[key] = scenes[key]
    arrays['pillar_grays'] = np.array(scenes['pillar_grays'])
    arrays['metadata_json'] = np.array(json.dumps(scenes['metadata']))
    arrays['scene_configs_json'] = np.array(json.dumps(scenes['scene_configs']))
    arrays['lightings_json'] = np.array(json.dumps(scenes['lightings']))
    np.savez_compressed(path, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t-mid', type=int, required=True)
    ap.add_argument('--t-late', type=int, required=True)
    ap.add_argument('--fov', type=float, default=60.0)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n-scenes', type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    n_scenes = args.n_scenes if args.n_scenes is not None else cfg['n_scenes']
    print(f"3-frame generator: t_mid={args.t_mid}, t_late={args.t_late}, "
          f"fov={args.fov}, n_scenes={n_scenes}, n_timesteps={cfg['n_timesteps']}")

    t0 = time.time()
    scenes = generate_3frame_scenes(
        n_scenes, cfg['random_seed'], cfg['n_timesteps'],
        args.t_mid, args.t_late, args.fov,
    )
    print(f"Generation time: {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_3frame_scenes(scenes, args.output)
    print(f"Saved → {args.output}")


if __name__ == '__main__':
    main()
