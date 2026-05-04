"""Static numerosity scene generator (PyBullet).

Builds N=low vs N=high scenes of non-overlapping spheres on a ground plane —
no occluder, no motion, single rendered frame per scene. Two regimes:

  * confounded:        all spheres share a fixed radius. Total surface area,
                       edge density, and luminance scale linearly with N.
  * area_controlled:   per-sphere radius r = sqrt(target_total_area / (N*pi))
                       so the cumulative projected area is matched between
                       conditions. Edge density and other low-level features
                       still differ — but we have given the scientist their
                       best shot at canceling the most-cited confound.

Used by the subtractive-analysis pipeline (specs/subtractive_analysis.md).
The existing scene_generator.py (physics scenes) is untouched.
"""

import json

import numpy as np
import pybullet as p
import pybullet_data

from config import IMAGE_SIZE


_DEFAULT_LIGHTING = {
    'lightDirection': [1.0, -1.0, 2.0],
    'lightColor': [1.0, 1.0, 1.0],
    'lightDistance': 5.0,
}


def _radius_for(regime, N, base_radius, total_area):
    if regime == 'confounded':
        return float(base_radius)
    if regime == 'area_controlled':
        return float(np.sqrt(total_area / (N * np.pi)))
    raise ValueError(f"unknown regime: {regime!r}")


def _sample_positions(rng, N, radius, xy_extent, z_height, max_attempts):
    """Rejection-sample N (x, y) centers on the plane such that no pair is
    closer than 2*radius + 1e-3. Returns list of [x, y, z]."""
    lo, hi = xy_extent
    positions = []
    for _ in range(N):
        for _ in range(max_attempts):
            x = float(rng.uniform(lo, hi))
            y = float(rng.uniform(lo, hi))
            ok = True
            for px, py, _ in positions:
                if (px - x) ** 2 + (py - y) ** 2 < (2 * radius + 1e-3) ** 2:
                    ok = False
                    break
            if ok:
                positions.append([x, y, z_height])
                break
        else:
            # Couldn't place this sphere without overlap; shrink xy_extent
            # is the user's responsibility — fail loudly.
            raise RuntimeError(
                f"Failed to place {N} spheres of radius {radius:.3f} in "
                f"xy_extent={xy_extent} after {max_attempts} attempts. "
                f"Reduce N, reduce radius, or widen xy_extent."
            )
    return positions


def _render(physics_client, lighting):
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0, -3, 2],
        cameraTargetPosition=[0, 0, 0.3],
        cameraUpVector=[0, 0, 1],
        physicsClientId=physics_client,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=90, aspect=1.0, nearVal=0.1, farVal=10.0,
        physicsClientId=physics_client,
    )
    _, _, rgba, depth, seg = p.getCameraImage(
        width=IMAGE_SIZE, height=IMAGE_SIZE,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        shadow=1,
        lightDirection=lighting['lightDirection'],
        lightColor=lighting['lightColor'],
        lightDistance=lighting['lightDistance'],
        physicsClientId=physics_client,
    )
    rgba_arr = np.array(rgba, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 4)
    depth_arr = np.array(depth, dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
    seg_arr = np.array(seg, dtype=np.int32).reshape(IMAGE_SIZE, IMAGE_SIZE)
    return rgba_arr.tobytes(), depth_arr.tobytes(), seg_arr.tobytes()


def _build_program_state(rgba_bytes, depth_bytes, seg_bytes):
    """Concatenate render bytes (uint8 -> float32) into a single 1-D array.

    Matches the byte layout of scene_generator.py so the same render-PCA
    machinery applies. No physics_labels / scene_config / lighting tail —
    those are physics-pipeline-specific and irrelevant here.
    """
    return np.frombuffer(rgba_bytes + depth_bytes + seg_bytes,
                         dtype=np.uint8).astype(np.float32)


def generate_numerosity_scenes(*, regime, n_scenes_per_condition, n_low, n_high,
                               base_radius, area_controlled_total_area,
                               xy_extent, z_height, placement_max_attempts,
                               base_color, ground_color, seed):
    """Generate (n_scenes_per_condition * 2) static numerosity scenes.

    Returns a dict with:
        program_states  : float32 [n_scenes, D_render]   render bytes only
        condition       : int8 [n_scenes]    0 = low (N=n_low), 1 = high (N=n_high)
        N               : int32 [n_scenes]   actual sphere count
        radius          : float32 [n_scenes] per-scene sphere radius
        total_area      : float32 [n_scenes] sum_i pi * r_i^2
        regime          : str                "confounded" or "area_controlled"
        rgba_initial    : float32 [n_scenes, IMAGE_SIZE**2 * 4]   t=0 RGBA bytes
                          (used by the cardinality MLP's pixel PCA)
        metadata        : dict with byte counts + render_indices slice
    """
    n_scenes = 2 * n_scenes_per_condition
    rng = np.random.default_rng(seed)

    rgba_bytes_count  = IMAGE_SIZE * IMAGE_SIZE * 4
    depth_bytes_count = IMAGE_SIZE * IMAGE_SIZE * 4
    seg_bytes_count   = IMAGE_SIZE * IMAGE_SIZE * 4
    D_render = rgba_bytes_count + depth_bytes_count + seg_bytes_count

    program_states = np.zeros((n_scenes, D_render), dtype=np.float32)
    rgba_initial   = np.zeros((n_scenes, rgba_bytes_count), dtype=np.float32)
    condition = np.zeros(n_scenes, dtype=np.int8)
    N_arr     = np.zeros(n_scenes, dtype=np.int32)
    radius_arr = np.zeros(n_scenes, dtype=np.float32)
    total_area_arr = np.zeros(n_scenes, dtype=np.float32)

    # Interleave low/high to keep dataset balanced even if generation aborts mid-way.
    plan = []
    for _ in range(n_scenes_per_condition):
        plan.append((0, n_low))
        plan.append((1, n_high))
    rng.shuffle(plan)

    color = list(base_color)
    ground_rgba = list(ground_color)

    for i, (cond, N) in enumerate(plan):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Generating numerosity scene {i+1}/{n_scenes} "
                  f"(regime={regime}, N={N})...")
        radius = _radius_for(regime, N,
                             base_radius=base_radius,
                             total_area=area_controlled_total_area)
        positions = _sample_positions(rng, N, radius, xy_extent, z_height,
                                      placement_max_attempts)

        pc = p.connect(p.DIRECT)
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                      physicsClientId=pc)
            # Constant ground appearance; load plane then recolor.
            plane_id = p.loadURDF("plane.urdf", physicsClientId=pc)
            p.changeVisualShape(plane_id, -1, rgbaColor=ground_rgba,
                                physicsClientId=pc)

            vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=radius, rgbaColor=color,
                specularColor=[0.4, 0.4, 0.4],
                physicsClientId=pc,
            )
            for pos in positions:
                p.createMultiBody(
                    baseMass=0,                # static — no physics needed
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=vis,
                    basePosition=pos,
                    physicsClientId=pc,
                )

            rgba_b, depth_b, seg_b = _render(pc, _DEFAULT_LIGHTING)
        finally:
            p.disconnect(pc)

        program_states[i] = _build_program_state(rgba_b, depth_b, seg_b)
        rgba_initial[i] = np.frombuffer(rgba_b, dtype=np.uint8).astype(np.float32)
        condition[i] = cond
        N_arr[i] = N
        radius_arr[i] = radius
        total_area_arr[i] = float(N * np.pi * radius ** 2)

    metadata = {
        'D_render_bytes': D_render,
        'render_indices': slice(0, D_render),
        'pixel_indices':  slice(0, rgba_bytes_count),
        'rgba_bytes_count': rgba_bytes_count,
        'image_size': IMAGE_SIZE,
        'regime': regime,
        'n_low': n_low,
        'n_high': n_high,
    }

    print(f"  Numerosity scene generation complete ({regime}): "
          f"{n_scenes} scenes, {n_scenes_per_condition} per condition.")

    return {
        'program_states': program_states,
        'rgba_initial':   rgba_initial,
        'condition':      condition,
        'N':              N_arr,
        'radius':         radius_arr,
        'total_area':     total_area_arr,
        'regime':         regime,
        'metadata':       metadata,
    }


def save_numerosity_scenes(scenes, path):
    arrays = {
        'program_states': scenes['program_states'],
        'rgba_initial':   scenes['rgba_initial'],
        'condition':      scenes['condition'],
        'N':              scenes['N'],
        'radius':         scenes['radius'],
        'total_area':     scenes['total_area'],
        'regime':         np.array(scenes['regime']),
    }
    meta = dict(scenes['metadata'])
    pi = meta['pixel_indices']
    meta['pixel_indices'] = [pi.start, pi.stop]
    ri = meta['render_indices']
    meta['render_indices'] = [ri.start, ri.stop]
    arrays['metadata_json'] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def load_numerosity_scenes(path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data['metadata_json']))
    pi = meta['pixel_indices']
    meta['pixel_indices'] = slice(pi[0], pi[1])
    ri = meta['render_indices']
    meta['render_indices'] = slice(ri[0], ri[1])
    return {
        'program_states': data['program_states'],
        'rgba_initial':   data['rgba_initial'],
        'condition':      data['condition'],
        'N':              data['N'],
        'radius':         data['radius'],
        'total_area':     data['total_area'],
        'regime':         str(data['regime']),
        'metadata':       meta,
    }
