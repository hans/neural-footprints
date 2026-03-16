"""
Scene generation using PyBullet.

Generates physics scenes with a central vertical occluding pillar, captures
raw program state (render buffers + saveBullet blob), and collects API-level
physics labels for analysis (never used in neural generation).

Key design choices that make pixels insufficient for behavior prediction:
  1. A central pillar at x=0 may occlude the object in the final frame
  2. Behavior label (KE) uses final velocities — invisible in pixels
  3. Initial pixels cannot predict whether the object ends up behind the pillar
"""

import os
import tempfile
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from config import N_OBJECTS, IMAGE_SIZE, N_TIMESTEPS


# Central vertical pillar at x=0 — occluder from camera's perspective
PILLAR_X = 0.0
PILLAR_WIDTH = 0.6    # total width in x
PILLAR_DEPTH = 2.0    # total depth in y — covers full object y range
PILLAR_HEIGHT = 1.5   # total height in z
PILLAR_Y_CENTER = -1.0
PILLAR_Z_CENTER = 0.75


def _create_scene(physics_client, rng):
    """
    Spawn ground plane + central occluding pillar + single rigid body.

    The object starts clearly to the left or right of the pillar (x=0) and
    moves with a random x-only velocity. Depending on direction and speed,
    it may end up behind the pillar in the final frame.

    Varying per scene: shape (sphere/box), color, x-velocity direction.
    """
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

    # Ground plane
    p.loadURDF("plane.urdf", physicsClientId=physics_client)

    # Central vertical pillar at x=0: VISUAL ONLY (no collision).
    # Objects pass through it freely — physics is unaffected.
    # Camera (at y=-3) cannot see objects behind it when they cross x=0.
    pillar_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[PILLAR_WIDTH / 2, PILLAR_DEPTH / 2, PILLAR_HEIGHT / 2],
        rgbaColor=[0.5, 0.5, 0.5, 1.0],
        physicsClientId=physics_client,
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,  # no collision — visual only
        baseVisualShapeIndex=pillar_vis,
        basePosition=[PILLAR_X, PILLAR_Y_CENTER, PILLAR_Z_CENTER],
        physicsClientId=physics_client,
    )

    body_ids = []
    masses = []
    frictions = []
    is_occluded = []
    shape_configs = []

    mass = rng.uniform(0.5, 5.0)
    friction = rng.uniform(0.1, 1.0)
    color = list(rng.uniform(0.1, 1.0, size=3)) + [1.0]

    # Random shape: sphere or box
    if rng.random() < 0.5:
        radius = float(rng.uniform(0.1, 0.35))
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius,
                                           physicsClientId=physics_client)
        vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                        rgbaColor=color,
                                        physicsClientId=physics_client)
        shape_cfg = {'shape': 'sphere', 'params': {'radius': radius}, 'color': list(color)}
    else:
        half_extents = [float(v) for v in rng.uniform(0.1, 0.35, size=3)]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                           physicsClientId=physics_client)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                        rgbaColor=color,
                                        physicsClientId=physics_client)
        shape_cfg = {'shape': 'box', 'params': {'half_extents': half_extents}, 'color': list(color)}

    # Start clearly on one side of the pillar; random y depth and z height
    side = rng.choice([-1, 1])
    x = side * rng.uniform(0.6, 1.5)
    y = rng.uniform(-1.5, -0.5)
    z = rng.uniform(0.4, 0.8)
    pos = [x, y, z]
    orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=pos,
        baseOrientation=orn,
        physicsClientId=physics_client,
    )
    p.changeDynamics(body_id, -1, lateralFriction=friction,
                     physicsClientId=physics_client)

    body_ids.append(body_id)
    masses.append(mass)
    frictions.append(friction)
    is_occluded.append(False)
    shape_configs.append(shape_cfg)

    # x-only velocity (left or right); gravity handles vertical fall
    x_vel = float(rng.uniform(-5.0, 5.0))
    p.resetBaseVelocity(body_ids[0], linearVelocity=[x_vel, 0.0, 0.0],
                        physicsClientId=physics_client)

    return body_ids, masses, frictions, is_occluded, shape_configs


def _get_initial_positions(body_ids, physics_client):
    """Record starting positions for behavior label computation."""
    positions = []
    for bid in body_ids:
        pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=physics_client)
        positions.append(np.array(pos))
    return positions


def _get_current_positions(body_ids, physics_client):
    """Get current positions of all objects."""
    positions = []
    for bid in body_ids:
        pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=physics_client)
        positions.append(np.array(pos))
    return positions


def _collect_physics_labels(body_ids, masses, frictions, physics_client):
    """
    Collect per-object physics labels from the API.
    Per object: position(3), orientation(4), linear_vel(3), angular_vel(3), mass(1), friction(1) = 15
    Total: 15 * N_OBJECTS = 75 floats
    """
    labels = []
    for i, bid in enumerate(body_ids):
        pos, orn = p.getBasePositionAndOrientation(bid, physicsClientId=physics_client)
        lin_vel, ang_vel = p.getBaseVelocity(bid, physicsClientId=physics_client)
        labels.extend(pos)        # 3
        labels.extend(orn)        # 4
        labels.extend(lin_vel)    # 3
        labels.extend(ang_vel)    # 3
        labels.append(masses[i])  # 1
        labels.append(frictions[i])  # 1
    return np.array(labels, dtype=np.float32)


def _compute_total_kinetic_energy(body_ids, masses, physics_client):
    """
    Total kinetic energy of all objects: KE = Σ 0.5 * mass_i * |lin_vel_i|².

    Directly computable from physics API labels (mass + final linear velocity).
    Not recoverable from pixel renders (pixels carry no velocity signal).

    Returns a float. The binary behavior label is computed as a median split
    across all scenes after generation.
    """
    ke = 0.0
    for i, bid in enumerate(body_ids):
        lin_vel, _ = p.getBaseVelocity(bid, physicsClientId=physics_client)
        ke += 0.5 * masses[i] * float(np.dot(lin_vel, lin_vel))
    return ke


def _render_scene(physics_client):
    """Render 64x64 image, return RGBA, depth, segmentation as raw bytes."""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0, -3, 2],
        cameraTargetPosition=[0, 0, 0.3],
        cameraUpVector=[0, 0, 1],
        physicsClientId=physics_client,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=10.0,
        physicsClientId=physics_client,
    )

    _, _, rgba, depth, seg = p.getCameraImage(
        width=IMAGE_SIZE, height=IMAGE_SIZE,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        physicsClientId=physics_client,
    )

    rgba_arr = np.array(rgba, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 4)
    depth_arr = np.array(depth, dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
    seg_arr = np.array(seg, dtype=np.int32).reshape(IMAGE_SIZE, IMAGE_SIZE)

    rgba_bytes = rgba_arr.tobytes()
    depth_bytes = depth_arr.tobytes()
    seg_bytes = seg_arr.tobytes()

    return rgba_bytes, depth_bytes, seg_bytes


def resimulate_scene(shape_configs, initial_physics_row):
    """
    Rebuild a scene from stored shape configs + initial physics state, step
    N_TIMESTEPS, and return the rendered RGBA image as uint8 [IMAGE_SIZE, IMAGE_SIZE, 4].

    Used for oracle physics-model prediction: given the full initial state
    (position, velocity, mass, friction, shape, color), the simulation is
    deterministic and produces a pixel-perfect final frame.

    Args:
        shape_configs:       list of dicts (one per object) with keys
                             'shape' ('sphere'|'box'), 'params', 'color'
        initial_physics_row: 1-D array of length 15*N_OBJECTS:
                             per object: pos(3), orn(4), lin_vel(3), ang_vel(3), mass(1), friction(1)
    """
    pc = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pc)
    p.setGravity(0, 0, -9.81, physicsClientId=pc)
    p.loadURDF("plane.urdf", physicsClientId=pc)

    pillar_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[PILLAR_WIDTH / 2, PILLAR_DEPTH / 2, PILLAR_HEIGHT / 2],
        rgbaColor=[0.5, 0.5, 0.5, 1.0],
        physicsClientId=pc,
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=pillar_vis,
        basePosition=[PILLAR_X, PILLAR_Y_CENTER, PILLAR_Z_CENTER],
        physicsClientId=pc,
    )

    for i, cfg in enumerate(shape_configs):
        off = i * 15
        pos = initial_physics_row[off:off + 3].tolist()
        orn = initial_physics_row[off + 3:off + 7].tolist()
        lin_vel = initial_physics_row[off + 7:off + 10].tolist()
        ang_vel = initial_physics_row[off + 10:off + 13].tolist()
        mass = float(initial_physics_row[off + 13])
        friction = float(initial_physics_row[off + 14])
        color = cfg['color']

        if cfg['shape'] == 'sphere':
            col = p.createCollisionShape(p.GEOM_SPHERE,
                                         radius=cfg['params']['radius'],
                                         physicsClientId=pc)
            vis = p.createVisualShape(p.GEOM_SPHERE,
                                      radius=cfg['params']['radius'],
                                      rgbaColor=color, physicsClientId=pc)
        else:
            col = p.createCollisionShape(p.GEOM_BOX,
                                         halfExtents=cfg['params']['half_extents'],
                                         physicsClientId=pc)
            vis = p.createVisualShape(p.GEOM_BOX,
                                      halfExtents=cfg['params']['half_extents'],
                                      rgbaColor=color, physicsClientId=pc)

        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=pc,
        )
        p.changeDynamics(body_id, -1, lateralFriction=friction, physicsClientId=pc)
        p.resetBaseVelocity(body_id, linearVelocity=lin_vel,
                            angularVelocity=ang_vel, physicsClientId=pc)

    for _ in range(N_TIMESTEPS):
        p.stepSimulation(physicsClientId=pc)

    rgba_bytes, _, _ = _render_scene(pc)
    p.disconnect(pc)

    return np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 4)


def _save_bullet_state(physics_client):
    """Save bullet state to temp file, read back as raw bytes."""
    tmp_path = os.path.join(tempfile.gettempdir(), "scene_state.bullet")
    p.saveBullet(tmp_path, physicsClientId=physics_client)
    with open(tmp_path, "rb") as f:
        bullet_bytes = f.read()
    os.remove(tmp_path)
    return bullet_bytes


def _build_program_state(rgba_bytes, depth_bytes, seg_bytes, bullet_bytes, bullet_k):
    """
    Concatenate all raw bytes, pad/truncate bullet to K, cast to float32.
    """
    # Pad or truncate bullet bytes
    if len(bullet_bytes) < bullet_k:
        bullet_bytes = bullet_bytes + b'\x00' * (bullet_k - len(bullet_bytes))
    else:
        bullet_bytes = bullet_bytes[:bullet_k]

    all_bytes = rgba_bytes + depth_bytes + seg_bytes + bullet_bytes
    # Cast to uint8 array, then to float32
    uint8_arr = np.frombuffer(all_bytes, dtype=np.uint8)
    return uint8_arr.astype(np.float32)


def calibrate_bullet_size(n_samples=50, seed=0):
    """Run n_samples scenes, return max .bullet file size + 20% buffer for padding constant K."""
    rng = np.random.default_rng(seed)
    max_size = 0

    for i in range(n_samples):
        pc = p.connect(p.DIRECT)
        scene_seed = rng.integers(0, 2**31)
        scene_rng = np.random.default_rng(scene_seed)
        body_ids, masses, frictions, is_occluded, _ = _create_scene(pc, scene_rng)

        # Step physics
        for _ in range(N_TIMESTEPS):
            p.stepSimulation(physicsClientId=pc)

        bullet_bytes = _save_bullet_state(pc)
        max_size = max(max_size, len(bullet_bytes))
        p.disconnect(pc)

    k = int(max_size * 1.2)
    # Round up to multiple of 4 for float32 alignment
    k = ((k + 3) // 4) * 4
    print(f"Calibration: max .bullet size = {max_size} bytes, K = {k} bytes (with 20% buffer)")
    return k


def generate_scenes(n_scenes, seed, bullet_k):
    """
    Generate n_scenes PyBullet scenes, returning program states and analysis labels.

    Returns dict with:
      'program_states':         ndarray [n_scenes x D]   — final render + bullet blob
      'physics_labels':         ndarray [n_scenes x 15*N_OBJECTS]  — final-state API labels
      'initial_physics_labels': ndarray [n_scenes x 15*N_OBJECTS]  — t=0 API labels
      'initial_renders':        ndarray [n_scenes x IMAGE_SIZE**2*4]  — t=0 RGBA bytes
      'behavior_labels':        ndarray [n_scenes]       — binary, KE median split
      'kinetic_energies':       ndarray [n_scenes]       — continuous final KE
      'metadata': dict with dimension info
    """
    rng = np.random.default_rng(seed)

    # Render byte counts
    rgba_bytes_count = IMAGE_SIZE * IMAGE_SIZE * 4  # uint8
    depth_bytes_count = IMAGE_SIZE * IMAGE_SIZE * 4  # float32 bytes
    seg_bytes_count = IMAGE_SIZE * IMAGE_SIZE * 4    # int32 bytes
    render_bytes_total = rgba_bytes_count + depth_bytes_count + seg_bytes_count
    total_bytes = render_bytes_total + bullet_k
    D = total_bytes  # each byte becomes one float32

    physics_dim = 15 * N_OBJECTS
    program_states = np.zeros((n_scenes, D), dtype=np.float32)
    physics_labels = np.zeros((n_scenes, physics_dim), dtype=np.float32)
    initial_physics_labels = np.zeros((n_scenes, physics_dim), dtype=np.float32)
    initial_renders = np.zeros((n_scenes, rgba_bytes_count), dtype=np.float32)
    kinetic_energies = np.zeros(n_scenes, dtype=np.float32)
    all_scene_configs = []

    for i in range(n_scenes):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Generating scene {i+1}/{n_scenes}...")

        pc = p.connect(p.DIRECT)
        scene_seed = rng.integers(0, 2**31)
        scene_rng = np.random.default_rng(scene_seed)

        body_ids, masses, frictions, is_occluded, shape_configs = _create_scene(pc, scene_rng)
        all_scene_configs.append(shape_configs)

        # Capture initial state (t=0) before stepping
        initial_physics_labels[i] = _collect_physics_labels(body_ids, masses, frictions, pc)
        init_rgba_bytes, _, _ = _render_scene(pc)
        initial_renders[i] = np.frombuffer(init_rgba_bytes, dtype=np.uint8).astype(np.float32)

        # Step physics
        for _ in range(N_TIMESTEPS):
            p.stepSimulation(physicsClientId=pc)

        # Collect final-state analysis labels (NOT used in neural generation)
        physics_labels[i] = _collect_physics_labels(body_ids, masses, frictions, pc)
        kinetic_energies[i] = _compute_total_kinetic_energy(body_ids, masses, pc)

        # Render final frame
        rgba_bytes, depth_bytes, seg_bytes = _render_scene(pc)

        # Save bullet state
        bullet_bytes = _save_bullet_state(pc)

        # Build program state
        program_states[i] = _build_program_state(
            rgba_bytes, depth_bytes, seg_bytes, bullet_bytes, bullet_k
        )

        p.disconnect(pc)

    # Behavior label: median split on total final kinetic energy.
    # KE is directly recoverable from physics labels (mass + lin_vel).
    # Pixels carry no velocity signal → render model stays at chance.
    median_ke = np.median(kinetic_energies)
    behavior_labels = (kinetic_energies > median_ke).astype(np.int32)

    # Metadata
    pixel_end = rgba_bytes_count  # RGBA is first in concatenation
    metadata = {
        'D_render_bytes': render_bytes_total,
        'D_physics_bytes': bullet_k,
        'D_total': D,
        'pixel_indices': slice(0, pixel_end),  # RGBA slice in float32 vector
    }

    behavior_rate = behavior_labels.mean()
    print(f"  Scene generation complete.")
    print(f"    Behavior label rate: {behavior_rate:.2%} (median-split on final KE)")
    print(f"    Median final KE: {median_ke:.4f}")

    return {
        'program_states': program_states,
        'physics_labels': physics_labels,
        'initial_physics_labels': initial_physics_labels,
        'initial_renders': initial_renders,
        'behavior_labels': behavior_labels,
        'kinetic_energies': kinetic_energies,
        'scene_configs': all_scene_configs,
        'metadata': metadata,
    }


def save_sample_renders(scenes, fig_dir, n_samples=16):
    """
    Save a grid of n_samples scenes showing initial (t=0) and final (t=N) renders.

    Helps assess scene visual complexity and feasibility of next-frame pixel prediction.
    Saved to {fig_dir}/sample_scenes.png.
    """
    initial_renders = scenes['initial_renders']
    program_states = scenes['program_states']
    pixel_indices = scenes['metadata']['pixel_indices']

    n = min(n_samples, len(initial_renders))
    fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title('t = 0 (initial)', fontsize=9)
    axes[0, 1].set_title(f't = {N_TIMESTEPS} (final)', fontsize=9)

    for i in range(n):
        # Initial RGBA
        init_rgba = initial_renders[i].astype(np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 4)
        # Final RGBA (first slice of program_states)
        final_rgba = program_states[i, pixel_indices].astype(np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 4)

        axes[i, 0].imshow(init_rgba)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(final_rgba)
        axes[i, 1].axis('off')

    plt.tight_layout(pad=0.3)
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'sample_scenes.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Sample renders saved: {fig_path}")
