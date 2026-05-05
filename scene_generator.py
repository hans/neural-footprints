"""
Scene generation using PyBullet.

Generates physics scenes with a central vertical occluding pillar, captures
raw program state (render buffers + physics labels + scene config), and
collects API-level physics labels for analysis.

Key design choices that make pixels insufficient for behavior prediction:
  1. A central pillar at x=0 may occlude the object in the final frame
  2. Behavior label (KE) uses final velocities — invisible in pixels
  3. Initial pixels cannot predict whether the object ends up behind the pillar

The program_state contains everything sufficient to resimulate the scene:
  render_bytes + physics_labels + scene_config + scene_lighting.
Non-render variables are a tiny fraction of the signal — swamped by pixels
in the random projection.
"""

import numpy as np
import pybullet as p
from config import (
    N_OBJECTS,
    IMAGE_SIZE,
    N_TIMESTEPS as _CFG_N_TIMESTEPS,
    PP_EARLY_FRAME as _CFG_PP_EARLY_FRAME,
    PP_LATE_FRAME as _CFG_PP_LATE_FRAME,
    CAMERA_FOV as _CFG_CAMERA_FOV,
    LINVEL_X_MAX as _CFG_LINVEL_X_MAX,
    X_ACCEL_MAX as _CFG_X_ACCEL_MAX,
)


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
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

    # Ground: infinite collision plane (physics) + solid-color visual box.
    # plane.urdf has a baked checkerboard texture that changeVisualShape
    # cannot override, so we separate the two bodies: GEOM_PLANE for
    # collision and a large GEOM_BOX for the visible surface.
    ground_col = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 0, 1],
                                        physicsClientId=physics_client)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_col,
                      basePosition=[0, 0, 0], physicsClientId=physics_client)
    ground_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[20, 20, 0.001],
                                     rgbaColor=[0.6, 0.6, 0.6, 1.0],
                                     physicsClientId=physics_client)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                      baseVisualShapeIndex=ground_vis,
                      basePosition=[0, 0, 0], physicsClientId=physics_client)

    # Central vertical pillar at x=0: VISUAL ONLY (no collision).
    # Objects pass through it freely — physics is unaffected.
    # Camera (at y=-3) cannot see objects behind it when they cross x=0.
    pillar_gray = float(rng.uniform(0.3, 0.8))
    pillar_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[PILLAR_WIDTH / 2, PILLAR_DEPTH / 2, PILLAR_HEIGHT / 2],
        rgbaColor=[pillar_gray, pillar_gray, pillar_gray, 1.0],
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
                                        specularColor=[0.4, 0.4, 0.4],
                                        physicsClientId=physics_client)
        shape_cfg = {'shape': 'sphere', 'params': {'radius': radius}, 'color': list(color)}
    else:
        half_extents = [float(v) for v in rng.uniform(0.1, 0.35, size=3)]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                           physicsClientId=physics_client)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                        rgbaColor=color,
                                        specularColor=[0.4, 0.4, 0.4],
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

    # x-only velocity (left or right); gravity handles vertical fall.
    # Range is configurable so the longer-window scene-gen experiments can
    # tighten linvel_x to keep the object on-screen.
    x_vel = float(rng.uniform(-_CFG_LINVEL_X_MAX, _CFG_LINVEL_X_MAX))
    p.resetBaseVelocity(body_ids[0], linearVelocity=[x_vel, 0.0, 0.0],
                        physicsClientId=physics_client)

    # Random x-acceleration (invisible in initial frame, breaks pixel predictability)
    x_accel = float(rng.uniform(-_CFG_X_ACCEL_MAX, _CFG_X_ACCEL_MAX))
    shape_configs[0]['x_accel'] = x_accel

    return body_ids, masses, frictions, is_occluded, shape_configs, pillar_gray


def _lock_rotation(body_ids, physics_client):
    """Zero angular velocity and reset orientation to identity for every body.

    Called after each simulation step. Free rigid bodies tumble under
    friction torque, but rotation is not a target of the inverse model
    (orn / angvel are not in observable_offsets) and only adds non-
    observable noise to the pixel features at t=mid and t=late.

    Note: resetBasePositionAndOrientation zeros linear velocity as a
    side-effect, so we capture lin_vel before the orientation reset and
    restore it via resetBaseVelocity.
    """
    for bid in body_ids:
        pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=physics_client)
        lin_vel, _ = p.getBaseVelocity(bid, physicsClientId=physics_client)
        p.resetBasePositionAndOrientation(bid, pos, [0.0, 0.0, 0.0, 1.0],
                                          physicsClientId=physics_client)
        p.resetBaseVelocity(bid, linearVelocity=lin_vel,
                            angularVelocity=[0.0, 0.0, 0.0],
                            physicsClientId=physics_client)


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


def _collect_physics_labels(body_ids, masses, frictions, physics_client,
                            applied_accels=None):
    """
    Collect per-object physics labels from the API.
    Per object: pos(3), orn(4), lin_vel(3), ang_vel(3), mass(1), friction(1), x_accel(1) = 16
    Total: 16 * N_OBJECTS floats.

    `applied_accels` is the per-object x-acceleration (in m/s²) being injected
    as an external force this scene; PyBullet has no notion of "acceleration"
    as state, so the caller must pass it. If None, defaults to zero per object
    (off-pipeline callers that don't apply acceleration).
    """
    if applied_accels is None:
        applied_accels = [0.0] * len(body_ids)
    labels = []
    for i, bid in enumerate(body_ids):
        pos, orn = p.getBasePositionAndOrientation(bid, physicsClientId=physics_client)
        lin_vel, ang_vel = p.getBaseVelocity(bid, physicsClientId=physics_client)
        labels.extend(pos)                  # 3
        labels.extend(orn)                  # 4
        labels.extend(lin_vel)              # 3
        labels.extend(ang_vel)              # 3
        labels.append(masses[i])            # 1
        labels.append(frictions[i])         # 1
        labels.append(applied_accels[i])    # 1
    return np.array(labels, dtype=np.float32)


def _encode_scene_config(shape_configs):
    """
    Encode scene shape configs into a fixed-length float32 vector.

    Per object: shape_is_box(1), radius(1), half_extents(3), color(4) = 9 floats.
    Unused fields zeroed (radius=0 for box, half_extents=[0,0,0] for sphere).

    Note: x_accel used to live here but moved to physics_labels — it is a
    kinematic state field (recoverable from three frames), not a render-
    determining scene parameter.
    """
    vec = []
    for cfg in shape_configs:
        if cfg['shape'] == 'box':
            vec.append(1.0)
            vec.append(0.0)
            vec.extend(cfg['params']['half_extents'])
        else:
            vec.append(0.0)
            vec.append(cfg['params']['radius'])
            vec.extend([0.0, 0.0, 0.0])
        vec.extend(cfg['color'])
    return np.array(vec, dtype=np.float32)


SCENE_CONFIG_DIM = 9  # per object


def _encode_scene_lighting(pillar_gray, lighting):
    """
    Encode per-scene lighting + camera parameters into a fixed-length float32 vector.

    pillar_gray(1), lightDirection(3), lightColor(3), lightDistance(1),
    camJitter(3) = 11 floats.
    """
    vec = [pillar_gray]
    vec.extend(lighting['lightDirection'])
    vec.extend(lighting['lightColor'])
    vec.append(lighting['lightDistance'])
    vec.extend(lighting.get('camJitter', [0.0, 0.0, 0.0]))
    return np.array(vec, dtype=np.float32)


SCENE_LIGHTING_DIM = 11


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


def _sample_lighting(rng):
    """Sample random lighting and camera parameters for a scene."""
    return {
        'lightDirection': [float(rng.uniform(-2, 2)),
                           float(rng.uniform(-2, 0)),
                           float(rng.uniform(1, 3))],
        'lightColor': [float(c) for c in rng.uniform(0.6, 1.0, size=3)],
        'lightDistance': float(rng.uniform(3.0, 8.0)),
        'camJitter': [float(v) for v in rng.uniform(-0.2, 0.2, size=3)],
    }


_DEFAULT_LIGHTING = {
    'lightDirection': [1, -1, 2],
    'lightColor': [1.0, 1.0, 1.0],
    'lightDistance': 5.0,
    'camJitter': [0.0, 0.0, 0.0],
}


def _render_scene(physics_client, lighting=None):
    """Render 64x64 image, return RGBA, depth, segmentation as raw bytes."""
    if lighting is None:
        lighting = _DEFAULT_LIGHTING
    jitter = lighting.get('camJitter', [0.0, 0.0, 0.0])
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0 + jitter[0], -3 + jitter[1], 2 + jitter[2]],
        cameraTargetPosition=[0, 0, 0.3],
        cameraUpVector=[0, 0, 1],
        physicsClientId=physics_client,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=_CFG_CAMERA_FOV, aspect=1.0, nearVal=0.1, farVal=10.0,
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

    rgba_bytes = rgba_arr.tobytes()
    depth_bytes = depth_arr.tobytes()
    seg_bytes = seg_arr.tobytes()

    return rgba_bytes, depth_bytes, seg_bytes


def resimulate_scene(shape_configs, initial_physics_row, *,
                     n_timesteps=None, return_program_state=False,
                     pillar_gray=0.5, lighting=None):
    """
    Rebuild a scene from stored shape configs + initial physics state, step
    N_TIMESTEPS, and return the rendered result.

    Used for oracle physics-model prediction: given the full initial state
    (position, velocity, mass, friction, shape, color), the simulation is
    deterministic.

    Args:
        shape_configs:       list of dicts (one per object) with keys
                             'shape' ('sphere'|'box'), 'params', 'color'
        initial_physics_row: 1-D array of length 16*N_OBJECTS:
                             per object: pos(3), orn(4), lin_vel(3), ang_vel(3), mass(1), friction(1), x_accel(1)
        return_program_state: if True, return full program_state float32 vector
                             (3-frame render buffers + physics labels + scene config + lighting).

    Returns:
        If return_program_state=False: RGBA uint8 [IMAGE_SIZE, IMAGE_SIZE, 4]
            of the BEHAVIORAL TARGET frame (rendered at t=n_timesteps).
        If return_program_state=True: float32 [D] program_state vector with
            three brain-input frames concatenated (t=0, t=PP_EARLY_FRAME,
            t=PP_LATE_FRAME).
    """
    if n_timesteps is None:
        n_timesteps = _CFG_N_TIMESTEPS
    pc = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=pc)
    ground_col = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 0, 1],
                                        physicsClientId=pc)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_col,
                      basePosition=[0, 0, 0], physicsClientId=pc)
    ground_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[20, 20, 0.001],
                                     rgbaColor=[0.6, 0.6, 0.6, 1.0],
                                     physicsClientId=pc)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                      baseVisualShapeIndex=ground_vis,
                      basePosition=[0, 0, 0], physicsClientId=pc)

    pillar_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[PILLAR_WIDTH / 2, PILLAR_DEPTH / 2, PILLAR_HEIGHT / 2],
        rgbaColor=[pillar_gray, pillar_gray, pillar_gray, 1.0],
        physicsClientId=pc,
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=pillar_vis,
        basePosition=[PILLAR_X, PILLAR_Y_CENTER, PILLAR_Z_CENTER],
        physicsClientId=pc,
    )

    body_ids = []
    masses_list = []
    frictions_list = []
    for i, cfg in enumerate(shape_configs):
        off = i * 16
        pos = initial_physics_row[off:off + 3].tolist()
        orn = initial_physics_row[off + 3:off + 7].tolist()
        lin_vel = initial_physics_row[off + 7:off + 10].tolist()
        ang_vel = initial_physics_row[off + 10:off + 13].tolist()
        mass = float(initial_physics_row[off + 13])
        friction = float(initial_physics_row[off + 14])
        # off + 15 is x_accel (the per-scene applied acceleration), read from
        # the cfg dict below for consistency with how _create_scene applies it.
        color = cfg['color']

        if cfg['shape'] == 'sphere':
            col = p.createCollisionShape(p.GEOM_SPHERE,
                                         radius=cfg['params']['radius'],
                                         physicsClientId=pc)
            vis = p.createVisualShape(p.GEOM_SPHERE,
                                      radius=cfg['params']['radius'],
                                      rgbaColor=color,
                                      specularColor=[0.4, 0.4, 0.4],
                                      physicsClientId=pc)
        else:
            col = p.createCollisionShape(p.GEOM_BOX,
                                         halfExtents=cfg['params']['half_extents'],
                                         physicsClientId=pc)
            vis = p.createVisualShape(p.GEOM_BOX,
                                      halfExtents=cfg['params']['half_extents'],
                                      rgbaColor=color,
                                      specularColor=[0.4, 0.4, 0.4],
                                      physicsClientId=pc)

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
        body_ids.append(body_id)
        masses_list.append(mass)
        frictions_list.append(friction)

    # Render the brain-input frames (t=0, t=PP_EARLY_FRAME, t=PP_LATE_FRAME)
    # only when building a full program_state. For RGBA-target callers we
    # only need the behavioral target at t=n_timesteps.
    initial_frame = _render_scene(pc, lighting=lighting) if return_program_state else None
    early_frame = None
    late_frame = None

    for t in range(n_timesteps):
        for i, (bid, cfg) in enumerate(zip(body_ids, shape_configs)):
            x_accel = cfg.get('x_accel', 0.0)
            if x_accel != 0.0:
                p.applyExternalForce(bid, -1,
                                     [x_accel * masses_list[i], 0, 0],
                                     [0, 0, 0], p.WORLD_FRAME,
                                     physicsClientId=pc)
        p.stepSimulation(physicsClientId=pc)
        _lock_rotation(body_ids, pc)
        if return_program_state and t + 1 == _CFG_PP_EARLY_FRAME:
            early_frame = _render_scene(pc, lighting=lighting)
        if return_program_state and t + 1 == _CFG_PP_LATE_FRAME:
            late_frame = _render_scene(pc, lighting=lighting)

    if return_program_state:
        applied_accels = [cfg.get('x_accel', 0.0) for cfg in shape_configs]
        final_physics = _collect_physics_labels(
            body_ids, masses_list, frictions_list, pc,
            applied_accels=applied_accels,
        )
        scene_config_vec = _encode_scene_config(shape_configs)
        lighting_vec = _encode_scene_lighting(pillar_gray, lighting or _DEFAULT_LIGHTING)
        p.disconnect(pc)
        return _build_program_state(
            [initial_frame, early_frame, late_frame],
            final_physics, scene_config_vec, lighting_vec,
        )
    else:
        # Behavioral target frame at t=n_timesteps.
        rgba_bytes, _, _ = _render_scene(pc, lighting=lighting)
        p.disconnect(pc)
        return np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(
            IMAGE_SIZE, IMAGE_SIZE, 4)


def _frame_render_vec(rgba_bytes, depth_bytes, seg_bytes):
    """One frame's RGBA+depth+seg bytes cast to float32 vector."""
    return np.frombuffer(rgba_bytes + depth_bytes + seg_bytes,
                         dtype=np.uint8).astype(np.float32)


def extract_brain_pixels(states, metadata):
    """RGBA bytes from all three brain-input frames, concatenated.

    `states` is any 2D array laid out like `program_states` (or `resim_*`
    program states): the leading block is three per-frame chunks of
    [RGBA | depth | seg]. This pulls just the RGBA portion of each frame.

    The result is what every prediction analysis (encoding, RSA, residual,
    dynamics, dissociation) consumes — it's the input a real scientist
    would have, given only camera output.
    """
    fri = metadata['frame_render_indices']
    rgba_bytes = (metadata['target_pixel_indices'].stop
                  - metadata['target_pixel_indices'].start)
    return np.concatenate(
        [states[:, s.start:s.start + rgba_bytes]
         for s in (fri['initial'], fri['early'], fri['late'])],
        axis=1,
    )


def extract_frame_pixels(frame_data, metadata):
    """RGBA bytes from a single-frame render array (initial/early/late/target_renders)."""
    s = metadata['target_pixel_indices']
    return frame_data[:, s]


def _build_program_state(frame_renders, physics_labels,
                         scene_config_vec, lighting_vec):
    """
    Concatenate three frames of render bytes (uint8->float32) with physics
    labels, scene config, and lighting parameters (native float32).

    Args:
        frame_renders: list of (rgba_bytes, depth_bytes, seg_bytes) tuples,
                       one per brain-input frame (initial / early / late).
        physics_labels, scene_config_vec, lighting_vec: per-scene 1-D arrays.

    The z-scoring in neural_model.py handles the scale difference between
    render bytes (0-255) and native float32 values.
    """
    render_vecs = [_frame_render_vec(*frame) for frame in frame_renders]
    return np.concatenate(render_vecs + [physics_labels, scene_config_vec,
                                          lighting_vec])


def generate_scenes(n_scenes, seed, *, n_timesteps=None):
    """
    Generate n_scenes PyBullet scenes, returning program states and analysis labels.

    Brain-input frames (full RGBA+depth+seg, concatenated into program_state):
      t=0 (initial), t=PP_EARLY_FRAME (early), t=PP_LATE_FRAME (late).

    The render captured at t=N_TIMESTEPS is held out as the behavioral
    prediction target (`target_renders`) and is *not* part of program_state,
    so it cannot leak into brain data.

    Returns dict with:
      'program_states':         ndarray [n_scenes x D]   — 3-frame render + physics_labels + scene_config + scene_lighting
      'physics_labels':         ndarray [n_scenes x 16*N_OBJECTS]  — final-state API labels (incl. x_accel)
      'initial_physics_labels': ndarray [n_scenes x 16*N_OBJECTS]  — t=0 API labels (incl. x_accel)
      'initial_renders':        ndarray [n_scenes x render_bytes_per_frame]  — t=0 full render
      'early_renders':          ndarray [n_scenes x render_bytes_per_frame]  — t=PP_EARLY_FRAME full render
      'late_renders':           ndarray [n_scenes x render_bytes_per_frame]  — t=PP_LATE_FRAME full render
      'target_renders':         ndarray [n_scenes x render_bytes_per_frame]  — t=N_TIMESTEPS full render (behavioral target only)
      'behavior_labels':        ndarray [n_scenes]       — binary, KE median split
      'kinetic_energies':       ndarray [n_scenes]       — continuous final KE
      'metadata': dict with dimension info
    """
    if n_timesteps is None:
        n_timesteps = _CFG_N_TIMESTEPS
    rng = np.random.default_rng(seed)

    # Render byte counts
    rgba_bytes_count = IMAGE_SIZE * IMAGE_SIZE * 4  # uint8
    depth_bytes_count = IMAGE_SIZE * IMAGE_SIZE * 4  # float32 bytes
    seg_bytes_count = IMAGE_SIZE * IMAGE_SIZE * 4    # int32 bytes
    render_bytes_per_frame = rgba_bytes_count + depth_bytes_count + seg_bytes_count
    n_brain_frames = 3
    render_bytes_total = n_brain_frames * render_bytes_per_frame

    physics_dim = 16 * N_OBJECTS
    config_dim = SCENE_CONFIG_DIM * N_OBJECTS
    D = render_bytes_total + physics_dim + config_dim + SCENE_LIGHTING_DIM
    program_states = np.zeros((n_scenes, D), dtype=np.float32)
    physics_labels = np.zeros((n_scenes, physics_dim), dtype=np.float32)
    initial_physics_labels = np.zeros((n_scenes, physics_dim), dtype=np.float32)
    initial_renders = np.zeros((n_scenes, render_bytes_per_frame), dtype=np.float32)
    early_renders = np.zeros((n_scenes, render_bytes_per_frame), dtype=np.float32)
    late_renders = np.zeros((n_scenes, render_bytes_per_frame), dtype=np.float32)
    target_renders = np.zeros((n_scenes, render_bytes_per_frame), dtype=np.float32)
    kinetic_energies = np.zeros(n_scenes, dtype=np.float32)
    all_scene_configs = []
    all_pillar_grays = []
    all_lightings = []

    for i in range(n_scenes):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Generating scene {i+1}/{n_scenes}...")

        pc = p.connect(p.DIRECT)
        scene_seed = rng.integers(0, 2**31)
        scene_rng = np.random.default_rng(scene_seed)

        body_ids, masses, frictions, is_occluded, shape_configs, pillar_gray = _create_scene(pc, scene_rng)
        all_scene_configs.append(shape_configs)
        all_pillar_grays.append(pillar_gray)

        # Sample lighting once per scene (consistent across all frames)
        lighting = _sample_lighting(scene_rng)
        all_lightings.append(lighting)

        # Capture initial state (t=0) before stepping
        applied_accels = [cfg.get('x_accel', 0.0) for cfg in shape_configs]
        initial_physics_labels[i] = _collect_physics_labels(
            body_ids, masses, frictions, pc, applied_accels=applied_accels,
        )
        init_frame = _render_scene(pc, lighting=lighting)
        initial_renders[i] = _frame_render_vec(*init_frame)

        # Step physics (with per-scene random x-acceleration as external force).
        # Capture full RGBA+depth+seg frames at t=PP_EARLY_FRAME and
        # t=PP_LATE_FRAME (the brain's later observations) and at
        # t=n_timesteps (the behavioral prediction target, not in brain data).
        early_frame = None
        late_frame = None
        target_frame = None
        for t in range(n_timesteps):
            for obj_idx, bid in enumerate(body_ids):
                x_accel = shape_configs[obj_idx].get('x_accel', 0.0)
                if x_accel != 0.0:
                    p.applyExternalForce(bid, -1,
                                         [x_accel * masses[obj_idx], 0, 0],
                                         [0, 0, 0], p.WORLD_FRAME,
                                         physicsClientId=pc)
            p.stepSimulation(physicsClientId=pc)
            _lock_rotation(body_ids, pc)
            if t + 1 == _CFG_PP_EARLY_FRAME:
                early_frame = _render_scene(pc, lighting=lighting)
                early_renders[i] = _frame_render_vec(*early_frame)
            if t + 1 == _CFG_PP_LATE_FRAME:
                late_frame = _render_scene(pc, lighting=lighting)
                late_renders[i] = _frame_render_vec(*late_frame)

        # Collect final-state analysis labels (NOT used in neural generation)
        physics_labels[i] = _collect_physics_labels(
            body_ids, masses, frictions, pc, applied_accels=applied_accels,
        )
        kinetic_energies[i] = _compute_total_kinetic_energy(body_ids, masses, pc)

        # Render behavioral target frame at t=n_timesteps (held out from brain)
        target_frame = _render_scene(pc, lighting=lighting)
        target_renders[i] = _frame_render_vec(*target_frame)

        # Build program state from the three brain-input frames.
        scene_config_vec = _encode_scene_config(shape_configs)
        lighting_vec = _encode_scene_lighting(pillar_gray, lighting)
        program_states[i] = _build_program_state(
            [init_frame, early_frame, late_frame],
            physics_labels[i], scene_config_vec, lighting_vec,
        )

        p.disconnect(pc)

    # Behavior label: median split on total final kinetic energy.
    # KE is directly recoverable from physics labels (mass + lin_vel).
    # Pixels carry no velocity signal → render model stays at chance.
    median_ke = np.median(kinetic_energies)
    behavior_labels = (kinetic_energies > median_ke).astype(np.int32)

    # Per-frame slices into program_state's render block.
    initial_slice = slice(0, render_bytes_per_frame)
    early_slice = slice(render_bytes_per_frame, 2 * render_bytes_per_frame)
    late_slice = slice(2 * render_bytes_per_frame, 3 * render_bytes_per_frame)
    # `pixel_indices` retains its old meaning (a single frame's RGBA inside
    # program_state) but now points at the LATE frame's RGBA — the latest
    # observation the brain has.
    late_rgba_start = late_slice.start
    pixel_indices = slice(late_rgba_start, late_rgba_start + rgba_bytes_count)

    metadata = {
        'D_render_bytes': render_bytes_total,
        'D_render_per_frame': render_bytes_per_frame,
        'D_physics_labels': physics_dim,
        'D_scene_config': config_dim,
        'D_scene_lighting': SCENE_LIGHTING_DIM,
        'D_total': D,
        'pixel_indices': pixel_indices,                  # RGBA of the LATE frame (inside program_state)
        'render_indices': slice(0, render_bytes_total),  # 3-frame full render block
        'frame_render_indices': {
            'initial': initial_slice,
            'early': early_slice,
            'late': late_slice,
        },
        'target_pixel_indices': slice(0, rgba_bytes_count),  # RGBA inside target_renders
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
        'early_renders': early_renders,
        'late_renders': late_renders,
        'target_renders': target_renders,
        'behavior_labels': behavior_labels,
        'kinetic_energies': kinetic_energies,
        'scene_configs': all_scene_configs,
        'pillar_grays': all_pillar_grays,
        'lightings': all_lightings,
        'metadata': metadata,
    }
