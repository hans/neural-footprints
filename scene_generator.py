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

# Visual-only background spheres (no collision, no physics) for extra pixel variation
N_BACKGROUND_OBJECTS = 5


def _create_ground(physics_client, ground_color):
    """Infinite collision plane (physics) + solid-color visual box (render).

    Returns the visual body ID so callers can apply per-scene styling.
    """
    ground_col = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 0, 1],
                                        physicsClientId=physics_client)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_col,
                      basePosition=[0, 0, 0], physicsClientId=physics_client)
    ground_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[20, 20, 0.001],
                                     rgbaColor=ground_color + [1.0],
                                     specularColor=[0, 0, 0],
                                     physicsClientId=physics_client)
    ground_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                       baseVisualShapeIndex=ground_vis,
                                       basePosition=[0, 0, 0],
                                       physicsClientId=physics_client)
    return ground_body_id


def _create_background_objects(physics_client, background_objects):
    """Instantiate visual-only static spheres from pre-sampled specs.

    Returns a list of body IDs (None where radius == 0, i.e. unused slot).
    """
    body_ids = []
    for obj in background_objects:
        if obj.get('radius', 0.0) <= 0.0:
            body_ids.append(None)
            continue
        specular = obj.get('specular', [0.0, 0.0, 0.0])
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=obj['radius'],
            rgbaColor=obj['color'] + [1.0],
            specularColor=specular,
            physicsClientId=physics_client,
        )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[obj['x'], obj['y'], obj['z']],
            physicsClientId=physics_client,
        )
        body_ids.append(bid)
    return body_ids


def _create_backdrop(physics_client, backdrop_color, backdrop_specular):
    """Large visual-only backdrop wall behind the scene.

    Fills the upper-background region of the camera view with a varying
    color/specular patch — physics-irrelevant, pixel-relevant.
    """
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[10.0, 0.05, 3.0],
        rgbaColor=backdrop_color + [1.0],
        specularColor=backdrop_specular,
        physicsClientId=physics_client,
    )
    bid = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 1.5, 2.0],
        physicsClientId=physics_client,
    )
    return bid


def _create_scene(physics_client, rng, lighting):
    """
    Spawn ground plane + central occluding pillar + single rigid body.

    The object starts clearly to the left or right of the pillar (x=0) and
    moves with a random x-only velocity. Depending on direction and speed,
    it may end up behind the pillar in the final frame.

    Varying per scene: shape (sphere/box/cylinder), color, specular,
    x-velocity direction, ground color (from `lighting`).
    """
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

    ground_body_id = _create_ground(physics_client,
                                    lighting.get('groundColor', [0.6, 0.6, 0.6]))

    # Central vertical pillar at x=0: VISUAL ONLY (no collision).
    # Objects pass through it freely — physics is unaffected.
    # Camera (at y=-3) cannot see objects behind it when they cross x=0.
    pillar_gray = float(rng.uniform(0.3, 0.8))
    pillar_specular = [float(v) for v in rng.uniform(0.0, 0.5, size=3)]
    pillar_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[PILLAR_WIDTH / 2, PILLAR_DEPTH / 2, PILLAR_HEIGHT / 2],
        rgbaColor=[pillar_gray, pillar_gray, pillar_gray, 1.0],
        specularColor=pillar_specular,
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
    # Per-scene specular for the foreground object: ranges from matte (0) to
    # glossy+tinted (1); physics-irrelevant, changes highlight appearance.
    specular = [float(v) for v in rng.uniform(0.0, 1.0, size=3)]

    # Random shape: sphere, box, or cylinder (equal probability)
    shape_roll = rng.random()
    if shape_roll < 1 / 3:
        radius = float(rng.uniform(0.07, 0.5))
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius,
                                           physicsClientId=physics_client)
        vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                        rgbaColor=color,
                                        specularColor=specular,
                                        physicsClientId=physics_client)
        shape_cfg = {'shape': 'sphere', 'params': {'radius': radius},
                     'color': list(color), 'specular': specular}
    elif shape_roll < 2 / 3:
        half_extents = [float(v) for v in rng.uniform(0.07, 0.5, size=3)]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                           physicsClientId=physics_client)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                        rgbaColor=color,
                                        specularColor=specular,
                                        physicsClientId=physics_client)
        shape_cfg = {'shape': 'box', 'params': {'half_extents': half_extents},
                     'color': list(color), 'specular': specular}
    else:
        radius = float(rng.uniform(0.07, 0.35))
        length = float(rng.uniform(0.15, 0.6))
        col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius,
                                           height=length,
                                           physicsClientId=physics_client)
        vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius,
                                        length=length,
                                        rgbaColor=color,
                                        specularColor=specular,
                                        physicsClientId=physics_client)
        shape_cfg = {'shape': 'cylinder',
                     'params': {'radius': radius, 'length': length},
                     'color': list(color), 'specular': specular}

    # Start clearly on one side of the pillar; random y depth and z height
    side = rng.choice([-1, 1])
    x = side * rng.uniform(0.6, 1.5)
    y = rng.uniform(-1.5, -0.5)
    z = rng.uniform(0.4, 0.8)
    pos = [x, y, z]
    # Random initial orientation (varies the t=0 frame; _lock_rotation resets
    # to identity from t=1 onward, so physics is unaffected).
    orn_angles = [float(v) for v in rng.uniform(-np.pi, np.pi, size=3)]
    orn = p.getQuaternionFromEuler(orn_angles)

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

    return body_ids, masses, frictions, is_occluded, shape_configs, pillar_gray, ground_body_id


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

    Per object:
      shape_is_box(1), shape_is_cylinder(1), radius(1), half_extents(3),
      color(4), specular(3) = 13 floats.
    Unused fields zeroed (e.g. radius=0 for box, half_extents=[0,0,0] for sphere).

    Note: x_accel lives in physics_labels (recoverable from three frames),
    not here.
    """
    vec = []
    for cfg in shape_configs:
        shape = cfg['shape']
        vec.append(1.0 if shape == 'box' else 0.0)
        vec.append(1.0 if shape == 'cylinder' else 0.0)
        if shape in ('sphere', 'cylinder'):
            vec.append(cfg['params']['radius'])
            vec.extend([0.0, 0.0, 0.0])
        else:
            vec.append(0.0)
            vec.extend(cfg['params']['half_extents'])
        vec.extend(cfg['color'])
        vec.extend(cfg.get('specular', [0.4, 0.4, 0.4]))
    return np.array(vec, dtype=np.float32)


SCENE_CONFIG_DIM = 13  # per object


def _encode_scene_lighting(pillar_gray, lighting):
    """
    Encode per-scene lighting, camera, ground, and background-object parameters.

    pillar_gray(1), lightDirection(3), lightColor(3), lightDistance(1),
    camJitter(3), camTargetJitter(3), lightAmbientCoeff(1),
    groundColor(3), backdropColor(3), backdropSpecular(3),
    per background sphere: x(1), y(1), z(1), radius(1), color(3), specular(3)
                           = 10 floats × N_BACKGROUND_OBJECTS.
    Total = 24 + 10 * N_BACKGROUND_OBJECTS floats.
    """
    vec = [pillar_gray]
    vec.extend(lighting['lightDirection'])
    vec.extend(lighting['lightColor'])
    vec.append(lighting['lightDistance'])
    vec.extend(lighting.get('camJitter', [0.0, 0.0, 0.0]))
    vec.extend(lighting.get('camTargetJitter', [0.0, 0.0, 0.0]))
    vec.append(lighting.get('lightAmbientCoeff', 0.4))
    vec.extend(lighting.get('groundColor', [0.6, 0.6, 0.6]))
    vec.extend(lighting.get('backdropColor', [0.2, 0.2, 0.4]))
    vec.extend(lighting.get('backdropSpecular', [0.0, 0.0, 0.0]))

    bg_objects = lighting.get('backgroundObjects', [])
    for i in range(N_BACKGROUND_OBJECTS):
        if i < len(bg_objects):
            obj = bg_objects[i]
            vec.extend([obj['x'], obj['y'], obj['z'], obj['radius']])
            vec.extend(obj['color'])
            vec.extend(obj.get('specular', [0.0, 0.0, 0.0]))
        else:
            vec.extend([0.0] * 10)

    return np.array(vec, dtype=np.float32)


# 24 base + 10 per background object
SCENE_LIGHTING_DIM = 24 + 10 * N_BACKGROUND_OBJECTS


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
    """
    Sample random lighting, camera, ground, and background-object parameters.

    Background objects are visual-only static spheres placed around the scene.
    Their positions and colors are physics-irrelevant but create independent
    pixel variation, increasing effective pixel-space dimensionality.
    """
    # Wider light-color gamut (0.4–1.0) with occasional warm/cool tints
    light_color = [float(c) for c in rng.uniform(0.4, 1.0, size=3)]

    background_objects = [
        {
            'x': float(rng.uniform(-3.0, 3.0)),
            'y': float(rng.uniform(-2.5, -0.2)),
            'z': float(rng.uniform(0.05, 0.8)),
            'radius': float(rng.uniform(0.15, 0.65)),
            'color': [float(c) for c in rng.uniform(0.05, 1.0, size=3)],
            'specular': [float(v) for v in rng.uniform(0.0, 0.9, size=3)],
        }
        for _ in range(N_BACKGROUND_OBJECTS)
    ]

    return {
        'lightDirection': [float(rng.uniform(-2.5, 2.5)),
                           float(rng.uniform(-2.5, -0.5)),
                           float(rng.uniform(0.5, 4.0))],
        'lightColor': light_color,
        'lightDistance': float(rng.uniform(2.5, 10.0)),
        'camJitter': [float(v) for v in rng.uniform(-0.35, 0.35, size=3)],
        'camTargetJitter': [float(rng.uniform(-0.2, 0.2)),
                            0.0,
                            float(rng.uniform(-0.15, 0.15))],
        'lightAmbientCoeff': float(rng.uniform(0.1, 0.7)),
        'groundColor': [float(c) for c in rng.uniform(0.10, 0.90, size=3)],
        'backdropColor': [float(c) for c in rng.uniform(0.05, 0.95, size=3)],
        'backdropSpecular': [float(v) for v in rng.uniform(0.0, 0.4, size=3)],
        'backgroundObjects': background_objects,
    }


_DEFAULT_LIGHTING = {
    'lightDirection': [1, -1, 2],
    'lightColor': [1.0, 1.0, 1.0],
    'lightDistance': 5.0,
    'camJitter': [0.0, 0.0, 0.0],
    'camTargetJitter': [0.0, 0.0, 0.0],
    'lightAmbientCoeff': 0.4,
    'groundColor': [0.6, 0.6, 0.6],
    'backdropColor': [0.2, 0.2, 0.4],
    'backdropSpecular': [0.0, 0.0, 0.0],
    'backgroundObjects': [
        {'x': 0.0, 'y': -1.0, 'z': 0.0, 'radius': 0.0,
         'color': [0.0, 0.0, 0.0], 'specular': [0.0, 0.0, 0.0]}
        for _ in range(N_BACKGROUND_OBJECTS)
    ],
}


def _render_scene(physics_client, lighting=None, render_size=None,
                  use_opengl=False):
    """Render image, return RGBA, depth, segmentation as raw bytes.

    render_size overrides IMAGE_SIZE when set; only use for visualization,
    not for building program_state (which must match IMAGE_SIZE).
    use_opengl uses ER_BULLET_HARDWARE_OPENGL (shadows); requires a GUI
    connection — only safe for visualization renders, not the pipeline.
    """
    if lighting is None:
        lighting = _DEFAULT_LIGHTING
    size = render_size if render_size is not None else IMAGE_SIZE
    jitter = lighting.get('camJitter', [0.0, 0.0, 0.0])
    tj = lighting.get('camTargetJitter', [0.0, 0.0, 0.0])
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0 + jitter[0], -3 + jitter[1], 2 + jitter[2]],
        cameraTargetPosition=[0 + tj[0], 0 + tj[1], 0.3 + tj[2]],
        cameraUpVector=[0, 0, 1],
        physicsClientId=physics_client,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=_CFG_CAMERA_FOV, aspect=1.0, nearVal=0.1, farVal=10.0,
        physicsClientId=physics_client,
    )

    kwargs = dict(
        width=size, height=size,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        shadow=1,
        lightDirection=lighting['lightDirection'],
        lightColor=lighting['lightColor'],
        lightDistance=lighting['lightDistance'],
        lightAmbientCoeff=lighting.get('lightAmbientCoeff', 0.4),
        physicsClientId=physics_client,
    )
    if use_opengl:
        kwargs['renderer'] = p.ER_BULLET_HARDWARE_OPENGL

    _, _, rgba, depth, seg = p.getCameraImage(**kwargs)

    rgba_arr = np.array(rgba, dtype=np.uint8).reshape(size, size, 4)
    depth_arr = np.array(depth, dtype=np.float32).reshape(size, size)
    seg_arr = np.array(seg, dtype=np.int32).reshape(size, size)

    rgba_bytes = rgba_arr.tobytes()
    depth_bytes = depth_arr.tobytes()
    seg_bytes = seg_arr.tobytes()

    return rgba_bytes, depth_bytes, seg_bytes


def open_render_client(use_gui=False):
    """Open a PyBullet physics client for rendering. Caller must disconnect."""
    pc = p.connect(p.GUI if use_gui else p.DIRECT,
                   options="--width=64 --height=64" if use_gui else "")
    if use_gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=pc)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=pc)
    return pc


def resimulate_scene(shape_configs, initial_physics_row, *,
                     n_timesteps=None, return_program_state=False,
                     pillar_gray=0.5, lighting=None, render_size=None,
                     use_gui=False, physics_client=None):
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
        physics_client:      existing client ID to reuse (caller manages lifecycle;
                             resetSimulation is called between scenes). If None,
                             a fresh client is opened and closed by this function.

    Returns:
        If return_program_state=False: RGBA uint8 [IMAGE_SIZE, IMAGE_SIZE, 4]
            of the BEHAVIORAL TARGET frame (rendered at t=n_timesteps).
        If return_program_state=True: float32 [D] program_state vector with
            three brain-input frames concatenated (t=0, t=PP_EARLY_FRAME,
            t=PP_LATE_FRAME).
    """
    if n_timesteps is None:
        n_timesteps = _CFG_N_TIMESTEPS
    owns_client = physics_client is None
    if owns_client:
        pc = open_render_client(use_gui)
    else:
        pc = physics_client
        p.resetSimulation(physicsClientId=pc)
    p.setGravity(0, 0, -9.81, physicsClientId=pc)
    _bg = lighting if lighting is not None else _DEFAULT_LIGHTING
    _create_ground(pc, _bg.get('groundColor', [0.6, 0.6, 0.6]))

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

    _create_background_objects(pc, _bg.get('backgroundObjects', []))
    _create_backdrop(pc,
                     _bg.get('backdropColor', [0.2, 0.2, 0.4]),
                     _bg.get('backdropSpecular', [0.0, 0.0, 0.0]))

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
        specular = cfg.get('specular', [0.4, 0.4, 0.4])

        if cfg['shape'] == 'sphere':
            col = p.createCollisionShape(p.GEOM_SPHERE,
                                         radius=cfg['params']['radius'],
                                         physicsClientId=pc)
            vis = p.createVisualShape(p.GEOM_SPHERE,
                                      radius=cfg['params']['radius'],
                                      rgbaColor=color,
                                      specularColor=specular,
                                      physicsClientId=pc)
        elif cfg['shape'] == 'cylinder':
            col = p.createCollisionShape(p.GEOM_CYLINDER,
                                         radius=cfg['params']['radius'],
                                         height=cfg['params']['length'],
                                         physicsClientId=pc)
            vis = p.createVisualShape(p.GEOM_CYLINDER,
                                      radius=cfg['params']['radius'],
                                      length=cfg['params']['length'],
                                      rgbaColor=color,
                                      specularColor=specular,
                                      physicsClientId=pc)
        else:
            col = p.createCollisionShape(p.GEOM_BOX,
                                         halfExtents=cfg['params']['half_extents'],
                                         physicsClientId=pc)
            vis = p.createVisualShape(p.GEOM_BOX,
                                      halfExtents=cfg['params']['half_extents'],
                                      rgbaColor=color,
                                      specularColor=specular,
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
    _rnd = lambda: _render_scene(pc, lighting=lighting, use_opengl=use_gui)
    initial_frame = _rnd() if return_program_state else None
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
            early_frame = _rnd()
        if return_program_state and t + 1 == _CFG_PP_LATE_FRAME:
            late_frame = _rnd()

    if return_program_state:
        applied_accels = [cfg.get('x_accel', 0.0) for cfg in shape_configs]
        final_physics = _collect_physics_labels(
            body_ids, masses_list, frictions_list, pc,
            applied_accels=applied_accels,
        )
        scene_config_vec = _encode_scene_config(shape_configs)
        lighting_vec = _encode_scene_lighting(pillar_gray, lighting or _DEFAULT_LIGHTING)
        if owns_client:
            p.disconnect(pc)
        return _build_program_state(
            [initial_frame, early_frame, late_frame],
            final_physics, scene_config_vec, lighting_vec,
        )
    else:
        # Behavioral target frame at t=n_timesteps.
        size = render_size if render_size is not None else IMAGE_SIZE
        rgba_bytes, _, _ = _render_scene(pc, lighting=lighting,
                                         render_size=render_size, use_opengl=use_gui)
        if owns_client:
            p.disconnect(pc)
        return np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(size, size, 4)


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


def generate_scenes(n_scenes, seed, *, n_timesteps=None, use_gui=False):
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

    # Reuse one connection for all scenes (resetSimulation between scenes).
    # GUI mode enables OpenGL shadow rendering; DIRECT is faster but shadowless.
    pc = p.connect(p.GUI if use_gui else p.DIRECT,
                    options="--width=64 --height=64" if use_gui else "")
    if use_gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=pc)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=pc)

    try:
        for i in range(n_scenes):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Generating scene {i+1}/{n_scenes}...")

            p.resetSimulation(physicsClientId=pc)
            scene_seed = rng.integers(0, 2**31)
            scene_rng = np.random.default_rng(scene_seed)

            # Sample lighting first so _create_scene can use groundColor.
            lighting = _sample_lighting(scene_rng)
            all_lightings.append(lighting)

            (body_ids, masses, frictions, is_occluded, shape_configs,
             pillar_gray, _ground_body_id) = _create_scene(pc, scene_rng, lighting)
            all_scene_configs.append(shape_configs)
            all_pillar_grays.append(pillar_gray)

            _create_background_objects(pc, lighting.get('backgroundObjects', []))
            _create_backdrop(
                pc,
                lighting.get('backdropColor', [0.2, 0.2, 0.4]),
                lighting.get('backdropSpecular', [0.0, 0.0, 0.0]),
            )

            _rnd = lambda: _render_scene(pc, lighting=lighting, use_opengl=use_gui)

            # Capture initial state (t=0) before stepping
            applied_accels = [cfg.get('x_accel', 0.0) for cfg in shape_configs]
            initial_physics_labels[i] = _collect_physics_labels(
                body_ids, masses, frictions, pc, applied_accels=applied_accels,
            )
            init_frame = _rnd()
            initial_renders[i] = _frame_render_vec(*init_frame)

            # Step physics (with per-scene random x-acceleration as external force).
            # Capture full RGBA+depth+seg frames at t=PP_EARLY_FRAME and
            # t=PP_LATE_FRAME (the brain's later observations) and at
            # t=n_timesteps (the behavioral prediction target, not in brain data).
            early_frame = None
            late_frame = None
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
                    early_frame = _rnd()
                    early_renders[i] = _frame_render_vec(*early_frame)
                if t + 1 == _CFG_PP_LATE_FRAME:
                    late_frame = _rnd()
                    late_renders[i] = _frame_render_vec(*late_frame)

            # Collect final-state analysis labels (NOT used in neural generation)
            physics_labels[i] = _collect_physics_labels(
                body_ids, masses, frictions, pc, applied_accels=applied_accels,
            )
            kinetic_energies[i] = _compute_total_kinetic_energy(body_ids, masses, pc)

            # Render behavioral target frame at t=n_timesteps (held out from brain)
            target_frame = _rnd()
            target_renders[i] = _frame_render_vec(*target_frame)

            # Build program state from the three brain-input frames.
            scene_config_vec = _encode_scene_config(shape_configs)
            lighting_vec = _encode_scene_lighting(pillar_gray, lighting)
            program_states[i] = _build_program_state(
                [init_frame, early_frame, late_frame],
                physics_labels[i], scene_config_vec, lighting_vec,
            )
    finally:
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
