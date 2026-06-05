"""
Scene generation using MuJoCo.

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

import os
import sys

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa" if sys.platform != "darwin" else "glfw"

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
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
PILLAR_WIDTH = 0.6  # total width in x
PILLAR_DEPTH = 2.0  # total depth in y — covers full object y range
PILLAR_HEIGHT = 1.5  # total height in z
PILLAR_Y_CENTER = -1.0
PILLAR_Z_CENTER = 0.75

N_BACKGROUND_OBJECTS = 5

SCENE_CONFIG_DIM = 11  # per object: shape_is_box(1), shape_is_cylinder(1), radius(1), half_extents(3), color(4), specular(1)
SCENE_LIGHTING_DIM = 57  # 15 base + 7 new (groundColor(3)+backdropColor(3)+backdropSpecular(1)) + 5×7 per bg sphere


_DEFAULT_LIGHTING = {
    "lightDirection": [0.5, -1, 3],
    "lightColor": [1.0, 1.0, 1.0],
    "lightDistance": 5.0,
    "camJitter": [0.0, 0.0, 0.0],
    "camTargetJitter": [0.0, 0.0, 0.0],
    "lightAmbientCoeff": 0.3,
}


def _sample_lighting(rng):
    """Sample random lighting and camera parameters for a scene."""
    return {
        "lightDirection": [
            float(rng.uniform(-1, 1)),
            float(rng.uniform(-2, 0)),
            float(rng.uniform(2, 4)),
        ],
        "lightColor": [float(c) for c in rng.uniform(0.6, 1.0, size=3)],
        "lightDistance": float(rng.uniform(3.0, 8.0)),
        "camJitter": [float(v) for v in rng.uniform(-0.5, 0.5, size=3)],
        "camTargetJitter": [
            float(rng.uniform(-0.25, 0.25)),
            0.0,
            float(rng.uniform(-0.1, 0.1)),
        ],
        "lightAmbientCoeff": float(rng.uniform(0.2, 0.4)),
        "groundColor": [float(c) for c in rng.uniform(0.3, 0.9, size=3)],
        "backdropColor": [float(c) for c in rng.uniform(0.1, 0.8, size=3)],
        "backdropSpecular": [float(v) for v in rng.uniform(0.0, 0.3, size=3)],
        "backgroundObjects": [
            {
                "x": float(rng.uniform(-3.0, 3.0)),
                "y": float(rng.uniform(0.5, 2.5)),
                "z": float(rng.uniform(0.1, 2.5)),
                "radius": float(rng.uniform(0.05, 0.4)),
                "color": [float(c) for c in rng.uniform(0.1, 1.0, size=3)],
                "specular": float(rng.uniform(0.0, 0.8)),
            }
            for _ in range(N_BACKGROUND_OBJECTS)
        ],
    }


def _look_at_quat(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.array(v, float) for v in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z_ax = -fwd
    x_ax = np.cross(fwd, up)
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    R = np.column_stack([x_ax, y_ax, z_ax])
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]


def _build_mjspec(rng):
    """Build a MuJoCo model/data for one randomized scene.

    Returns (model, data, body_id, qvel_offset, mass, friction, x_accel,
             shape_config, pillar_gray, lighting).
    """
    pillar_gray = float(rng.uniform(0.3, 0.8))
    lighting = _sample_lighting(rng)

    spec = mujoco.MjSpec()
    spec.option.gravity = [0, 0, -9.81]

    sky_tex = spec.add_texture()
    sky_tex.name = "sky"
    sky_tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky_tex.rgb1 = [0.3, 0.5, 0.8]
    sky_tex.rgb2 = [0.65, 0.8, 1.0]
    sky_tex.width = 512
    sky_tex.height = 512

    floor_tex = spec.add_texture()
    floor_tex.name = "checker"
    floor_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_tex.rgb1 = [0.08, 0.08, 0.08]
    floor_tex.rgb2 = [0.75, 0.70, 0.60]
    floor_tex.width = 512
    floor_tex.height = 512

    floor_mat = spec.add_material()
    floor_mat.name = "floor_mat"
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "checker"
    floor_mat.texrepeat = [5.0, 5.0]
    floor_mat.shininess = 0.0
    floor_mat.specular = 0.0

    # Ground plane
    g = spec.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [0, 0, 0.01]
    g.material = "floor_mat"

    # Pillar (visual only)
    pil = spec.worldbody.add_geom()
    pil.type = mujoco.mjtGeom.mjGEOM_BOX
    pil.size = [PILLAR_WIDTH / 2, PILLAR_DEPTH / 2, PILLAR_HEIGHT / 2]
    pil.pos = [PILLAR_X, PILLAR_Y_CENTER, PILLAR_Z_CENTER]
    pil.rgba = [pillar_gray, pillar_gray, pillar_gray, 1.0]
    pil.contype = 0
    pil.conaffinity = 0

    # Light
    lt = spec.worldbody.add_light()
    lt.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    lt.dir = [-d for d in lighting["lightDirection"]]
    lt.diffuse = lighting["lightColor"]
    lt.ambient = [lighting["lightAmbientCoeff"]] * 3
    lt.specular = [0.5, 0.5, 0.5]
    lt.castshadow = True

    # Camera
    jitter = lighting.get("camJitter", [0, 0, 0])
    tj = lighting.get("camTargetJitter", [0, 0, 0])
    eye = [jitter[0], -3 + jitter[1], 2 + jitter[2]]
    target = [tj[0], tj[1], 0.3 + tj[2]]
    cam = spec.worldbody.add_camera()
    cam.name = "scene_cam"
    cam.pos = eye
    cam.quat = _look_at_quat(eye, target)
    cam.fovy = _CFG_CAMERA_FOV

    # Background spheres (visual only)
    bg_objects = lighting.get("backgroundObjects", [])
    for i, obj in enumerate(bg_objects):
        if obj.get("radius", 0.0) <= 0.0:
            continue
        mat = spec.add_material()
        mat.name = f"bg_mat_{i}"
        mat.rgba = [*obj["color"], 1.0]
        mat.shininess = obj.get("specular", 0.0)
        bg_geom = spec.worldbody.add_geom()
        bg_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        bg_geom.size = [obj["radius"], 0, 0]
        bg_geom.pos = [obj["x"], obj["y"], obj["z"]]
        bg_geom.material = f"bg_mat_{i}"
        bg_geom.contype = 0
        bg_geom.conaffinity = 0

    # Backdrop wall (visual only)
    backdrop_color = lighting.get("backdropColor", [0.2, 0.2, 0.4])
    backdrop_spec = lighting.get("backdropSpecular", [0.0, 0.0, 0.0])
    bd_mat = spec.add_material()
    bd_mat.name = "backdrop_mat"
    bd_mat.rgba = [*backdrop_color, 1.0]
    bd_mat.shininess = float(np.mean(backdrop_spec))
    bd_geom = spec.worldbody.add_geom()
    bd_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    bd_geom.size = [10.0, 0.05, 3.0]
    bd_geom.pos = [0.0, 2.0, 2.0]
    bd_geom.material = "backdrop_mat"
    bd_geom.contype = 0
    bd_geom.conaffinity = 0

    # Object randomization (same distributions as old _create_scene)
    mass = rng.uniform(0.5, 5.0)
    friction = rng.uniform(0.1, 1.0)
    color = list(rng.uniform(0.1, 1.0, size=3)) + [1.0]
    side = rng.choice([-1, 1])
    x = side * rng.uniform(0.6, 1.5)
    y = rng.uniform(-1.5, -0.5)
    z = rng.uniform(0.4, 0.8)
    x_vel = float(rng.uniform(-_CFG_LINVEL_X_MAX, _CFG_LINVEL_X_MAX))
    x_accel = float(rng.uniform(-_CFG_X_ACCEL_MAX, _CFG_X_ACCEL_MAX))
    specular = float(rng.uniform(0.0, 0.8))

    shape_roll = rng.random()
    if shape_roll < 1 / 3:  # sphere
        radius = float(rng.uniform(0.07, 0.5))
        shape_cfg = {
            "shape": "sphere",
            "params": {"radius": radius},
            "color": list(color),
            "specular": specular,
            "x_accel": x_accel,
        }
    elif shape_roll < 2 / 3:  # box
        half_extents = [float(v) for v in rng.uniform(0.07, 0.5, size=3)]
        shape_cfg = {
            "shape": "box",
            "params": {"half_extents": half_extents},
            "color": list(color),
            "specular": specular,
            "x_accel": x_accel,
        }
    else:  # cylinder
        radius = float(rng.uniform(0.07, 0.35))
        half_length = float(rng.uniform(0.075, 0.3))
        shape_cfg = {
            "shape": "cylinder",
            "params": {"radius": radius, "half_length": half_length},
            "color": list(color),
            "specular": specular,
            "x_accel": x_accel,
        }

    # Object material (handles color + specular)
    obj_mat = spec.add_material()
    obj_mat.name = "obj_mat"
    obj_mat.rgba = color
    obj_mat.shininess = specular

    # Object body — 3 slide joints (no rotation DOF)
    body = spec.worldbody.add_body()
    body.name = "object"
    body.pos = [x, y, z]
    for name, axis in [
        ("obj_sx", [1, 0, 0]),
        ("obj_sy", [0, 1, 0]),
        ("obj_sz", [0, 0, 1]),
    ]:
        jnt = body.add_joint()
        jnt.name = name
        jnt.type = mujoco.mjtJoint.mjJNT_SLIDE
        jnt.axis = axis

    geom = body.add_geom()
    if shape_cfg["shape"] == "sphere":
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        geom.size = [shape_cfg["params"]["radius"], 0, 0]
    elif shape_cfg["shape"] == "cylinder":
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [
            shape_cfg["params"]["radius"],
            shape_cfg["params"]["half_length"],
            0,
        ]
    else:  # box
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = shape_cfg["params"]["half_extents"]
    geom.material = "obj_mat"
    geom.friction = [friction, 0.005, 0.0001]
    geom.mass = mass  # NOTE: if MuJoCo requires density instead, compute density = mass / volume

    model = spec.compile()
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    qvel_offset = int(
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_sx")]
    )

    # Set initial velocity
    data.qvel[qvel_offset : qvel_offset + 3] = [x_vel, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    return (
        model,
        data,
        body_id,
        qvel_offset,
        mass,
        friction,
        x_accel,
        shape_cfg,
        pillar_gray,
        lighting,
    )


def _render_frame(renderer, model, data, lighting=None):
    """Returns (rgba_bytes, depth_bytes, seg_bytes)."""
    renderer.update_scene(data, camera="scene_cam")

    # Pass 1: RGB → RGBA
    rgb = renderer.render()  # (H, W, 3) uint8
    H, W = rgb.shape[:2]
    alpha = np.full((H, W, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=2)
    rgba_bytes = rgba.tobytes()

    # Pass 2: depth
    renderer.enable_depth_rendering()
    depth = renderer.render()  # (H, W) float32
    renderer.disable_depth_rendering()
    depth_bytes = depth.tobytes()

    # Pass 3: segmentation
    renderer.enable_segmentation_rendering()
    seg_raw = renderer.render()  # (H, W, 2) int32
    renderer.disable_segmentation_rendering()
    seg = seg_raw[:, :, 0].astype(np.int32)
    seg_bytes = seg.tobytes()

    return rgba_bytes, depth_bytes, seg_bytes


def _collect_physics_labels(body_id, qvel_offset, mass, friction, x_accel, data):
    """Returns float32 array of shape (16,): pos(3), orn(4), lin_vel(3), ang_vel(3), mass(1), friction(1), x_accel(1)."""
    pos = data.xpos[body_id].tolist()
    orn = [0.0, 0.0, 0.0, 1.0]  # identity — slide joints, no rotation
    lin_vel = data.qvel[qvel_offset : qvel_offset + 3].tolist()
    ang_vel = [0.0, 0.0, 0.0]
    labels = pos + orn + lin_vel + ang_vel + [mass, friction, x_accel]
    return np.array(labels, dtype=np.float32)


def _compute_total_kinetic_energy(qvel_offset, mass, data):
    lin_vel = data.qvel[qvel_offset : qvel_offset + 3]
    return float(0.5 * mass * np.dot(lin_vel, lin_vel))


def _encode_scene_config(shape_configs):
    """
    Encode scene shape configs into a fixed-length float32 vector.

    Per object: shape_is_box(1), shape_is_cylinder(1), radius(1),
                half_extents(3), color(4), specular(1) = 13 floats.
    Unused fields zeroed (radius=0 for box, half_extents=[0,0,0] for sphere/cylinder).

    Note: x_accel used to live here but moved to physics_labels — it is a
    kinematic state field (recoverable from three frames), not a render-
    determining scene parameter.
    """
    vec = []
    for cfg in shape_configs:
        shape = cfg["shape"]
        vec.append(1.0 if shape == "box" else 0.0)
        vec.append(1.0 if shape == "cylinder" else 0.0)
        if shape == "box":
            vec.append(0.0)
            vec.extend(cfg["params"]["half_extents"])
        else:  # sphere or cylinder
            vec.append(cfg["params"]["radius"])
            vec.extend([0.0, 0.0, 0.0])
        vec.extend(cfg["color"])
        vec.append(cfg.get("specular", 0.4))
    return np.array(vec, dtype=np.float32)


def _encode_scene_lighting(pillar_gray, lighting):
    """
    Encode per-scene lighting and camera parameters into a fixed-length float32 vector.

    pillar_gray(1), lightDirection(3), lightColor(3), lightDistance(1),
    camJitter(3), camTargetJitter(3), lightAmbientCoeff(1) = 15 base floats.
    groundColor(3), backdropColor(3), backdropSpecular(1) = 7 new floats → subtotal 22.
    5 background spheres × 7 floats each (x,y,z,radius,color[3]) = 35 → total 57.
    """
    vec = [pillar_gray]
    vec.extend(lighting["lightDirection"])  # 3
    vec.extend(lighting["lightColor"])  # 3
    vec.append(lighting["lightDistance"])  # 1
    vec.extend(lighting.get("camJitter", [0.0, 0.0, 0.0]))  # 3
    vec.extend(lighting.get("camTargetJitter", [0.0, 0.0, 0.0]))  # 3
    vec.append(lighting.get("lightAmbientCoeff", 0.4))  # 1 → subtotal 15
    vec.extend(lighting.get("groundColor", [0.6, 0.6, 0.6]))  # 3
    vec.extend(lighting.get("backdropColor", [0.2, 0.2, 0.4]))  # 3
    vec.append(
        float(np.mean(lighting.get("backdropSpecular", [0.0, 0.0, 0.0])))
    )  # 1 → subtotal 22
    # 5 background spheres × 7 floats each = 35
    bg = lighting.get("backgroundObjects", [])
    for i in range(N_BACKGROUND_OBJECTS):
        if i < len(bg):
            o = bg[i]
            vec.extend([o["x"], o["y"], o["z"], o["radius"]])  # 4
            vec.extend(o["color"])  # 3
        else:
            vec.extend([0.0] * 7)
    return np.array(vec, dtype=np.float32)


def _frame_render_vec(rgba_bytes, depth_bytes, seg_bytes):
    """One frame's RGBA+depth+seg bytes cast to float32 vector."""
    return np.frombuffer(rgba_bytes + depth_bytes + seg_bytes, dtype=np.uint8).astype(
        np.float32
    )


def _build_program_state(frame_renders, physics_labels, scene_config_vec, lighting_vec):
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
    return np.concatenate(
        render_vecs + [physics_labels, scene_config_vec, lighting_vec]
    )


def extract_brain_pixels(states, metadata):
    """RGBA bytes from all three brain-input frames, concatenated.

    `states` is any 2D array laid out like `program_states` (or `resim_*`
    program states): the leading block is three per-frame chunks of
    [RGBA | depth | seg]. This pulls just the RGBA portion of each frame.

    The result is what every prediction analysis (encoding, RSA, residual,
    dynamics, dissociation) consumes — it's the input a real scientist
    would have, given only camera output.
    """
    fri = metadata["frame_render_indices"]
    rgba_bytes = (
        metadata["target_pixel_indices"].stop - metadata["target_pixel_indices"].start
    )
    return np.concatenate(
        [
            states[:, s.start : s.start + rgba_bytes]
            for s in (fri["initial"], fri["early"], fri["late"])
        ],
        axis=1,
    )


def extract_frame_pixels(frame_data, metadata):
    """RGBA bytes from a single-frame render array (initial/early/late/target_renders)."""
    s = metadata["target_pixel_indices"]
    return frame_data[:, s]


def open_render_client(use_gui=False):
    """Stub — MuJoCo does not use persistent render clients."""
    return None


def generate_scenes(n_scenes, seed, *, n_timesteps=None, use_gui=False):
    """
    Generate n_scenes MuJoCo scenes, returning program states and analysis labels.

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
    seg_bytes_count = IMAGE_SIZE * IMAGE_SIZE * 4  # int32 bytes
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

    # Create renderer once — OSMesa/EGL context init costs seconds per call.
    # All scenes share the same model structure (ngeom/nbody/ncam identical),
    # so we reuse the GL context and swap renderer._model each iteration.
    renderer = None
    try:
        for i in range(n_scenes):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Generating scene {i+1}/{n_scenes}...")

            scene_seed = rng.integers(0, 2**31)
            scene_rng = np.random.default_rng(scene_seed)

            (
                model,
                data,
                body_id,
                qvel_offset,
                mass,
                friction,
                x_accel,
                shape_cfg,
                pillar_gray,
                lighting,
            ) = _build_mjspec(scene_rng)
            all_scene_configs.append([shape_cfg])
            all_pillar_grays.append(pillar_gray)
            all_lightings.append(lighting)

            if renderer is None:
                renderer = mujoco.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE)
            else:
                renderer._model = model

            applied_accels = [x_accel]
            initial_physics_labels[i] = _collect_physics_labels(
                body_id, qvel_offset, mass, friction, x_accel, data
            )
            init_frame = _render_frame(renderer, model, data, lighting)
            initial_renders[i] = _frame_render_vec(*init_frame)

            early_frame = None
            late_frame = None
            for t in range(n_timesteps):
                data.xfrc_applied[body_id, 0] = x_accel * mass
                mujoco.mj_step(model, data)
                data.xfrc_applied[body_id, 0] = 0.0
                if t + 1 == _CFG_PP_EARLY_FRAME:
                    early_frame = _render_frame(renderer, model, data, lighting)
                    early_renders[i] = _frame_render_vec(*early_frame)
                if t + 1 == _CFG_PP_LATE_FRAME:
                    late_frame = _render_frame(renderer, model, data, lighting)
                    late_renders[i] = _frame_render_vec(*late_frame)

            physics_labels[i] = _collect_physics_labels(
                body_id, qvel_offset, mass, friction, x_accel, data
            )
            kinetic_energies[i] = _compute_total_kinetic_energy(qvel_offset, mass, data)

            target_frame = _render_frame(renderer, model, data, lighting)
            target_renders[i] = _frame_render_vec(*target_frame)

            scene_config_vec = _encode_scene_config([shape_cfg])
            lighting_vec = _encode_scene_lighting(pillar_gray, lighting)
            program_states[i] = _build_program_state(
                [init_frame, early_frame, late_frame],
                physics_labels[i],
                scene_config_vec,
                lighting_vec,
            )
    finally:
        if renderer is not None:
            del renderer

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
        "D_render_bytes": render_bytes_total,
        "D_render_per_frame": render_bytes_per_frame,
        "D_physics_labels": physics_dim,
        "D_scene_config": config_dim,
        "D_scene_lighting": SCENE_LIGHTING_DIM,
        "D_total": D,
        "pixel_indices": pixel_indices,  # RGBA of the LATE frame (inside program_state)
        "render_indices": slice(0, render_bytes_total),  # 3-frame full render block
        "frame_render_indices": {
            "initial": initial_slice,
            "early": early_slice,
            "late": late_slice,
        },
        "target_pixel_indices": slice(
            0, rgba_bytes_count
        ),  # RGBA inside target_renders
    }

    behavior_rate = behavior_labels.mean()
    print(f"  Scene generation complete.")
    print(f"    Behavior label rate: {behavior_rate:.2%} (median-split on final KE)")
    print(f"    Median final KE: {median_ke:.4f}")

    return {
        "program_states": program_states,
        "physics_labels": physics_labels,
        "initial_physics_labels": initial_physics_labels,
        "initial_renders": initial_renders,
        "early_renders": early_renders,
        "late_renders": late_renders,
        "target_renders": target_renders,
        "behavior_labels": behavior_labels,
        "kinetic_energies": kinetic_energies,
        "scene_configs": all_scene_configs,
        "pillar_grays": all_pillar_grays,
        "lightings": all_lightings,
        "metadata": metadata,
    }


def _build_scene_model(
    shape_configs,
    initial_physics_row,
    pillar_gray,
    lighting,
    offscreen=None,
    camera_fov=None,
):
    """Rebuild and compile the MjModel/MjData for a stored scene.

    Extracted verbatim from resimulate_scene's spec-building block so the same
    deterministic reconstruction can be reused for full-animation rendering.
    Returns (model, data, body_id, qvel_offset), with the object's initial
    linear velocity applied and mj_forward called.

    offscreen: if not None, raise spec.visual.global_.offwidth/offheight to this
        many pixels before compile, so render sizes above MuJoCo's default
        640x480 offscreen framebuffer cap are supported.
    """
    # Extract initial state from physics row (single object, off=0)
    off = 0
    pos = initial_physics_row[off : off + 3].tolist()
    lin_vel = initial_physics_row[off + 7 : off + 10].tolist()
    mass = float(initial_physics_row[off + 13])
    friction = float(initial_physics_row[off + 14])
    cfg = shape_configs[0]

    # Build spec (same as _build_mjspec but with fixed params from initial_physics_row)
    spec = mujoco.MjSpec()
    spec.option.gravity = [0, 0, -9.81]

    sky_tex = spec.add_texture()
    sky_tex.name = "sky"
    sky_tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky_tex.rgb1 = [0.3, 0.5, 0.8]
    sky_tex.rgb2 = [0.65, 0.8, 1.0]
    sky_tex.width = 512
    sky_tex.height = 512

    floor_tex = spec.add_texture()
    floor_tex.name = "checker"
    floor_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_tex.rgb1 = [0.08, 0.08, 0.08]
    floor_tex.rgb2 = [0.75, 0.70, 0.60]
    floor_tex.width = 512
    floor_tex.height = 512

    floor_mat = spec.add_material()
    floor_mat.name = "floor_mat"
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "checker"
    floor_mat.texrepeat = [5.0, 5.0]
    floor_mat.shininess = 0.0
    floor_mat.specular = 0.0

    g = spec.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [0, 0, 0.01]
    g.material = "floor_mat"

    pil = spec.worldbody.add_geom()
    pil.type = mujoco.mjtGeom.mjGEOM_BOX
    pil.size = [PILLAR_WIDTH / 2, PILLAR_DEPTH / 2, PILLAR_HEIGHT / 2]
    pil.pos = [PILLAR_X, PILLAR_Y_CENTER, PILLAR_Z_CENTER]
    pil.rgba = [pillar_gray, pillar_gray, pillar_gray, 1.0]
    pil.contype = 0
    pil.conaffinity = 0

    lt = spec.worldbody.add_light()
    lt.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    lt.dir = [-d for d in lighting["lightDirection"]]
    lt.diffuse = lighting["lightColor"]
    lt.ambient = [lighting["lightAmbientCoeff"]] * 3
    lt.specular = [0.5, 0.5, 0.5]
    lt.castshadow = True

    jitter = lighting.get("camJitter", [0, 0, 0])
    tj = lighting.get("camTargetJitter", [0, 0, 0])
    eye = [jitter[0], -3 + jitter[1], 2 + jitter[2]]
    target_pt = [tj[0], tj[1], 0.3 + tj[2]]
    cam = spec.worldbody.add_camera()
    cam.name = "scene_cam"
    cam.pos = eye
    cam.quat = _look_at_quat(eye, target_pt)
    cam.fovy = camera_fov if camera_fov is not None else _CFG_CAMERA_FOV

    # Background spheres (visual only) — must match _build_mjspec order
    bg_objects = lighting.get("backgroundObjects", [])
    for i, obj in enumerate(bg_objects):
        if obj.get("radius", 0.0) <= 0.0:
            continue
        mat = spec.add_material()
        mat.name = f"bg_mat_{i}"
        mat.rgba = [*obj["color"], 1.0]
        mat.shininess = obj.get("specular", 0.0)
        bg_geom = spec.worldbody.add_geom()
        bg_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        bg_geom.size = [obj["radius"], 0, 0]
        bg_geom.pos = [obj["x"], obj["y"], obj["z"]]
        bg_geom.material = f"bg_mat_{i}"
        bg_geom.contype = 0
        bg_geom.conaffinity = 0

    # Backdrop wall (visual only) — must match _build_mjspec order
    backdrop_color = lighting.get("backdropColor", [0.2, 0.2, 0.4])
    backdrop_spec = lighting.get("backdropSpecular", [0.0, 0.0, 0.0])
    bd_mat = spec.add_material()
    bd_mat.name = "backdrop_mat"
    bd_mat.rgba = [*backdrop_color, 1.0]
    bd_mat.shininess = float(np.mean(backdrop_spec))
    bd_geom = spec.worldbody.add_geom()
    bd_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    bd_geom.size = [10.0, 0.05, 3.0]
    bd_geom.pos = [0.0, 2.0, 2.0]
    bd_geom.material = "backdrop_mat"
    bd_geom.contype = 0
    bd_geom.conaffinity = 0

    # Object material (handles color + specular)
    specular = cfg.get("specular", 0.4)
    obj_mat = spec.add_material()
    obj_mat.name = "obj_mat"
    obj_mat.rgba = cfg["color"]
    obj_mat.shininess = specular

    body = spec.worldbody.add_body()
    body.name = "object"
    body.pos = pos
    for name, axis in [
        ("obj_sx", [1, 0, 0]),
        ("obj_sy", [0, 1, 0]),
        ("obj_sz", [0, 0, 1]),
    ]:
        jnt = body.add_joint()
        jnt.name = name
        jnt.type = mujoco.mjtJoint.mjJNT_SLIDE
        jnt.axis = axis

    geom = body.add_geom()
    if cfg["shape"] == "sphere":
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        geom.size = [cfg["params"]["radius"], 0, 0]
    elif cfg["shape"] == "cylinder":
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [cfg["params"]["radius"], cfg["params"]["half_length"], 0]
    else:  # box
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = cfg["params"]["half_extents"]
    geom.material = "obj_mat"
    geom.friction = [friction, 0.005, 0.0001]
    geom.mass = mass

    # MuJoCo's default offscreen framebuffer is 640x480, which already covers
    # render sizes <=480 (e.g. the 384/256 animation defaults). Only raise the
    # cap when a larger render is requested.
    if offscreen is not None and offscreen > 480:
        spec.visual.global_.offwidth = offscreen
        spec.visual.global_.offheight = offscreen

    model = spec.compile()
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    qvel_offset = int(
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_sx")]
    )

    data.qvel[qvel_offset : qvel_offset + 3] = lin_vel
    mujoco.mj_forward(model, data)
    return model, data, body_id, qvel_offset


def resimulate_scene(
    shape_configs,
    initial_physics_row,
    *,
    n_timesteps=None,
    return_program_state=False,
    pillar_gray=0.5,
    lighting=None,
    render_size=None,
    use_gui=False,
    physics_client=None,
):
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
        physics_client:      accepted and silently ignored (backward compat).

    Returns:
        If return_program_state=False: RGBA uint8 [IMAGE_SIZE, IMAGE_SIZE, 4]
            of the BEHAVIORAL TARGET frame (rendered at t=n_timesteps).
        If return_program_state=True: float32 [D] program_state vector with
            three brain-input frames concatenated (t=0, t=PP_EARLY_FRAME,
            t=PP_LATE_FRAME).
    """
    if n_timesteps is None:
        n_timesteps = _CFG_N_TIMESTEPS
    if lighting is None:
        lighting = _DEFAULT_LIGHTING

    # Scalars needed for the render loop and physics-label collection.
    mass = float(initial_physics_row[13])
    friction = float(initial_physics_row[14])
    x_accel = float(initial_physics_row[15])

    model, data, body_id, qvel_offset = _build_scene_model(
        shape_configs, initial_physics_row, pillar_gray, lighting
    )

    render_h = render_w = render_size if render_size is not None else IMAGE_SIZE
    renderer = mujoco.Renderer(model, height=render_h, width=render_w)
    try:
        initial_frame = (
            _render_frame(renderer, model, data, lighting)
            if return_program_state
            else None
        )
        early_frame = None
        late_frame = None

        for t in range(n_timesteps):
            data.xfrc_applied[body_id, 0] = x_accel * mass
            mujoco.mj_step(model, data)
            data.xfrc_applied[body_id, 0] = 0.0
            if return_program_state and t + 1 == _CFG_PP_EARLY_FRAME:
                early_frame = _render_frame(renderer, model, data, lighting)
            if return_program_state and t + 1 == _CFG_PP_LATE_FRAME:
                late_frame = _render_frame(renderer, model, data, lighting)

        if return_program_state:
            final_physics = _collect_physics_labels(
                body_id, qvel_offset, mass, friction, x_accel, data
            )
            scene_config_vec = _encode_scene_config(shape_configs)
            lighting_vec = _encode_scene_lighting(pillar_gray, lighting)
            del renderer
            return _build_program_state(
                [initial_frame, early_frame, late_frame],
                final_physics,
                scene_config_vec,
                lighting_vec,
            )
        else:
            rgba_bytes, _, _ = _render_frame(renderer, model, data, lighting)
            del renderer
            return np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(
                render_h, render_w, 4
            )
    except:
        del renderer
        raise


def render_scene_frames(
    shape_configs,
    initial_physics_row,
    *,
    n_timesteps=None,
    pillar_gray=0.5,
    lighting=None,
    render_size=384,
    stride=1,
    camera_fov=None,
):
    """Re-simulate a stored scene and render RGB at every captured timestep.

    Unlike resimulate_scene (which only renders the 3 brain-input frames or the
    single behavioral-target frame), this captures the full motion: t=0 plus
    every `stride`-th physics step, for slide-deck animations. RGB-only — it
    skips the depth/segmentation passes that _render_frame does.

    Args:
        shape_configs:       list of dicts (one per object), as stored in
                             scenes['scene_configs'].
        initial_physics_row: 1-D array length 16*N_OBJECTS, as stored in
                             scenes['initial_physics_labels'] (true initial state).
        render_size:         output frame size in px (square). Values >480 raise
                             MuJoCo's offscreen framebuffer cap automatically.
        stride:              capture every Nth step (1 = all intermediate frames).

    Returns:
        uint8 array [T, render_size, render_size, 3], T = 1 + n_timesteps // stride.
    """
    if n_timesteps is None:
        n_timesteps = _CFG_N_TIMESTEPS
    if lighting is None:
        lighting = _DEFAULT_LIGHTING

    mass = float(initial_physics_row[13])
    x_accel = float(initial_physics_row[15])

    model, data, body_id, qvel_offset = _build_scene_model(
        shape_configs, initial_physics_row, pillar_gray, lighting,
        offscreen=render_size, camera_fov=camera_fov,
    )

    renderer = mujoco.Renderer(model, height=render_size, width=render_size)
    frames = []
    try:
        renderer.update_scene(data, camera="scene_cam")
        frames.append(renderer.render().copy())  # t=0
        for t in range(n_timesteps):
            data.xfrc_applied[body_id, 0] = x_accel * mass
            mujoco.mj_step(model, data)
            data.xfrc_applied[body_id, 0] = 0.0
            if (t + 1) % stride == 0:
                renderer.update_scene(data, camera="scene_cam")
                frames.append(renderer.render().copy())
    finally:
        del renderer

    return np.stack(frames)
