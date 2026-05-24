# MuJoCo Migration Spec

Replace PyBullet with MuJoCo in `scene_generator.py`. This is a **plumbing-only swap** — no change to scene style, visual quality, or output data formats. The goal is faster physics simulation. Richer visuals (nvdiffrast, PBR materials) are deferred to a later phase.

## Scope

**Only `scene_generator.py` changes.** Everything else (pipeline scripts, `neural_model.py`, `evaluation.py`, tests, `config.yaml`) is untouched.

**Add `mujoco` to `pyproject.toml` dependencies. Remove `pybullet`.**

## Public API — must be preserved exactly

```python
generate_scenes(n_scenes, seed, *, n_timesteps=None, use_gui=False) -> dict
resimulate_scene(shape_configs, initial_physics_row, *, n_timesteps=None,
                 return_program_state=False, pillar_gray=0.5, lighting=None,
                 render_size=None, use_gui=False, physics_client=None) -> ndarray
extract_brain_pixels(states, metadata) -> ndarray
extract_frame_pixels(frame_data, metadata) -> ndarray
open_render_client(use_gui=False)  # see note below
```

`extract_brain_pixels` and `extract_frame_pixels` are pure array slicers — do not touch them.

`open_render_client` is called externally (e.g. in scripts). Stub it to return `None`; callers that pass `physics_client=None` to `resimulate_scene` will get a fresh internal model instead (no change in behavior). Do not break callers that pass the returned value to `resimulate_scene`'s `physics_client` parameter — `resimulate_scene` should silently ignore a non-None but semantically-empty value (or just accept and ignore it).

## Output format — must be preserved exactly

`generate_scenes` returns the same dict keys and array shapes as before:

```
program_states          [n_scenes, D]        float32
physics_labels          [n_scenes, 16]       float32   (16 = per-object stride)
initial_physics_labels  [n_scenes, 16]       float32
initial_renders         [n_scenes, F]        float32   (F = render_bytes_per_frame)
early_renders           [n_scenes, F]        float32
late_renders            [n_scenes, F]        float32
target_renders          [n_scenes, F]        float32
behavior_labels         [n_scenes]           int32
kinetic_energies        [n_scenes]           float32
scene_configs           list[list[dict]]
pillar_grays            list[float]
lightings               list[dict]
metadata                dict                 (same keys and slice objects)
```

`render_bytes_per_frame = IMAGE_SIZE * IMAGE_SIZE * (4 + 4 + 4)` bytes  
(RGBA uint8 → 4 bytes/px, depth float32 → 4 bytes/px, seg int32 → 4 bytes/px).  
This is identical to the PyBullet layout.

`physics_labels` per-object stride is 16 floats:  
`pos(3), orn(4), lin_vel(3), ang_vel(3), mass(1), friction(1), x_accel(1)`.  
With slide joints, `orn` is always identity `[0,0,0,1]` (PyBullet xyzw) and `ang_vel` is always `[0,0,0]` — same as PyBullet after rotation-locking.

`metadata` slice objects (`pixel_indices`, `render_indices`, `frame_render_indices`, `target_pixel_indices`) must be identical to what PyBullet produces for the same `IMAGE_SIZE`.

## Implementation

### Dependencies / imports

```python
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation  # for camera quat computation
```

Remove `import pybullet as p`.

### Scene construction — `_build_mjspec(rng)`

Returns `(model, data, body_id, mass, friction, shape_config, pillar_gray, lighting)`.

Use `mujoco.MjSpec()` (MuJoCo 3.x programmatic API):

```python
spec = mujoco.MjSpec()
spec.option.gravity = [0, 0, -9.81]

# Ground plane (infinite, visual + collision)
g = spec.worldbody.add_geom()
g.type = mujoco.mjtGeom.mjGEOM_PLANE
g.size = [0, 0, 0.01]
g.rgba = [ground_r, ground_r, ground_r, 1.0]  # use 0.6

# Pillar (visual only — contype=0, conaffinity=0)
p = spec.worldbody.add_geom()
p.type = mujoco.mjtGeom.mjGEOM_BOX
p.size = [PILLAR_WIDTH/2, PILLAR_DEPTH/2, PILLAR_HEIGHT/2]
p.pos = [PILLAR_X, PILLAR_Y_CENTER, PILLAR_Z_CENTER]
p.rgba = [pillar_gray, pillar_gray, pillar_gray, 1.0]
p.contype = 0
p.conaffinity = 0

# Light (directional, per-scene random)
lt = spec.worldbody.add_light()
lt.directional = True
lt.dir = [-d for d in lighting['lightDirection']]  # rays toward scene
lt.diffuse = lighting['lightColor']
lt.ambient = [lighting['lightAmbientCoeff']] * 3
lt.specular = [0.3, 0.3, 0.3]
lt.castshadow = False  # keep fast; no shadows in DIRECT mode anyway

# Camera (per-scene, with jitter)
jitter = lighting.get('camJitter', [0, 0, 0])
tj = lighting.get('camTargetJitter', [0, 0, 0])
eye = [jitter[0], -3 + jitter[1], 2 + jitter[2]]
target = [tj[0], tj[1], 0.3 + tj[2]]
cam = spec.worldbody.add_camera()
cam.name = "scene_cam"
cam.pos = eye
cam.quat = _look_at_quat(eye, target)  # see helper below
cam.fovy = _CFG_CAMERA_FOV  # degrees; matches PyBullet's vertical FOV

# Object body — 3 slide joints (x/y/z) so rotation is mechanically impossible
body = spec.worldbody.add_body()
body.name = "object"
body.pos = [x, y, z]   # randomised starting position
for axis in ([1,0,0], [0,1,0], [0,0,1]):
    jnt = body.add_joint()
    jnt.type = mujoco.mjtJoint.mjJNT_SLIDE
    jnt.axis = axis
    jnt.damping = 0.0

geom = body.add_geom()
if sphere:
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = [radius, 0, 0]
else:
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = half_extents          # half-extents, same as PyBullet
geom.rgba = color                     # [r, g, b, 1.0]
geom.friction = [friction, 0.005, 0.0001]   # sliding, torsional, rolling
geom.mass = mass                      # sets inertia from mass + geom geometry

model = spec.compile()
data = mujoco.MjData(model)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
```

**`_look_at_quat(eye, target, up=[0,0,1])`** — helper, no external calls needed:

```python
def _look_at_quat(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.array(v, float) for v in (eye, target, up))
    fwd = target - eye;  fwd /= np.linalg.norm(fwd)
    z_ax = -fwd                        # MuJoCo camera looks along -Z
    x_ax = np.cross(fwd, up);  x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    R = np.column_stack([x_ax, y_ax, z_ax])
    q_xyzw = Rotation.from_matrix(R).as_quat()   # scipy → [x,y,z,w]
    return [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # MuJoCo → [w,x,y,z]
```

### Rotation locking

**Eliminated.** With 3 slide joints the object has no rotational DOF. `_lock_rotation` is deleted. The simulation loop has no per-step orientation/angvel reset.

### External force (x_accel)

`data.xfrc_applied` persists across steps. Set it **before** each `mj_step`, then clear it.

```python
for t in range(n_timesteps):
    data.xfrc_applied[body_id, 0] = x_accel * mass   # force in world-X
    mujoco.mj_step(model, data)
    data.xfrc_applied[body_id, 0] = 0.0
```

### Reading physics state

```python
pos = data.xpos[body_id].copy()          # world position [x,y,z]
orn = [0.0, 0.0, 0.0, 1.0]              # always identity (no rotation DOF)
lin_vel = data.qvel[qvel_offset:qvel_offset+3].copy()  # slide joint velocities
ang_vel = [0.0, 0.0, 0.0]              # always zero
```

`qvel_offset` for the object body: since there are 3 slide joints and no other DOFs preceding them in the model, `qvel_offset = 0`.  
Verify with `mujoco.mj_name2id` + `model.jnt_qposadr` / `model.jnt_dofadr`.

For `_compute_total_kinetic_energy`: `KE = 0.5 * mass * norm(lin_vel)^2`.

### Rendering — `_render_frame(renderer, model, data, lighting=None)`

Returns `(rgba_bytes, depth_bytes, seg_bytes)` — same as old `_render_scene`.

```python
renderer.update_scene(data, camera="scene_cam")

# Pass 1: RGB → RGBA
rgb = renderer.render()                      # (H, W, 3) uint8
alpha = np.full((H, W, 1), 255, dtype=np.uint8)
rgba = np.concatenate([rgb, alpha], axis=2)  # (H, W, 4)
rgba_bytes = rgba.tobytes()

# Pass 2: depth
renderer.enable_depth_rendering()
depth = renderer.render()                    # (H, W) float32, values in [0,1]
renderer.disable_depth_rendering()
depth_bytes = depth.tobytes()

# Pass 3: segmentation
renderer.enable_segmentation_rendering()
seg_raw = renderer.render()                  # (H, W, 2) int32
renderer.disable_segmentation_rendering()
seg = seg_raw[:, :, 0].astype(np.int32)      # geom ID channel
seg_bytes = seg.tobytes()
```

`Renderer` is created once per `generate_scenes` call (not per frame) and reused:

```python
renderer = mujoco.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE)
```

Note: `mujoco.Renderer` is tied to a specific `MjModel`. Since each scene has a fresh `model`, a fresh `Renderer` must be created per scene. This is the main performance cost vs. PyBullet's `resetSimulation`; benchmark to confirm speed is still a win.

If per-scene renderer creation is a bottleneck, consider a fixed-geometry approach (one model template with adjustable fields) as a follow-on optimisation, but do not do it speculatively.

### `generate_scenes` structure

```python
for i in range(n_scenes):
    model, data, body_id, mass, friction, shape_config, pillar_gray, lighting = \
        _build_mjspec(rng)
    renderer = mujoco.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE)
    try:
        qvel_offset = int(model.jnt_dofadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_slide_x")
        ])  # or compute from model; see note

        # Set initial velocity
        data.qvel[qvel_offset:qvel_offset+3] = [x_vel, 0.0, 0.0]
        mujoco.mj_forward(model, data)    # propagate to xpos/xquat etc.

        # Capture t=0
        init_frame = _render_frame(renderer, model, data, lighting)
        ...

        for t in range(n_timesteps):
            data.xfrc_applied[body_id, 0] = x_accel * mass
            mujoco.mj_step(model, data)
            data.xfrc_applied[body_id, 0] = 0.0
            ...
    finally:
        del renderer   # or renderer.close() if available
```

**Joint naming**: name the joints in the spec so they can be looked up reliably:
```python
for name, axis in [("obj_sx", [1,0,0]), ("obj_sy", [0,1,0]), ("obj_sz", [0,0,1])]:
    jnt = body.add_joint(); jnt.name = name; jnt.axis = axis; ...
qvel_offset = int(model.jnt_dofadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_sx")])
```

### `resimulate_scene`

Same logic as `generate_scenes` but driven by `shape_configs` + `initial_physics_row`. The `physics_client` parameter is accepted and silently ignored (backward compat with callers that pass `None` or an old PyBullet ID).

Extract initial state:
```python
off = 0  # single object
pos      = initial_physics_row[off:off+3]
# orn[3:7] ignored — slide joints have no orientation DOF
lin_vel  = initial_physics_row[off+7:off+10]
# ang_vel[10:13] ignored
mass     = float(initial_physics_row[off+13])
friction = float(initial_physics_row[off+14])
x_accel  = float(initial_physics_row[off+15])
```

Set initial position via spec: `body.pos = pos`. After compile, `data.qpos[0:3]` starts at `[0,0,0]` (body is already at `pos`), and `data.qvel[qvel_offset:qvel_offset+3] = lin_vel`.

### `open_render_client`

```python
def open_render_client(use_gui=False):
    """Stub — MuJoCo does not use persistent render clients."""
    return None
```

### Cache invalidation

After merging, cached `data/*.npz` files were generated by PyBullet and are stale. Document in the commit message (or a comment in `scene_generator.py`) that running `rm data/*.npz` before the pipeline is required.

Optionally bump a config key (e.g. `scene_engine: mujoco`) that doesn't affect computation but can serve as a changelog marker.

## Tests

The fast unit tests (`test_dim_constants`, `_encode_*`, `_build_program_state`) don't touch the physics engine and require no changes.

The slow integration tests must still pass:

- `test_generate_scenes_shape` — output dict shapes unchanged ✓
- `test_generate_scenes_determinism` — MuJoCo is deterministic for same seed ✓
- `test_physics_labels_include_x_accel` — `x_accel` at offset 15 in stride-16 labels ✓
- `test_resimulate_scene_determinism` — same model + same state → byte-identical renders ✓

Update the `pytest.ini_options` marker description from "PyBullet" to "physics engine":
```
"slow: integration tests that boot the physics engine (deselect with -m 'not slow')"
```

## What NOT to do

- Do not change `IMAGE_SIZE`, `N_TIMESTEPS`, or any config values.
- Do not add textures, materials, HDR lighting, or mesh geometry — those belong to the nvdiffrast phase.
- Do not parallelize scene generation with MJX/JAX — that is a separate optimisation.
- Do not change the `program_state` layout or `metadata` slice structure.
- Do not add MuJoCo-specific fields to the returned dict.
