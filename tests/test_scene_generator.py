"""Tests for scene_generator.

Fast tests cover the pure helpers (_encode_scene_config, _encode_scene_lighting,
_build_program_state) without touching PyBullet. Slow tests booting the
PyBullet DIRECT renderer are gated under @pytest.mark.slow and exercise
generate_scenes and resimulate_scene end-to-end at small N.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scene_generator import (
    SCENE_CONFIG_DIM,
    SCENE_LIGHTING_DIM,
    _build_program_state,
    _encode_scene_config,
    _encode_scene_lighting,
)


# --- Constants ----------------------------------------------------------


def test_dim_constants():
    # Documented constants must match what the encoders actually emit.
    assert SCENE_CONFIG_DIM == 9
    assert SCENE_LIGHTING_DIM == 11


# --- _encode_scene_config -----------------------------------------------


def test_encode_scene_config_sphere():
    cfg = {"shape": "sphere", "params": {"radius": 0.3},
           "color": [0.1, 0.2, 0.3, 1.0], "x_accel": 4.0}
    vec = _encode_scene_config([cfg])
    assert vec.dtype == np.float32
    assert vec.shape == (SCENE_CONFIG_DIM,)
    # Layout: [shape_is_box, radius, *half_extents(3), *color(4)]
    # x_accel is no longer in scene_config — it is now in physics_labels.
    assert vec[0] == 0.0
    assert vec[1] == pytest.approx(0.3)
    np.testing.assert_array_equal(vec[2:5], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(vec[5:9], [0.1, 0.2, 0.3, 1.0])


def test_encode_scene_config_box():
    cfg = {"shape": "box", "params": {"half_extents": [0.2, 0.15, 0.3]},
           "color": [0.7, 0.7, 0.0, 1.0], "x_accel": -2.0}
    vec = _encode_scene_config([cfg])
    assert vec.shape == (SCENE_CONFIG_DIM,)
    assert vec[0] == 1.0
    assert vec[1] == 0.0
    np.testing.assert_allclose(vec[2:5], [0.2, 0.15, 0.3])
    np.testing.assert_allclose(vec[5:9], [0.7, 0.7, 0.0, 1.0])


def test_encode_scene_config_two_objects(tiny_shape_configs):
    vec = _encode_scene_config(tiny_shape_configs)
    assert vec.shape == (2 * SCENE_CONFIG_DIM,)
    # Second slice must equal a fresh single-object encoding of the second config.
    second = _encode_scene_config([tiny_shape_configs[1]])
    np.testing.assert_array_equal(vec[SCENE_CONFIG_DIM:], second)


def test_encode_scene_config_ignores_x_accel():
    # x_accel must not affect the encoded vec — it now lives in physics_labels.
    with_accel = {"shape": "sphere", "params": {"radius": 0.2},
                  "color": [0.0, 0.0, 0.0, 1.0], "x_accel": 7.5}
    without = {"shape": "sphere", "params": {"radius": 0.2},
               "color": [0.0, 0.0, 0.0, 1.0]}
    np.testing.assert_array_equal(
        _encode_scene_config([with_accel]),
        _encode_scene_config([without]),
    )


# --- _encode_scene_lighting ---------------------------------------------


def test_encode_scene_lighting(tiny_lighting):
    vec = _encode_scene_lighting(0.55, tiny_lighting)
    assert vec.dtype == np.float32
    assert vec.shape == (SCENE_LIGHTING_DIM,)
    # Layout: [pillar_gray, *lightDirection(3), *lightColor(3), lightDistance, *camJitter(3)]
    assert vec[0] == pytest.approx(0.55)
    np.testing.assert_allclose(vec[1:4], tiny_lighting["lightDirection"])
    np.testing.assert_allclose(vec[4:7], tiny_lighting["lightColor"])
    assert vec[7] == pytest.approx(tiny_lighting["lightDistance"])
    np.testing.assert_allclose(vec[8:11], tiny_lighting["camJitter"])


# --- _build_program_state ------------------------------------------------


def test_build_program_state_length_and_dtype():
    rgba_bytes = bytes(range(256)) * 4    # 1024 uint8 bytes per frame
    depth_bytes = (np.arange(16, dtype=np.float32)).tobytes()  # 64 bytes
    seg_bytes = (np.arange(8, dtype=np.int32)).tobytes()       # 32 bytes
    physics = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    config = np.array([0.5, 0.6], dtype=np.float32)
    lighting = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Three brain-input frames concatenated into the render block.
    frames = [(rgba_bytes, depth_bytes, seg_bytes)] * 3
    ps = _build_program_state(frames, physics, config, lighting)

    per_frame_render = len(rgba_bytes) + len(depth_bytes) + len(seg_bytes)
    expected_len = (
        3 * per_frame_render + len(physics) + len(config) + len(lighting)
    )
    assert ps.dtype == np.float32
    assert ps.shape == (expected_len,)
    # Each frame's leading RGBA bytes appear at the start of its slice,
    # cast from uint8 to float32.
    for i in range(3):
        offset = i * per_frame_render
        np.testing.assert_array_equal(
            ps[offset:offset + 256],
            np.arange(256, dtype=np.float32),
        )
    # Physics labels appear right after the 3-frame render block.
    render_total = 3 * per_frame_render
    np.testing.assert_array_equal(ps[render_total:render_total + 3], physics)


# --- Property-based -----------------------------------------------------


# Values are cast to float32 in the encoders, so generate float32-representable
# numbers to keep round-trip comparisons exact (no subnormal-underflow flake).
unit_floats = st.floats(min_value=0.0, max_value=1.0, allow_nan=False,
                        allow_infinity=False, width=32)
finite_floats = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False,
                          allow_infinity=False, width=32)
color_strategy = st.lists(unit_floats, min_size=4, max_size=4)


# 1/16 and 1/2 are both exactly representable in float32 — keeps hypothesis happy.
small_dim = st.floats(min_value=0.0625, max_value=0.5, allow_nan=False,
                      allow_infinity=False, width=32)


@settings(max_examples=30, deadline=None)
@given(
    is_box=st.booleans(),
    radius=small_dim,
    half_extents=st.lists(small_dim, min_size=3, max_size=3),
    color=color_strategy,
)
def test_property_encode_scene_config(is_box, radius, half_extents, color):
    if is_box:
        cfg = {"shape": "box", "params": {"half_extents": half_extents},
               "color": color}
    else:
        cfg = {"shape": "sphere", "params": {"radius": radius},
               "color": color}
    vec = _encode_scene_config([cfg])
    assert vec.dtype == np.float32
    assert vec.shape == (SCENE_CONFIG_DIM,)
    assert vec[0] == (1.0 if is_box else 0.0)


@settings(max_examples=30, deadline=None)
@given(
    pillar_gray=unit_floats,
    direction=st.lists(finite_floats, min_size=3, max_size=3),
    color=st.lists(unit_floats, min_size=3, max_size=3),
    distance=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, width=32),
    cam_jitter=st.lists(finite_floats, min_size=3, max_size=3),
)
def test_property_encode_scene_lighting(pillar_gray, direction, color, distance,
                                        cam_jitter):
    lighting = {
        "lightDirection": direction,
        "lightColor": color,
        "lightDistance": distance,
        "camJitter": cam_jitter,
    }
    vec = _encode_scene_lighting(pillar_gray, lighting)
    assert vec.shape == (SCENE_LIGHTING_DIM,)
    assert vec[0] == pytest.approx(pillar_gray)
    np.testing.assert_allclose(vec[1:4], direction)
    np.testing.assert_allclose(vec[4:7], color)
    assert vec[7] == pytest.approx(distance)
    np.testing.assert_allclose(vec[8:11], cam_jitter)


# --- Slow integration tests --------------------------------------------


@pytest.mark.slow
def test_generate_scenes_smoke():
    """3-scene smoke test: returned dict is well-formed with expected shapes."""
    from scene_generator import generate_scenes

    out = generate_scenes(n_scenes=3, seed=42)

    expected_keys = {
        "program_states", "physics_labels", "initial_physics_labels",
        "initial_renders", "early_renders", "late_renders", "target_renders",
        "behavior_labels", "kinetic_energies", "scene_configs",
        "pillar_grays", "lightings", "metadata",
    }
    assert expected_keys.issubset(out.keys())

    meta = out["metadata"]
    assert out["program_states"].shape == (3, meta["D_total"])
    assert out["program_states"].dtype == np.float32
    assert out["physics_labels"].shape == (3, meta["D_physics_labels"])
    assert out["initial_physics_labels"].shape == (3, meta["D_physics_labels"])
    assert out["kinetic_energies"].shape == (3,)
    assert np.all(out["kinetic_energies"] >= 0.0)

    assert out["behavior_labels"].dtype == np.int32
    assert set(np.unique(out["behavior_labels"]).tolist()).issubset({0, 1})

    assert len(out["scene_configs"]) == 3
    assert len(out["pillar_grays"]) == 3
    assert len(out["lightings"]) == 3

    # Render layout: three brain-input frames + a held-out behavioral target.
    per_frame = meta["D_render_per_frame"]
    assert meta["D_render_bytes"] == 3 * per_frame
    for key in ("initial_renders", "early_renders", "late_renders",
                "target_renders"):
        assert out[key].shape == (3, per_frame)

    # frame_render_indices must partition render_indices exactly.
    fri = meta["frame_render_indices"]
    assert fri["initial"] == slice(0, per_frame)
    assert fri["early"] == slice(per_frame, 2 * per_frame)
    assert fri["late"] == slice(2 * per_frame, 3 * per_frame)
    assert meta["render_indices"] == slice(0, 3 * per_frame)

    # pixel_indices points at the LATE frame's RGBA inside program_state.
    rgba_bytes = (meta["target_pixel_indices"].stop
                  - meta["target_pixel_indices"].start)
    assert meta["pixel_indices"].start == fri["late"].start
    assert meta["pixel_indices"].stop == fri["late"].start + rgba_bytes
    # target_pixel_indices must lie within the target_renders width.
    assert meta["target_pixel_indices"].stop <= per_frame


@pytest.mark.slow
def test_generate_scenes_determinism():
    """Two calls with the same seed must produce identical numerical output."""
    from scene_generator import generate_scenes

    a = generate_scenes(n_scenes=3, seed=99)
    b = generate_scenes(n_scenes=3, seed=99)

    np.testing.assert_array_equal(a["program_states"], b["program_states"])
    np.testing.assert_array_equal(a["kinetic_energies"], b["kinetic_energies"])
    np.testing.assert_array_equal(a["physics_labels"], b["physics_labels"])
    np.testing.assert_array_equal(a["behavior_labels"], b["behavior_labels"])


@pytest.mark.slow
def test_physics_labels_include_x_accel():
    """x_accel from the per-scene config must appear at offset 15 of each
    object's stride-16 physics_labels slot, in both initial and final labels."""
    from scene_generator import generate_scenes
    from config import N_OBJECTS

    out = generate_scenes(n_scenes=4, seed=123)
    expected = np.array(
        [[cfg.get('x_accel', 0.0) for cfg in scene_cfgs]
         for scene_cfgs in out['scene_configs']],
        dtype=np.float32,
    )  # (n_scenes, N_OBJECTS)

    for i in range(N_OBJECTS):
        col = i * 16 + 15
        np.testing.assert_allclose(out['physics_labels'][:, col], expected[:, i])
        np.testing.assert_allclose(out['initial_physics_labels'][:, col], expected[:, i])

    # And the layout: stride 16 per object.
    assert out['physics_labels'].shape[1] == 16 * N_OBJECTS


@pytest.mark.slow
def test_resimulate_scene_determinism():
    """resimulate_scene given identical inputs must produce byte-identical renders."""
    from scene_generator import generate_scenes, resimulate_scene

    out = generate_scenes(n_scenes=1, seed=7)
    shape_configs = out["scene_configs"][0]
    initial_physics = out["initial_physics_labels"][0]
    pillar_gray = out["pillar_grays"][0]
    lighting = out["lightings"][0]

    img_a = resimulate_scene(
        shape_configs, initial_physics,
        pillar_gray=pillar_gray, lighting=lighting,
    )
    img_b = resimulate_scene(
        shape_configs, initial_physics,
        pillar_gray=pillar_gray, lighting=lighting,
    )
    np.testing.assert_array_equal(img_a, img_b)
