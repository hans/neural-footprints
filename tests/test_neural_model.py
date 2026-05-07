"""Tests for neural_model.generate_neural_activity.

The function is a deterministic random linear projection with per-block
operator-norm normalization and additive Gaussian noise. These tests pin down
its invariants so future refactors cannot silently change the data path that
every downstream analysis depends on.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.sparse.linalg import svds

from neural_model import generate_neural_activity


# --- Unit tests ---------------------------------------------------------


def test_output_shape_and_metadata_keys(tiny_program_states, rng_seed):
    activity, meta = generate_neural_activity(
        tiny_program_states, rng_seed, n_neurons=8, noise_level=0.1
    )
    n_scenes, D = tiny_program_states.shape
    assert activity.shape == (n_scenes, 8)
    assert np.all(np.isfinite(activity))

    for key in ("W", "means", "block_norms", "signal_std", "var_per_dim", "total_var"):
        assert key in meta, f"missing metadata key: {key}"
    assert meta["W"].shape == (8, D)
    assert meta["means"].shape == (D,)
    assert meta["var_per_dim"].shape == (D,)
    assert meta["block_norms"].shape == (1,)  # default: one block


def test_determinism_same_seed(tiny_program_states, rng_seed):
    a1, m1 = generate_neural_activity(
        tiny_program_states, rng_seed, n_neurons=8, noise_level=0.5
    )
    a2, m2 = generate_neural_activity(
        tiny_program_states, rng_seed, n_neurons=8, noise_level=0.5
    )
    assert np.array_equal(a1, a2)
    assert np.array_equal(m1["W"], m2["W"])


def test_different_seeds_differ(tiny_program_states):
    a1, _ = generate_neural_activity(tiny_program_states, 1, n_neurons=8, noise_level=0.1)
    a2, _ = generate_neural_activity(tiny_program_states, 2, n_neurons=8, noise_level=0.1)
    assert not np.allclose(a1, a2)


def test_means_match_input(tiny_program_states, rng_seed):
    _, meta = generate_neural_activity(
        tiny_program_states, rng_seed, n_neurons=4, noise_level=0.0
    )
    np.testing.assert_array_equal(meta["means"], tiny_program_states.mean(axis=0))


def test_constant_column_handled(tiny_program_states, rng_seed):
    # Column 5 is constant in the fixture. After centering it becomes zero,
    # contributing zero variance. The op norm should still be finite (driven by
    # the non-constant columns) and the output should be finite throughout.
    _, meta = generate_neural_activity(
        tiny_program_states, rng_seed, n_neurons=4, noise_level=0.0
    )
    assert meta["var_per_dim"][5] == 0.0
    assert np.all(np.isfinite(meta["block_norms"]))
    assert meta["block_norms"][0] > 0.0


def test_noise_level_zero_matches_signal(tiny_program_states, rng_seed):
    activity, meta = generate_neural_activity(
        tiny_program_states, rng_seed, n_neurons=6, noise_level=0.0
    )

    # Reproduce the noise-free signal manually using op-norm normalization.
    means = tiny_program_states.mean(axis=0)
    centered = tiny_program_states - means
    D = centered.shape[1]
    sigma = float(svds(centered.astype(np.float64), k=1,
                       return_singular_vectors=False, solver='arpack')[0])
    if sigma == 0.0:
        sigma = 1.0
    normalized = (centered / sigma).astype(tiny_program_states.dtype)
    expected = normalized @ meta["W"].T

    np.testing.assert_array_equal(activity, expected)


def test_translation_invariance_noise_zero(tiny_program_states, rng_seed):
    a1, _ = generate_neural_activity(
        tiny_program_states, rng_seed, n_neurons=6, noise_level=0.0
    )
    shifted = tiny_program_states + np.array([3.0] * tiny_program_states.shape[1],
                                              dtype=tiny_program_states.dtype)
    a2, _ = generate_neural_activity(shifted, rng_seed, n_neurons=6, noise_level=0.0)
    # Centering removes the constant offset; outputs should match within float
    # tolerance (subtraction of a large constant introduces small rounding error).
    np.testing.assert_allclose(a1, a2, rtol=1e-4, atol=1e-4)


def test_scale_invariance_noise_zero(tiny_program_states, rng_seed):
    a1, _ = generate_neural_activity(
        tiny_program_states, rng_seed, n_neurons=6, noise_level=0.0
    )
    scaled = (tiny_program_states * 5.0).astype(tiny_program_states.dtype)
    a2, _ = generate_neural_activity(scaled, rng_seed, n_neurons=6, noise_level=0.0)
    # Uniform scaling scales the operator norm by the same factor, which cancels.
    np.testing.assert_allclose(a1, a2, rtol=1e-4, atol=1e-4)


def test_multi_block_normalization(tiny_program_states, rng_seed):
    # Two-block split: first 32 dims and last 32 dims normalized independently.
    block_sizes = [32, 32]
    activity, meta = generate_neural_activity(
        tiny_program_states, rng_seed, n_neurons=6, noise_level=0.0,
        block_sizes=block_sizes,
    )
    assert meta["block_norms"].shape == (2,)
    assert np.all(meta["block_norms"] > 0)
    assert activity.shape == (tiny_program_states.shape[0], 6)
    assert np.all(np.isfinite(activity))

    # Each block should have been normalized to op norm ≈ 1 (i.e., its top
    # singular value equals its block_norm, not 1.0, but after dividing the
    # reconstructed block's op norm equals 1.0).
    means = tiny_program_states.mean(axis=0)
    centered = tiny_program_states - means
    for i, size in enumerate(block_sizes):
        start = sum(block_sizes[:i])
        block = centered[:, start:start + size].astype(np.float64)
        sigma_reconstructed = float(svds(block / meta["block_norms"][i], k=1,
                                         return_singular_vectors=False,
                                         solver='arpack')[0])
        np.testing.assert_allclose(sigma_reconstructed, 1.0, rtol=1e-5)


def test_single_scene_yields_zero_activity(rng_seed):
    # n_scenes=1: every column has std=0, centered block is all zeros,
    # op norm is 0 → substituted with 1.0 → normalized is all zeros → output zero.
    x = np.array([[1.0, 2.0, -3.0, 0.5]], dtype=np.float32)
    activity, _ = generate_neural_activity(x, rng_seed, n_neurons=4, noise_level=1.0)
    assert activity.shape == (1, 4)
    np.testing.assert_array_equal(activity, np.zeros_like(activity))


# --- Property-based tests -----------------------------------------------


@settings(max_examples=25, deadline=None)
@given(
    n_scenes=st.integers(min_value=3, max_value=30),
    D=st.integers(min_value=4, max_value=30),
    n_neurons=st.integers(min_value=2, max_value=20),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_shape_and_finite(n_scenes, D, n_neurons, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_scenes, D)).astype(np.float32)
    activity, meta = generate_neural_activity(
        x, seed, n_neurons=n_neurons, noise_level=0.1
    )
    assert activity.shape == (n_scenes, n_neurons)
    assert np.all(np.isfinite(activity))
    assert meta["W"].shape == (n_neurons, D)
    assert "block_norms" in meta


@settings(max_examples=25, deadline=None)
@given(
    n_scenes=st.integers(min_value=3, max_value=30),
    D=st.integers(min_value=4, max_value=30),
    n_neurons=st.integers(min_value=2, max_value=20),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_determinism(n_scenes, D, n_neurons, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_scenes, D)).astype(np.float32)
    a1, _ = generate_neural_activity(x, seed, n_neurons=n_neurons, noise_level=0.3)
    a2, _ = generate_neural_activity(x, seed, n_neurons=n_neurons, noise_level=0.3)
    assert np.array_equal(a1, a2)


@settings(max_examples=25, deadline=None)
@given(
    n_scenes=st.integers(min_value=3, max_value=30),
    D=st.integers(min_value=4, max_value=30),
    n_neurons=st.integers(min_value=2, max_value=20),
    seed=st.integers(min_value=0, max_value=10_000),
    scale=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_property_positive_scale_invariance_noise_zero(n_scenes, D, n_neurons, seed, scale):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_scenes, D)).astype(np.float32)
    a1, _ = generate_neural_activity(x, seed, n_neurons=n_neurons, noise_level=0.0)
    a2, _ = generate_neural_activity(
        (x * scale).astype(np.float32), seed, n_neurons=n_neurons, noise_level=0.0
    )
    np.testing.assert_allclose(a1, a2, rtol=1e-3, atol=1e-3)
