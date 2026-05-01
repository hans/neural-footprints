"""Shared fixtures for the test suite.

Tests build their own small fixtures and never read from the project's
data/ or outputs/ directories — they must be runnable on a fresh checkout.
"""

import numpy as np
import pytest


SEED = 12345


@pytest.fixture
def rng_seed() -> int:
    return SEED


@pytest.fixture
def tiny_program_states() -> np.ndarray:
    """Synthetic program-state matrix with mixed structure.

    20 scenes, 64 dims. Includes a constant column (index 5) and a
    near-constant column to exercise the std==0 branch in
    generate_neural_activity.
    """
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.0, 1.0, size=(20, 64)).astype(np.float32)
    x[:, 5] = 7.0  # constant column
    return x


@pytest.fixture
def tiny_shape_configs() -> list[dict]:
    """Two-object shape config in the exact format scene_generator produces."""
    return [
        {
            "shape": "sphere",
            "params": {"radius": 0.25},
            "color": [0.5, 0.2, 0.8, 1.0],
            "x_accel": 1.5,
        },
        {
            "shape": "box",
            "params": {"half_extents": [0.2, 0.15, 0.3]},
            "color": [0.1, 0.7, 0.4, 1.0],
            "x_accel": -2.0,
        },
    ]


@pytest.fixture
def tiny_lighting() -> dict:
    return {
        "lightDirection": [1.0, -1.5, 2.0],
        "lightColor": [0.9, 0.85, 0.95],
        "lightDistance": 4.5,
    }
