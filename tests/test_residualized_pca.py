"""Tests for analyses.residualized_pca.

The analysis must show that motion decodability from neural PCs is a *render
confound*: when motion is injected into neural activity only through the pixel
features, residualizing those pixels out should collapse motion decoding toward
chance, while decoding motion directly from the pixel PCs (positive control)
stays high. These tests build a synthetic dataset with exactly that structure
and pin the expected ordering of results. No physics engine is booted.
"""

import numpy as np
import pytest

from analyses.residualized_pca import run_residualized_pca_analysis

SEED = 7


def _make_dataset(n=160, d_pix=8, n_neurons=30, noise=0.1):
    """Synthetic data where motion (vx) reaches neural ONLY via the pixels."""
    rng = np.random.default_rng(SEED)

    vx = rng.normal(size=n)

    # Pixel PCA features: first column carries vx (forward render of motion),
    # the rest are unrelated render structure.
    raw_pixel_pca = rng.normal(size=(n, d_pix))
    raw_pixel_pca[:, 0] = vx + 0.25 * rng.normal(size=n)
    # Forward-model predicted frames carry vx too (also a forward render).
    predicted_pixel_pca = rng.normal(size=(n, d_pix))
    predicted_pixel_pca[:, 0] = vx + 0.25 * rng.normal(size=n)

    # Neural activity is a random linear map of the pixels plus noise — so the
    # ONLY route from motion to neural is through the (residualizable) pixels.
    W = rng.normal(size=(d_pix, n_neurons))
    neural_activity = raw_pixel_pca @ W + noise * rng.normal(size=(n, n_neurons))

    physics_labels = np.zeros((n, 16))
    physics_labels[:, 7] = vx  # motion_dir = (vx > 0)
    scenes = {
        "initial_physics_labels": physics_labels,
        "pillar_grays": rng.normal(size=n),
        "lightings": [{"camJitter": [0.0, 0.0, float(z)]} for z in rng.normal(size=n)],
    }
    return neural_activity, scenes, raw_pixel_pca, predicted_pixel_pca


def _run(**kwargs):
    neural, scenes, X, S = _make_dataset(**kwargs)
    return run_residualized_pca_analysis(
        neural,
        scenes,
        {},
        raw_pixel_pca=X,
        predicted_pixel_pca=S,
        compute_null=False,
    )


def _all_pc_motion(result, cond):
    return result[cond]["decode_accs_per_target"]["motion_dir"][-1]


def test_result_structure():
    result = _run()
    for cond in ("raw", "resid_X", "resid_XS", "pixel"):
        assert cond in result, f"missing condition {cond}"
        block = result[cond]
        for target in ("cam_height", "pillar_gray", "motion_dir"):
            accs = block["decode_accs_per_target"][target]
            assert len(accs) == len(block["pc_counts"])
            assert len(block["chance_per_target"][target]["hi"]) == len(accs)
    assert 0.0 < result["residual_variance_fraction_X"] < 1.0
    assert 0.0 < result["residual_variance_fraction_XS"] < 1.0


def test_motion_is_decodable_from_raw_neural():
    # The phenomenon must exist before residualization can remove it.
    result = _run()
    assert _all_pc_motion(result, "raw") > 0.65


def test_residualization_collapses_motion_decoding():
    result = _run()
    raw = _all_pc_motion(result, "raw")
    resid_x = _all_pc_motion(result, "resid_X")
    resid_xs = _all_pc_motion(result, "resid_XS")
    # Removing the pixel-explainable component drops motion decodability...
    assert resid_x < raw
    assert resid_xs < raw
    # ...down toward chance (no non-pixel route to motion in this dataset).
    assert resid_xs < 0.62


def test_pixel_positive_control_decodes_motion():
    # The renders genuinely carry motion — the confound is real.
    result = _run()
    assert _all_pc_motion(result, "pixel") > 0.65


def test_resid_xs_optional_when_no_predicted_frames():
    neural, scenes, X, _ = _make_dataset()
    result = run_residualized_pca_analysis(
        neural, scenes, {}, raw_pixel_pca=X, compute_null=False
    )
    assert "resid_XS" not in result
    assert "residual_variance_fraction_XS" not in result
    assert "resid_X" in result and "pixel" in result
