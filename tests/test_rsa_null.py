"""Tests for RSA Mantel permutation null helpers."""
import numpy as np
import pytest
from analyses.rsa import _rsa_null_summary


def test_rsa_null_summary_one_sided_pvalue():
    # All null values below observed → p = 0.0
    perm_values = np.array([0.01, 0.02, 0.03])
    result = _rsa_null_summary("corr_neural_P", perm_values, observed=0.10, two_sided=False)
    assert result["null_corr_neural_P_pvalue"] == pytest.approx(0.0)
    assert result["null_corr_neural_P_observed"] == pytest.approx(0.10)
    assert "null_corr_neural_P_ci_lo" in result
    assert "null_corr_neural_P_ci_hi" in result
    assert "null_corr_neural_P_mean" in result
    assert "null_corr_neural_P_perm_values" in result


def test_rsa_null_summary_one_sided_all_above():
    # All null values at or above observed → p = 1.0
    perm_values = np.array([0.10, 0.20, 0.30])
    result = _rsa_null_summary("corr_neural_P", perm_values, observed=0.05, two_sided=False)
    assert result["null_corr_neural_P_pvalue"] == pytest.approx(1.0)


def test_rsa_null_summary_two_sided_pvalue():
    # Two-sided: |perm| >= |observed|. observed=0.0, all perm |values| > 0 → p = 1.0
    perm_values = np.array([-0.1, 0.05, 0.15])
    result = _rsa_null_summary("partial_P_given_XS", perm_values, observed=0.0, two_sided=True)
    assert result["null_partial_P_given_XS_pvalue"] == pytest.approx(1.0)


def test_rsa_null_summary_two_sided_none_exceed():
    # observed is very large, no null value exceeds it → p = 0.0
    perm_values = np.array([0.01, 0.02, 0.03])
    result = _rsa_null_summary("partial_P_given_XS", perm_values, observed=0.99, two_sided=True)
    assert result["null_partial_P_given_XS_pvalue"] == pytest.approx(0.0)


from analyses.rsa import _compute_rsa_null_distribution, _empty_rsa_null_results, _compute_rdm
from scipy.stats import spearmanr


def _make_rsa_fixtures(n_sub=30, n_phys=3, seed=99):
    rng = np.random.default_rng(seed)
    physics_scaled_sub = rng.standard_normal((n_sub, n_phys))
    neural_sub = rng.standard_normal((n_sub, 10))
    X_sub = rng.standard_normal((n_sub, 5))
    S_sub = rng.standard_normal((n_sub, 5))
    rdm_neural = _compute_rdm(neural_sub)
    rdm_X = _compute_rdm(X_sub)
    rdm_S = _compute_rdm(S_sub)
    rdm_neural[np.isnan(rdm_neural)] = 0.0
    rdm_X[np.isnan(rdm_X)] = 0.0
    rdm_S[np.isnan(rdm_S)] = 0.0
    obs_corr_P = spearmanr(rdm_neural, _compute_rdm(physics_scaled_sub))[0]
    return physics_scaled_sub, rdm_neural, rdm_X, rdm_S, obs_corr_P


def test_compute_rsa_null_keys_gt_only():
    physics_scaled_sub, rdm_neural, rdm_X, rdm_S, obs_corr_P = _make_rsa_fixtures()
    result = _compute_rsa_null_distribution(
        physics_scaled_sub,
        rdm_neural,
        rdm_X,
        rdm_S=rdm_S,
        physics_inf_scaled_sub=None,
        observed_corr_P=obs_corr_P,
        observed_partial_P_given_X=0.1,
        observed_partial_P_given_XS=0.01,
        n_permutations=10,
        seed=0,
    )
    for prefix in ("corr_neural_P", "partial_P_given_X", "partial_P_given_XS"):
        assert f"null_{prefix}_pvalue" in result, f"missing null_{prefix}_pvalue"
        assert f"null_{prefix}_perm_values" in result
    # No inf keys expected
    assert "null_corr_neural_P_inf_pvalue" not in result


def test_compute_rsa_null_keys_with_inf():
    rng = np.random.default_rng(7)
    physics_scaled_sub, rdm_neural, rdm_X, rdm_S, obs_corr_P = _make_rsa_fixtures()
    physics_inf_sub = rng.standard_normal((30, 3))
    result = _compute_rsa_null_distribution(
        physics_scaled_sub,
        rdm_neural,
        rdm_X,
        rdm_S=rdm_S,
        physics_inf_scaled_sub=physics_inf_sub,
        observed_corr_P=obs_corr_P,
        observed_partial_P_given_X=0.1,
        observed_partial_P_given_XS=0.01,
        observed_corr_P_inf=0.05,
        observed_partial_P_inf_given_X=0.08,
        observed_partial_P_inf_given_XS=0.01,
        n_permutations=10,
        seed=0,
    )
    for prefix in ("corr_neural_P_inf", "partial_P_inf_given_X", "partial_P_inf_given_XS"):
        assert f"null_{prefix}_pvalue" in result, f"missing null_{prefix}_pvalue"


def test_compute_rsa_null_perm_length():
    physics_scaled_sub, rdm_neural, rdm_X, rdm_S, obs_corr_P = _make_rsa_fixtures()
    n_perms = 7
    result = _compute_rsa_null_distribution(
        physics_scaled_sub,
        rdm_neural,
        rdm_X,
        observed_corr_P=obs_corr_P,
        observed_partial_P_given_X=0.1,
        n_permutations=n_perms,
        seed=0,
    )
    assert len(result["null_corr_neural_P_perm_values"]) == n_perms


def test_empty_rsa_null_results_keys():
    result = _empty_rsa_null_results()
    for prefix in (
        "corr_neural_P", "partial_P_given_X", "partial_P_given_XS",
        "corr_neural_P_inf", "partial_P_inf_given_X", "partial_P_inf_given_XS",
    ):
        assert f"null_{prefix}_pvalue" in result
        assert np.isnan(result[f"null_{prefix}_pvalue"])


from analyses.rsa import run_rsa_analysis


def _make_run_rsa_inputs(n_scenes=60, n_neurons=20, n_phys=3, n_pix=8, seed=42):
    rng = np.random.default_rng(seed)
    neural_activity = rng.standard_normal((n_scenes, n_neurons)).astype(np.float32)
    physics_labels = rng.standard_normal((n_scenes, n_phys)).astype(np.float32)
    scenes = {"physics_labels": physics_labels}
    raw_pixel_pca = rng.standard_normal((n_scenes, n_pix)).astype(np.float32)
    predicted_pixel_pca = rng.standard_normal((n_scenes, n_pix)).astype(np.float32)
    inferred_physics_labels = rng.standard_normal((n_scenes, n_phys)).astype(np.float32)
    return neural_activity, scenes, {}, raw_pixel_pca, predicted_pixel_pca, inferred_physics_labels


def test_run_rsa_analysis_includes_null_pvalues():
    neural, scenes, meta, raw_pca, pred_pca, inf_phys = _make_run_rsa_inputs()
    result = run_rsa_analysis(
        neural, scenes, meta,
        raw_pixel_pca=raw_pca,
        rsa_subsample=40,
        predicted_pixel_pca=pred_pca,
        inferred_physics_labels=inf_phys,
        compute_null=True,
        n_null_permutations=5,
        null_seed=0,
    )
    for prefix in ("corr_neural_P", "partial_P_given_X", "partial_P_given_XS",
                   "corr_neural_P_inf", "partial_P_inf_given_X", "partial_P_inf_given_XS"):
        key = f"null_{prefix}_pvalue"
        assert key in result, f"missing {key}"
        assert not np.isnan(result[key]), f"{key} is NaN but compute_null=True"


def test_run_rsa_analysis_no_null_when_disabled():
    neural, scenes, meta, raw_pca, pred_pca, inf_phys = _make_run_rsa_inputs()
    result = run_rsa_analysis(
        neural, scenes, meta,
        raw_pixel_pca=raw_pca,
        rsa_subsample=40,
        compute_null=False,
    )
    assert np.isnan(result["null_corr_neural_P_pvalue"])
