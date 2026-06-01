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
