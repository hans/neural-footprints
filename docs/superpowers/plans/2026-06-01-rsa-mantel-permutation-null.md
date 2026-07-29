# RSA Mantel Permutation Null Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Mantel permutation null to RSA partial-correlation results, paired with the existing encoding null, so p-values can be reported for all six RSA metrics.

**Architecture:** `_compute_rsa_null_distribution` in `analyses/rsa.py` permutes scene-label rows (not RDM cells), re-derives the physics RDM each permutation, and computes the same partial-Spearman statistics on the null RDM — mirroring how `_compute_null_distribution` in `analyses/encoding.py` works. The same default seed (`0`) is used for reproducibility. Results are merged into the RSA result dict, read by `evaluation.py`, and controlled by two new config keys.

**Tech Stack:** NumPy, SciPy (spearmanr), scikit-learn (StandardScaler already used), pytest via `uv run --with pytest`

---

## File Map

| File | Change |
|---|---|
| `analyses/rsa.py` | Add `_rsa_null_summary`, `_compute_rsa_null_distribution`, `_empty_rsa_null_results`; update `run_rsa_analysis` signature |
| `config.yaml` | Add `rsa_compute_null: true` and `rsa_n_null_permutations: 300` |
| `scripts/run_rsa.py` | Pass new config keys to `run_rsa_analysis` |
| `evaluation.py` | Add RSA null p-value checks after existing RSA checks |
| `tests/test_rsa_null.py` | Unit tests for null functions |

---

### Task 1: Config keys

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add the two new config keys**

Open `config.yaml`. After the existing `rsa_subsample: 500` line (line 25), add:

```yaml
rsa_compute_null: true
rsa_n_null_permutations: 300
```

- [ ] **Step 2: Verify the file looks correct**

Run: `grep -n "rsa_" config.yaml`

Expected output:
```
25:rsa_subsample: 500
26:rsa_compute_null: true
27:rsa_n_null_permutations: 300
```

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "feat: add rsa_compute_null + rsa_n_null_permutations config keys"
```

---

### Task 2: `_rsa_null_summary` helper

**Files:**
- Modify: `analyses/rsa.py` (add after `_partial_spearman_2`, before `run_rsa_analysis`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_rsa_null.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest python -m pytest tests/test_rsa_null.py::test_rsa_null_summary_one_sided_pvalue -v`

Expected: `ImportError` or `AttributeError` — `_rsa_null_summary` does not exist yet.

- [ ] **Step 3: Implement `_rsa_null_summary` in `analyses/rsa.py`**

Insert after `_partial_spearman_2` (after line 63), before `run_rsa_analysis`:

```python
def _rsa_null_summary(prefix, perm_values, observed, two_sided=False):
    """Per-perm scalar null → 95% CI + p-value.

    two_sided: use |perm| >= |observed| (for KEY ≈-0 tests);
               else one-sided (perm >= observed, for expected-positive tests).
    """
    lo, hi = np.percentile(perm_values, [2.5, 97.5])
    if two_sided:
        pvalue = float((np.abs(perm_values) >= abs(observed)).mean())
    else:
        pvalue = float((perm_values >= observed).mean())
    return {
        f"null_{prefix}_perm_values": perm_values,
        f"null_{prefix}_ci_lo": float(lo),
        f"null_{prefix}_ci_hi": float(hi),
        f"null_{prefix}_mean": float(perm_values.mean()),
        f"null_{prefix}_pvalue": pvalue,
        f"null_{prefix}_observed": float(observed),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest python -m pytest tests/test_rsa_null.py -v`

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add analyses/rsa.py tests/test_rsa_null.py
git commit -m "feat: add _rsa_null_summary helper with one/two-sided support"
```

---

### Task 3: `_compute_rsa_null_distribution` and `_empty_rsa_null_results`

**Files:**
- Modify: `analyses/rsa.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rsa_null.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest python -m pytest tests/test_rsa_null.py -k "test_compute_rsa_null or test_empty" -v`

Expected: `ImportError` — functions don't exist yet.

- [ ] **Step 3: Implement `_compute_rsa_null_distribution` in `analyses/rsa.py`**

Insert after `_rsa_null_summary`, before `run_rsa_analysis`:

```python
def _compute_rsa_null_distribution(
    physics_scaled_sub,
    rdm_neural,
    rdm_X,
    *,
    rdm_S=None,
    physics_inf_scaled_sub=None,
    observed_corr_P,
    observed_partial_P_given_X,
    observed_partial_P_given_XS=None,
    observed_corr_P_inf=None,
    observed_partial_P_inf_given_X=None,
    observed_partial_P_inf_given_XS=None,
    n_permutations=300,
    seed=0,
):
    """Mantel permutation null for RSA partial correlations.

    Permutes scene rows of physics_scaled_sub (not RDM cells), re-derives the
    physics RDM, then recomputes partial Spearman correlations against the fixed
    rdm_neural / rdm_X / rdm_S. The same row permutation is applied to
    physics_inf_scaled_sub for apples-to-apples inferred-physics comparison.
    """
    n_sub = physics_scaled_sub.shape[0]
    has_S = rdm_S is not None
    has_inf = physics_inf_scaled_sub is not None

    rng = np.random.default_rng(seed)

    corr_P_null = np.empty(n_permutations)
    partial_X_null = np.empty(n_permutations)
    partial_XS_null = np.empty(n_permutations) if has_S else None
    if has_inf:
        corr_P_inf_null = np.empty(n_permutations)
        partial_inf_X_null = np.empty(n_permutations)
        partial_inf_XS_null = np.empty(n_permutations) if has_S else None

    for p in range(n_permutations):
        perm = rng.permutation(n_sub)
        physics_perm = physics_scaled_sub[perm]
        rdm_physics_perm = _compute_rdm(physics_perm)
        rdm_physics_perm[np.isnan(rdm_physics_perm)] = 0.0

        corr_P_null[p] = spearmanr(rdm_neural, rdm_physics_perm)[0]
        partial_X_null[p] = _partial_spearman(rdm_neural, rdm_physics_perm, rdm_X)[0]
        if has_S:
            partial_XS_null[p] = _partial_spearman_2(
                rdm_neural, rdm_physics_perm, rdm_X, rdm_S
            )[0]

        if has_inf:
            inf_perm = physics_inf_scaled_sub[perm]
            rdm_physics_inf_perm = _compute_rdm(inf_perm)
            rdm_physics_inf_perm[np.isnan(rdm_physics_inf_perm)] = 0.0
            corr_P_inf_null[p] = spearmanr(rdm_neural, rdm_physics_inf_perm)[0]
            partial_inf_X_null[p] = _partial_spearman(
                rdm_neural, rdm_physics_inf_perm, rdm_X
            )[0]
            if has_S:
                partial_inf_XS_null[p] = _partial_spearman_2(
                    rdm_neural, rdm_physics_inf_perm, rdm_X, rdm_S
                )[0]

    out = {}
    out.update(_rsa_null_summary("corr_neural_P", corr_P_null, observed_corr_P, two_sided=False))
    out.update(_rsa_null_summary("partial_P_given_X", partial_X_null, observed_partial_P_given_X, two_sided=False))
    if has_S and observed_partial_P_given_XS is not None:
        out.update(_rsa_null_summary("partial_P_given_XS", partial_XS_null, observed_partial_P_given_XS, two_sided=True))
    if has_inf:
        out.update(_rsa_null_summary("corr_neural_P_inf", corr_P_inf_null, observed_corr_P_inf, two_sided=False))
        out.update(_rsa_null_summary("partial_P_inf_given_X", partial_inf_X_null, observed_partial_P_inf_given_X, two_sided=False))
        if has_S and observed_partial_P_inf_given_XS is not None:
            out.update(_rsa_null_summary("partial_P_inf_given_XS", partial_inf_XS_null, observed_partial_P_inf_given_XS, two_sided=True))
    return out
```

- [ ] **Step 4: Implement `_empty_rsa_null_results` in `analyses/rsa.py`**

Insert directly after `_compute_rsa_null_distribution`:

```python
def _empty_rsa_null_results():
    nan = float("nan")
    out = {}
    for prefix in (
        "corr_neural_P",
        "partial_P_given_X",
        "partial_P_given_XS",
        "corr_neural_P_inf",
        "partial_P_inf_given_X",
        "partial_P_inf_given_XS",
    ):
        out.update({
            f"null_{prefix}_perm_values": np.empty(0),
            f"null_{prefix}_ci_lo": nan,
            f"null_{prefix}_ci_hi": nan,
            f"null_{prefix}_mean": nan,
            f"null_{prefix}_pvalue": nan,
            f"null_{prefix}_observed": nan,
        })
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --with pytest python -m pytest tests/test_rsa_null.py -v`

Expected: All tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add analyses/rsa.py tests/test_rsa_null.py
git commit -m "feat: add _compute_rsa_null_distribution and _empty_rsa_null_results"
```

---

### Task 4: Wire null into `run_rsa_analysis`

**Files:**
- Modify: `analyses/rsa.py` (`run_rsa_analysis` function)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rsa_null.py`:

```python
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
    # All null pvalues should be NaN when disabled
    assert np.isnan(result["null_corr_neural_P_pvalue"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest python -m pytest tests/test_rsa_null.py::test_run_rsa_analysis_includes_null_pvalues -v`

Expected: `TypeError` — `run_rsa_analysis` doesn't accept `compute_null` yet.

- [ ] **Step 3: Update `run_rsa_analysis` signature**

Change the function signature from:

```python
def run_rsa_analysis(
    neural_activity,
    scenes,
    neural_meta,
    *,
    raw_pixel_pca,
    rsa_subsample=None,
    predicted_pixel_pca=None,
    inferred_physics_labels=None,
):
```

to:

```python
def run_rsa_analysis(
    neural_activity,
    scenes,
    neural_meta,
    *,
    raw_pixel_pca,
    rsa_subsample=None,
    predicted_pixel_pca=None,
    inferred_physics_labels=None,
    compute_null=True,
    n_null_permutations=300,
    null_seed=0,
):
```

- [ ] **Step 4: Call the null distribution and merge results into `result`**

At the end of `run_rsa_analysis`, just before `return result`, replace the bare `return result` with:

```python
    # --- Permutation null ---
    if compute_null:
        null_results = _compute_rsa_null_distribution(
            physics_scaled,
            rdm_neural,
            rdm_X,
            rdm_S=result.get("rdm_S"),
            physics_inf_scaled_sub=physics_inf_scaled,
            observed_corr_P=corr_neural_P,
            observed_partial_P_given_X=partial_P_given_X,
            observed_partial_P_given_XS=result.get("partial_P_given_XS"),
            observed_corr_P_inf=result.get("corr_neural_P_inf"),
            observed_partial_P_inf_given_X=result.get("partial_P_inf_given_X"),
            observed_partial_P_inf_given_XS=result.get("partial_P_inf_given_XS"),
            n_permutations=n_null_permutations,
            seed=null_seed,
        )
    else:
        null_results = _empty_rsa_null_results()
    result.update(null_results)

    return result
```

Note: `physics_scaled` and `physics_inf_scaled` are already in scope — they're the standardized, subsampled physics arrays computed earlier in `run_rsa_analysis`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --with pytest python -m pytest tests/test_rsa_null.py -v`

Expected: All tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add analyses/rsa.py tests/test_rsa_null.py
git commit -m "feat: wire RSA Mantel null into run_rsa_analysis"
```

---

### Task 5: Pass config knobs from `scripts/run_rsa.py`

**Files:**
- Modify: `scripts/run_rsa.py`

- [ ] **Step 1: Update `run_rsa_analysis` call to pass null config**

In `scripts/run_rsa.py`, change the `run_rsa_analysis` call from:

```python
results = run_rsa_analysis(
    neural,
    scenes,
    neural_meta,
    raw_pixel_pca=raw_pixel_pca,
    rsa_subsample=cfg["rsa_subsample"],
    predicted_pixel_pca=predicted_pixel_pca,
    inferred_physics_labels=inferred_physics_labels,
)
```

to:

```python
results = run_rsa_analysis(
    neural,
    scenes,
    neural_meta,
    raw_pixel_pca=raw_pixel_pca,
    rsa_subsample=cfg["rsa_subsample"],
    predicted_pixel_pca=predicted_pixel_pca,
    inferred_physics_labels=inferred_physics_labels,
    compute_null=cfg.get("rsa_compute_null", True),
    n_null_permutations=cfg.get("rsa_n_null_permutations", 300),
    null_seed=0,
)
```

- [ ] **Step 2: Verify syntax**

Run: `uv run python -c "import ast, sys; ast.parse(open('scripts/run_rsa.py').read()); print('OK')" `

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/run_rsa.py
git commit -m "feat: pass rsa_compute_null/n_null_permutations from config to run_rsa_analysis"
```

---

### Task 6: Wire RSA null p-values into `evaluation.py`

**Files:**
- Modify: `evaluation.py`

- [ ] **Step 1: Locate the insertion point**

In `evaluation.py`, find the block starting at the existing RSA check for `partial_P_given_XS` (around line 243). New null p-value checks go immediately after each corresponding point estimate check.

- [ ] **Step 2: Add null checks for GT physics**

After the existing block:
```python
    check(
        "Neural ↔ physics correlation present",
        corr_P > 0.05,
        f"r = {corr_P:.4f}",
        "expect > 0.05",
    )
```
add:
```python
    if rsa_results.get("null_corr_neural_P_pvalue") is not None and not np.isnan(rsa_results.get("null_corr_neural_P_pvalue", float("nan"))):
        pval = float(rsa_results["null_corr_neural_P_pvalue"])
        check(
            "corr_neural_P significantly above null (p < 0.05, one-sided)",
            pval < 0.05,
            f"p = {pval:.3f}",
            "PASS iff p < 0.05 one-sided vs Mantel permutation null",
        )
```

After the existing block:
```python
    check(
        "Partial neural↔physics | X > 0 (naive positive)",
        partial_X > 0.02,
        f"r = {partial_X:.4f}",
        "expect > 0.02",
    )
```
add:
```python
    if rsa_results.get("null_partial_P_given_X_pvalue") is not None and not np.isnan(rsa_results.get("null_partial_P_given_X_pvalue", float("nan"))):
        pval = float(rsa_results["null_partial_P_given_X_pvalue"])
        check(
            "partial_P_given_X significantly above null (p < 0.05, one-sided)",
            pval < 0.05,
            f"p = {pval:.3f}",
            "PASS iff p < 0.05 one-sided vs Mantel permutation null",
        )
```

After the existing `partial_P_given_XS` point estimate check inside `if rsa_results.get("partial_P_given_XS") is not None:`, add:
```python
        if rsa_results.get("null_partial_P_given_XS_pvalue") is not None and not np.isnan(rsa_results.get("null_partial_P_given_XS_pvalue", float("nan"))):
            pval = float(rsa_results["null_partial_P_given_XS_pvalue"])
            check(
                "partial_P_given_XS significantly above null (KEY, p < 0.05, two-sided)",
                pval < 0.05,
                f"p = {pval:.3f}",
                "PASS iff p < 0.05; PASS means physics distinguishable from random pairing (bad news)",
            )
```

- [ ] **Step 3: Add null checks for inferred physics**

After the existing inferred-physics RSA check for `corr_neural_P_inf`, add:
```python
    if rsa_results.get("null_corr_neural_P_inf_pvalue") is not None and not np.isnan(rsa_results.get("null_corr_neural_P_inf_pvalue", float("nan"))):
        pval = float(rsa_results["null_corr_neural_P_inf_pvalue"])
        check(
            "corr_neural_P_inf significantly above null (p < 0.05, one-sided)",
            pval < 0.05,
            f"p = {pval:.3f}",
            "PASS iff p < 0.05 one-sided vs Mantel permutation null",
        )
```

After the existing check for `partial_P_inf_given_X`, add:
```python
    if rsa_results.get("null_partial_P_inf_given_X_pvalue") is not None and not np.isnan(rsa_results.get("null_partial_P_inf_given_X_pvalue", float("nan"))):
        pval = float(rsa_results["null_partial_P_inf_given_X_pvalue"])
        check(
            "partial_P_inf_given_X significantly above null (p < 0.05, one-sided)",
            pval < 0.05,
            f"p = {pval:.3f}",
            "PASS iff p < 0.05 one-sided vs Mantel permutation null",
        )
```

After the existing check for `partial_P_inf_given_XS`, add:
```python
    if rsa_results.get("null_partial_P_inf_given_XS_pvalue") is not None and not np.isnan(rsa_results.get("null_partial_P_inf_given_XS_pvalue", float("nan"))):
        pval = float(rsa_results["null_partial_P_inf_given_XS_pvalue"])
        check(
            "partial_P_inf_given_XS significantly above null (KEY, p < 0.05, two-sided)",
            pval < 0.05,
            f"p = {pval:.3f}",
            "PASS iff p < 0.05; PASS means inferred physics distinguishable from random pairing (bad news)",
        )
```

- [ ] **Step 4: Verify syntax**

Run: `uv run python -c "import ast; ast.parse(open('evaluation.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 5: Smoke-test evaluation with stubs**

Run: `uv run python -c "
from evaluation import evaluate
import numpy as np

enc = {
    'r2_X': np.array([0.4]),
    'r2_P': np.array([0.2]),
    'control_accuracy': 0.95,
}
rsa = {
    'corr_neural_X': 0.3,
    'corr_neural_P': 0.1,
    'partial_P_given_X': 0.05,
    'partial_P_given_XS': 0.01,
    'corr_neural_P_inf': 0.08,
    'partial_P_inf_given_X': 0.04,
    'partial_P_inf_given_XS': 0.01,
    'null_corr_neural_P_pvalue': 0.01,
    'null_partial_P_given_X_pvalue': 0.02,
    'null_partial_P_given_XS_pvalue': 0.80,
    'null_corr_neural_P_inf_pvalue': 0.03,
    'null_partial_P_inf_given_X_pvalue': 0.04,
    'null_partial_P_inf_given_XS_pvalue': 0.75,
}
diss = {
    'mean_r2_pixel': 0.5,
    'mean_r2_physics': 0.2,
    'pixel_behavioral_score': 0.1,
    'physics_behavioral_score': 0.95,
    'metric_label': 'R2',
    'objective': 'next_frame_pixels',
}
evaluate(enc, rsa, diss)
print('OK')
"`

Expected: prints the evaluation report then `OK` with no exception.

- [ ] **Step 6: Commit**

```bash
git add evaluation.py
git commit -m "feat: add RSA null p-value checks to evaluation.py"
```

---

## Self-Review Against Spec

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Mantel permutation null on partial correlations (permute labels → re-derive RDM → recompute partial r) | Task 3 |
| Same RNG seed as encoding null (`seed=0`) | Task 3 (seed default = 0) |
| Permute scenes, not RDM cells | Task 3 (permutes `physics_scaled_sub[perm]`, then calls `_compute_rdm`) |
| Two-sided for KEY tests (`partial_P_given_XS`, `partial_P_inf_given_XS`) | Task 2 (`two_sided=True`) |
| One-sided for raw correlations (`corr_neural_P`, `corr_neural_P_inf`) | Task 2 (`two_sided=False`) |
| Coverage: all 6 metrics with nulls + p-values | Tasks 3–4 |
| Wire p-values into `evaluation.py` | Task 6 |
| `rsa_n_null_permutations` config (default 300) | Task 1 |
| `rsa_compute_null: true` toggle | Task 1 |
| Caveat: Mantel non-independence makes p-values optimistic | Not in code (paper text only, not code concern) |

**Placeholder scan:** No TBDs or "similar to Task N" references found.

**Type consistency:** `_rsa_null_summary` returns keys `null_{prefix}_pvalue` — used consistently in Tasks 4, 5, 6. `_compute_rsa_null_distribution` returns `_rsa_null_summary` output — consumed correctly in Task 4. `run_rsa_analysis` uses `result.get("rdm_S")` — `rdm_S` is set into result via `result["rdm_S"] = rdm_S` at line 186 of `rsa.py` before the null call.

**Gap check:** The spec says "partial_P_given_X" is one-sided but does NOT list it in the "Coverage" bullet list explicitly. Looking at the bullet list again:
- `corr_neural_P` (one-sided) ✓
- `partial_P_given_X` (one-sided) ✓ (listed in "Sidedness" section)  
- `partial_P_given_XS` (two-sided, KEY) ✓
- `corr_neural_P_inf` (one-sided) ✓
- `partial_P_inf_given_X` (one-sided) ✓
- `partial_P_inf_given_XS` (two-sided, KEY) ✓

All 6 metrics covered.
