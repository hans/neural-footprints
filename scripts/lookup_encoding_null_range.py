"""
Look up the permutation-null R² range from the brain encoding analysis.

The encoding analysis (analyses/encoding.py) row-shuffles the physics labels
n_permutations times and refits the ridge encoder each time, giving a null
distribution of mean-across-neurons R². This is the reference "chance" band for
*any* physics-decoding R² in the study, including the residualization R²s in
evaluation.py (r2_P | X, r2_P | X,S). A residual R² that sits inside this band
is statistically indistinguishable from chance.

Usage:
    uv run python scripts/lookup_encoding_null_range.py [norm]
        norm defaults to "zscore"; pass e.g. "truncated_svd" for another run.
"""

import json
import sys

import numpy as np


def _summarize(perm_means, observed, observed_label):
    """Print the null range for one per-permutation mean array."""
    perm_means = np.asarray(perm_means, dtype=float)
    lo95, hi95 = np.percentile(perm_means, [2.5, 97.5])
    p_one_sided = float((perm_means >= observed).mean()) if observed is not None else None

    print(f"    n_permutations      = {perm_means.size}")
    print(f"    null min .. max     = {perm_means.min():+.4f} .. {perm_means.max():+.4f}")
    print(f"    null 95% CI         = [{lo95:+.4f}, {hi95:+.4f}]")
    print(f"    null mean           = {perm_means.mean():+.4f}")
    if observed is not None:
        print(f"    observed ({observed_label}) = {observed:+.4f}")
        print(f"    one-sided p (obs>=null) = {p_one_sided:.3f}")


def main():
    norm = sys.argv[1] if len(sys.argv) > 1 else "zscore"
    path = f"outputs/{norm}/encoding_results.json"
    print(f"Loading {path}\n")
    with open(path) as f:
        d = json.load(f)

    # (key prefix, human label, observed-value key)
    blocks = [
        ("r2_physics_only_null", "GT physics encoding null (r2_P)", "null_physics_observed"),
        ("r2_P_inf_null", "Inferred physics encoding null (r2_P_inf)", "null_r2_P_inf_observed"),
    ]

    for null_key, label, obs_key in blocks:
        if null_key not in d:
            continue
        print(f"=== {label} ===")
        # Prefer the full [n_perm x n_neurons] array; collapse to per-perm means.
        arr = np.asarray(d[null_key], dtype=float)
        perm_means = arr.mean(axis=1) if arr.ndim == 2 else arr
        observed = d.get(obs_key)
        _summarize(perm_means, observed, label.split("(")[-1].rstrip(") "))
        print()

    print(
        "Interpretation: chance-level physics R² is small and *negative* (~-0.0005),\n"
        "so a residualization R² near zero/slightly-negative is the signature of no\n"
        "detectable signal, not a bug. Caveat: this null predicts RAW neural from\n"
        "shuffled physics; the residualization analysis predicts the X-RESIDUAL of\n"
        "neural, a different (noisier) target — so this band is a ballpark reference,\n"
        "not an exact null for r2_P | X. For a rigorous 'at chance' claim on the\n"
        "residual, build a permutation null inside run_residual_analysis."
    )


if __name__ == "__main__":
    main()
