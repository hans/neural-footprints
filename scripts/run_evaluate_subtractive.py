"""Numerically evaluate the subtractive findings (mirrors run_evaluate.py).

Loads outputs/subtractive_{regime}_{mode}_results.json for both regimes,
calls evaluate_subtractive(), and writes outputs/evaluation_subtractive_{mode}.json.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from io_utils import load_results, save_results
from evaluation_subtractive import evaluate_subtractive


mode = snakemake.wildcards.mode  # noqa: F821

per_regime = {}
for path in snakemake.input.results:  # noqa: F821
    results = load_results(path)
    regime = results['regime']
    per_regime[regime] = results

n_passed, n_total, checks = evaluate_subtractive(per_regime, mode=mode)

save_results({
    'mode': mode,
    'n_passed': n_passed,
    'n_total': n_total,
    'checks': checks,
}, snakemake.output[0])  # noqa: F821
