import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from io_utils import load_results, save_results
from evaluation import evaluate

encoding = load_results(snakemake.input.encoding)
rsa = load_results(snakemake.input.rsa)
dissociation = load_results(snakemake.input.dissociation)
dynamics = load_results(snakemake.input.dynamics)

# Convert lists back to arrays where evaluate() expects them
for key in ['r2_pixels_only', 'r2_physics_only', 'r2_combined', 'delta_r2']:
    if key in encoding:
        encoding[key] = np.array(encoding[key])

n_passed, n_total, checks = evaluate(encoding, rsa, dissociation,
                                     dynamics_results=dynamics)

save_results({
    'n_passed': n_passed,
    'n_total': n_total,
    'checks': checks,
}, snakemake.output[0])
