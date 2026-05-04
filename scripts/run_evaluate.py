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
for key in ['r2_render_only', 'r2_physics_only', 'r2_combined', 'delta_r2']:
    if key in encoding:
        encoding[key] = np.array(encoding[key])

residual_results = None
if hasattr(snakemake.input, 'residual'):
    residual_results = load_results(snakemake.input.residual)
    for key in ['r2_raw_render', 'r2_raw_physics_gt',
                'r2_resid_render', 'r2_resid_physics_gt']:
        if key in residual_results and residual_results[key] is not None:
            residual_results[key] = np.array(residual_results[key])

n_passed, n_total, checks = evaluate(encoding, rsa, dissociation,
                                     dynamics_results=dynamics,
                                     residual_results=residual_results)

save_results({
    'n_passed': n_passed,
    'n_total': n_total,
    'checks': checks,
}, snakemake.output[0])
