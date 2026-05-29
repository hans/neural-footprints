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
for key in [
    "r2_X",
    "r2_P",
    "r2_XP",
    "r2_S",
    "r2_XS",
    "r2_XPS",
    "delta_P_given_X",
    "delta_P_given_XS",
    # backward-compat aliases
    "r2_pixel_only",
    "r2_physics_only",
    "r2_combined",
    "delta_r2",
    # inferred-physics variants
    "r2_P_inf",
    "r2_XP_inf",
    "delta_P_inf_given_X",
    "r2_XPS_inf",
    "delta_P_inf_given_XS",
]:
    if key in encoding:
        encoding[key] = np.array(encoding[key])

residual_results = None
if hasattr(snakemake.input, "residual"):
    residual_results = load_results(snakemake.input.residual)
    for key in ["r2_P_given_X", "r2_P_given_XS",
                "r2_P_inf_given_X", "r2_P_inf_given_XS"]:
        if key in residual_results and residual_results[key] is not None:
            residual_results[key] = np.array(residual_results[key])

n_passed, n_total, checks = evaluate(
    encoding,
    rsa,
    dissociation,
    dynamics_results=dynamics,
    residual_results=residual_results,
)

save_results(
    {
        "n_passed": n_passed,
        "n_total": n_total,
        "checks": checks,
    },
    snakemake.output[0],
)
