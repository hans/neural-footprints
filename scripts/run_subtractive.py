"""Run the subtractive analysis on block-structured neural activity.

Outputs:
  outputs/subtractive_{regime}_{mode}_results.json
  data/subtractive_{regime}_{mode}_plot_data.npz
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from load_config import load_config
from io_utils import save_results
from analyses.subtractive import run_subtractive_analysis


cfg = load_config()
sub_cfg = cfg['subtractive']
regime = snakemake.wildcards.regime  # noqa: F821
mode   = snakemake.wildcards.mode    # noqa: F821

print(f"\nRunning subtractive analysis (regime={regime}, mode={mode})")
print("=" * 60)

data = np.load(snakemake.input.neural)  # noqa: F821
neural_activity = data['neural_activity']
block_meta = {
    'block_assignment': data['block_assignment'],
    'block_names': [str(s) for s in data['block_names']],
    'grid_shape': tuple(int(x) for x in data['grid_shape']),
    'grid_positions': data['grid_positions'],
}
condition = data['condition']
inferred_N = data['inferred_N']
cardinality_val_r2 = float(data['cardinality_val_r2'])
print(f"  neural_activity: {neural_activity.shape}")
print(f"  cardinality val R²: {cardinality_val_r2:.3f}")
print(f"  condition counts: low={int((condition == 0).sum())}, "
      f"high={int((condition == 1).sum())}")

results = run_subtractive_analysis(
    neural_activity, condition, block_meta,
    fdr_alpha=sub_cfg['fdr_alpha'],
    thresholds_uncorrected=tuple(sub_cfg['thresholds_uncorrected']),
)
results['regime'] = regime
results['mode'] = mode
results['cardinality_val_r2'] = cardinality_val_r2
results['inferred_N_low_mean']  = float(inferred_N[condition == 0].mean())
results['inferred_N_high_mean'] = float(inferred_N[condition == 1].mean())

print(f"\n  HEADLINE: {results['headline']}")
for pb in results['per_block']:
    print(f"  per-block ({pb['name']:>8}): mean|t|={pb['mean_abs_t']:.2f}  "
          f"d={pb['mean_cohens_d']:+.2f}  n={pb['n_neurons']}")
print(f"  Threshold sweep:")
for thr_name, rec in results['thresholds'].items():
    fracs = ", ".join(
        f"{block_meta['block_names'][b]}={rec[f'frac_significant_block_{b}']:.1%}"
        for b in range(len(block_meta['block_names']))
    )
    print(f"    {thr_name:>14}: total={rec['n_significant_total']:>4}  ({fracs})")

# Split plot_data out so the JSON stays small
plot_data = results.pop('plot_data')
plot_data['cardinality_val_r2'] = np.array(cardinality_val_r2)
plot_data['inferred_N_low_mean']  = np.array(results['inferred_N_low_mean'])
plot_data['inferred_N_high_mean'] = np.array(results['inferred_N_high_mean'])
plot_data['regime'] = np.array(regime)
plot_data['mode'] = np.array(mode)

save_results(results, snakemake.output.results)  # noqa: F821
np.savez_compressed(snakemake.output.plot_data, **plot_data)  # noqa: F821
print(f"\n  Saved results   -> {snakemake.output.results}")  # noqa: F821
print(f"  Saved plot data -> {snakemake.output.plot_data}")  # noqa: F821
