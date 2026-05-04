"""Subtractive (cognitive-subtraction) analysis.

Per-neuron Welch's two-sample t-test between high-N and low-N conditions,
then a threshold sweep over uncorrected, FDR-BH, and Bonferroni cutoffs.
The headline question — does the abstract block ever dominate the surviving
"brain blob"? — is computed here.

See specs/subtractive_analysis.md for the scientific motivation.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Per-neuron Welch's t and effect sizes
# ---------------------------------------------------------------------------

def _welch_t(neural_activity, condition_labels):
    """Per-neuron Welch's two-sample t-statistic and p-value.

    condition_labels: 0 = low (control), 1 = high (task). Returned t is
    signed in the (high - low) direction.
    """
    cond = np.asarray(condition_labels)
    high = neural_activity[cond == 1]
    low  = neural_activity[cond == 0]
    if high.shape[0] < 2 or low.shape[0] < 2:
        raise ValueError("Each condition needs at least 2 scenes for Welch's t.")
    if neural_activity.std(axis=0).min() == 0:
        raise ValueError(
            "At least one neuron has zero variance across scenes; refusing to "
            "compute t-statistic. Investigate the projection."
        )
    t, p = stats.ttest_ind(high, low, axis=0, equal_var=False)
    return np.asarray(t, dtype=np.float64), np.asarray(p, dtype=np.float64)


def _cohens_d(neural_activity, condition_labels):
    """Per-neuron Cohen's d using pooled SD (high - low)."""
    cond = np.asarray(condition_labels)
    high = neural_activity[cond == 1]
    low  = neural_activity[cond == 0]
    nh, nl = high.shape[0], low.shape[0]
    s2 = ((nh - 1) * high.var(axis=0, ddof=1) +
          (nl - 1) * low.var(axis=0, ddof=1)) / max(nh + nl - 2, 1)
    s = np.sqrt(np.maximum(s2, 1e-12))
    return (high.mean(axis=0) - low.mean(axis=0)) / s


# ---------------------------------------------------------------------------
# Multiple-comparison corrections
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values):
    """BH-FDR adjusted q-values. Returns array same shape as p_values."""
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity (cumulative min from the right)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return np.minimum(out, 1.0)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def _threshold_summary(p_values, q_values, block_assignment, n_blocks,
                       *, fdr_alpha, thresholds_uncorrected):
    """Build per-threshold dict of significance counts (total + per-block)."""
    n = p_values.size
    out = {}

    def _block_split(mask, key):
        rec = {
            f'n_significant_total': int(mask.sum()),
        }
        for b_idx in range(n_blocks):
            b_mask = (block_assignment == b_idx)
            n_b = int(b_mask.sum())
            n_sig_b = int((mask & b_mask).sum())
            rec[f'n_significant_block_{b_idx}'] = n_sig_b
            rec[f'frac_significant_block_{b_idx}'] = (n_sig_b / n_b) if n_b else 0.0
        return rec

    for alpha in thresholds_uncorrected:
        out[f'p<{alpha}'] = _block_split(p_values < alpha, key=f'p<{alpha}')

    bonf_alpha = 0.05 / max(n, 1)
    out[f'bonferroni'] = _block_split(p_values < bonf_alpha, key='bonferroni')
    out[f'fdr_bh'] = _block_split(q_values < fdr_alpha, key='fdr_bh')
    return out


# ---------------------------------------------------------------------------
# Continuous-sweep curve
# ---------------------------------------------------------------------------

def _continuous_sweep(t_stats, block_assignment, n_blocks, *, n_points=200):
    """Returns (thresholds, counts_per_block) for a |t|-threshold sweep.

    counts_per_block is shape (n_blocks, n_points) of int counts of neurons
    surviving |t| >= threshold.
    """
    abs_t = np.abs(t_stats)
    if abs_t.size == 0:
        return np.empty(0), np.empty((n_blocks, 0), dtype=np.int64)
    t_max = float(abs_t.max())
    thresholds = np.linspace(0.0, t_max, n_points)
    counts = np.zeros((n_blocks, n_points), dtype=np.int64)
    for b in range(n_blocks):
        b_t = abs_t[block_assignment == b]
        for i, thr in enumerate(thresholds):
            counts[b, i] = int((b_t >= thr).sum())
    return thresholds, counts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_subtractive_analysis(neural_activity, condition_labels, block_meta, *,
                             fdr_alpha=0.05,
                             thresholds_uncorrected=(0.05, 0.01, 0.001)):
    """Run the per-neuron subtraction + threshold sweep.

    Parameters
    ----------
    neural_activity : float [n_scenes, n_neurons]
    condition_labels : int [n_scenes]   0 = low, 1 = high
    block_meta : dict
        From generate_block_structured_neural; must contain 'block_assignment',
        'block_names', 'grid_shape', 'grid_positions'.
    fdr_alpha : float
    thresholds_uncorrected : iterable[float]

    Returns
    -------
    dict suitable for json-dump (results) and a 'plot_data' subdict of arrays.
    """
    block_assignment = np.asarray(block_meta['block_assignment'])
    block_names = list(block_meta['block_names'])
    grid_shape = tuple(block_meta['grid_shape'])
    grid_positions = np.asarray(block_meta['grid_positions'])
    n_blocks = len(block_names)

    t_stats, p_values = _welch_t(neural_activity, condition_labels)
    q_values = benjamini_hochberg(p_values)
    d_per_neuron = _cohens_d(neural_activity, condition_labels)

    # Per-block summary stats
    per_block = []
    for b_idx, name in enumerate(block_names):
        mask = (block_assignment == b_idx)
        per_block.append({
            'name': name,
            'n_neurons': int(mask.sum()),
            'mean_abs_t': float(np.abs(t_stats[mask]).mean()),
            'median_abs_t': float(np.median(np.abs(t_stats[mask]))),
            'mean_cohens_d': float(d_per_neuron[mask].mean()),
            'median_cohens_d': float(np.median(d_per_neuron[mask])),
        })

    thresholds_summary = _threshold_summary(
        p_values, q_values, block_assignment, n_blocks,
        fdr_alpha=fdr_alpha,
        thresholds_uncorrected=thresholds_uncorrected,
    )

    # Headline call: at FDR-q<0.05, which block has the higher fraction
    # surviving? (i.e. which block "dominates" the apparent brain blob?)
    fdr_rec = thresholds_summary['fdr_bh']
    fracs = [fdr_rec[f'frac_significant_block_{b}'] for b in range(n_blocks)]
    winner_idx = int(np.argmax(fracs))
    headline = (f"{block_names[winner_idx]} dominates "
                f"(FDR q<{fdr_alpha}: {fracs[winner_idx]:.1%})")

    sweep_thresholds, sweep_counts = _continuous_sweep(
        t_stats, block_assignment, n_blocks
    )

    # Reshape t-stats to grid for the headline heatmap
    t_grid = np.full(grid_shape, np.nan, dtype=np.float64)
    for i, (r, c) in enumerate(grid_positions):
        t_grid[int(r), int(c)] = t_stats[i]

    cond = np.asarray(condition_labels)
    high_mask = cond == 1
    low_mask = cond == 0

    results = {
        'n_neurons': int(neural_activity.shape[1]),
        'n_scenes_high': int(high_mask.sum()),
        'n_scenes_low': int(low_mask.sum()),
        'block_names': block_names,
        'per_block': per_block,
        'thresholds': thresholds_summary,
        'fdr_alpha': float(fdr_alpha),
        'headline': headline,

        'plot_data': {
            't_stats':        t_stats.astype(np.float32),
            'p_values':       p_values.astype(np.float32),
            'q_values_fdr':   q_values.astype(np.float32),
            'cohens_d':       d_per_neuron.astype(np.float32),
            'block_assignment': block_assignment.astype(np.int8),
            'block_names':    np.array(block_names),
            'grid_shape':     np.array(grid_shape, dtype=np.int32),
            'grid_positions': grid_positions.astype(np.int32),
            't_grid':         t_grid.astype(np.float32),
            'sweep_thresholds': sweep_thresholds.astype(np.float32),
            'sweep_counts':     sweep_counts.astype(np.int32),
            'condition_means': np.stack([
                neural_activity[low_mask].mean(axis=0),
                neural_activity[high_mask].mean(axis=0),
            ]).astype(np.float32),
            'condition_stds': np.stack([
                neural_activity[low_mask].std(axis=0),
                neural_activity[high_mask].std(axis=0),
            ]).astype(np.float32),
        },
    }
    return results
