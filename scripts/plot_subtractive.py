"""Plotting for the subtractive-analysis pipeline.

Inputs (Snakemake): list of data/subtractive_{regime}_{mode}_plot_data.npz
files (one per regime; all sharing the same mode).
Outputs:
    figures/subtractive_{regime}_{mode}.pdf      (per-regime 2x3 figure)
    figures/subtractive_headline_{mode}.pdf      (cross-regime comparison)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from analyses.plot_style import paper_style, COLORS, FULL_WIDTH, COL_WIDTH


REGIMES_ORDER = ['confounded', 'area_controlled']


def _load(path):
    d = dict(np.load(path, allow_pickle=False))
    d['regime'] = str(d['regime'])
    d['mode'] = str(d['mode']) if 'mode' in d else 'inferred'
    d['block_names'] = [str(s) for s in d['block_names']]
    d['grid_shape'] = tuple(int(x) for x in d['grid_shape'])
    return d


def _mode_label(mode):
    return {
        'ground_truth': 'ground-truth N',
        'inferred':     'inferential $\\hat N$',
    }.get(mode, mode)


def _block_color(name):
    return COLORS['physics'] if name == 'sensory' else COLORS['pixels']


def _draw_block_divider(ax, grid_shape, block_assignment, grid_positions):
    """Vertical line between sensory (left) and abstract (right) regions."""
    abstract_cols = grid_positions[block_assignment == 1, 1]
    if abstract_cols.size:
        col_split = abstract_cols.min() - 0.5
        ax.axvline(col_split, color='black', lw=0.8, alpha=0.6)


def _heatmap(ax, t_grid, *, vmax, mask=None, title=None):
    """t-stat heatmap on a (rows, cols) grid. Diverging cmap centered at 0."""
    arr = t_grid.copy()
    if mask is not None:
        arr = np.where(mask, arr, np.nan)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(arr, cmap='RdBu_r', norm=norm, aspect='equal',
                   interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return im


def _critical_t(alpha, df):
    """Two-tailed Welch's t critical value for given alpha and df."""
    return float(stats.t.ppf(1 - alpha / 2, df))


def _per_regime_figure(data, *, out_path):
    grid_shape = data['grid_shape']
    block_assignment = data['block_assignment']
    block_names = data['block_names']
    grid_positions = data['grid_positions']
    t_grid = data['t_grid']
    t_stats = data['t_stats']
    p_values = data['p_values']
    q_values = data['q_values_fdr']
    sweep_t = data['sweep_thresholds']
    sweep_counts = data['sweep_counts']
    val_r2 = float(data['cardinality_val_r2'])
    inf_low  = float(data['inferred_N_low_mean'])
    inf_high = float(data['inferred_N_high_mean'])
    regime = data['regime']
    mode = data['mode']

    # Layout grid as a heatmap; build masks for FDR / Bonferroni significance
    n_neurons = t_stats.size
    df_approx = n_neurons - 2  # rough; actual df varies per neuron with Welch
    sig_fdr = (q_values < 0.05)
    sig_bonf = (p_values < 0.05 / max(n_neurons, 1))

    sig_fdr_grid = np.zeros(grid_shape, dtype=bool)
    sig_bonf_grid = np.zeros(grid_shape, dtype=bool)
    for i, (r, c) in enumerate(grid_positions):
        sig_fdr_grid[int(r), int(c)] = sig_fdr[i]
        sig_bonf_grid[int(r), int(c)] = sig_bonf[i]

    vmax = float(np.nanmax(np.abs(t_grid)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    with paper_style():
        fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.62))
        gs = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.45)

        ax = fig.add_subplot(gs[0, 0])
        _heatmap(ax, t_grid, vmax=vmax, title='Welch $t$ (high $-$ low $N$)')
        _draw_block_divider(ax, grid_shape, block_assignment, grid_positions)
        ax.set_xlabel('grid col'); ax.set_ylabel('grid row')

        ax = fig.add_subplot(gs[0, 1])
        _heatmap(ax, t_grid, vmax=vmax, mask=sig_fdr_grid,
                 title=f'FDR $q<0.05$ (n={int(sig_fdr.sum())})')
        _draw_block_divider(ax, grid_shape, block_assignment, grid_positions)

        ax = fig.add_subplot(gs[0, 2])
        _heatmap(ax, t_grid, vmax=vmax, mask=sig_bonf_grid,
                 title=f'Bonferroni $p<0.05/N$ (n={int(sig_bonf.sum())})')
        _draw_block_divider(ax, grid_shape, block_assignment, grid_positions)

        # Continuous threshold sweep
        ax = fig.add_subplot(gs[1, 0])
        for b_idx, name in enumerate(block_names):
            ax.plot(sweep_t, sweep_counts[b_idx], lw=1.2,
                    color=_block_color(name), label=name)
        # Mark p<0.05/0.01/0.001 critical |t|
        for alpha, ls in zip([0.05, 0.01, 0.001], ['--', '-.', ':']):
            tc = _critical_t(alpha, df_approx)
            ax.axvline(tc, color='gray', lw=0.6, ls=ls, alpha=0.7)
            ax.text(tc, sweep_counts.max() * 0.95, f'p<{alpha}', rotation=90,
                    fontsize=5, color='gray', va='top')
        bonf_t = _critical_t(0.05 / max(n_neurons, 1), df_approx)
        ax.axvline(bonf_t, color='black', lw=0.6, ls='-', alpha=0.7)
        ax.text(bonf_t, sweep_counts.max() * 0.50, 'Bonf', rotation=90,
                fontsize=5, color='black', va='top')
        ax.set_xlabel('threshold $|t|$'); ax.set_ylabel('# neurons surviving')
        ax.legend(frameon=False)
        ax.set_title('Threshold sweep')

        # |t| histogram by block
        ax = fig.add_subplot(gs[1, 1])
        bins = np.linspace(0, np.abs(t_stats).max() + 1e-6, 40)
        for b_idx, name in enumerate(block_names):
            ax.hist(np.abs(t_stats[block_assignment == b_idx]),
                    bins=bins, alpha=0.6, color=_block_color(name),
                    label=name, edgecolor='white', linewidth=0.3)
        ax.set_xlabel('$|t|$'); ax.set_ylabel('# neurons')
        ax.set_title('$|t|$ distribution')
        ax.legend(frameon=False)

        # Bar chart: fraction significant per block per discrete threshold
        ax = fig.add_subplot(gs[1, 2])
        thr_labels = ['p<0.05', 'p<0.01', 'p<0.001', 'FDR', 'Bonf']
        thr_keys   = ['p<0.05', 'p<0.01', 'p<0.001', 'fdr_bh', 'bonferroni']
        # Recompute fractions from stored t_stats / p / q
        bonf_alpha = 0.05 / max(n_neurons, 1)
        masks = [
            p_values < 0.05,
            p_values < 0.01,
            p_values < 0.001,
            q_values < 0.05,
            p_values < bonf_alpha,
        ]
        x = np.arange(len(thr_labels))
        width = 0.4
        for offset, b_idx, name in [(- width / 2, 0, block_names[0]),
                                    (+ width / 2, 1, block_names[1])]:
            fracs = []
            for m in masks:
                b_mask = (block_assignment == b_idx)
                n_b = int(b_mask.sum())
                fracs.append(int((m & b_mask).sum()) / n_b if n_b else 0.0)
            ax.bar(x + offset, fracs, width, color=_block_color(name),
                   label=name, edgecolor='white', linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(thr_labels, rotation=15)
        ax.set_ylabel('fraction surviving')
        ax.set_title('Per-block fraction surviving')
        ax.legend(frameon=False)
        ax.set_ylim(0, 1.0)

        if mode == 'ground_truth':
            subtitle = f"abstract input: ground-truth N (oracle)"
        else:
            subtitle = (
                f"cognitive model val $R^2$ = {val_r2:.2f}  |  "
                f"$\\hat N$ low = {inf_low:.1f}, high = {inf_high:.1f}"
            )
        fig.suptitle(
            f"Subtractive analysis ({regime}, {_mode_label(mode)})  --  {subtitle}",
            fontsize=8, y=1.02,
        )
        fig.savefig(out_path)
        plt.close(fig)


def _headline_figure(per_regime, *, mode, out_path):
    """Cross-regime comparison: FDR-thresholded brain blob side by side."""
    n = len(per_regime)
    with paper_style():
        fig, axes = plt.subplots(1, n, figsize=(FULL_WIDTH, FULL_WIDTH * 0.45),
                                 squeeze=False)
        axes = axes[0]
        # Shared colorbar range across regimes
        vmax = max(
            float(np.nanmax(np.abs(d['t_grid']))) for d in per_regime.values()
        )
        for ax, regime in zip(axes, [r for r in REGIMES_ORDER
                                     if r in per_regime]):
            d = per_regime[regime]
            grid_shape = d['grid_shape']
            block_assignment = d['block_assignment']
            grid_positions = d['grid_positions']
            block_names = d['block_names']
            t_stats = d['t_stats']
            q_values = d['q_values_fdr']

            sig = q_values < 0.05
            sig_grid = np.zeros(grid_shape, dtype=bool)
            for i, (r, c) in enumerate(grid_positions):
                sig_grid[int(r), int(c)] = sig[i]

            _heatmap(ax, d['t_grid'], vmax=vmax, mask=sig_grid)
            _draw_block_divider(ax, grid_shape, block_assignment, grid_positions)
            # Per-block fractions for the inset bar
            fracs = []
            for b_idx, name in enumerate(block_names):
                b_mask = (block_assignment == b_idx)
                n_b = int(b_mask.sum())
                fracs.append(int((sig & b_mask).sum()) / n_b if n_b else 0.0)
            ax.set_title(
                f'{regime}\n'
                f'sensory: {fracs[0]:.0%}  abstract: {fracs[1]:.0%}\n'
                f'(FDR $q<0.05$)'
            )
            ax.set_xlabel('grid col')
            if ax is axes[0]:
                ax.set_ylabel('grid row')

        fig.suptitle(
            f'No threshold rescues the abstract block  ({_mode_label(mode)})',
            fontsize=9, y=1.04,
        )
        fig.savefig(out_path)
        plt.close(fig)


# ----- Main -------------------------------------------------------------

inputs = list(snakemake.input)  # noqa: F821
figure_outputs = list(snakemake.output.figures)  # noqa: F821
headline_output = snakemake.output.headline  # noqa: F821
mode_wc = snakemake.wildcards.mode  # noqa: F821

# Map each input plot_data to its regime-specific output by matching the
# regime substring; this avoids relying on Snakemake's input/output ordering.
loaded = [_load(p) for p in inputs]
per_regime = {d['regime']: d for d in loaded}

modes_seen = {d['mode'] for d in loaded}
assert modes_seen == {mode_wc}, (
    f"plot_subtractive expects all inputs to share mode={mode_wc!r}, "
    f"got {modes_seen}"
)

for fig_path in figure_outputs:
    # Filename pattern: figures/subtractive_<regime>_<mode>.pdf
    base = (
        os.path.basename(fig_path)
        .removeprefix('subtractive_')
        .removesuffix('.pdf')
        .removesuffix(f'_{mode_wc}')
    )
    d = per_regime[base]
    print(f"Plotting per-regime figure ({base}, {mode_wc}) -> {fig_path}")
    _per_regime_figure(d, out_path=fig_path)

print(f"Plotting headline cross-regime figure ({mode_wc}) -> {headline_output}")
_headline_figure(per_regime, mode=mode_wc, out_path=headline_output)

print("Done.")
