"""Plotting functions for all analyses. Separated from computation for fast iteration."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from analyses.plot_style import COLORS, COL_WIDTH, FULL_WIDTH, paper_style

# Model names
LABEL_SENSORY = "Sensory"
LABEL_PHYSICS = "Physics"
LABEL_SENSORY_PLUS_PHYSICS = "Full"


def _null_ci_from_perm_array(null_array, lo_pct=2.5, hi_pct=97.5):
    """Per-perm mean across neurons → (lo, mid, hi). Returns None if empty."""
    if null_array is None or null_array.size == 0:
        return None
    perm_means = null_array.mean(axis=1)
    lo, hi = np.percentile(perm_means, [lo_pct, hi_pct])
    return float(lo), float(perm_means.mean()), float(hi)


def _draw_null_marker(ax, bar, null_ci, *, label=None):
    """
    Render the null distribution at the bar's x-position as a horizontal mean
    line with vertical CI caps. Used because the typical null CI is thinner
    than a pixel on the bar's R² axis — a translucent rectangle would be
    invisible. Returns the artist used for the legend.
    """
    if null_ci is None:
        return None
    lo, mid, hi = null_ci
    x0 = bar.get_x()
    x1 = x0 + bar.get_width()
    color = "#222222"
    ax.plot(
        [x0, x1],
        [mid, mid],
        color=color,
        linewidth=0.9,
        zorder=4,
        solid_capstyle="butt",
    )
    cx = 0.5 * (x0 + x1)
    ax.plot([cx, cx], [lo, hi], color=color, linewidth=0.9, zorder=4)
    cap_w = 0.18 * (x1 - x0)
    for y in (lo, hi):
        ax.plot([cx - cap_w, cx + cap_w], [y, y], color=color, linewidth=0.9, zorder=4)
    proxy = ax.plot([], [], color=color, linewidth=0.9, label=label)[0]
    return proxy


def plot_encoding_bars(plot_data, fig_dir="figures"):
    """Five encoding bar plots focused on inferred-physics variance partitioning.

    Requires r2_P_inf and delta_P_inf_given_X in plot_data (produced when
    run_encoding.py has inferred_physics_labels). Silently skips if absent.
    """
    keys = plot_data.files if hasattr(plot_data, "files") else plot_data
    if "r2_P_inf" not in keys:
        return

    r2_X = plot_data["r2_X"]
    r2_P_inf = plot_data["r2_P_inf"]
    delta_P_inf = plot_data["delta_P_inf_given_X"]
    n_neurons = len(r2_X)
    sem = lambda x: float(x.std() / np.sqrt(n_neurons))

    mean_r2_X = float(r2_X.mean())
    mean_r2_P_inf = float(r2_P_inf.mean())
    mean_delta = float(delta_P_inf.mean())

    # Per-perm null means (n_perms,) derived from raw null arrays
    r2_P_inf_null_means = delta_null_means = None
    r2_P_inf_null_ci = delta_null_ci = None

    if "r2_P_inf_null" in keys:
        r2_P_inf_null = plot_data["r2_P_inf_null"]  # (n_perms, n_neurons)
        r2_P_inf_null_means = r2_P_inf_null.mean(axis=1)
        r2_P_inf_null_ci = np.percentile(r2_P_inf_null_means, [2.5, 97.5])

    if "delta_P_inf_given_X_null" in keys:
        delta_P_inf_null = plot_data["delta_P_inf_given_X_null"]
        delta_null_means = delta_P_inf_null.mean(axis=1)
        delta_null_ci = np.percentile(delta_null_means, [2.5, 97.5])

    bar_w = 0.5
    null_color = "#888888"

    def _bar_label(ax, bar, val, fmt="{:.3f}"):
        ylo, yhi = ax.get_ylim()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005 * (yhi - ylo),
            fmt.format(val),
            ha="center", va="bottom", fontweight="bold",
        )

    # --- Plot 1: r2_X vs r2_P_inf with permutation null ---
    with paper_style():
        fig, ax = plt.subplots(figsize=(COL_WIDTH * 0.65, 2.0))
        labels = [LABEL_SENSORY, "Inferred\nphysics"]
        heights = [mean_r2_X, mean_r2_P_inf]
        errs = [sem(r2_X), sem(r2_P_inf)]
        colors = [COLORS["sensory"], COLORS["control"]]
        bars = ax.bar(labels, heights, yerr=errs, color=colors, capsize=3, width=bar_w)
        if r2_P_inf_null_ci is not None:
            null_ci_full = (
                float(r2_P_inf_null_ci[0]),
                float(r2_P_inf_null_means.mean()),
                float(r2_P_inf_null_ci[1]),
            )
            _draw_null_marker(ax, bars[1], null_ci_full, label="Permutation null")
            ax.legend(loc="upper right", frameon=False, fontsize=5, handlelength=1.5)
        ax.set_ylabel("Mean R²")
        ax.set_title("Encoding R²: sensory vs. inferred physics")
        ax.set_ylim(0, max(heights) * 1.18)
        for bar, val in zip(bars, heights):
            _bar_label(ax, bar, val)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/encoding_r2_physics_null.pdf")
        plt.close()

    # --- Plot 2: r2_X vs delta_P_inf (unique inferred physics) ---
    with paper_style():
        fig, ax = plt.subplots(figsize=(COL_WIDTH * 0.65, 2.0))
        labels = [LABEL_SENSORY, "Unique\ninferred physics"]
        heights = [mean_r2_X, mean_delta]
        errs = [sem(r2_X), sem(delta_P_inf)]
        colors = [COLORS["sensory"], COLORS["control"]]
        bars = ax.bar(labels, heights, yerr=errs, color=colors, capsize=3, width=bar_w)
        ax.set_ylabel("Mean R²")
        ax.set_title("R²: sensory vs. unique inferred physics")
        ax.set_ylim(0, max(heights) * 1.18)
        for bar, val in zip(bars, heights):
            _bar_label(ax, bar, val)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/encoding_r2_unique_physics.pdf")
        plt.close()

    # --- Plot 3: unique inferred physics zoomed in, with null axhspan ---
    with paper_style():
        fig, ax = plt.subplots(figsize=(COL_WIDTH * 0.5, 2.0))
        ax.bar(
            ["Unique\ninferred physics"],
            [mean_delta],
            yerr=[sem(delta_P_inf)],
            color=COLORS["control"],
            capsize=3,
            width=bar_w,
            zorder=3,
        )
        if delta_null_ci is not None:
            lo, hi = float(delta_null_ci[0]), float(delta_null_ci[1])
            ax.axhspan(lo, hi, color=null_color, alpha=0.20, label="Null 95% CI", zorder=0)
            ax.axhline(
                float(delta_null_means.mean()),
                color=null_color, linewidth=0.8, linestyle="--", zorder=1,
            )
            ax.legend(loc="upper right", frameon=False, fontsize=5)
            pad = abs(hi - lo) * 0.8
            y_lo = min(lo, 0, mean_delta) - pad
            y_hi = max(hi, mean_delta + sem(delta_P_inf)) + pad * 3
            ax.set_ylim(y_lo, y_hi)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylabel("Mean ΔR²")
        ax.set_title("Unique inferred-physics R²\nvs. permutation null")
        ax.text(
            0, mean_delta + sem(delta_P_inf) * 1.5,
            f"{mean_delta:.2e}", ha="center", va="bottom", fontsize=6,
        )
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/encoding_r2_unique_physics_zoomed.pdf")
        plt.close()

    # --- Plot 4: density of r2_P_inf null + observed ---
    if r2_P_inf_null_means is not None:
        with paper_style():
            fig, ax = plt.subplots(figsize=(COL_WIDTH, 1.9))
            ax.hist(
                r2_P_inf_null_means,
                bins=12,
                color=null_color,
                alpha=0.65,
                density=True,
                label="Null distribution",
            )
            ax.axvline(
                mean_r2_P_inf,
                color=COLORS["control"],
                linewidth=1.2,
                label=f"Observed = {mean_r2_P_inf:.3f}",
            )
            ax.set_xlabel("Mean R² per permutation")
            ax.set_ylabel("Density")
            ax.set_title("Null distribution: R²(Pᴵⁿᶠ)")
            ax.legend(frameon=False, fontsize=5)
            plt.tight_layout()
            plt.savefig(f"{fig_dir}/encoding_null_r2_physics_density.pdf")
            plt.close()

    # --- Plot 5: density of delta_P_inf null + observed ---
    if delta_null_means is not None:
        with paper_style():
            fig, ax = plt.subplots(figsize=(COL_WIDTH, 1.9))
            ax.hist(
                delta_null_means,
                bins=12,
                color=null_color,
                alpha=0.65,
                density=True,
                label="Null distribution",
            )
            ax.axvline(
                mean_delta,
                color=COLORS["control"],
                linewidth=1.2,
                label=f"Observed = {mean_delta:.2e}",
            )
            ax.set_xlabel("Mean ΔR² per permutation")
            ax.set_ylabel("Density")
            ax.set_title("Null distribution: unique R²(Pᴵⁿᶠ)")
            ax.legend(frameon=False, fontsize=5)
            plt.tight_layout()
            plt.savefig(f"{fig_dir}/encoding_null_delta_density.pdf")
            plt.close()


def plot_encoding(plot_data, fig_dir="figures"):
    r2_pixel_only = plot_data["r2_pixel_only"]
    r2_combined = plot_data["r2_combined"]
    n_neurons = len(r2_pixel_only)
    mean_r2_pixel = r2_pixel_only.mean()
    mean_r2_comb = r2_combined.mean()
    mean_delta = (r2_combined - r2_pixel_only).mean()
    neuron_counts = plot_data["neuron_counts"]
    subsample_means = plot_data["subsample_means"]
    subsample_sems = plot_data["subsample_sems"]
    control_acc = float(plot_data["control_accuracy"])
    control_acc_std = float(plot_data["control_accuracy_std"])

    keys = plot_data.files if hasattr(plot_data, "files") else plot_data
    has_predicted = "r2_predicted_pixel" in keys
    r2_predicted = plot_data["r2_predicted_pixel"] if has_predicted else None

    with paper_style():
        fig, axes = plt.subplots(3, 1, figsize=(COL_WIDTH, 5.5))

        # Panel A: R² bar plot
        ax = axes[0]
        bar_labels = [LABEL_SENSORY, LABEL_SENSORY_PLUS_PHYSICS]
        bar_heights = [mean_r2_pixel, mean_r2_comb]
        bar_errs = [
            r2_pixel_only.std() / np.sqrt(n_neurons),
            r2_combined.std() / np.sqrt(n_neurons),
        ]
        bar_colors = [COLORS["sensory"], COLORS["physics"]]
        # if has_predicted:
        #     bar_labels.append("Predicted S")
        #     bar_heights.append(float(r2_predicted.mean()))
        #     bar_errs.append(float(r2_predicted.std() / np.sqrt(n_neurons)))
        #     bar_colors.append(COLORS["control"])
        ax.bar(
            bar_labels,
            bar_heights,
            yerr=bar_errs,
            color=bar_colors,
            capsize=3,
            width=0.6,
        )
        ax.set_ylabel("Mean R\u00b2")
        ax.set_title("Encoding model: R\u00b2 \u00b1 physics labels")
        ymax = max(bar_heights) * 1.12
        ax.annotate(
            f"\u0394R\u00b2 = {mean_delta:.6f}",
            xy=(0.5, ymax),
            ha="center",
            style="italic",
        )

        # Panel B: Subsampling curve
        ax = axes[1]
        ax.errorbar(
            neuron_counts,
            subsample_means,
            yerr=subsample_sems,
            marker="o",
            color=COLORS["sensory"],
            capsize=2,
        )
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Number of neurons sampled")
        ax.set_ylabel("Mean \u0394R\u00b2")
        ax.set_title("\u0394R\u00b2 vs. neuron subsampling")

        # Panel C: Control accuracy
        ax = axes[2]
        ax.bar(
            ["Physics \u2192 Behavior"],
            [control_acc],
            yerr=[control_acc_std],
            color=COLORS["control"],
            capsize=3,
            width=0.5,
        )
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
        ax.set_ylabel("Accuracy")
        ax.set_title("Control: physics labels predict behavior")
        ax.set_ylim(0, 1)
        ax.legend()

        fig.align_ylabels(axes)
        plt.tight_layout()
        fig_path = f"{fig_dir}/encoding_analysis.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_rsa(plot_data, fig_dir="figures"):
    rdm_neural = plot_data["rdm_neural"]
    rdm_pixel = plot_data["rdm_X"]
    rdm_physics_gt = plot_data["rdm_physics"]
    n_sub = int(plot_data["n_sub"])
    corr_neural_pixel = float(plot_data["corr_neural_X"])
    corr_neural_physics_gt = float(plot_data["corr_neural_P"])

    keys = plot_data.files if hasattr(plot_data, "files") else plot_data
    has_inferred = "rdm_physics_inf" in keys
    rdm_physics_inf = plot_data["rdm_physics_inf"] if has_inferred else None
    corr_neural_physics_inf = (
        float(plot_data["corr_neural_P_inf"]) if has_inferred else None
    )
    has_predicted = "rdm_predicted" in keys
    rdm_predicted_flat = plot_data["rdm_predicted"] if has_predicted else None
    corr_neural_predicted = (
        float(plot_data["corr_neural_predicted"]) if has_predicted else None
    )

    with paper_style():
        # Reorder scenes via hierarchical clustering on pixel RDM
        # so shared structure between Neural and Sensory is visible
        rdm_pixel_full = squareform(rdm_pixel)
        Z = linkage(rdm_pixel, method="average")
        order = leaves_list(Z)

        n_show = min(40, n_sub)
        order = order[:n_show]
        rdm_neural_sq = squareform(rdm_neural)[np.ix_(order, order)]
        rdm_pixel_sq = rdm_pixel_full[np.ix_(order, order)]
        rdm_physics_gt_sq = squareform(rdm_physics_gt)[np.ix_(order, order)]
        rdm_physics_inf_sq = (
            squareform(rdm_physics_inf)[np.ix_(order, order)] if has_inferred else None
        )

        all_rdms = [rdm_neural_sq, rdm_pixel_sq, rdm_physics_gt_sq]
        if has_inferred:
            all_rdms.append(rdm_physics_inf_sq)
        vmin = min(r.min() for r in all_rdms)
        vmax = max(r.max() for r in all_rdms)

        # 3 rows: top = Neural | cbar | Sensory; mid = GT Phys | _ | Inferred Phys;
        # bottom = bar chart spanning all columns
        fig = plt.figure(figsize=(COL_WIDTH, COL_WIDTH * 1.35))
        gs = fig.add_gridspec(
            3,
            3,
            width_ratios=[1, 0.05, 1],
            height_ratios=[1, 1, 0.8],
            hspace=0.5,
            wspace=0.15,
        )

        ax_nr = fig.add_subplot(gs[0, 0])
        im = ax_nr.imshow(
            rdm_neural_sq, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax
        )
        ax_nr.set_title("Neural RDM")
        ax_nr.set_xticks([])
        ax_nr.set_yticks([])

        ax_rr = fig.add_subplot(gs[0, 2])
        ax_rr.imshow(rdm_pixel_sq, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        ax_rr.set_title(f"{LABEL_SENSORY} RDM")
        ax_rr.set_xticks([])
        ax_rr.set_yticks([])

        cax = fig.add_subplot(gs[0, 1])
        cb = fig.colorbar(im, cax=cax, orientation="vertical")
        cb.ax.tick_params(labelsize=4, length=1.5, pad=1)
        cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cb.set_ticklabels([f"{vmin:.1f}", f"{(vmin+vmax)/2:.1f}", f"{vmax:.1f}"])

        ax_pg = fig.add_subplot(gs[1, 0])
        ax_pg.imshow(
            rdm_physics_gt_sq, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax
        )
        ax_pg.set_title(f"{LABEL_PHYSICS} RDM (GT)")
        ax_pg.set_xticks([])
        ax_pg.set_yticks([])

        if has_inferred:
            ax_pi = fig.add_subplot(gs[1, 2])
            ax_pi.imshow(
                rdm_physics_inf_sq, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax
            )
            ax_pi.set_title(f"{LABEL_PHYSICS} RDM (inferred)")
            ax_pi.set_xticks([])
            ax_pi.set_yticks([])

        ax = fig.add_subplot(gs[2, :])
        labels = [
            f"Neural\u2013\n{LABEL_SENSORY}",
            f"Neural\u2013\n{LABEL_PHYSICS}\n(GT)",
        ]
        values = [corr_neural_pixel, corr_neural_physics_gt]
        colors = [COLORS["sensory"], COLORS["physics"]]
        if has_inferred:
            labels.append(f"Neural\u2013\n{LABEL_PHYSICS}\n(inferred)")
            values.append(corr_neural_physics_inf)
            colors.append(COLORS["control"])
        if has_predicted:
            labels.append(f"Neural\u2013\nPredicted S")
            values.append(corr_neural_predicted)
            colors.append("#E67E22")
        bars = ax.bar(labels, values, color=colors, width=0.6)
        ax.set_ylabel("Spearman r")
        ax.set_title("RSA correlations")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )

        fig_path = f"{fig_dir}/rsa_analysis.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_dissociation(plot_data, fig_dir="figures"):
    r2_pixel = plot_data["r2_pixel"]
    r2_physics = plot_data["r2_physics"]
    n_neurons = len(r2_pixel)
    mean_r2_pixel = r2_pixel.mean()
    mean_r2_physics = r2_physics.mean()
    pixel_score = float(plot_data["pixel_score"])
    physics_score = float(plot_data["physics_score"])
    metric_label = str(plot_data["metric_label"])
    chance_val = float(plot_data["chance"])
    chance = None if np.isnan(chance_val) else chance_val

    keys = plot_data.files if hasattr(plot_data, "files") else plot_data
    has_predicted = "r2_predicted_pixel" in keys and "predicted_pixel_score" in keys
    has_inferred = "inferred_physics_score" in keys and not np.isnan(
        float(plot_data["inferred_physics_score"])
    )
    physics_null_ci = (
        _null_ci_from_perm_array(plot_data["r2_physics_null"])
        if "r2_physics_null" in keys
        else None
    )
    if has_predicted:
        r2_predicted = plot_data["r2_predicted_pixel"]
        predicted_score = float(plot_data["predicted_pixel_score"])
        mean_r2_predicted = r2_predicted.mean()
    if has_inferred:
        inferred_physics_score = float(plot_data["inferred_physics_score"])

    with paper_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_WIDTH, 2.0))

        bar_width = 0.5
        r2_colors = [COLORS["sensory"], COLORS["physics"]]
        r2_labels = [LABEL_SENSORY, LABEL_PHYSICS]
        r2_heights = [mean_r2_pixel, mean_r2_physics]
        r2_errs = [
            r2_pixel.std() / np.sqrt(n_neurons),
            r2_physics.std() / np.sqrt(n_neurons),
        ]
        behav_colors = [COLORS["sensory"], COLORS["physics"]]
        behav_labels = [LABEL_SENSORY, LABEL_PHYSICS]
        behav_heights = [pixel_score, physics_score]
        # if has_predicted:
        #     r2_colors.append("#E67E22")
        #     r2_labels.append("Predicted S")
        #     r2_heights.append(mean_r2_predicted)
        #     r2_errs.append(r2_predicted.std() / np.sqrt(n_neurons))
        #     behav_colors.append("#E67E22")
        #     behav_labels.append("Predicted S")
        #     behav_heights.append(predicted_score)
        if has_inferred:
            # Inferred physics has no separate neural R² in dissociation data
            # (it's the injected feature; neural R² lives in PP analysis), so
            # the bar appears only on the behavioral panel — a hatched fill
            # distinguishes it visually from the GT-physics oracle bar.
            behav_colors.append(COLORS["physics"])
            behav_labels.append("Physics\n(inferred)")
            behav_heights.append(inferred_physics_score)

        bars1 = ax1.bar(
            r2_labels,
            r2_heights,
            width=bar_width,
            color=r2_colors,
            yerr=r2_errs,
            capsize=3,
        )
        ax1.set_ylabel("Neural\nR\u00b2", rotation=0, labelpad=20)
        ax1.set_title("Encoding performance")
        print(physics_null_ci)
        # ax1.axvspan(physics_null_ci[0], physics_null_ci[2], color="#222222", alpha=0.15)
        # _draw_null_marker(ax1, bars1[1], physics_null_ci, label="Permutation null")
        # if physics_null_ci is not None:
        #     ax1.legend(
        #         loc="center right",
        #         frameon=False,
        #         fontsize=5,
        #         handlelength=1.5,
        #         borderpad=0.2,
        #     )
        #     ymin, ymax = ax1.get_ylim()
        #     if physics_null_ci[0] < ymin:
        #         ax1.set_ylim(physics_null_ci[0] - 0.02 * (ymax - ymin), ymax)
        for bar, val in zip(bars1, r2_heights):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        bars2 = ax2.bar(
            behav_labels,
            behav_heights,
            width=bar_width,
            color=behav_colors,
            capsize=3,
        )
        if has_inferred:
            # Hatch the inferred-physics bar to distinguish it from the GT
            # oracle (same color, different surface).
            bars2[-1].set_hatch("//")
            bars2[-1].set_edgecolor("white")
        if chance is not None:
            ax2.axhline(chance, color="gray", linestyle="--", alpha=0.5, label="Chance")
            ax2.set_ylim(0, 1.1)
            ax2.legend()
        ax2.set_ylabel(metric_label)
        ax2.set_title("Computational sufficiency")
        for bar, val in zip(bars2, behav_heights):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        fig_path = f"{fig_dir}/dissociation.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_dissociation_combined(plot_data, fig_dir="figures"):
    """Dissociation figure comparing pixel vs. pixel+physics models.

    The two encoding bars are nearly identical height, showing that adding
    physics to the pixel model barely changes neural R² — even though
    physics dramatically improves behavioral prediction.
    """
    r2_pixel = plot_data["r2_pixel"]
    r2_combined = plot_data["r2_combined"]
    n_neurons = len(r2_pixel)
    mean_r2_pixel = r2_pixel.mean()
    mean_r2_combined = r2_combined.mean()
    pixel_score = float(plot_data["pixel_score"])
    combined_score = float(plot_data["combined_score"])
    metric_label = str(plot_data["metric_label"])
    chance_val = float(plot_data["chance"])
    chance = None if np.isnan(chance_val) else chance_val
    keys = plot_data.files if hasattr(plot_data, "files") else plot_data
    combined_null_ci = (
        _null_ci_from_perm_array(plot_data["r2_combined_null"])
        if "r2_combined_null" in keys
        else None
    )

    with paper_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_WIDTH, 2.0))

        bar_width = 0.5
        colors = [COLORS["sensory"], COLORS["combined"]]
        labels = [LABEL_SENSORY, LABEL_SENSORY_PLUS_PHYSICS]

        bars1 = ax1.bar(
            labels,
            [mean_r2_pixel, mean_r2_combined],
            width=bar_width,
            color=colors,
            yerr=[
                r2_pixel.std() / np.sqrt(n_neurons),
                r2_combined.std() / np.sqrt(n_neurons),
            ],
            capsize=3,
        )
        ax1.set_ylabel("Neural R\u00b2")
        ax1.set_title("Encoding performance")
        _draw_null_marker(ax1, bars1[1], combined_null_ci, label="Permutation null")
        if combined_null_ci is not None:
            ax1.legend(
                loc="center right",
                frameon=False,
                fontsize=5,
                handlelength=1.5,
                borderpad=0.2,
            )
        for bar, val in zip(bars1, [mean_r2_pixel, mean_r2_combined]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        bars2 = ax2.bar(
            labels,
            [pixel_score, combined_score],
            width=bar_width,
            color=colors,
            capsize=3,
        )
        if chance is not None:
            ax2.axhline(chance, color="gray", linestyle="--", alpha=0.5, label="Chance")
            ax2.set_ylim(0, 1.1)
            ax2.legend()
        ax2.set_ylabel(metric_label)
        ax2.set_title("Computational sufficiency")
        for bar, val in zip(bars2, [pixel_score, combined_score]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        fig_path = f"{fig_dir}/dissociation_combined.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_predicted_frames(plot_data, fig_dir="figures"):
    """Predicted frames grid — uses figure* (full width) for legibility."""
    init_imgs = plot_data["predicted_init_imgs"]
    pixel_imgs = plot_data["predicted_pixel_imgs"]
    physics_imgs = plot_data["predicted_physics_imgs"]
    final_imgs = plot_data["predicted_final_imgs"]
    n = len(init_imgs)

    keys = plot_data.files if hasattr(plot_data, "files") else plot_data
    fwd_imgs = plot_data["predicted_fwd_imgs"] if "predicted_fwd_imgs" in keys else None

    col_titles = [
        "t=0 (input)",
        f"{LABEL_SENSORY} model\nprediction",
        f"{LABEL_PHYSICS} model\nprediction",
        "t=N (actual)",
    ]
    cols = [init_imgs, pixel_imgs, physics_imgs, final_imgs]
    if fwd_imgs is not None:
        col_titles.insert(3, "Cog. fwd model\n(late frame)")
        cols.insert(3, fwd_imgs[:n])

    n_cols = len(cols)
    with paper_style():
        fig, axes = plt.subplots(n, n_cols, figsize=(FULL_WIDTH * n_cols / 4, 1.6 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for col_idx, (title, imgs) in enumerate(zip(col_titles, cols)):
            axes[0, col_idx].set_title(title)
            for row_idx in range(n):
                axes[row_idx, col_idx].imshow(imgs[row_idx])
                axes[row_idx, col_idx].axis("off")

        plt.tight_layout(pad=0.3)
        fig_path = f"{fig_dir}/predicted_frames.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_predicted_frames_compact(plot_data, fig_dir="figures", scene_idx=1):
    """Predicted frames as a 2x2 grid for a single-column figure."""
    init_imgs = plot_data["predicted_init_imgs"]
    pixel_imgs = plot_data["predicted_pixel_imgs"]
    physics_imgs = plot_data["predicted_physics_imgs"]
    final_imgs = plot_data["predicted_final_imgs"]

    titles = [
        "t=0 (input)",
        f"{LABEL_SENSORY} model\nprediction",
        f"{LABEL_PHYSICS} model\nprediction",
        "t=N (actual)",
    ]
    imgs = [
        init_imgs[scene_idx],
        pixel_imgs[scene_idx],
        physics_imgs[scene_idx],
        final_imgs[scene_idx],
    ]

    with paper_style():
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(COL_WIDTH, COL_WIDTH * 0.95),
            gridspec_kw={"hspace": 0.15, "wspace": 0.05},
        )

        for ax, title, img in zip(axes.flat, titles, imgs):
            ax.imshow(img)
            ax.set_title(title, pad=2)
            ax.axis("off")

        fig_path = f"{fig_dir}/predicted_frames_compact.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_dynamics(plot_data, fig_dir="figures"):
    r2_physics_forward = plot_data["r2_physics_forward"]
    r2_pixel_forward = plot_data["r2_pixel_forward"]
    r2_inferred_forward = (
        plot_data["r2_inferred_forward"] if "r2_inferred_forward" in plot_data else None
    )
    n_neurons = len(r2_physics_forward)
    mean_r2_physics = r2_physics_forward.mean()
    mean_r2_pixel = r2_pixel_forward.mean()

    sem = lambda x: x.std() / np.sqrt(n_neurons)

    with paper_style():
        fig, ax1 = plt.subplots(1, 1, figsize=(COL_WIDTH * 0.7, 2.2))

        bar_width = 0.5
        if r2_inferred_forward is not None:
            mean_r2_inferred = r2_inferred_forward.mean()
            colors = [COLORS["sensory"], COLORS["control"], COLORS["physics"]]
            labels = [LABEL_SENSORY, "Inferred\nphysics", f"{LABEL_PHYSICS}\n(GT)"]
            heights = [mean_r2_pixel, mean_r2_inferred, mean_r2_physics]
            errs = [
                sem(r2_pixel_forward),
                sem(r2_inferred_forward),
                sem(r2_physics_forward),
            ]
        else:
            colors = [COLORS["sensory"], COLORS["physics"]]
            labels = [LABEL_SENSORY, LABEL_PHYSICS]
            heights = [mean_r2_pixel, mean_r2_physics]
            errs = [sem(r2_pixel_forward), sem(r2_physics_forward)]

        bars1 = ax1.bar(
            labels, heights, width=bar_width, color=colors, yerr=errs, capsize=3
        )
        ax1.set_ylabel("Future neural R\u00b2")
        ax1.set_title("Future brain state prediction")
        for bar, val in zip(bars1, heights):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        fig_path = f"{fig_dir}/dynamics_analysis.pdf"
        plt.savefig(fig_path)
        plt.close()


# Decoding targets shown on the PCA panels. Order controls draw order
# and legend order: positive controls first (largest pixel footprint at
# top), negative control (motion) last in physics red.
PCA_TARGET_STYLE = [
    ("cam_height", "Camera height", "#6ACC65"),  # green
    ("pillar_gray", "Pillar gray", "#9B59B6"),  # purple
    ("motion_dir", "Motion direction", "#D65F5F"),  # physics red
]


def _draw_pca_decoding(ax, pc_counts, plot_data):
    """Plot per-target decoding curves, each with its own chance band."""
    for name, label, color in PCA_TARGET_STYLE:
        key = f"decode_accs__{name}"
        if key not in plot_data:
            continue
        accs = plot_data[key]
        lo_key, hi_key = f"chance_lo__{name}", f"chance_hi__{name}"
        if lo_key in plot_data and hi_key in plot_data:
            ax.fill_between(
                pc_counts,
                plot_data[lo_key],
                plot_data[hi_key],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        linestyle = "--" if name == "motion_dir" else "-"
        ax.plot(
            pc_counts,
            accs,
            marker="o",
            markersize=3,
            color=color,
            linestyle=linestyle,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))


def plot_pca(plot_data, fig_dir="figures"):
    cumvar = plot_data["cumvar"]
    neural_pca_2d = plot_data["neural_pca_2d"]
    motion_dir = plot_data["motion_dir"]
    pc_counts = plot_data["pc_counts"].astype(int)
    n_neurons = int(plot_data["n_neurons"])
    motion_decode_accs = plot_data["decode_accs__motion_dir"]
    all_pc_acc = float(motion_decode_accs[-1])
    pc1, pc2 = neural_pca_2d[:, 0], neural_pca_2d[:, 1]

    # Figure 1: elbow + scatter + multi-target decoding (stacked vertically)
    with paper_style():
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(COL_WIDTH, 6.0))

        # Elbow + motion decoding reference line
        ax1.plot(range(1, n_neurons + 1), cumvar, color=COLORS["sensory"])
        ax1.axhline(
            all_pc_acc,
            color=COLORS["physics"],
            linestyle="--",
            linewidth=1.0,
            label=f"Motion decoding (all PCs): {all_pc_acc:.1%}",
        )
        ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance (50%)")
        ax1.set_xlabel("Number of principal components")
        ax1.set_ylabel("Cumulative explained variance")
        ax1.set_title("PCA elbow plot + motion decoding")
        ax1.legend()
        ax1.set_xlim(1, n_neurons)

        # PC1/PC2 scatter colored by motion direction
        lo, hi = 1, 99
        pc1_lim = np.percentile(pc1, [lo, hi])
        pc2_lim = np.percentile(pc2, [lo, hi])
        colors = np.where(motion_dir == 1, COLORS["physics"], COLORS["sensory"])
        ax2.scatter(pc1, pc2, c=colors, alpha=0.3, s=4, edgecolors="none")
        pad1 = 0.05 * (pc1_lim[1] - pc1_lim[0])
        pad2 = 0.05 * (pc2_lim[1] - pc2_lim[0])
        ax2.set_xlim(pc1_lim[0] - pad1, pc1_lim[1] + pad1)
        ax2.set_ylim(pc2_lim[0] - pad2, pc2_lim[1] + pad2)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("PC1 vs PC2 (colored by motion direction)")
        ax2.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=COLORS["sensory"],
                    markersize=5,
                    label="Left",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=COLORS["physics"],
                    markersize=5,
                    label="Right",
                ),
            ]
        )

        # Multi-target decoding sweep: sensory positive controls + motion
        _draw_pca_decoding(ax3, pc_counts, plot_data)
        ax3.set_xlabel("Number of principal components")
        ax3.set_ylabel("Decoding accuracy")
        ax3.set_title("Decoding from top-k PCs")
        ax3.legend(loc="center right", frameon=False)

        fig.align_ylabels([ax1, ax2, ax3])
        plt.tight_layout()
        fig_path = f"{fig_dir}/pca_analysis.pdf"
        plt.savefig(fig_path)
        plt.close()

    # Figure 2: elbow + multi-target decoding overlay with twinx
    with paper_style():
        fig, ax_var = plt.subplots(figsize=(COL_WIDTH, 2.0))
        ax_dec = ax_var.twinx()

        ax_var.plot(
            range(1, n_neurons + 1),
            cumvar,
            color=COLORS["neutral"],
            linewidth=0.8,
            label="Cumulative variance",
        )
        ax_var.set_xlabel("Number of principal components")
        ax_var.set_ylabel("Cumulative explained variance", color=COLORS["neutral"])
        ax_var.tick_params(axis="y", labelcolor=COLORS["neutral"])
        ax_var.set_xlim(1, n_neurons)
        ax_var.set_ylim(0, 1.05)
        ax_var.set_xscale("log")
        ax_var.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

        _draw_pca_decoding(ax_dec, pc_counts, plot_data)
        ax_dec.set_ylabel("Decoding accuracy")

        lines_var, labels_var = ax_var.get_legend_handles_labels()
        lines_dec, labels_dec = ax_dec.get_legend_handles_labels()
        ax_var.legend(
            lines_var + lines_dec,
            labels_var + labels_dec,
            loc="center right",
            frameon=False,
            fontsize=5,
        )

        fig_path2 = f"{fig_dir}/pca_variance_decoding.pdf"
        plt.savefig(fig_path2)
        plt.close()


def plot_sample_scenes(
    initial_renders,
    target_renders,
    rgba_bytes,
    image_size,
    n_timesteps,
    fig_dir="figures",
    n_samples=6,
    fwd_renders=None,
):
    n = min(n_samples, len(initial_renders))
    n_cols = 3 if fwd_renders is not None else 2
    with paper_style():
        fig, axes = plt.subplots(n, n_cols, figsize=(COL_WIDTH * n_cols / 2, 1.6 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        axes[0, 0].set_title("t = 0 (actual)")
        axes[0, 1].set_title(f"t = {n_timesteps} (target)")
        if fwd_renders is not None:
            axes[0, 2].set_title("Fwd model (t=0 pred)")

        for i in range(n):
            init_rgba = (
                initial_renders[i, :rgba_bytes]
                .astype(np.uint8)
                .reshape(image_size, image_size, 4)
            )
            target_rgba = (
                target_renders[i, :rgba_bytes]
                .astype(np.uint8)
                .reshape(image_size, image_size, 4)
            )
            axes[i, 0].imshow(init_rgba)
            axes[i, 0].axis("off")
            axes[i, 1].imshow(target_rgba)
            axes[i, 1].axis("off")
            if fwd_renders is not None:
                axes[i, 2].imshow(fwd_renders[i])
                axes[i, 2].axis("off")

        plt.tight_layout(pad=0.3)
        fig_path = f"{fig_dir}/sample_scenes.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_residual(plot_data, fig_dir="figures"):
    r2_P_given_X = plot_data["r2_P_given_X"]
    var_kept_X = float(plot_data["residual_variance_fraction_X"])
    n = len(r2_P_given_X)

    keys = plot_data.files if hasattr(plot_data, "files") else plot_data
    has_XS = "r2_P_given_XS" in keys
    r2_P_given_XS = plot_data["r2_P_given_XS"] if has_XS else None
    var_kept_XS = float(plot_data["residual_variance_fraction_XS"]) if has_XS else None

    # Pre-residualization baseline R²(P → neural), from encoding analysis.
    has_pre = "r2_P_neural" in keys
    r2_P_neural = plot_data["r2_P_neural"] if has_pre else None

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4))

        # Panel A: per-neuron scatter — raw R²(P) vs R²(P|X,S)
        ax = axes[0]
        if has_pre and has_XS:
            x_vals, y_vals = r2_P_neural, r2_P_given_XS
            ax.scatter(
                x_vals,
                y_vals,
                s=6,
                alpha=0.5,
                color=COLORS["physics"],
                edgecolors="none",
            )
            lo = float(min(x_vals.min(), y_vals.min()))
            hi = float(max(x_vals.max(), y_vals.max()))
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            ax.plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                color="gray",
                linestyle="--",
                linewidth=0.8,
                label="y = x",
            )
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel("R² (P, raw)")
            ax.set_ylabel("R² (P | X+S-residual neural)")
            ax.set_title("Physics collapses after removing X+S")
            ax.legend(loc="upper left")
        elif has_XS:
            ax.scatter(
                r2_P_given_X,
                r2_P_given_XS,
                s=6,
                alpha=0.5,
                color=COLORS["physics"],
                edgecolors="none",
            )
            lo = float(min(r2_P_given_X.min(), r2_P_given_XS.min()))
            hi = float(max(r2_P_given_X.max(), r2_P_given_XS.max()))
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            ax.plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                color="gray",
                linestyle="--",
                linewidth=0.8,
                label="y = x",
            )
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel("R² (P | X-residual neural)")
            ax.set_ylabel("R² (P | X+S-residual neural)")
            ax.set_title("Physics collapses after removing X+S")
            ax.legend(loc="upper left")
        else:
            n_bins = max(20, n // 50)
            ax.hist(r2_P_given_X, bins=n_bins, color=COLORS["physics"], alpha=0.7)
            ax.axvline(0, color="gray", linestyle=":", linewidth=0.6)
            ax.set_xlabel("R² (P | X-residual neural)")
            ax.set_ylabel("Neuron count")
            ax.set_title(
                f"Physics R² after X-residualization\n(var kept={var_kept_X:.2f})"
            )

        # Panel B: mean physics R² — R²(P|X), R²(P) raw, R²(P|X,S)
        ax = axes[1]
        labels = ["R²(P|X)"]
        means = [r2_P_given_X.mean()]
        sems = [r2_P_given_X.std() / np.sqrt(n)]
        bar_colors = [COLORS["sensory"]]

        if has_pre:
            labels.append("R²(P)")
            means.append(r2_P_neural.mean())
            sems.append(r2_P_neural.std() / np.sqrt(n))
            bar_colors.append(COLORS["physics"])

        if has_XS:
            labels.append("R²(P|X,S)")
            means.append(r2_P_given_XS.mean())
            sems.append(r2_P_given_XS.std() / np.sqrt(n))
            bar_colors.append(COLORS["control"])

        bars = ax.bar(
            labels,
            means,
            yerr=sems,
            color=bar_colors,
            capsize=3,
            width=0.5,
        )

        title_parts = []
        if has_pre:
            raw_mean = float(r2_P_neural.mean())
            title_parts.append(f"R²(P)={raw_mean:.4f}")
        if has_XS:
            title_parts.append(f"var kept: X={var_kept_X:.2f}, XS={var_kept_XS:.2f}")
        else:
            title_parts.append(f"var kept={var_kept_X:.2f}")
        ax.set_title("  ".join(title_parts))

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_ylabel("Mean R²")
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        plt.tight_layout()
        fig_path = f"{fig_dir}/residual_analysis.pdf"
        plt.savefig(fig_path)
        plt.close()


# Condition styling for the residualized-PCA motion decoding figure.
RESIDUALIZED_PCA_STYLE = [
    ("raw", "Neural (raw)", "physics", "-"),
    ("resid_X", "Neural | X-residual", "control", "--"),
    ("resid_XS", "Neural | X+S-residual", "neutral", ":"),
    ("pixel", "Pixels (positive control)", "sensory", "-"),
]


def plot_residualized_pca(plot_data, fig_dir="figures"):
    """Motion decodability across residualization conditions.

    Shows that motion decoding from neural PCs (which rises with #PCs) collapses
    to chance once the pixel-explainable component is regressed out, while a
    direct pixel-PC decode stays high — i.e. the decodability was a render
    confound. See analyses/residualized_pca.py.
    """
    keys = plot_data.files if hasattr(plot_data, "files") else plot_data
    present = [c for (c, _, _, _) in RESIDUALIZED_PCA_STYLE
               if f"decode__{c}__motion_dir" in keys]

    with paper_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4))

        # Panel A: motion decoding vs #PCs, one curve per condition.
        # Reference chance band from the raw-neural permutation null.
        if "chance_lo__raw__motion_dir" in keys:
            ax1.fill_between(
                plot_data["pc_counts__raw"],
                plot_data["chance_lo__raw__motion_dir"],
                plot_data["chance_hi__raw__motion_dir"],
                color="gray",
                alpha=0.15,
                linewidth=0,
                label="Chance (null)",
            )
        for cond, label, color_key, linestyle in RESIDUALIZED_PCA_STYLE:
            key = f"decode__{cond}__motion_dir"
            if key not in keys:
                continue
            ax1.plot(
                plot_data[f"pc_counts__{cond}"],
                plot_data[key],
                marker="o",
                markersize=3,
                color=COLORS[color_key],
                linestyle=linestyle,
                label=label,
            )
        ax1.set_xscale("log")
        ax1.set_ylim(0.4, 1.02)
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax1.set_xlabel("Number of principal components")
        ax1.set_ylabel("Motion decoding accuracy")
        ax1.set_title("Motion decodability is a render confound")
        ax1.legend(loc="upper left", frameon=False, fontsize=5)

        # Panel B: all-PC motion accuracy per condition (headline bars).
        labels, means, bar_colors = [], [], []
        for cond, label, color_key, _ in RESIDUALIZED_PCA_STYLE:
            key = f"decode__{cond}__motion_dir"
            if key not in keys:
                continue
            labels.append(label.replace(" ", "\n"))
            means.append(float(plot_data[key][-1]))
            bar_colors.append(COLORS[color_key])
        bars = ax2.bar(labels, means, color=bar_colors, width=0.6)
        ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, label="Chance")
        ax2.set_ylim(0.4, 1.02)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax2.set_ylabel("Motion accuracy (all PCs)")
        ax2.set_title("All-PC motion decoding")
        for bar, val in zip(bars, means):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

        plt.tight_layout()
        fig_path = f"{fig_dir}/residualized_pca.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_pp(plot_data, fig_dir="figures"):
    """Bar charts summarizing the predictive processing analysis."""
    prior_r2 = float(plot_data["prior_r2"])
    oracle_r2 = float(plot_data["oracle_r2"])
    pp_r2 = float(plot_data["pp_r2"])
    render_r2 = float(plot_data["render_r2"])
    full_per_dim_r2 = plot_data["full_per_dim_r2"]

    with paper_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.2))

        # Panel A: model comparison bar
        labels = ["Prior", "Render\nbaseline", "InverseModel\n(PP)", "Oracle"]
        heights = [prior_r2, render_r2, pp_r2, oracle_r2]
        colors = [
            COLORS["neutral"],
            COLORS["sensory"],
            COLORS["control"],
            COLORS["physics"],
        ]
        bars = ax1.bar(labels, heights, color=colors, width=0.6)
        ax1.set_ylabel("Physics inference R²")
        ax1.set_title("PP model vs. baselines")
        for bar, val in zip(bars, heights):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=5,
            )

        # Panel B: per-dimension R²
        x = np.arange(len(full_per_dim_r2))
        ax2.bar(x, full_per_dim_r2, color=COLORS["control"], width=0.8)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax2.set_xlabel("Physics dimension")
        ax2.set_ylabel("R²")
        ax2.set_title("InverseModel per-dimension R²")

        plt.tight_layout()
        fig_path = f"{fig_dir}/predictive_processing.pdf"
        plt.savefig(fig_path)
        plt.close()


def plot_pp_frames(plot_data, fig_dir="figures", fwd_frame_imgs=None):
    """Frame grid for the PP analysis: input frames → PP sim → render baseline → target."""
    init_imgs = plot_data["init_frame_imgs"]
    early_imgs = plot_data["early_frame_imgs"]
    pp_imgs = plot_data["pp_frame_imgs"]
    render_imgs = plot_data["render_frame_imgs"]
    final_imgs = plot_data["final_frame_imgs"]
    n = len(init_imgs)

    col_titles = [
        "t=0 (input)",
        "Early frame",
        "PP inferred\nsim",
        "Render\nprediction",
        "t=N (target)",
    ]
    cols = [init_imgs, early_imgs, pp_imgs, render_imgs, final_imgs]
    if fwd_frame_imgs is not None:
        col_titles.append("Cog. fwd\n(t=0 pred)")
        cols.append(fwd_frame_imgs[:n])

    n_cols = len(cols)
    with paper_style():
        fig, axes = plt.subplots(n, n_cols, figsize=(FULL_WIDTH * n_cols / 4, 1.4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for col_idx, (title, imgs) in enumerate(zip(col_titles, cols)):
            axes[0, col_idx].set_title(title, fontsize=5)
            for row_idx in range(n):
                img = imgs[row_idx]
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                axes[row_idx, col_idx].imshow(img)
                axes[row_idx, col_idx].axis("off")

        plt.tight_layout(pad=0.2)
        fig_path = f"{fig_dir}/pp_frames.pdf"
        plt.savefig(fig_path)
        plt.close()
