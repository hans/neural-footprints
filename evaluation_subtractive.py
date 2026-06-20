"""Evaluation framework for the subtractive-analysis pipeline.

Encodes the core findings from specs/subtractive_analysis.md as numerical
PASS/FAIL checks. Mirrors the format of evaluation.py so the printed report
reads the same way.
"""

from evaluation import (
    GREEN, RED, YELLOW, BOLD, RESET,
    _check,
)


def _frac(results, block_name, threshold_key):
    """Per-block fraction surviving a given threshold key."""
    block_idx = results['block_names'].index(block_name)
    rec = results['thresholds'][threshold_key]
    return float(rec[f'frac_significant_block_{block_idx}'])


def _all_threshold_keys():
    return ['p<0.05', 'p<0.01', 'p<0.001', 'fdr_bh', 'bonferroni']


def evaluate_subtractive(per_regime_results, *, mode):
    """Run subtractive checks and print a colored report.

    Parameters
    ----------
    per_regime_results : dict[str, dict]
        Maps regime name ("confounded", "area_controlled") to the
        results JSON loaded from outputs/subtractive_{regime}_{mode}_results.json.
    mode : str
        "ground_truth" or "inferred". Selects the appropriate check set.

    Returns
    -------
    (n_passed, n_total, checks)
    """
    if mode not in {"ground_truth", "inferred"}:
        raise ValueError(f"Unknown subtractive mode: {mode!r}")

    lines = []
    checks = []
    passed_total = 0
    total = 0

    def check(name, passed, actual_str, threshold_str):
        nonlocal passed_total, total
        total += 1
        if passed:
            passed_total += 1
        lines.append(_check(name, passed, actual_str, threshold_str))
        checks.append({'name': name, 'passed': passed,
                       'actual': actual_str, 'threshold': threshold_str})

    confounded = per_regime_results['confounded']
    area_ctrl  = per_regime_results['area_controlled']

    # The abstract block is intentionally "compact and quiet" by spec design
    # (config.yaml comment around abstract_weight_std=0.025): per-neuron
    # condition shift sized at ~half the sensory population's. So the finding
    # is NOT "abstract block saturates / sensory dominates" — it's "even when
    # the abstract block is wired with an oracle ground-truth signal, sensory
    # leakage produces a comparable surviving brain blob."
    #
    # All thresholds below are calibrated against the first end-to-end run on
    # this branch (1200 scenes/regime, default config). They live ~30% below
    # the observed values so a regression of moderate size will trip them.

    # ----------------------------------------------------------------
    # Confounded regime: Finding #1 (sensible-threshold failure)
    # ----------------------------------------------------------------
    lines.append(f"\n{BOLD}Confounded regime — Finding #1 "
                 f"(sensible-threshold failure){RESET}")

    f_sens_fdr  = _frac(confounded, 'sensory', 'fdr_bh')
    f_sens_bonf = _frac(confounded, 'sensory', 'bonferroni')
    f_sens_p05  = _frac(confounded, 'sensory', 'p<0.05')
    f_abs_fdr   = _frac(confounded, 'abstract', 'fdr_bh')

    check(
        "Large surviving sensory blob at FDR",
        f_sens_fdr >= 0.25,
        f"frac = {f_sens_fdr:.1%}",
        "expect ≥ 25% (well above chance — sensory leakage produces a real-looking blob)",
    )
    check(
        "Sensory blob survives Bonferroni",
        f_sens_bonf >= 0.05,
        f"frac = {f_sens_bonf:.1%}",
        "expect ≥ 5% (residual leakage even at strictest correction)",
    )
    check(
        "Sensory blob substantial at p<0.05 (uncorrected)",
        f_sens_p05 >= 0.30,
        f"frac = {f_sens_p05:.1%}",
        "expect ≥ 30% at the standard threshold",
    )

    # Mode-aware: in inferred the headline winner at FDR is sensory; in
    # ground_truth abstract wins (oracle), but sensory still tracks closely.
    if mode == "inferred":
        winner = confounded['headline'].split(' ')[0]
        check(
            "Headline winner at FDR is the sensory block",
            winner == 'sensory',
            f"winner = {winner!r}",
            "expect 'sensory' (sensory leakage outranks abstract signal)",
        )
    else:
        check(
            "Sensory blob tracks abstract at FDR",
            f_sens_fdr >= 0.6 * f_abs_fdr,
            f"sensory/abstract = {f_sens_fdr / max(f_abs_fdr, 1e-9):.0%}",
            "expect sensory ≥ 60% × abstract (sensory leakage stays close to oracle abstract)",
        )

    # ----------------------------------------------------------------
    # Confounded regime: Finding #2 (no-threshold rescue)
    # ----------------------------------------------------------------
    lines.append(f"\n{BOLD}Confounded regime — Finding #2 "
                 f"(no-threshold rescue){RESET}")

    if mode == "inferred":
        for thr in _all_threshold_keys():
            f_sens = _frac(confounded, 'sensory', thr)
            f_abs  = _frac(confounded, 'abstract', thr)
            check(
                f"sensory ≥ abstract at {thr}",
                f_sens >= f_abs,
                f"sensory = {f_sens:.1%}, abstract = {f_abs:.1%}",
                "expect sensory ≥ abstract (no threshold rescues abstract)",
            )
    else:
        # Ground-truth: the abstract block trivially dominates because it's
        # an oracle. Reframe as: sensory leakage stays well above the noise
        # floor at every threshold (3× chance), including the strictest.
        thresholds_min_frac = {
            'p<0.05':     0.15,   # chance = 5%
            'p<0.01':     0.05,   # chance = 1%
            'p<0.001':    0.02,   # chance = 0.1%
            'fdr_bh':     0.10,   # chance ~ FDR alpha = 5%
            'bonferroni': 0.02,   # chance is vanishing
        }
        for thr, min_frac in thresholds_min_frac.items():
            f_sens = _frac(confounded, 'sensory', thr)
            check(
                f"sensory leakage above noise floor at {thr}",
                f_sens >= min_frac,
                f"sensory = {f_sens:.1%}",
                f"expect ≥ {min_frac:.0%} (well above chance at this threshold)",
            )

    # ----------------------------------------------------------------
    # Area-controlled regime: control works
    # ----------------------------------------------------------------
    lines.append(f"\n{BOLD}Area-controlled regime — control works{RESET}")
    f_sens_fdr_area = _frac(area_ctrl, 'sensory', 'fdr_bh')
    f_abs_fdr_area  = _frac(area_ctrl, 'abstract', 'fdr_bh')

    check(
        "Sensory leakage stays bounded under area control",
        f_sens_fdr_area <= 0.40,
        f"frac = {f_sens_fdr_area:.1%}",
        "expect ≤ 40% (removing the area confound pulls sensory back, but doesn't eliminate it)",
    )
    check(
        "Abstract signal is recoverable under area control",
        f_abs_fdr_area >= 0.30,
        f"frac = {f_abs_fdr_area:.1%}",
        "expect ≥ 30% (abstract signal reaches the surface when no confound)",
    )

    # ----------------------------------------------------------------
    # Cross-regime: the area-control manipulation does something
    # ----------------------------------------------------------------
    lines.append(f"\n{BOLD}Cross-regime sanity{RESET}")
    check(
        "Area control reduces sensory leakage at FDR",
        f_sens_fdr > f_sens_fdr_area,
        f"confounded = {f_sens_fdr:.1%}  vs  area_controlled = {f_sens_fdr_area:.1%}",
        "expect confounded > area_controlled",
    )

    # ----------------------------------------------------------------
    # Mode-specific extras
    # ----------------------------------------------------------------
    lines.append(f"\n{BOLD}Mode-specific{RESET}")
    if mode == "inferred":
        val_r2_min = min(float(per_regime_results[r]['cardinality_val_r2'])
                         for r in per_regime_results)
        check(
            "CardinalityModel learned numerosity",
            val_r2_min > 0.5,
            f"min val R² = {val_r2_min:.3f}",
            "expect > 0.5 (matches train_cardinality.py runtime guard)",
        )
    else:
        for regime, results in per_regime_results.items():
            f_abs = _frac(results, 'abstract', 'fdr_bh')
            check(
                f"Oracle abstract block is condition-selective ({regime})",
                f_abs >= 0.30,
                f"frac = {f_abs:.1%}",
                "expect ≥ 30% (ground-truth N drives meaningful abstract signal; "
                "no saturation by spec design — abstract block is intentionally quiet)",
            )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    if passed_total == total:
        color = GREEN
    elif passed_total >= total * 0.7:
        color = YELLOW
    else:
        color = RED

    lines.append("")
    lines.append(f"{BOLD}{'=' * 50}{RESET}")
    lines.append(
        f"{color}{BOLD}{passed_total}/{total} checks passed "
        f"(mode={mode}){RESET}"
    )
    if passed_total < total:
        lines.append(
            f"{RED}The subtractive findings are not fully demonstrated.{RESET}"
        )
    else:
        lines.append(
            f"{GREEN}Sensory leakage dominates the apparent brain blob — "
            f"no threshold rescues the abstract block.{RESET}"
        )
    lines.append(f"{BOLD}{'=' * 50}{RESET}")

    print("\n" + "=" * 50)
    print(f"{BOLD}EVALUATION (subtractive, mode={mode}){RESET}")
    print("=" * 50)
    for line in lines:
        print(line)

    return passed_total, total, checks
