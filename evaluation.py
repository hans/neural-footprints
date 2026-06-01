"""
Evaluation framework for Neural Footprints simulation.

Defines explicit pass/fail criteria for each analysis and prints
a colored report.
"""

import numpy as np

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _check(name, passed, actual_str, threshold_str):
    """Format a single check line with pass/fail coloring."""
    icon = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    return f"  [{icon}] {name}: {actual_str}  {DIM}({threshold_str}){RESET}"


def evaluate(
    encoding_results,
    rsa_results,
    dissociation_results,
    pp_results=None,
    dynamics_results=None,
    residual_results=None,
):
    """Run all checks and print colored evaluation report. Returns (n_passed, n_total)."""

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
        checks.append(
            {
                "name": name,
                "passed": passed,
                "actual": actual_str,
                "threshold": threshold_str,
            }
        )

    # --- Variance Partitioning (Encoding Model) ---
    r2_X = np.asarray(encoding_results["r2_X"]).mean()
    r2_P = np.asarray(encoding_results["r2_P"]).mean()
    ctrl = encoding_results["control_accuracy"]

    lines.append(f"\n{BOLD}Variance Partitioning{RESET}")
    check(
        "X (raw frames) explains neural activity",
        r2_X > 0.30,
        f"R² = {r2_X:.4f}",
        "expect > 0.30",
    )
    check(
        "P (physics) explains neural activity",
        r2_P > 0.10,
        f"R² = {r2_P:.4f}",
        "expect > 0.10",
    )
    if encoding_results.get("null_r2_P_pvalue") is not None:
        pval = float(encoding_results["null_r2_P_pvalue"])
        check(
            "r2_P significantly above null (p < 0.05)",
            pval < 0.05,
            f"p = {pval:.3f}",
            "PASS iff p < 0.05 one-sided vs permutation null",
        )
    check(
        "Control: physics predicts behavior",
        ctrl > 0.90,
        f"accuracy = {ctrl:.1%}",
        "expect > 90%",
    )

    if encoding_results.get("r2_S") is not None:
        r2_S = np.asarray(encoding_results["r2_S"]).mean()
        check(
            "S (predicted frames) explains neural activity",
            r2_S > 0.30,
            f"R² = {r2_S:.4f}",
            "expect > 0.30",
        )

    if encoding_results.get("delta_P_given_X") is not None:
        dpx = np.asarray(encoding_results["delta_P_given_X"]).mean()
        check(
            "delta_P | X > 0 (naive positive: physics adds beyond raw frames)",
            dpx > 0.005,
            f"ΔR² = {dpx:.6f}",
            "expect > 0.005",
        )
        if encoding_results.get("null_delta_P_given_X_pvalue") is not None:
            pval = float(encoding_results["null_delta_P_given_X_pvalue"])
            check(
                "delta_P | X significantly above null (p < 0.05)",
                pval < 0.05,
                f"p = {pval:.3f}",
                "PASS iff p < 0.05 one-sided vs permutation null",
            )

    if encoding_results.get("delta_P_given_XS") is not None:
        dpxs = np.asarray(encoding_results["delta_P_given_XS"]).mean()
        check(
            "delta_P | X,S ≈ 0 (KEY: physics adds nothing beyond frames + model)",
            dpxs < 0.005,
            f"ΔR² = {dpxs:.6f}",
            "expect < 0.005 KEY",
        )
        if encoding_results.get("null_delta_P_given_XS_pvalue") is not None:
            pval = float(encoding_results["null_delta_P_given_XS_pvalue"])
            check(
                "delta_P | X,S significantly above null (KEY significance test, p < 0.05)",
                pval < 0.05,
                f"p = {pval:.3f}",
                "PASS iff p < 0.05; PASS here means physics genuinely adds beyond X+S (bad news)",
            )

    # Inferred-physics parallel checks (same thresholds as GT)
    if encoding_results.get("r2_P_inf") is not None:
        r2_P_inf = np.asarray(encoding_results["r2_P_inf"]).mean()
        check(
            "P_inf (inferred physics) explains neural activity",
            r2_P_inf > 0.10,
            f"R² = {r2_P_inf:.4f}",
            "expect > 0.10 (same threshold as GT; pass/fail asymmetry is the finding)",
        )
        if encoding_results.get("null_r2_P_inf_pvalue") is not None:
            pval = float(encoding_results["null_r2_P_inf_pvalue"])
            check(
                "r2_P_inf significantly above null (p < 0.05)",
                pval < 0.05,
                f"p = {pval:.3f}",
                "PASS iff p < 0.05 one-sided vs permutation null",
            )
    if encoding_results.get("delta_P_inf_given_X") is not None:
        dpx_inf = np.asarray(encoding_results["delta_P_inf_given_X"]).mean()
        check(
            "delta_P_inf | X > 0 (naive positive: inferred physics adds beyond raw frames)",
            dpx_inf > 0.005,
            f"ΔR² = {dpx_inf:.6f}",
            "expect > 0.005",
        )
        if encoding_results.get("null_delta_P_inf_given_X_pvalue") is not None:
            pval = float(encoding_results["null_delta_P_inf_given_X_pvalue"])
            check(
                "delta_P_inf | X significantly above null (p < 0.05)",
                pval < 0.05,
                f"p = {pval:.3f}",
                "PASS iff p < 0.05 one-sided vs permutation null",
            )
    if encoding_results.get("delta_P_inf_given_XS") is not None:
        dpxs_inf = np.asarray(encoding_results["delta_P_inf_given_XS"]).mean()
        check(
            "delta_P_inf | X,S ≈ 0 (KEY: inferred physics adds nothing beyond frames + model)",
            dpxs_inf < 0.005,
            f"ΔR² = {dpxs_inf:.6f}",
            "expect < 0.005 KEY",
        )
        if encoding_results.get("null_delta_P_inf_given_XS_pvalue") is not None:
            pval = float(encoding_results["null_delta_P_inf_given_XS_pvalue"])
            check(
                "delta_P_inf | X,S significantly above null (KEY significance test, p < 0.05)",
                pval < 0.05,
                f"p = {pval:.3f}",
                "PASS iff p < 0.05; PASS here means inferred physics genuinely adds beyond X+S (bad news)",
            )

    # --- Residualization ---
    if residual_results is not None:
        lines.append(f"\n{BOLD}Residualization{RESET}")
        if residual_results.get("r2_P_given_X") is not None:
            r2_PgX = float(np.asarray(residual_results["r2_P_given_X"]).mean())
            check(
                "r2_P | X > 0 (physics survives X-only residualization)",
                r2_PgX > 0.01,
                f"R² = {r2_PgX:.4f}",
                "expect > 0.01",
            )
        if residual_results.get("r2_P_given_XS") is not None:
            r2_PgXS = float(np.asarray(residual_results["r2_P_given_XS"]).mean())
            check(
                "r2_P | X,S ≈ 0 (KEY: physics collapses after X+S residualization)",
                r2_PgXS < 0.01,
                f"R² = {r2_PgXS:.4f}",
                "expect < 0.01 KEY",
            )
        # Inferred-physics parallel checks (same thresholds as GT)
        if residual_results.get("r2_P_inf_given_X") is not None:
            r2_PinfgX = float(np.asarray(residual_results["r2_P_inf_given_X"]).mean())
            check(
                "r2_P_inf | X > 0 (inferred physics survives X-only residualization)",
                r2_PinfgX > 0.01,
                f"R² = {r2_PinfgX:.4f}",
                "expect > 0.01 (same threshold as GT; asymmetry is the finding)",
            )
        if residual_results.get("r2_P_inf_given_XS") is not None:
            r2_PinfgXS = float(np.asarray(residual_results["r2_P_inf_given_XS"]).mean())
            check(
                "r2_P_inf | X,S ≈ 0 (KEY: inferred physics collapses after X+S residualization)",
                r2_PinfgXS < 0.01,
                f"R² = {r2_PinfgXS:.4f}",
                "expect < 0.01 KEY",
            )

    # --- RSA ---
    corr_X = rsa_results["corr_neural_X"]
    corr_P = rsa_results["corr_neural_P"]
    partial_X = rsa_results["partial_P_given_X"]

    lines.append(f"\n{BOLD}RSA{RESET}")
    check(
        "Neural ↔ X correlation is dominant",
        corr_X > 0.10,
        f"r = {corr_X:.4f}",
        "expect > 0.10",
    )
    check(
        "Neural ↔ physics correlation present",
        corr_P > 0.05,
        f"r = {corr_P:.4f}",
        "expect > 0.05",
    )
    check(
        "Partial neural↔physics | X > 0 (naive positive)",
        partial_X > 0.02,
        f"r = {partial_X:.4f}",
        "expect > 0.02",
    )

    if rsa_results.get("partial_P_given_XS") is not None:
        partial_XS = rsa_results["partial_P_given_XS"]
        check(
            "Partial neural↔physics | X,S ≈ 0 (KEY)",
            abs(partial_XS) < 0.05,
            f"r = {partial_XS:.4f}",
            "expect |r| < 0.05 KEY",
        )

    # Inferred-physics parallel RSA checks (same thresholds as GT)
    if rsa_results.get("corr_neural_P_inf") is not None:
        corr_P_inf = rsa_results["corr_neural_P_inf"]
        check(
            "Neural ↔ inferred-physics correlation present",
            corr_P_inf > 0.05,
            f"r = {corr_P_inf:.4f}",
            "expect > 0.05 (same threshold as GT; asymmetry is the finding)",
        )
    if rsa_results.get("partial_P_inf_given_X") is not None:
        partial_inf_X = rsa_results["partial_P_inf_given_X"]
        check(
            "Partial neural↔inferred-physics | X > 0 (naive positive)",
            partial_inf_X > 0.02,
            f"r = {partial_inf_X:.4f}",
            "expect > 0.02",
        )
    if rsa_results.get("partial_P_inf_given_XS") is not None:
        partial_inf_XS = rsa_results["partial_P_inf_given_XS"]
        check(
            "Partial neural↔inferred-physics | X,S ≈ 0 (KEY)",
            abs(partial_inf_XS) < 0.05,
            f"r = {partial_inf_XS:.4f}",
            "expect |r| < 0.05 KEY",
        )

    # --- Dissociation ---
    r2_pix = dissociation_results["mean_r2_pixel"]
    r2_phys = dissociation_results["mean_r2_physics"]
    beh_pix = dissociation_results["pixel_behavioral_score"]
    beh_phys = dissociation_results["physics_behavioral_score"]
    beh_phys_inf = dissociation_results.get(
        "inferred_physics_behavioral_score", float("nan")
    )
    metric = dissociation_results["metric_label"]
    obj = dissociation_results["objective"]

    lines.append(f"\n{BOLD}Dissociation (objective: {obj}){RESET}")
    check(
        "Pixel model has higher neural R²",
        r2_pix > r2_phys,
        f"pixel R² = {r2_pix:.4f} vs physics R² = {r2_phys:.4f}",
        "expect pixel > physics",
    )
    check(
        "Physics model has higher behavioral score",
        beh_phys > beh_pix,
        f"physics {metric} = {beh_phys:.4f} vs pixel {metric} = {beh_pix:.4f}",
        "expect physics > pixel",
    )
    check(
        "Pixel behavioral score is poor",
        beh_pix < 0.30 if obj == "next_frame_pixels" else beh_pix < 0.70,
        f"{metric} = {beh_pix:.4f}",
        "expect low" if obj == "next_frame_pixels" else "expect < 0.70",
    )
    if obj == "next_frame_pixels":
        # Oracle resimulates the held-out t=N_TIMESTEPS RGBA target
        # deterministically; with the 3-frame brain block, this should
        # remain essentially perfect.
        check(
            "Physics oracle behavioral score is near-perfect",
            beh_phys > 0.90,
            f"{metric} = {beh_phys:.4f}",
            "expect > 0.90 (oracle re-renders the held-out target)",
        )
        if beh_phys_inf == beh_phys_inf:  # not nan
            check(
                "Inferred-physics behavioral score is high",
                beh_phys_inf > 0.70,
                f"inferred {metric} = {beh_phys_inf:.4f}  "
                f"(GT oracle = {beh_phys:.4f}, gap = {beh_phys - beh_phys_inf:+.4f})",
                "expect > 0.70 — inferred physics + PyBullet should approach the GT oracle",
            )
        delta_pix = dissociation_results.get(
            "delta_pixel_behavioral_score", float("nan")
        )
        delta_phys = dissociation_results.get(
            "delta_physics_behavioral_score", float("nan")
        )
        if delta_pix == delta_pix:  # not nan
            check(
                "Delta-frame: physics oracle near-perfect in delta space",
                delta_phys > 0.90,
                f"delta R² = {delta_phys:.4f}",
                "expect > 0.90 — static bg cancels in frame4−frame1 delta, oracle is exact",
            )
            check(
                "Delta-frame: pixel model poor in delta space",
                delta_pix < 0.70,
                f"delta R² = {delta_pix:.4f}",
                "expect < 0.70 — blurry PCA prediction cannot match sharp object-displacement delta",
            )

    # --- Dynamics (future brain state) ---
    if dynamics_results is not None:
        r2_phys_fwd = dynamics_results["mean_r2_physics_forward"]
        r2_pix_fwd = dynamics_results["mean_r2_pixel_forward"]
        fwd_gap = dynamics_results["forward_gap"]

        lines.append(f"\n{BOLD}Future Brain State (Dynamics){RESET}")
        check(
            "Physics forward model predicts future brain state",
            r2_phys_fwd > 0.30,
            f"R² = {r2_phys_fwd:.4f}",
            "expect > 0.30",
        )
        check(
            "Physics forward beats pixel forward",
            r2_phys_fwd > r2_pix_fwd,
            f"physics R² = {r2_phys_fwd:.4f} vs pixel R² = {r2_pix_fwd:.4f}",
            "expect physics > pixel",
        )
        check(
            "Forward model gap is substantial",
            fwd_gap > 0.10,
            f"gap = {fwd_gap:.4f}",
            "expect > 0.10",
        )
        # With the 3-frame brain block, the pixel forward model can fill
        # at most one frame's RGBA from initial pixels — the other two
        # frames stay at training mean, so prediction should be clearly
        # poor in absolute terms, not just relatively.
        check(
            "Pixel forward model is poor in absolute terms",
            r2_pix_fwd < 0.20,
            f"R² = {r2_pix_fwd:.4f}",
            "expect < 0.20",
        )

    # --- Summary ---
    if passed_total == total:
        color = GREEN
    elif passed_total >= total * 0.7:
        color = YELLOW
    else:
        color = RED

    lines.append("")
    lines.append(f"{BOLD}{'=' * 50}{RESET}")
    lines.append(f"{color}{BOLD}{passed_total}/{total} checks passed{RESET}")
    if passed_total < total:
        lines.append(
            f"{RED}The simulation does not fully demonstrate the expected dissociation.{RESET}"
        )
    else:
        lines.append(
            f"{GREEN}Physics is functionally operative but methodologically invisible.{RESET}"
        )
    lines.append(f"{BOLD}{'=' * 50}{RESET}")

    print("\n" + "=" * 50)
    print(f"{BOLD}EVALUATION{RESET}")
    print("=" * 50)
    for line in lines:
        print(line)

    return passed_total, total, checks
