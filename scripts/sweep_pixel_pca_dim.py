"""Sweep pixel_pca_dim for the InverseModel and downstream encoding/residual R².

Per the spec at specs/pixel_pca_dim_sweep.md, this script holds everything fixed
except `pixel_pca_dim` ∈ {50, 200, 500, optionally 1000} and reports:

  - per-dim R² for each observable physics dim
  - mean per-dim R² (the headline behavioral metric)
  - inferred-physics encoding R²        (`r2_inferred` from analyses.encoding)
  - residualized inferred-physics R²    (`r2_resid_inferred` from analyses.residual)
  - wall-clock training time

Decision matrix (spec):

    | outcome                                    | next move
    |--------------------------------------------|--------------------------
    | mean ≥ 0.65 AND pos_z + x_accel ≥ 0.5      | bump config + ship
    | mean climbs but x_accel stays ≤ 0          | structural change (CNN / diff)
    | per-dim R² flat                            | MLP / timing is bottleneck
    | mixed                                      | per-dim slope decides

Standalone: builds its own scenes fixture if `data/scenes.npz` is absent (the
fixture is not coupled to Snakemake state). Run as:

    uv run python scripts/sweep_pixel_pca_dim.py
    uv run python scripts/sweep_pixel_pca_dim.py --dims 50 200 500 1000
    uv run python scripts/sweep_pixel_pca_dim.py --skip-residual  # faster smoke

Output: outputs/pca_dim_sweep.json + summary table to stdout.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from scripts.load_config import load_config
from scripts.io_utils import load_scenes, save_scenes
from analyses.predictive_processing import InverseModel, build_pp_features
from analyses.pp_io import extract_activations
from analyses.encoding import run_encoding_analysis
from analyses.residual import run_residual_analysis
from neural_model import generate_neural_activity

# Stride 16 per object: pos(0:3), orn(3:7), linvel(7:10), angvel(10:13),
# mass(13), friction(14), x_accel(15). With N_OBJECTS=1, the 5 observable
# dims with variance are: pos_x(0), pos_y(1), pos_z(2), linvel_x(7), x_accel(15).
PHYSICS_LABELS = [
    "pos_x",
    "pos_y",
    "pos_z",
    "orn_x",
    "orn_y",
    "orn_z",
    "orn_w",
    "linvel_x",
    "linvel_y",
    "linvel_z",
    "angvel_x",
    "angvel_y",
    "angvel_z",
    "mass",
    "friction",
    "x_accel",
]
# Spec uses vel_x as the alias for linvel_x.
SPEC_DIM_NAMES = {
    0: "pos_x",
    1: "pos_y",
    2: "pos_z",
    7: "vel_x",
    15: "x_accel",
}


def _ensure_scenes(path, cfg):
    """Return loaded scenes; build a fresh fixture (small) if `path` is missing.

    The fixture build does NOT touch Snakemake or rewrite `data/scenes.npz` if
    it already exists — this stays a side-effect-free diagnostic.
    """
    if os.path.exists(path):
        print(f"  loading scenes from {path}")
        return load_scenes(path)
    print(
        f"  {path} not found — generating a fresh fixture (this takes a minute or two)..."
    )
    from scene_generator import generate_scenes

    n = min(cfg.get("n_scenes", 2000), 2000)
    scenes = generate_scenes(n, cfg["random_seed"], n_timesteps=cfg["n_timesteps"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_scenes(scenes, path)
    print(f"  saved fresh fixture → {path}")
    return scenes


def _sweep_one(scenes, neural, neural_meta, pixel_pca_dim, *, cfg, run_residual=True):
    """Train an InverseModel at one pixel_pca_dim setting; return metrics dict."""
    print(f"\n{'=' * 60}")
    print(f"pixel_pca_dim = {pixel_pca_dim}")
    print("=" * 60)

    initial_physics = scenes["initial_physics_labels"]

    # Rebuild three-frame whitened PCA features at this dim setting.
    print(f"  building features (3 frames × {pixel_pca_dim} PCs)...")
    feats = build_pp_features(scenes, pixel_pca_dim=pixel_pca_dim)
    pixel_pca_concat = feats["pixel_pca_concat"]
    print(f"  feature shape = {pixel_pca_concat.shape}")

    # Train InverseModel — held-fixed schedule per spec.
    print(
        f"  training InverseModel (300 epochs, lr 1e-3, dropout 0.05, patience 50)..."
    )
    t0 = time.time()
    inv = InverseModel()
    inv.fit(
        pixel_pca_concat,
        initial_physics,
        n_epochs=300,
        batch_size=64,
        lr=1e-3,
        val_frac=0.15,
        patience=50,
    )
    train_time_s = time.time() - t0
    print(f"  trained in {train_time_s:.1f}s")

    # Per-dim R² (val split, 5 observable dims).
    full_per_dim_r2 = np.zeros(inv.full_physics_dim_)
    full_per_dim_r2[inv.valid_dims_] = inv.per_dim_r2_
    per_dim_r2 = {
        name: float(full_per_dim_r2[idx]) for idx, name in SPEC_DIM_NAMES.items()
    }
    mean_r2 = float(inv.per_dim_r2_.mean())

    print(f"\n  per-dim R² (val):")
    for name, r2 in per_dim_r2.items():
        print(f"    {name:8s}  {r2:+.4f}")
    print(f"    mean      {mean_r2:+.4f}")

    # Downstream encoding R² on the same neural activity (which was generated
    # from the original PP checkpoint, NOT this sweep's checkpoint). The
    # `r2_inferred` we report is "how well does the inferred-physics from THIS
    # sweep predict the existing neural activity" — interpreted as: did the
    # inferred-physics signal change qualitatively when we increased PCA dim?
    inferred_physics = inv.predict(pixel_pca_concat)

    print(f"\n  encoding analysis: neural ~ inferred_physics ...")
    enc = run_encoding_analysis(
        neural,
        scenes,
        neural_meta,
        render_pca_dim=cfg["render_pca_dim"],
        inferred_physics=inferred_physics,
    )
    r2_inferred = float(np.mean(enc["r2_inferred"]))
    r2_inferred_combined = float(np.mean(enc["r2_inferred_combined"]))
    delta_r2_inferred = float(np.mean(enc["delta_r2_inferred"]))
    print(f"    r2_inferred           = {r2_inferred:.4f}")
    print(f"    r2_inferred_combined  = {r2_inferred_combined:.4f}")
    print(f"    delta_r2_inferred     = {delta_r2_inferred:+.6f}")

    r2_resid_inferred = None
    if run_residual:
        print(
            f"\n  residual analysis: render-residualized neural ~ inferred_physics ..."
        )
        resid = run_residual_analysis(
            neural,
            scenes,
            neural_meta,
            render_pca_dim=cfg["render_pca_dim"],
            inferred_physics=inferred_physics,
        )
        r2_resid_inferred = float(np.mean(resid["r2_resid_inferred"]))
        print(
            f"    r2_resid_inferred     = {r2_resid_inferred:.4f}  "
            "(should be ~0 — residualization collapses inferred signal)"
        )

    return {
        "pixel_pca_dim": int(pixel_pca_dim),
        "train_time_s": float(train_time_s),
        "per_dim_r2": per_dim_r2,
        "mean_r2": mean_r2,
        "r2_inferred": r2_inferred,
        "r2_inferred_combined": r2_inferred_combined,
        "delta_r2_inferred": delta_r2_inferred,
        "r2_resid_inferred": r2_resid_inferred,
        "feature_shape": list(pixel_pca_concat.shape),
        "n_observable_dims": int(inv.valid_dims_.sum()),
    }


def _gen_neural_for_sweep(scenes, cfg):
    """Generate the neural activity used as the encoding/residual target.

    Mirrors scripts/gen_neural.py: takes the pretrained inverse-model checkpoint
    activations if available, else trains a default-config InverseModel here.
    The encoding/residual numbers we report are "inferred-physics from THIS
    sweep predicting that fixed neural population" — the population itself does
    not change across sweep settings. This is the correct comparator: a moving
    target (regenerating neural per sweep step) would confound the question.
    """
    render_indices = scenes["metadata"]["render_indices"]
    render = scenes["program_states"][:, render_indices]

    pp_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "pp_activations.npz"
    )
    if os.path.exists(pp_path):
        print(f"  using cached PP activations from {pp_path}")
        pp = np.load(pp_path)
        hidden_acts = pp["hidden_acts"]
        inferred_physics = pp["inferred_physics"]
    else:
        print(
            f"  no cached PP activations — training default-config InverseModel for neural projection..."
        )
        feats = build_pp_features(scenes, pixel_pca_dim=cfg["pp_pixel_pca_dim"])
        inv_default = InverseModel()
        inv_default.fit(feats["pixel_pca_concat"], scenes["initial_physics_labels"])
        layer = cfg.get("pp_neural_layer", "h2")
        hidden_acts = extract_activations(
            inv_default, feats["pixel_pca_concat"], layer=layer
        )
        inferred_physics = inv_default.predict(feats["pixel_pca_concat"])

    neural_input = np.concatenate(
        [render, hidden_acts, inferred_physics], axis=1
    ).astype(np.float32)

    neural, neural_meta = generate_neural_activity(
        neural_input,
        cfg["random_seed"],
        n_neurons=cfg["n_neurons"],
        noise_level=cfg["noise_level"],
    )
    return neural, neural_meta


def _print_summary(results):
    """ASCII summary table to stdout."""
    print("\n" + "=" * 90)
    print("SWEEP SUMMARY")
    print("=" * 90)
    cols = [
        "pca_dim",
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "x_accel",
        "mean",
        "r2_inf",
        "r2_inf_resid",
        "train_s",
    ]
    widths = [8, 8, 8, 8, 8, 9, 8, 8, 13, 9]
    header = "".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in results:
        row_vals = [
            f"{r['pixel_pca_dim']:>5d}",
            f"{r['per_dim_r2']['pos_x']:+.3f}",
            f"{r['per_dim_r2']['pos_y']:+.3f}",
            f"{r['per_dim_r2']['pos_z']:+.3f}",
            f"{r['per_dim_r2']['vel_x']:+.3f}",
            f"{r['per_dim_r2']['x_accel']:+.3f}",
            f"{r['mean_r2']:+.3f}",
            f"{r['r2_inferred']:+.3f}",
            (
                f"{r['r2_resid_inferred']:+.3f}"
                if r["r2_resid_inferred"] is not None
                else "  --"
            ),
            f"{r['train_time_s']:.1f}",
        ]
        widths_str = [8, 8, 8, 8, 8, 9, 8, 8, 13, 9]
        print("".join(f"{v:>{w}}" for v, w in zip(row_vals, widths_str)))
    print("=" * 90)


def _interpret(results):
    """Print decision-matrix interpretation."""
    print("\nInterpretation (decision matrix from specs/pixel_pca_dim_sweep.md):")
    if not results:
        print("  no results — nothing to interpret")
        return

    means = [r["mean_r2"] for r in results]
    pos_z = [r["per_dim_r2"]["pos_z"] for r in results]
    x_acc = [r["per_dim_r2"]["x_accel"] for r in results]

    final = results[-1]
    base = results[0]

    mean_lift = final["mean_r2"] - base["mean_r2"]
    pos_z_lift = final["per_dim_r2"]["pos_z"] - base["per_dim_r2"]["pos_z"]
    x_acc_final = final["per_dim_r2"]["x_accel"]

    print(
        f"  base (dim={base['pixel_pca_dim']}):  mean={base['mean_r2']:+.3f}  "
        f"pos_z={base['per_dim_r2']['pos_z']:+.3f}  x_accel={base['per_dim_r2']['x_accel']:+.3f}"
    )
    print(
        f"  final (dim={final['pixel_pca_dim']}): mean={final['mean_r2']:+.3f}  "
        f"pos_z={final['per_dim_r2']['pos_z']:+.3f}  x_accel={final['per_dim_r2']['x_accel']:+.3f}"
    )
    print(
        f"  mean lift = {mean_lift:+.3f}  pos_z lift = {pos_z_lift:+.3f}  "
        f"x_accel final = {x_acc_final:+.3f}"
    )

    if (
        final["mean_r2"] >= 0.65
        and final["per_dim_r2"]["pos_z"] >= 0.5
        and x_acc_final >= 0.5
    ):
        verdict = (
            "ROW 1 — linear basis was capacity-starved, MLP is fine. "
            "Recommend: bump pp_pixel_pca_dim in config.yaml and ship."
        )
    elif mean_lift > 0.05 and x_acc_final <= 0.0:
        verdict = (
            "ROW 2 — basis can't carry second-difference info regardless of dim. "
            "Recommend: structural change — CNN on raw 3-frame stack, OR "
            "frame-difference features (pca(t0), pca(t1)−pca(t0), pca(t2)−pca(t1))."
        )
    elif abs(mean_lift) < 0.03:
        verdict = (
            "ROW 3 — per-dim R² is largely flat across settings; basis isn't the "
            "bottleneck. Recommend: revisit MLP capacity / 3-frame timing / "
            "scene generative ranges."
        )
    else:
        verdict = (
            "ROW 4 — mixed verdict: some dims climb, some plateau. "
            "Use per-dim slopes (above) to choose between further dim bumps vs. "
            "structural change. Specifically: x_accel slope is the deciding factor."
        )
    print(f"  → {verdict}")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[50, 200, 500],
        help="pixel_pca_dim values to sweep (spec default: 50 200 500; add 1000 if still climbing)",
    )
    ap.add_argument("--scenes", default=os.path.join("data", "scenes.npz"))
    ap.add_argument("--out", default=os.path.join("outputs", "pca_dim_sweep.json"))
    ap.add_argument(
        "--skip-residual",
        action="store_true",
        help="Skip residual analysis (faster smoke runs)",
    )
    ap.add_argument(
        "--auto-extend",
        action="store_true",
        help="If mean R² is still climbing meaningfully (≥0.03) at the largest "
        "setting and 1000 is not already in --dims, run it too.",
    )
    args = ap.parse_args()

    print("=" * 60)
    print("pixel_pca_dim sweep — InverseModel quality")
    print("=" * 60)
    print(f"  dims to sweep: {args.dims}")
    print(f"  output JSON  : {args.out}")
    print(f"  residual     : {'skipped' if args.skip_residual else 'on'}")

    print(f"\nLoading scenes...")
    scenes = _ensure_scenes(args.scenes, cfg)
    print(f"  n = {len(scenes['initial_renders'])}")

    print(f"\nGenerating neural activity (fixed across all sweep settings)...")
    neural, neural_meta = _gen_neural_for_sweep(scenes, cfg)
    print(f"  neural shape = {neural.shape}")

    results = []
    dims_to_run = list(args.dims)
    i = 0
    while i < len(dims_to_run):
        d = dims_to_run[i]
        r = _sweep_one(
            scenes, neural, neural_meta, d, cfg=cfg, run_residual=not args.skip_residual
        )
        results.append(r)
        i += 1

        # Auto-extend to 1000 if mean still climbing meaningfully at the last dim.
        if (
            args.auto_extend
            and i == len(dims_to_run)
            and 1000 not in dims_to_run
            and len(results) >= 2
        ):
            lift = results[-1]["mean_r2"] - results[-2]["mean_r2"]
            if lift >= 0.03:
                print(
                    f"\n  mean R² lift {lift:+.3f} ≥ 0.03 between dim={dims_to_run[-2]} "
                    f"and dim={dims_to_run[-1]} — auto-extending to 1000."
                )
                dims_to_run.append(1000)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "sweep": results,
        "config": {
            "n_scenes": len(scenes["initial_renders"]),
            "pp_hidden_dim": cfg["pp_hidden_dim"],
            "pp_dropout_rate": cfg["pp_dropout_rate"],
            "pp_early_frame": cfg["pp_early_frame"],
            "pp_late_frame": cfg["pp_late_frame"],
            "render_pca_dim": cfg["render_pca_dim"],
            "n_neurons": cfg["n_neurons"],
            "noise_level": cfg["noise_level"],
            "random_seed": cfg["random_seed"],
        },
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {args.out}")

    _print_summary(results)
    _interpret(results)


if __name__ == "__main__":
    main()
