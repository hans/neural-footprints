"""
Render the softmax-sweep summary as a side-by-side table.

Reads outputs/pp_cnn_softmax_sweep.json (or whatever --summary points at) and
prints a comparison table including the v1 softmax / gridpool baselines from
the simple-CNN diagnostic.

Usage
-----
    uv run python scripts/report_softmax_sweep.py
    uv run python scripts/report_softmax_sweep.py --summary outputs/pp_cnn_softmax_sweep.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PER_DIM_COLS = ["pos_x", "pos_y", "pos_z", "linvel_x", "x_accel"]

# Reference rows from earlier diagnostics (same val split).
REFERENCE_ROWS = [
    # (label, n_params, mean_r2, per-dim dict)
    ("pca50_mlp (3-frame)", None, 0.6553, None),
    (
        "gridpool (simple)",
        1700197,
        0.7963,
        {
            "pos_x": 0.9813,
            "pos_y": 0.8715,
            "pos_z": 0.6375,
            "linvel_x": 0.8410,
            "x_accel": 0.6502,
        },
    ),
    (
        "spatial_softmax v1",
        34405,
        0.7564,
        {
            "pos_x": 0.9764,
            "pos_y": 0.8107,
            "pos_z": 0.5717,
            "linvel_x": 0.8418,
            "x_accel": 0.5812,
        },
    ),
]


def fmt_int(n):
    return "—" if n is None else f"{n:,}"


def fmt_r2(v):
    return "—" if v is None else f"{v:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="outputs/pp_cnn_softmax_sweep.json")
    args = ap.parse_args()

    with open(args.summary) as f:
        d = json.load(f)

    rows = list(REFERENCE_ROWS)
    for name, r in d["results"].items():
        per_dim = r["per_dim_r2"]
        rows.append(
            (
                name,
                r["n_params"],
                r["mean_valid_dim_r2"],
                {k: per_dim.get(k) for k in PER_DIM_COLS},
            )
        )

    # Header
    head = f"  {'config':24s}  {'params':>10s}  {'mean R²':>8s}  " + "  ".join(
        f"{c:>8s}" for c in PER_DIM_COLS
    )
    print(head)
    print("  " + "-" * (len(head) - 2))

    for label, n_params, mean_r2, per_dim in rows:
        cells = "  ".join(
            fmt_r2(per_dim[c]) if per_dim and per_dim.get(c) is not None else "       —"
            for c in PER_DIM_COLS
        )
        print(
            f"  {label:24s}  {fmt_int(n_params):>10s}  {fmt_r2(mean_r2):>8s}  {cells}"
        )

    # Highlight the best run
    best = max(d["results"].items(), key=lambda kv: kv[1]["mean_valid_dim_r2"])
    print(
        f"\n  Best sweep config: {best[0]}  "
        f"mean R² = {best[1]['mean_valid_dim_r2']:+.4f}  "
        f"({best[1]['n_params']:,} params)"
    )

    # Compute relative numbers vs gridpool / softmax_v1 for the headline
    gp_mean = next(r for l, _, r, _ in REFERENCE_ROWS if "gridpool" in l)
    gp_params = next(p for l, p, _, _ in REFERENCE_ROWS if "gridpool" in l)
    print(
        f"  vs gridpool:  perf = {best[1]['mean_valid_dim_r2']/gp_mean*100:.1f}%  "
        f"params = {best[1]['n_params']/gp_params*100:.2f}%"
    )


if __name__ == "__main__":
    main()
