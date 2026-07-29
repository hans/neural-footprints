"""
Matched permutation null for the residualization R² (r2_P | X and r2_P | X,S).

The encoding analysis's physics null is built with a different estimator on a
different target (raw neural, closed-form ridge), so the residualization point
estimate (~-0.0023) cannot be read off it. This builds the *matched* null:
the exact two-stage estimator from analyses/residual.py, shuffling physics rows
in stage 2 only (stage-1 residual is physics-independent, so it's computed once).

Memory-careful for a 6 GB box: the two pixel-PCAs are built in float32 and
cached to disk BEFORE the 557 MB neural array is loaded, so the big render
matrices and neural are never co-resident.

    uv run python scripts/residual_matched_null.py [n_perms] [norm]
"""

import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from analyses.residual import _r2_per_neuron, _ridge_cv_predict
from io_utils import load_neural, load_scenes
from load_config import load_config

N_PERMS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
NORM = sys.argv[2] if len(sys.argv) > 2 else "zscore"
PCA_CACHE = f"data/{NORM}/_matched_null_pixel_pca.npz"


def extract_brain_pixels(states, metadata):
    """Inlined from scene_generator (avoids its mujoco import; no GL in container)."""
    fri = metadata["frame_render_indices"]
    rgba_bytes = (
        metadata["target_pixel_indices"].stop - metadata["target_pixel_indices"].start
    )
    return np.concatenate(
        [states[:, s.start : s.start + rgba_bytes]
         for s in (fri["initial"], fri["early"], fri["late"])],
        axis=1,
    )


def pca_f32(data, n_components):
    """float32, in-place-standardized, randomized PCA — bounds peak memory.

    Numerically ~identical to encoding.pca_reduce_pixels (StandardScaler+PCA);
    float32 vs float64 differences are well below the 4th decimal we report.
    """
    data = np.asarray(data, dtype=np.float32)
    mu = data.mean(axis=0)
    sd = data.std(axis=0)
    sd[sd == 0] = 1.0
    data -= mu
    data /= sd
    pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized")
    out = pca.fit_transform(data).astype(np.float32)
    del data
    gc.collect()
    return out


cfg = load_config()
print(f"Matched residualization null: {N_PERMS} perms, norm={NORM}")

# --- Pass 1: build & cache the two pixel-PCAs (no neural loaded) ---
if os.path.exists(PCA_CACHE):
    print(f"\nLoading cached pixel-PCAs from {PCA_CACHE}")
    z = np.load(PCA_CACHE)
    raw_pixel_pca, predicted_pixel_pca, physics_labels = z["X"], z["S"], z["phys"]
    metadata_present = True
else:
    print("\nPass 1: building pixel-PCAs in float32...")
    scenes = load_scenes("data/scenes.npz")
    physics_labels = np.asarray(scenes["physics_labels"])

    raw_frames = np.concatenate(
        [scenes["initial_renders"], scenes["early_renders"], scenes["late_renders"]],
        axis=1,
    )
    raw_pixel_pca = pca_f32(raw_frames, cfg["pixel_pca_dim"])
    del raw_frames
    print(f"  raw_pixel_pca: {raw_pixel_pca.shape}")

    fwd = np.load("data/forward_renders.npz")
    predicted_brain_pixels = extract_brain_pixels(fwd["forward_program_states"], scenes["metadata"])
    del fwd, scenes
    gc.collect()
    predicted_pixel_pca = pca_f32(predicted_brain_pixels, cfg["pixel_pca_dim"])
    del predicted_brain_pixels
    gc.collect()
    print(f"  predicted_pixel_pca: {predicted_pixel_pca.shape}")

    np.savez_compressed(PCA_CACHE, X=raw_pixel_pca, S=predicted_pixel_pca, phys=physics_labels)
    print(f"  cached -> {PCA_CACHE}")

# --- Pass 2: load neural, build residuals once, run the null ---
print("\nPass 2: loading neural + building residual targets...")
neural, _ = load_neural(f"data/{NORM}/neural.npz")
physics_scaled = StandardScaler().fit_transform(physics_labels)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
alphas = np.logspace(-2, 6, 20)

y_resid_X = neural - _ridge_cv_predict(raw_pixel_pca, neural, cv=cv, alphas=alphas)
XS = np.hstack([raw_pixel_pca, predicted_pixel_pca])
y_resid_XS = neural - _ridge_cv_predict(XS, neural, cv=cv, alphas=alphas)
del neural
gc.collect()


def stage2_mean_r2(y_resid, phys):
    """Stage-2 mean-across-neurons R²: decode physics from the residual."""
    return _r2_per_neuron(y_resid, _ridge_cv_predict(phys, y_resid, cv=cv, alphas=alphas)).mean()


obs_X = stage2_mean_r2(y_resid_X, physics_scaled)
obs_XS = stage2_mean_r2(y_resid_XS, physics_scaled)
print(f"\nObserved  r2_P | X   = {obs_X:+.4f}  (analysis reported ~-0.0023)")
print(f"Observed  r2_P | X,S = {obs_XS:+.4f}")

print(f"\nRunning {N_PERMS} shuffles (stage 2 only)...")
rng = np.random.default_rng(0)
null_X = np.empty(N_PERMS)
null_XS = np.empty(N_PERMS)
n = physics_scaled.shape[0]
for p in range(N_PERMS):
    phys_perm = physics_scaled[rng.permutation(n)]
    null_X[p] = stage2_mean_r2(y_resid_X, phys_perm)
    null_XS[p] = stage2_mean_r2(y_resid_XS, phys_perm)
    if (p + 1) % 10 == 0 or p == 0:
        print(f"  perm {p + 1}/{N_PERMS}: null r2_P|X = {null_X[p]:+.4f}, r2_P|X,S = {null_XS[p]:+.4f}")


def report(label, observed, null):
    lo, hi = np.percentile(null, [2.5, 97.5])
    # one-sided p that observed is NOT below chance: P(null <= observed)
    p_below = float((null <= observed).mean())
    print(f"\n=== {label} ===")
    print(f"  observed            = {observed:+.4f}")
    print(f"  null min .. max     = {null.min():+.4f} .. {null.max():+.4f}")
    print(f"  null 95% CI         = [{lo:+.4f}, {hi:+.4f}]")
    print(f"  null mean           = {null.mean():+.4f}")
    print(f"  P(null <= observed) = {p_below:.3f}  (high => observed at/above chance; ~0 => at the null floor)")
    print(f"  observed vs null mean: gap = {observed - null.mean():+.5f} (negative => slightly below chance)")


report("r2_P | X (matched null)", obs_X, null_X)
report("r2_P | X,S (matched null)", obs_XS, null_XS)

# Patch null CI scalars into the residual plot_data npz so the zoomed plot works.
npz_path = f"data/{NORM}/residual_plot_data.npz"
if os.path.exists(npz_path):
    existing = dict(np.load(npz_path, allow_pickle=False))
    lo_XS, hi_XS = np.percentile(null_XS, [2.5, 97.5])
    existing["null_r2_P_given_XS_ci_lo"] = np.array(float(lo_XS))
    existing["null_r2_P_given_XS_ci_hi"] = np.array(float(hi_XS))
    existing["null_r2_P_given_XS_mean"] = np.array(float(null_XS.mean()))
    existing["null_r2_P_given_XS_observed"] = np.array(float(obs_XS))
    np.savez_compressed(npz_path, **existing)
    print(f"\nPatched null CI into {npz_path}")
    print(f"  95% CI = [{lo_XS:+.4f}, {hi_XS:+.4f}], mean = {null_XS.mean():+.4f}, observed = {obs_XS:+.4f}")
    print("Run: snakemake --touch figures/{norm}/residual_null_zoomed.pdf")
