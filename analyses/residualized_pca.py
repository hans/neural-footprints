"""
Residualized PCA: motion decodability from neural PCs is a render confound.

The plain PCA negative control (``analyses/pca_analysis.py``) shows motion
direction is only weakly decodable from the top-k neural PCs — but it *is*
"sort of" decodable, especially as more (higher) PCs are added. This module
asks *why*.

Causal chain: physics -> forward renders -> pixels. The 3-frame brain renders
carry the object's displacement, so any neural PC that captures render variance
inherits motion-correlated structure. We test whether the residual motion
decodability survives once the pixel-explainable component of neural activity
is removed.

Procedure (decoding analog of ``analyses/residual.py``):

  1. raw      : decode each target from the neural activity's own top-k PCs.
  2. resid_X  : regress raw-frame pixel PCA out of neural activity
                (cross-validated ridge), then decode from the residual's PCs.
  3. resid_XS : same, residualizing on [observed pixels X, forward-render
                predictions S] (the stronger control).
  4. pixel    : positive control — decode each target directly from the
                pixel-PCA's own top-k PCs (proves the renders carry motion).

Key conceptual point: ``vx`` is *also* present directly in the program_state
physics block, so residualizing on pixels does NOT remove the genuine causal
motion footprint — only the render-mediated one. If motion decoding still
collapses to chance under resid_X / resid_XS, that is the *stronger* result:
the true physics footprint sits below the noise floor and the apparent
decodability rode entirely on the render confound. The ``pixel`` positive
control confirms the renders genuinely carry motion.
"""

import numpy as np

from analyses.pca_analysis import _build_targets, _decode_pc_sweep
from analyses.residual import _ridge_cv_predict
from sklearn.model_selection import KFold


def _sweep_to_dict(sweep):
    """Pack a _decode_pc_sweep return tuple into a JSON-friendly dict."""
    pc_counts, cumvar, _scores, decode_accs, chance = sweep
    return {
        "pc_counts": list(pc_counts),
        "cumulative_variance": [float(v) for v in cumvar],
        "decode_accs_per_target": {
            name: [float(a) for a in accs] for name, accs in decode_accs.items()
        },
        "chance_per_target": chance,
    }


def run_residualized_pca_analysis(
    neural_activity,
    scenes,
    neural_meta,
    *,
    raw_pixel_pca,
    predicted_pixel_pca=None,
    n_permutations=50,
    compute_null=True,
    n_splits=5,
    random_state=42,
):
    """Run the residualized PCA decoding analysis (see module docstring).

    Returns a dict keyed by condition (``raw``, ``resid_X``, ``resid_XS`` when
    S is provided, and ``pixel``), each holding ``pc_counts``,
    ``cumulative_variance``, ``decode_accs_per_target`` and
    ``chance_per_target``. Also reports the residual variance fractions.
    """
    print("\n" + "=" * 60)
    print("RESIDUALIZED PCA: motion decodability is a render confound")
    print("=" * 60)

    n_scenes, n_neurons = neural_activity.shape
    print(f"  n_scenes={n_scenes}, n_neurons={n_neurons}")

    targets = _build_targets(scenes)
    for name, y in targets.items():
        n_pos = int(y.sum())
        print(f"  {name}: {n_scenes - n_pos} class-0 / {n_pos} class-1")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    alphas = np.logspace(-2, 6, 20)

    def sweep(feature_matrix):
        return _decode_pc_sweep(
            feature_matrix,
            targets,
            n_permutations=n_permutations,
            compute_null=compute_null,
        )

    result = {}

    # --- raw neural baseline ---
    print("\n[raw] decode from neural activity's own PCs...")
    result["raw"] = _sweep_to_dict(sweep(neural_activity))

    # --- resid_X: residualize on observed pixels X ---
    print("\n[resid_X] residualize neural on X (observed pixels), then decode...")
    y_pred_X = _ridge_cv_predict(raw_pixel_pca, neural_activity, cv=cv, alphas=alphas)
    y_resid_X = neural_activity - y_pred_X
    var_kept_X = y_resid_X.var(axis=0).mean() / neural_activity.var(axis=0).mean()
    print(f"  residual variance fraction = {var_kept_X:.4f}")
    result["resid_X"] = _sweep_to_dict(sweep(y_resid_X))
    result["residual_variance_fraction_X"] = float(var_kept_X)

    # --- resid_XS: residualize on [X, S] (stronger control) ---
    if predicted_pixel_pca is not None:
        print("\n[resid_XS] residualize neural on [X, S], then decode...")
        XS = np.hstack([raw_pixel_pca, predicted_pixel_pca])
        y_pred_XS = _ridge_cv_predict(XS, neural_activity, cv=cv, alphas=alphas)
        y_resid_XS = neural_activity - y_pred_XS
        var_kept_XS = y_resid_XS.var(axis=0).mean() / neural_activity.var(axis=0).mean()
        print(f"  residual variance fraction = {var_kept_XS:.4f}")
        result["resid_XS"] = _sweep_to_dict(sweep(y_resid_XS))
        result["residual_variance_fraction_XS"] = float(var_kept_XS)

    # --- pixel positive control: decode from pixel PCA's own PCs ---
    print("\n[pixel] positive control: decode directly from pixel PCs...")
    result["pixel"] = _sweep_to_dict(sweep(raw_pixel_pca))

    result["n_neurons"] = int(n_neurons)
    return result
