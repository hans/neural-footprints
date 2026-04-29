"""Cardinality-inference cognitive model for the subtractive-analysis pipeline.

Mirrors the role InverseModel plays in the PP pipeline: a learned MLP that
maps pixel-PCA features to a latent state — here, the scalar count N. Its h2
hidden activations and its scalar prediction together form the input pool for
the "Abstract Block" of the block-structured population in
analyses/block_projection.py.

The MLP architecture is identical to InverseMLPNet (three hidden layers with
dropout, named blocks for activation taps); only the target preprocessing
differs (1-D z-scored integer N instead of multi-dim physics).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analyses.inferential_mlp import (
    InferentialMLPNet,
    InferentialModel,
)


class CardinalityMLPNet(InferentialMLPNet):
    """Identical architecture to InverseMLPNet, output_dim is fixed at 1."""
    pass


class CardinalityModel(InferentialModel):
    """Pixel PCA features -> scalar cardinality N.

    The target N is z-scored on fit (mean ~ (n_low + n_high)/2, std ~ |n_high - n_low|/2
    for a balanced two-class design) and inverse-transformed on predict so the
    returned values are in native integer-units. Use predict_round() to snap
    the prediction to the nearest integer for sanity checking.
    """

    NET_CLASS = CardinalityMLPNet

    def __init__(self, hidden_dim, dropout_rate):
        super().__init__(hidden_dim=hidden_dim, dropout_rate=dropout_rate)

    def _preprocess_target(self, N):
        """Accept N as a 1-D array of integers; return a 2-D z-scored target."""
        N_arr = np.asarray(N, dtype=np.float32).reshape(-1, 1)
        self.target_scaler_ = StandardScaler()
        return self.target_scaler_.fit_transform(N_arr)

    def _postprocess_prediction(self, y_scaled):
        """Inverse z-score back to native units; squeeze to 1-D for callers."""
        return self.target_scaler_.inverse_transform(y_scaled).ravel()

    def predict_round(self, X):
        return np.round(self.predict(X)).astype(np.int32)


def whitened_pca_features(rgba_initial, pixel_pca_dim, *, fit_pca=True,
                          existing_scaler=None, existing_pca=None):
    """Build whitened pixel-PCA features from RGBA initial frames.

    Parameters
    ----------
    rgba_initial : float32 [n_scenes, IMAGE_SIZE**2 * 4]
    pixel_pca_dim : int
    fit_pca : bool
        If True, fit a fresh StandardScaler + PCA. If False, the caller must
        supply ``existing_scaler`` and ``existing_pca`` (used to apply the
        same transform to a held-out / downstream batch).

    Returns
    -------
    features : float32 [n_scenes, pixel_pca_dim]
    scaler   : the StandardScaler used (newly fit or passed through)
    pca      : the PCA used (newly fit or passed through)
    """
    if fit_pca:
        scaler = StandardScaler()
        pca = PCA(n_components=pixel_pca_dim, whiten=True, random_state=42)
        feats = pca.fit_transform(scaler.fit_transform(rgba_initial))
    else:
        if existing_scaler is None or existing_pca is None:
            raise ValueError("fit_pca=False requires existing_scaler and existing_pca.")
        feats = existing_pca.transform(existing_scaler.transform(rgba_initial))
        scaler, pca = existing_scaler, existing_pca
    return feats.astype(np.float32), scaler, pca
