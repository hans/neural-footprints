"""
Predictive Processing model of visual perception.

Two-stage model:
  (1) InverseModel: image(t=0) + image(t=early) → inferred physical state
      - Two-frame input enables velocity inference from optical displacement
      - MC Dropout: dropout active at inference time → distribution over physical states
  (2) Forward model: PyBullet resimulate_scene (exact physics engine, not learned)

Behavioral comparison (next-frame pixel R²):
  Prior MLP:    true physics_labels → [MLP] → final pixels        (architecture ceiling)
  Render-only:  pixel_pca_two_frame → [MLP] → final pixels         (no physics intermediate)
  PP chain:     pixel_pca_two_frame → [InverseModel] → physics → PyBullet → final pixels
  Oracle:       true physics_labels → PyBullet → final pixels      (deterministic upper bound)

Scientific question: does the PP model's explicit physics intermediate become
detectable in neural encoding analyses? Expected: no — inferred physics has
low neural R², mirroring true physics_labels in the dissociation analysis.

This module returns numerical results + plot data; figure rendering lives in
scripts/plot_pp.py to match the pre-pp pipeline pattern.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    N_OBJECTS,
    IMAGE_SIZE,
    PP_HIDDEN_DIM as _CFG_PP_HIDDEN_DIM,
    PP_PIXEL_PCA_DIM as _CFG_PP_PIXEL_PCA_DIM,
    PP_DROPOUT_RATE as _CFG_PP_DROPOUT_RATE,
)
from models import (
    InverseMLPNet, InverseCNNNet, SpatialSoftmaxV2, build_frame_stack,
)


# ---------------------------------------------------------------------------
# Neural R² helper (RidgeCV per neuron, cross-validated)
# ---------------------------------------------------------------------------

_RIDGE_ALPHAS = np.logspace(-2, 6, 20)


def _mean_neural_r2(features, neural_activity):
    """Cross-validated Ridge R², averaged over neurons. Returns (mean, per_neuron)."""
    ridge = RidgeCV(alphas=_RIDGE_ALPHAS, alpha_per_target=True)
    predictions = cross_val_predict(ridge, features, neural_activity, cv=5)
    ss_res = ((neural_activity - predictions) ** 2).sum(axis=0)
    ss_tot = ((neural_activity - neural_activity.mean(axis=0)) ** 2).sum(axis=0)
    per_neuron = 1 - ss_res / ss_tot
    return float(per_neuron.mean()), per_neuron


# ---------------------------------------------------------------------------
# Wrapper classes
# ---------------------------------------------------------------------------
#
# nn.Module architectures live in ``models/``. This module owns the
# fit/predict/scaler-aware wrappers around them.
#
# Public API contract (shared across all three wrappers):
#   .fit(X, physics_labels, ...)                  X = wrapper.prepare_input(scenes)
#   .predict(X)                  → (n, full_physics_dim)
#   .predict_stochastic(X, n_samples=N)
#                                → (N, n, full_physics_dim) — MC dropout samples
#   .extract_activations(X, layer)
#                                → (n, layer_dim)
#   .prepare_input(scenes)       → X (backbone-specific input array)
#
# State attrs (used by analyses/pp_io.py round-trip):
#   .net_, .phys_scaler_, .valid_dims_, .per_dim_r2_,
#   .full_physics_dim_, .const_values_, .input_scaler_ (mlp only — None elsewhere).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# InverseModel (MLP backbone)
# ---------------------------------------------------------------------------

class InverseModel:
    """
    Maps two-frame pixel PCA → inferred physical state.

    Trains with dropout. predict() turns dropout off; predict_stochastic()
    keeps it on for MC dropout sampling.

    Restricts the regression target to dimensions that are (a) in principle
    observable from the three pixel frames (position from one frame, linear
    velocity from two, x_accel from three via second-difference) and (b)
    actually have variance across the dataset. Mass, friction, orientation,
    and angular velocity are filled with the training-set mean on predict()
    since pixels can't recover them.

    Why mean-fill of the non-observable dims is dynamically safe in this
    scene generator (i.e. resimulating from inferred state still produces
    the right trajectory, even though four of the 15 dims per object are
    wrong per scene):

      - Orientation: scene_generator always initializes with the identity
        quaternion, so the dataset mean equals every scene's ground truth.
      - Angular velocity: never set at t=0 (PyBullet default = 0), so the
        initial value is always zero and the mean = GT.
      - Mass: cancels out of the dynamics. Gravity is mass-independent, and
        the per-scene x-acceleration is applied as force = x_accel * mass
        (scene_generator._create_scene + resimulate_scene), so PyBullet's
        a = F/m yields a = x_accel regardless of the mass passed in. There
        are no inter-object collisions (single object per scene; pillar has
        no collision shape), so there's no regime where mass matters.
      - Friction: only enters during ground contact. Most 30-step rollouts
        don't reach the ground; when contact does occur, sliding decel is
        μ·g, and the gap between true μ and the dataset mean produces only
        sub-pixel-scale trajectory deviations.

    Consequence: the headline pp_r2 metric explicitly fills non-observable
    dims with GT before resimulating, but the frame visualization in
    run_predictive_processing_analysis can safely skip that step — the
    resulting figure is unbiased relative to the metric (only difference
    is single-MC-sample vs. 20-sample mean for the observable dims).

    If scene_generator changes in ways that break the above (inter-object
    collisions, longer rollouts that guarantee ground contact, randomized
    initial orientation, non-zero initial angular velocity), revisit this:
    mean-fill would then introduce bias and the visualization should mirror
    the pp_r2 GT-fill loop.
    """

    def __init__(self, pixel_pca_dim=None, hidden_dim=None, dropout_rate=None):
        self.pixel_pca_dim = pixel_pca_dim if pixel_pca_dim is not None else _CFG_PP_PIXEL_PCA_DIM
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.net_ = None
        self.input_scaler_ = None
        self.phys_scaler_ = None
        self.per_dim_r2_ = None
        self.valid_dims_ = None
        self.full_physics_dim_ = None
        self.const_values_ = None

    def prepare_input(self, scenes):
        """Three-frame whitened pixel-PCA concatenation — the InverseModel input."""
        return build_pp_features(scenes, pixel_pca_dim=self.pixel_pca_dim)['pixel_pca_concat']

    def fit(self, pixel_pca_two_frame, physics_labels,
            n_epochs=300, batch_size=64, lr=1e-3, val_frac=0.15, patience=50):
        self.full_physics_dim_ = physics_labels.shape[1]

        # Stride of 16 per object: pos(0:3), orn(3:7), lin_vel(7:10),
        # ang_vel(10:13), mass(13), friction(14), x_accel(15). Pixels can
        # recover position (directly visible from one frame), linear velocity
        # (from inter-frame displacement, two frames), and x_accel (from
        # second-difference, three frames at t={0, EARLY, LATE}). Orn / avel /
        # mass / friction are filtered out of the target — see the dynamics-
        # safety reasoning in the class docstring.
        observable_offsets = list(range(0, 3)) + list(range(7, 10)) + [15]
        observable_indices = []
        for i in range(N_OBJECTS):
            observable_indices.extend([i * 16 + j for j in observable_offsets])

        std_per_dim = physics_labels.std(axis=0)
        has_variance = std_per_dim > 1e-4
        observable_mask = np.zeros(self.full_physics_dim_, dtype=bool)
        observable_mask[observable_indices] = True

        self.valid_dims_ = observable_mask & has_variance
        self.const_values_ = physics_labels.mean(axis=0)
        n_valid = self.valid_dims_.sum()
        n_observable = observable_mask.sum()
        print(f"    InverseModel: {n_valid}/{n_observable} observable physics dims have variance "
              f"({self.full_physics_dim_} total)")

        physics_valid = physics_labels[:, self.valid_dims_]

        self.input_scaler_ = StandardScaler()
        X = self.input_scaler_.fit_transform(pixel_pca_two_frame)
        self.phys_scaler_ = StandardScaler()
        y = self.phys_scaler_.fit_transform(physics_valid)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_frac, random_state=42
        )

        input_dim = X.shape[1]
        output_dim = y.shape[1]
        self.net_ = InverseMLPNet(
            input_dim, output_dim,
            hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate,
        )

        optimizer = torch.optim.Adam(self.net_.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10
        )
        loss_fn = nn.MSELoss()

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_count = 0
        best_state = None

        for epoch in range(n_epochs):
            self.net_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(self.net_(xb), yb).backward()
                optimizer.step()

            self.net_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net_(X_val_t), y_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"    InverseModel early stop at epoch {epoch+1} (val loss={best_val_loss:.4f})")
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)

        self.net_.eval()
        with torch.no_grad():
            y_pred_val = self.net_(X_val_t).numpy()
        self.per_dim_r2_ = r2_score(y_val, y_pred_val, multioutput='raw_values')

        print(f"    InverseModel val MSE={best_val_loss:.4f}  "
              f"mean per-dim R²={self.per_dim_r2_.mean():.4f}  "
              f"max={self.per_dim_r2_.max():.4f}")
        return self

    def _expand_to_full(self, valid_predictions):
        n = valid_predictions.shape[0]
        full = np.tile(self.const_values_, (n, 1))
        full[:, self.valid_dims_] = valid_predictions
        return full

    def predict(self, pixel_pca_two_frame):
        """Deterministic prediction (dropout off). Returns physics in original units, full dims."""
        self.net_.eval()
        X = torch.tensor(
            self.input_scaler_.transform(pixel_pca_two_frame), dtype=torch.float32
        )
        with torch.no_grad():
            pred_scaled = self.net_(X).numpy()
        valid_preds = self.phys_scaler_.inverse_transform(pred_scaled)
        return self._expand_to_full(valid_preds)

    def predict_stochastic(self, pixel_pca_two_frame, n_samples=8):
        """MC dropout: stochastic forward passes with dropout active."""
        self.net_.train()
        X = torch.tensor(
            self.input_scaler_.transform(pixel_pca_two_frame), dtype=torch.float32
        )
        samples_scaled = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples_scaled.append(self.net_(X).numpy())
        self.net_.eval()
        return np.stack([
            self._expand_to_full(self.phys_scaler_.inverse_transform(s))
            for s in samples_scaled
        ])

    def extract_activations(self, pixel_pca_two_frame, layer='h2'):
        """Deterministic post-ReLU activations of one hidden layer.

        layer: 'h1' | 'h2' | 'h3'. Dropout is off (eval mode), so the returned
        activations are the deterministic representation the net would use at
        inference time, not an MC sample.
        """
        if layer not in ('h1', 'h2', 'h3'):
            raise ValueError(f"layer must be 'h1', 'h2', or 'h3'; got {layer!r}")
        self.net_.eval()
        X = torch.tensor(
            self.input_scaler_.transform(pixel_pca_two_frame), dtype=torch.float32
        )
        with torch.no_grad():
            _, acts = self.net_.forward_with_activations(X)
        return acts[layer].numpy()


# ---------------------------------------------------------------------------
# CNN inverse model (Phase 2 of specs/inverse_model_input_repr.md)
#
# Skips pixel PCA entirely. Same head dims (256→256→128→n_observable) so the
# `extract_activations(layer='h2')` interface is interchangeable with
# InverseModel for downstream neural projection.
#
# Not currently in the pp_inverse_backbone dispatch — kept available for the
# off-pipeline diagnostic in scripts/eval_pp_cnn.py.
# ---------------------------------------------------------------------------


class InverseCNN:
    """CNN-based inverse model: raw 3-frame stack → physics.

    API mirrors :class:`InverseModel` (``fit`` / ``predict`` / ``extract_activations``)
    so it can be swapped in without changes to ``run_predictive_processing_analysis``
    or ``gen_neural.py`` if Phase 2 of ``specs/inverse_model_input_repr.md`` wins.

    Input convention for ``fit`` / ``predict`` / ``extract_activations``: a numpy
    array of shape ``(N, n_frames, n_channels, H, W)``. uint8 inputs are scaled
    to [0, 1]; float inputs are passed through (assumed pre-normalized).
    """

    def __init__(self, hidden_dim=None, dropout_rate=None):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.net_ = None
        self.phys_scaler_ = None
        self.per_dim_r2_ = None
        self.valid_dims_ = None
        self.full_physics_dim_ = None
        self.const_values_ = None
        self.history_ = None
        self.best_epoch_ = None

    @staticmethod
    def _prep_frames(frames):
        if frames.dtype == np.uint8 or frames.max() > 1.5:
            return frames.astype(np.float32) / 255.0
        return frames.astype(np.float32)

    def fit(self, frames, physics_labels,
            n_epochs=300, batch_size=64, lr=1e-3, val_frac=0.15, patience=50,
            verbose=True):
        frames = self._prep_frames(frames)
        self.full_physics_dim_ = physics_labels.shape[1]

        observable_offsets = list(range(0, 3)) + list(range(7, 10)) + [15]
        observable_indices = []
        for i in range(N_OBJECTS):
            observable_indices.extend([i * 16 + j for j in observable_offsets])

        std_per_dim = physics_labels.std(axis=0)
        has_variance = std_per_dim > 1e-4
        observable_mask = np.zeros(self.full_physics_dim_, dtype=bool)
        observable_mask[observable_indices] = True
        self.valid_dims_ = observable_mask & has_variance
        self.const_values_ = physics_labels.mean(axis=0)

        physics_valid = physics_labels[:, self.valid_dims_]
        self.phys_scaler_ = StandardScaler()
        y = self.phys_scaler_.fit_transform(physics_valid)

        idx = np.arange(frames.shape[0])
        idx_tr, idx_val = train_test_split(idx, test_size=val_frac, random_state=42)

        n_frames, n_channels = frames.shape[1], frames.shape[2]
        self.net_ = InverseCNNNet(
            n_frames=n_frames, n_channels=n_channels, output_dim=y.shape[1],
            hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate,
        )
        opt = torch.optim.Adam(self.net_.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
        loss_fn = nn.MSELoss()

        X_tr_t = torch.tensor(frames[idx_tr], dtype=torch.float32)
        y_tr_t = torch.tensor(y[idx_tr], dtype=torch.float32)
        X_val_t = torch.tensor(frames[idx_val], dtype=torch.float32)
        y_val_t = torch.tensor(y[idx_val], dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                            batch_size=batch_size, shuffle=True)

        best_val = float('inf')
        best_state = None
        best_epoch = 0
        bad_epochs = 0
        history = []
        for epoch in range(n_epochs):
            self.net_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.net_(xb), yb).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                vl = loss_fn(self.net_(X_val_t), y_val_t).item()
                tl = loss_fn(self.net_(X_tr_t), y_tr_t).item()
            history.append({'epoch': epoch + 1, 'train_loss': tl, 'val_loss': vl})
            sch.step(vl)
            if vl < best_val - 1e-5:
                best_val = vl
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                best_epoch = epoch + 1
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if verbose:
                        print(f"    InverseCNN early stop at epoch {epoch+1} "
                              f"(best val={best_val:.4f} @ epoch {best_epoch})")
                    break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        with torch.no_grad():
            y_pred_val = self.net_(X_val_t).numpy()
        self.per_dim_r2_ = r2_score(y[idx_val], y_pred_val, multioutput='raw_values')
        self.history_ = history
        self.best_epoch_ = best_epoch
        if verbose:
            print(f"    InverseCNN val MSE={best_val:.4f}  "
                  f"mean per-dim R²={self.per_dim_r2_.mean():.4f}  "
                  f"max={self.per_dim_r2_.max():.4f}")
        return self

    def _expand_to_full(self, valid_predictions):
        n = valid_predictions.shape[0]
        full = np.tile(self.const_values_, (n, 1))
        full[:, self.valid_dims_] = valid_predictions
        return full

    def _forward_in_batches(self, frames, batch_size=128, return_acts=None):
        """Run the net over frames in batches; return ndarray (or dict of arrays)."""
        self.net_.eval()
        X = torch.tensor(self._prep_frames(frames), dtype=torch.float32)
        outs = []
        acts = {k: [] for k in (return_acts or [])}
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                xb = X[i:i + batch_size]
                if return_acts:
                    out, a = self.net_.forward_with_activations(xb)
                    for k in return_acts:
                        acts[k].append(a[k].numpy())
                else:
                    out = self.net_(xb)
                outs.append(out.numpy())
        out = np.concatenate(outs, axis=0)
        if return_acts:
            return out, {k: np.concatenate(acts[k], axis=0) for k in return_acts}
        return out

    def prepare_input(self, scenes):
        """Stacked 3-frame uint8 tensor — fed straight into the conv tower."""
        return build_frame_stack(scenes)

    def predict(self, frames):
        pred_scaled = self._forward_in_batches(frames)
        return self._expand_to_full(self.phys_scaler_.inverse_transform(pred_scaled))

    def extract_activations(self, frames, layer='h2'):
        """Deterministic post-ReLU activations of one hidden layer (h1/h2/h3)."""
        if layer not in ('h1', 'h2', 'h3'):
            raise ValueError(f"layer must be 'h1', 'h2', or 'h3'; got {layer!r}")
        _, acts = self._forward_in_batches(frames, return_acts=[layer])
        return acts[layer]


# ---------------------------------------------------------------------------
# InverseSoftmaxCNN (SpatialSoftmaxV2 backbone)
#
# CLEANUP: physics-dim filtering (observable mask + variance + const-fill) is
# duplicated across InverseModel / InverseCNN / InverseSoftmaxCNN. Factoring
# it into a small helper is a separate refactor — explicitly out of scope for
# the current pluggable-backbone change.
# ---------------------------------------------------------------------------

class InverseSoftmaxCNN:
    """Spatial-softmax keypoint inverse model: raw 3-frame stack → physics.

    Backbone is :class:`models.spatial_softmax.SpatialSoftmaxV2`. API mirrors
    :class:`InverseModel` so it is interchangeable in
    ``run_predictive_processing_analysis`` and ``train_pp_for_neural``.

    MC-dropout uncertainty vs val-R² tradeoff: dropout lives between the
    head's hidden Linears. ``predict_stochastic`` puts the whole net in
    ``train()`` mode and pins the conv tower back to eval. Empirically,
    rate ≥ 0.05 costs ~0.2 val R² on the v2_128_temp_mlp config (initial
    plan estimate of "within sweep noise" was off — measured 0.62 vs the
    no-dropout sweep's 0.83). Default is 0.0 so the wrapper reproduces the
    sweep numbers; with that setting ``predict_stochastic`` returns
    ``n_samples`` identical forward passes (mean is correct, variance is
    zero). Dial up only if the analysis specifically needs MC-dropout
    uncertainty bands.
    """

    def __init__(self, *, n_filters=128, learned_temp=True, temp_per_channel=True,
                 include_variance=False, hidden_dim=256, head_depth=3,
                 dropout_rate=0.0):
        self.n_filters = n_filters
        self.learned_temp = learned_temp
        self.temp_per_channel = temp_per_channel
        self.include_variance = include_variance
        self.hidden_dim = hidden_dim
        self.head_depth = head_depth
        self.dropout_rate = dropout_rate

        self.net_ = None
        self.input_scaler_ = None  # raw frames; no scaler.
        self.phys_scaler_ = None
        self.per_dim_r2_ = None
        self.valid_dims_ = None
        self.full_physics_dim_ = None
        self.const_values_ = None
        self.history_ = None
        self.best_epoch_ = None

    @staticmethod
    def _prep_frames(frames):
        if frames.dtype == np.uint8 or frames.max() > 1.5:
            return frames.astype(np.float32) / 255.0
        return frames.astype(np.float32)

    def prepare_input(self, scenes):
        """Stacked 3-frame uint8 tensor — same convention as InverseCNN."""
        return build_frame_stack(scenes)

    def fit(self, frames, physics_labels,
            n_epochs=200, batch_size=64, lr=1e-3, val_frac=0.15,
            patience=30, min_epochs=60, verbose=True):
        frames = self._prep_frames(frames)
        self.full_physics_dim_ = physics_labels.shape[1]

        observable_offsets = list(range(0, 3)) + list(range(7, 10)) + [15]
        observable_indices = []
        for i in range(N_OBJECTS):
            observable_indices.extend([i * 16 + j for j in observable_offsets])

        std_per_dim = physics_labels.std(axis=0)
        has_variance = std_per_dim > 1e-4
        observable_mask = np.zeros(self.full_physics_dim_, dtype=bool)
        observable_mask[observable_indices] = True
        self.valid_dims_ = observable_mask & has_variance
        self.const_values_ = physics_labels.mean(axis=0)
        n_valid = int(self.valid_dims_.sum())
        n_observable = int(observable_mask.sum())
        if verbose:
            print(f"    InverseSoftmaxCNN: {n_valid}/{n_observable} observable physics dims have variance "
                  f"({self.full_physics_dim_} total)")

        physics_valid = physics_labels[:, self.valid_dims_]
        self.phys_scaler_ = StandardScaler()
        y = self.phys_scaler_.fit_transform(physics_valid)

        idx = np.arange(frames.shape[0])
        idx_tr, idx_val = train_test_split(idx, test_size=val_frac, random_state=42)

        # Match the sweep script's per-config seeding — keeps the val R² obtained
        # in scripts/eval_pp_cnn_softmax_sweep.py reproducible inside the pipeline.
        torch.manual_seed(42)
        n_frames, n_channels = frames.shape[1], frames.shape[2]
        self.net_ = SpatialSoftmaxV2(
            n_frames=n_frames, n_channels=n_channels, image_size=IMAGE_SIZE,
            output_dim=y.shape[1],
            n_filters=self.n_filters,
            learned_temp=self.learned_temp,
            temp_per_channel=self.temp_per_channel,
            include_variance=self.include_variance,
            hidden_dim=self.hidden_dim,
            head_depth=self.head_depth,
            dropout_rate=self.dropout_rate,
        )

        opt = torch.optim.Adam(self.net_.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
        loss_fn = nn.MSELoss()

        X_tr_t = torch.tensor(frames[idx_tr], dtype=torch.float32)
        y_tr_t = torch.tensor(y[idx_tr], dtype=torch.float32)
        X_val_t = torch.tensor(frames[idx_val], dtype=torch.float32)
        y_val_t = torch.tensor(y[idx_val], dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                            batch_size=batch_size, shuffle=True)

        best_val = float('inf')
        best_state = None
        best_epoch = 0
        bad_epochs = 0
        history = []
        for epoch in range(n_epochs):
            self.net_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.net_(xb), yb).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                vl = loss_fn(self.net_(X_val_t), y_val_t).item()
                tl = loss_fn(self.net_(X_tr_t), y_tr_t).item()
            history.append({'epoch': epoch + 1, 'train_loss': tl, 'val_loss': vl})
            sch.step(vl)
            if vl < best_val - 1e-5:
                best_val = vl
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                best_epoch = epoch + 1
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience and (epoch + 1) >= min_epochs:
                    if verbose:
                        print(f"    InverseSoftmaxCNN early stop at epoch {epoch+1} "
                              f"(best val={best_val:.4f} @ epoch {best_epoch})")
                    break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        with torch.no_grad():
            y_pred_val = self.net_(X_val_t).numpy()
        self.per_dim_r2_ = r2_score(y[idx_val], y_pred_val, multioutput='raw_values')
        self.history_ = history
        self.best_epoch_ = best_epoch
        if verbose:
            print(f"    InverseSoftmaxCNN val MSE={best_val:.4f}  "
                  f"mean per-dim R²={self.per_dim_r2_.mean():.4f}  "
                  f"max={self.per_dim_r2_.max():.4f}")
        return self

    def _expand_to_full(self, valid_predictions):
        n = valid_predictions.shape[0]
        full = np.tile(self.const_values_, (n, 1))
        full[:, self.valid_dims_] = valid_predictions
        return full

    def _forward_in_batches(self, frames, batch_size=128, return_acts=None,
                            stochastic=False):
        """Run the net over frames in batches; return ndarray (or dict of arrays).

        ``stochastic`` toggles the head's dropout on (whole net is set to
        train(), then conv tower is pinned back to eval — see class docstring).
        """
        if stochastic:
            self.net_.train()
            self.net_.conv.eval()
        else:
            self.net_.eval()
        X = torch.tensor(self._prep_frames(frames), dtype=torch.float32)
        outs = []
        acts = {k: [] for k in (return_acts or [])}
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                xb = X[i:i + batch_size]
                if return_acts:
                    out, a = self.net_.forward_with_activations(xb)
                    for k in return_acts:
                        acts[k].append(a[k].numpy())
                else:
                    out = self.net_(xb)
                outs.append(out.numpy())
        if stochastic:
            self.net_.eval()
        out = np.concatenate(outs, axis=0)
        if return_acts:
            return out, {k: np.concatenate(acts[k], axis=0) for k in return_acts}
        return out

    def predict(self, frames):
        pred_scaled = self._forward_in_batches(frames)
        return self._expand_to_full(self.phys_scaler_.inverse_transform(pred_scaled))

    def predict_stochastic(self, frames, n_samples=8):
        """MC-dropout: head dropout active for n_samples forward passes."""
        samples = []
        for _ in range(n_samples):
            pred_scaled = self._forward_in_batches(frames, stochastic=True)
            samples.append(self._expand_to_full(
                self.phys_scaler_.inverse_transform(pred_scaled)
            ))
        return np.stack(samples)

    def extract_activations(self, frames, layer='h2'):
        """Deterministic post-ReLU activations of one head layer (h1/h2/h3).

        Layer aliasing follows :class:`models.spatial_softmax.SpatialSoftmaxV2`:
        when ``head_depth < 3``, ``h3`` aliases to ``h2``; when
        ``head_depth < 2``, both ``h2`` and ``h3`` alias to ``h1``.
        """
        if layer not in ('h1', 'h2', 'h3'):
            raise ValueError(f"layer must be 'h1', 'h2', or 'h3'; got {layer!r}")
        _, acts = self._forward_in_batches(frames, return_acts=[layer])
        return acts[layer]


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

INVERSE_BACKBONES = ('mlp', 'softmax_cnn')


def make_inverse_model(backbone='mlp', **kwargs):
    """Build an inverse-model wrapper for the named backbone.

    Currently dispatches:
        'mlp'         → :class:`InverseModel`
        'softmax_cnn' → :class:`InverseSoftmaxCNN`

    The 'cnn' backbone (:class:`InverseCNN`) is intentionally excluded from
    the dispatch — it is kept available as a class for the off-pipeline
    diagnostic in scripts/eval_pp_cnn.py only.
    """
    if backbone == 'mlp':
        return InverseModel(**kwargs)
    if backbone == 'softmax_cnn':
        return InverseSoftmaxCNN(**kwargs)
    raise ValueError(
        f"unknown pp_inverse_backbone {backbone!r}; expected one of {INVERSE_BACKBONES}"
    )


# ---------------------------------------------------------------------------
# Helper MLPs and pixel R²
# ---------------------------------------------------------------------------

def _make_prior_mlp():
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(_CFG_PP_HIDDEN_DIM, _CFG_PP_HIDDEN_DIM),
            activation='relu', max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
        )
    )


def _make_render_mlp():
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(_CFG_PP_HIDDEN_DIM, _CFG_PP_HIDDEN_DIM),
            activation='relu', max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
        )
    )


def _eval_physics_to_pixel_mlp(physics_labels, final_pixel_pca,
                               train_idx, test_idx,
                               pca_final, scaler_pix,
                               actual_final_pixels_test):
    mlp = _make_prior_mlp()
    mlp.fit(physics_labels[train_idx], final_pixel_pca[train_idx])
    pred_pca = mlp.predict(physics_labels[test_idx])
    pred_raw = np.clip(
        scaler_pix.inverse_transform(pca_final.inverse_transform(pred_pca)), 0, 255
    ).astype(np.float32)
    actual = actual_final_pixels_test.astype(np.float32)
    ss_res = np.sum((actual - pred_raw) ** 2)
    ss_tot = np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def _pixel_r2(predicted_rgba, actual_raw_pixels):
    actual = actual_raw_pixels.astype(np.float32)
    predicted = predicted_rgba.astype(np.float32)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def physics_groups():
    """Maps semantic label type → list of dimension indices (stride 16 per object)."""
    return {
        'Position':    [i * 16 + j for i in range(N_OBJECTS) for j in range(3)],
        'Orientation': [i * 16 + j for i in range(N_OBJECTS) for j in range(3, 7)],
        'Lin. Vel.':   [i * 16 + j for i in range(N_OBJECTS) for j in range(7, 10)],
        'Ang. Vel.':   [i * 16 + j for i in range(N_OBJECTS) for j in range(10, 13)],
        'Mass':        [i * 16 + 13 for i in range(N_OBJECTS)],
        'Friction':    [i * 16 + 14 for i in range(N_OBJECTS)],
        'Accel':       [i * 16 + 15 for i in range(N_OBJECTS)],
    }


def build_pp_features(scenes, pixel_pca_dim=None):
    """Compute the InverseModel input features from a scenes dict.

    Three-frame whitened pixel PCAs at t={0, PP_EARLY_FRAME, PP_LATE_FRAME},
    concatenated. Used identically by scripts/train_pp_for_neural.py (to fit
    the model once early in the pipeline) and by run_predictive_processing_analysis
    (to apply the loaded checkpoint and produce inferred-physics for analysis).
    Keeping it in one place avoids the silent failure mode where mismatched PCAs
    let the loaded weights apply to the wrong feature basis.

    Returns
    -------
    dict with keys:
        pixel_pca_t0      : (n, pixel_pca_dim) array, t=0 PCA features
        pixel_pca_early   : (n, pixel_pca_dim) array, t=PP_EARLY_FRAME features
        pixel_pca_late    : (n, pixel_pca_dim) array, t=PP_LATE_FRAME features
        pixel_pca_concat  : (n, 3*pixel_pca_dim), the InverseModel input
    """
    if pixel_pca_dim is None:
        pixel_pca_dim = _CFG_PP_PIXEL_PCA_DIM

    initial_renders = scenes['initial_renders']
    early_renders   = scenes['early_renders']
    late_renders    = scenes['late_renders']

    def _whitened_pca(x):
        scaler = StandardScaler()
        pca = PCA(n_components=pixel_pca_dim, whiten=True, random_state=42)
        return pca.fit_transform(scaler.fit_transform(x))

    pixel_pca_t0    = _whitened_pca(initial_renders)
    pixel_pca_early = _whitened_pca(early_renders)
    pixel_pca_late  = _whitened_pca(late_renders)

    return {
        'pixel_pca_t0': pixel_pca_t0,
        'pixel_pca_early': pixel_pca_early,
        'pixel_pca_late': pixel_pca_late,
        'pixel_pca_concat': np.concatenate(
            [pixel_pca_t0, pixel_pca_early, pixel_pca_late], axis=1
        ),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_predictive_processing_analysis(neural_activity, scenes,
                                       *, pixel_pca_dim=None, n_oracle=200,
                                       inv_model=None):
    """
    Evaluate the predictive-processing pipeline against neural activity.

    If ``inv_model`` is provided (an already-trained ``InverseModel`` instance),
    use it for all inference and skip in-function training. This is the path
    taken when neural activity was generated from PP-model activations: the
    same checkpoint must be reused so the inferred-physics array fed to
    downstream encoding/RSA matches what was projected into neural activity.

    If ``inv_model`` is None, fit a fresh InverseModel on the analysis-side
    train split (legacy behavior — only useful when neural activity does not
    depend on PP).

    Returns a results dict with scalar metrics, the per-scene
    inferred_physics_all array (for downstream encoding/RSA), and
    plot_data with arrays needed by scripts/plot_pp.py.
    """
    from scene_generator import resimulate_scene, open_render_client
    import pybullet as p

    if pixel_pca_dim is None:
        pixel_pca_dim = _CFG_PP_PIXEL_PCA_DIM

    print("\n" + "=" * 60)
    print("SIMULATION 4: Predictive Processing Model")
    print("=" * 60)

    pixel_indices        = scenes['metadata']['pixel_indices']
    target_pixel_indices = scenes['metadata']['target_pixel_indices']
    initial_renders = scenes['initial_renders']
    early_renders   = scenes['early_renders']
    late_renders    = scenes['late_renders']
    target_renders  = scenes['target_renders']
    initial_physics = scenes['initial_physics_labels']
    scene_configs   = scenes['scene_configs']
    program_states  = scenes['program_states']
    pillar_grays    = scenes['pillar_grays']
    lightings       = scenes['lightings']
    n = len(initial_renders)

    # --- 1. Shared pixel PCAs ---
    print("\nPreparing pixel representations...")

    # Behavioral target = t=N_TIMESTEPS render (held out from program_state).
    # Match dissociation.py: compare against target_renders, not the LATE
    # (t=PP_LATE_FRAME) slice of program_state.
    final_pixels_raw = target_renders[:, target_pixel_indices]
    scaler_pix = StandardScaler()
    pix_scaled = scaler_pix.fit_transform(final_pixels_raw)
    pca_final = PCA(n_components=pixel_pca_dim, whiten=True, random_state=42)
    final_pixel_pca = pca_final.fit_transform(pix_scaled)

    # Pixel-PCA features used for the render-only baseline + neural-R² readouts.
    # The inverse model gets its own input via ``inv_model.prepare_input(scenes)``
    # below — the MLP backbone re-uses ``pixel_pca_concat``; raw-pixel backbones
    # ignore these PCA features and consume the 3-frame stack instead.
    feats = build_pp_features(scenes, pixel_pca_dim=pixel_pca_dim)
    pixel_pca_t0       = feats['pixel_pca_t0']
    pixel_pca_early    = feats['pixel_pca_early']
    pixel_pca_late     = feats['pixel_pca_late']
    pixel_pca_two_frame = feats['pixel_pca_concat']

    # --- 2. Train / test split ---
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_test = int(n * 0.2)
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    actual_test_raw = final_pixels_raw[test_idx].astype(np.float32)
    n_oracle = min(n_oracle, n_test)
    oracle_test_idx = test_idx[:n_oracle]

    # --- 3. Prior eval: true physics → pixels (architecture ceiling) ---
    print("\nPrior evaluation: MLP(true physics → pixels)...")
    prior_r2 = _eval_physics_to_pixel_mlp(
        initial_physics, final_pixel_pca,
        train_idx, test_idx,
        pca_final, scaler_pix,
        actual_test_raw,
    )
    print(f"  Prior MLP R²: {prior_r2:.4f}")

    # --- 4. InverseModel (load pretrained, or fit fresh on analysis train split) ---
    if inv_model is None:
        print("\nFitting InverseModel (two-frame pixels → physics, MC dropout)...")
        # Match the wrapper's pixel_pca_dim to the analyzer's so a later
        # inv_model.prepare_input(scenes) call rebuilds the same feature basis
        # the model was trained on.
        inv_model = InverseModel(pixel_pca_dim=pixel_pca_dim)
        inv_model.fit(pixel_pca_two_frame[train_idx], initial_physics[train_idx])
    else:
        print("\nReusing pretrained InverseModel (same checkpoint used for neural projection).")
        print(f"    InverseModel val per-dim R²: mean={inv_model.per_dim_r2_.mean():.4f}  "
              f"max={inv_model.per_dim_r2_.max():.4f}")

    # Backbone-specific input array for all subsequent inv_model.predict* calls.
    inv_input = inv_model.prepare_input(scenes)

    # --- 5. Render-only baseline ---
    print("\nFitting render-only baseline (two-frame pixels → pixels)...")
    render_mlp = _make_render_mlp()
    render_mlp.fit(pixel_pca_two_frame[train_idx], final_pixel_pca[train_idx])

    # --- 6. Oracle (deterministic PyBullet with true physics) ---
    # Renderer must match scene generation (use_gui=True → OpenGL with shadows).
    # Same pattern as dissociation.py / dynamics.py — share one client across all
    # resim calls in this analysis.
    print(f"\nComputing oracle R² ({n_oracle} test scenes)...")
    resim_pc = open_render_client(use_gui=True)
    try:
        oracle_preds = np.stack([
            resimulate_scene(scene_configs[i], initial_physics[i],
                             pillar_gray=pillar_grays[i],
                             lighting=lightings[i],
                             use_gui=True, physics_client=resim_pc).reshape(-1).astype(np.float32)
            for i in oracle_test_idx
        ])
        oracle_actual = final_pixels_raw[oracle_test_idx].astype(np.float32)
        oracle_r2 = _pixel_r2(oracle_preds, oracle_actual)
        print(f"  Oracle R²: {oracle_r2:.4f}")

        # --- 7. PP chain (InverseModel mean → PyBullet) ---
        # Inferred physics supplies observable dims; non-observable dims (mass,
        # friction, orientation, ang_vel) are pulled from ground truth since
        # they cannot be recovered from pixels.
        print(f"\nComputing PP chain R² ({n_oracle} test scenes, MC ensemble mean)...")
        mc_samples = inv_model.predict_stochastic(inv_input[oracle_test_idx], n_samples=20)
        inferred_mean_oracle = mc_samples.mean(axis=0)
        non_observable = ~inv_model.valid_dims_
        for j in range(n_oracle):
            gt = initial_physics[oracle_test_idx[j]]
            inferred_mean_oracle[j, non_observable] = gt[non_observable]
        pp_preds = np.stack([
            resimulate_scene(scene_configs[oracle_test_idx[j]], inferred_mean_oracle[j],
                             pillar_gray=pillar_grays[oracle_test_idx[j]],
                             lighting=lightings[oracle_test_idx[j]],
                             use_gui=True, physics_client=resim_pc).reshape(-1).astype(np.float32)
            for j in range(n_oracle)
        ])
        pp_r2 = _pixel_r2(pp_preds, oracle_actual)
    finally:
        p.disconnect(resim_pc)
    print(f"  PP chain R²: {pp_r2:.4f}  (gap from oracle: {oracle_r2 - pp_r2:.4f})")

    # --- 8. Render-only R² (same oracle scenes, fair comparison) ---
    print(f"\nComputing render-only R² ({n_oracle} oracle scenes)...")
    render_pred_pca = render_mlp.predict(pixel_pca_two_frame[oracle_test_idx])
    render_pred_raw = np.clip(
        scaler_pix.inverse_transform(pca_final.inverse_transform(render_pred_pca)), 0, 255
    ).astype(np.float32)
    render_r2 = _pixel_r2(render_pred_raw, oracle_actual)
    print(f"  Render-only R²: {render_r2:.4f}")

    print(f"\n  Behavioral summary:")
    print(f"    Prior MLP:   {prior_r2:.4f}  (architecture ceiling)")
    print(f"    Oracle:      {oracle_r2:.4f}  (PyBullet + true physics)")
    print(f"    PP chain:    {pp_r2:.4f}  (PyBullet + inferred physics)")
    print(f"    Render-only: {render_r2:.4f}  (MLP, no physics intermediate)")

    # --- 9. Neural R² on PP representations ---
    print("\nComputing neural R² for PP representations...")
    inferred_physics_all = inv_model.predict(inv_input)

    neural_r2_t0, _          = _mean_neural_r2(pixel_pca_t0, neural_activity)
    neural_r2_two_frame, _   = _mean_neural_r2(pixel_pca_two_frame, neural_activity)
    neural_r2_inv_physics, _ = _mean_neural_r2(inferred_physics_all, neural_activity)

    print(f"  Neural R²  t=0 pixel PCA:     {neural_r2_t0:.4f}")
    print(f"  Neural R²  two-frame PCA:     {neural_r2_two_frame:.4f}")
    print(f"  Neural R²  inferred physics:  {neural_r2_inv_physics:.4f}  ← should be low")

    # --- 10. Frame visualization data (PP chain MC dropout sample) ---
    n_frame_samples = min(8, n_oracle)
    frame_idx = oracle_test_idx[:n_frame_samples]
    frame_stochastic = inv_model.predict_stochastic(
        inv_input[frame_idx], n_samples=1
    )[0]
    viz_pc = open_render_client(use_gui=True)
    try:
        pp_frame_imgs = np.stack([
            resimulate_scene(scene_configs[frame_idx[j]], frame_stochastic[j],
                             pillar_gray=pillar_grays[frame_idx[j]],
                             lighting=lightings[frame_idx[j]],
                             use_gui=True, physics_client=viz_pc)
            for j in range(n_frame_samples)
        ])
    finally:
        p.disconnect(viz_pc)
    _rgba_s = scenes['metadata']['target_pixel_indices']  # RGBA slice within per-frame render vecs
    init_frame_imgs = initial_renders[frame_idx][:, _rgba_s].astype(np.uint8).reshape(
        n_frame_samples, IMAGE_SIZE, IMAGE_SIZE, 4)
    early_frame_imgs = early_renders[frame_idx][:, _rgba_s].astype(np.uint8).reshape(
        n_frame_samples, IMAGE_SIZE, IMAGE_SIZE, 4)
    final_frame_imgs = target_renders[frame_idx][:, target_pixel_indices].astype(np.uint8).reshape(
        n_frame_samples, IMAGE_SIZE, IMAGE_SIZE, 4)

    # Render-only baseline frames for the same scenes (two-frame MLP, no physics).
    render_frame_pred_pca = render_mlp.predict(pixel_pca_two_frame[frame_idx])
    render_frame_pred_raw = np.clip(
        scaler_pix.inverse_transform(pca_final.inverse_transform(render_frame_pred_pca)),
        0, 255,
    ).astype(np.uint8)
    render_frame_imgs = render_frame_pred_raw.reshape(
        n_frame_samples, IMAGE_SIZE, IMAGE_SIZE, 4)

    # --- 11. Per-dim InverseModel quality (full physics dim) ---
    full_per_dim_r2 = np.zeros(inv_model.full_physics_dim_)
    full_per_dim_r2[inv_model.valid_dims_] = inv_model.per_dim_r2_

    return {
        # scalar metrics (also persisted to outputs/pp_results.json)
        'prior_r2': prior_r2,
        'oracle_r2': oracle_r2,
        'pp_r2': pp_r2,
        'render_r2': render_r2,
        'neural_r2_t0': neural_r2_t0,
        'neural_r2_two_frame': neural_r2_two_frame,
        'neural_r2_inferred_physics': neural_r2_inv_physics,
        'inverse_mean_r2': float(inv_model.per_dim_r2_.mean()),
        'inverse_per_dim_r2': inv_model.per_dim_r2_,

        # downstream artifact (consumed by encoding/rsa)
        'inferred_physics_all': inferred_physics_all,

        # plot_data: arrays needed by scripts/plot_pp.py
        'plot_data': {
            'prior_r2': prior_r2,
            'oracle_r2': oracle_r2,
            'pp_r2': pp_r2,
            'render_r2': render_r2,
            'neural_r2_t0': neural_r2_t0,
            'neural_r2_two_frame': neural_r2_two_frame,
            'neural_r2_inferred_physics': neural_r2_inv_physics,
            'full_per_dim_r2': full_per_dim_r2,
            'init_frame_imgs': init_frame_imgs,
            'early_frame_imgs': early_frame_imgs,
            'pp_frame_imgs': pp_frame_imgs,
            'render_frame_imgs': render_frame_imgs,
            'final_frame_imgs': final_frame_imgs,
            'frame_idx': frame_idx,
        },
    }
