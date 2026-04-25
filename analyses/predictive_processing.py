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
# PyTorch network
# ---------------------------------------------------------------------------

class InverseMLPNet(nn.Module):
    """Three-hidden-layer MLP with dropout, kept active at inference for MC sampling."""
    def __init__(self, input_dim, output_dim,
                 hidden_dim=None, dropout_rate=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = _CFG_PP_HIDDEN_DIM
        if dropout_rate is None:
            dropout_rate = _CFG_PP_DROPOUT_RATE
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# InverseModel
# ---------------------------------------------------------------------------

class InverseModel:
    """
    Maps two-frame pixel PCA → inferred physical state.

    Trains with dropout. predict() turns dropout off; predict_stochastic()
    keeps it on for MC dropout sampling.

    Restricts the regression target to dimensions that are (a) in principle
    observable from two pixel frames (position + linear velocity) and
    (b) actually have variance across the dataset. Mass, friction,
    orientation, and angular velocity are filled with the training-set mean
    on predict() since pixels can't recover them.
    """

    def __init__(self):
        self.net_ = None
        self.input_scaler_ = None
        self.phys_scaler_ = None
        self.per_dim_r2_ = None
        self.valid_dims_ = None
        self.full_physics_dim_ = None
        self.const_values_ = None

    def fit(self, pixel_pca_two_frame, physics_labels,
            n_epochs=300, batch_size=64, lr=1e-3, val_frac=0.15, patience=50):
        self.full_physics_dim_ = physics_labels.shape[1]

        # Stride of 15 per object: pos(0:3), orn(3:7), lin_vel(7:10),
        # ang_vel(10:13), mass(13), friction(14). Pixels can recover
        # position (directly visible) and linear velocity (from inter-frame
        # displacement). Everything else is filtered out of the target.
        observable_offsets = list(range(0, 3)) + list(range(7, 10))
        observable_indices = []
        for i in range(N_OBJECTS):
            observable_indices.extend([i * 15 + j for j in observable_offsets])

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
        self.net_ = InverseMLPNet(input_dim, output_dim)

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
    """Maps semantic label type → list of dimension indices (stride 15 per object)."""
    return {
        'Position':    [i * 15 + j for i in range(N_OBJECTS) for j in range(3)],
        'Orientation': [i * 15 + j for i in range(N_OBJECTS) for j in range(3, 7)],
        'Lin. Vel.':   [i * 15 + j for i in range(N_OBJECTS) for j in range(7, 10)],
        'Ang. Vel.':   [i * 15 + j for i in range(N_OBJECTS) for j in range(10, 13)],
        'Mass':        [i * 15 + 13 for i in range(N_OBJECTS)],
        'Friction':    [i * 15 + 14 for i in range(N_OBJECTS)],
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_predictive_processing_analysis(neural_activity, scenes,
                                       *, pixel_pca_dim=None, n_oracle=200):
    """
    Train and evaluate the predictive-processing pipeline.

    Returns a results dict with scalar metrics, the per-scene
    inferred_physics_all array (for downstream encoding/RSA), and
    plot_data with arrays needed by scripts/plot_pp.py.
    """
    from scene_generator import resimulate_scene

    if pixel_pca_dim is None:
        pixel_pca_dim = _CFG_PP_PIXEL_PCA_DIM

    print("\n" + "=" * 60)
    print("SIMULATION 4: Predictive Processing Model")
    print("=" * 60)

    pixel_indices   = scenes['metadata']['pixel_indices']
    initial_renders = scenes['initial_renders']
    early_renders   = scenes['early_renders']
    initial_physics = scenes['initial_physics_labels']
    scene_configs   = scenes['scene_configs']
    program_states  = scenes['program_states']
    pillar_grays    = scenes['pillar_grays']
    lightings       = scenes['lightings']
    n = len(initial_renders)

    # --- 1. Shared pixel PCAs (RGBA-only, three time points) ---
    print("\nPreparing pixel representations...")

    final_pixels_raw = program_states[:, pixel_indices]  # final-frame RGBA bytes

    scaler_pix = StandardScaler()
    pix_scaled = scaler_pix.fit_transform(final_pixels_raw)
    pca_final = PCA(n_components=pixel_pca_dim, whiten=True, random_state=42)
    final_pixel_pca = pca_final.fit_transform(pix_scaled)

    scaler_t0 = StandardScaler()
    pca_t0 = PCA(n_components=pixel_pca_dim, whiten=True, random_state=42)
    pixel_pca_t0 = pca_t0.fit_transform(scaler_t0.fit_transform(initial_renders))

    scaler_early = StandardScaler()
    pca_early = PCA(n_components=pixel_pca_dim, whiten=True, random_state=42)
    pixel_pca_early = pca_early.fit_transform(scaler_early.fit_transform(early_renders))

    pixel_pca_two_frame = np.concatenate([pixel_pca_t0, pixel_pca_early], axis=1)

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

    # --- 4. Fit InverseModel ---
    print("\nFitting InverseModel (two-frame pixels → physics, MC dropout)...")
    inv_model = InverseModel()
    inv_model.fit(pixel_pca_two_frame[train_idx], initial_physics[train_idx])

    # --- 5. Render-only baseline ---
    print("\nFitting render-only baseline (two-frame pixels → pixels)...")
    render_mlp = _make_render_mlp()
    render_mlp.fit(pixel_pca_two_frame[train_idx], final_pixel_pca[train_idx])

    # --- 6. Oracle (deterministic PyBullet with true physics) ---
    print(f"\nComputing oracle R² ({n_oracle} test scenes)...")
    oracle_preds = np.stack([
        resimulate_scene(scene_configs[i], initial_physics[i],
                         pillar_gray=pillar_grays[i],
                         lighting=lightings[i]).reshape(-1).astype(np.float32)
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
    mc_samples = inv_model.predict_stochastic(pixel_pca_two_frame[oracle_test_idx], n_samples=20)
    inferred_mean_oracle = mc_samples.mean(axis=0)
    non_observable = ~inv_model.valid_dims_
    for j in range(n_oracle):
        gt = initial_physics[oracle_test_idx[j]]
        inferred_mean_oracle[j, non_observable] = gt[non_observable]
    pp_preds = np.stack([
        resimulate_scene(scene_configs[oracle_test_idx[j]], inferred_mean_oracle[j],
                         pillar_gray=pillar_grays[oracle_test_idx[j]],
                         lighting=lightings[oracle_test_idx[j]]).reshape(-1).astype(np.float32)
        for j in range(n_oracle)
    ])
    pp_r2 = _pixel_r2(pp_preds, oracle_actual)
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
    inferred_physics_all = inv_model.predict(pixel_pca_two_frame)

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
        pixel_pca_two_frame[frame_idx], n_samples=1
    )[0]
    pp_frame_imgs = np.stack([
        resimulate_scene(scene_configs[frame_idx[j]], frame_stochastic[j],
                         pillar_gray=pillar_grays[frame_idx[j]],
                         lighting=lightings[frame_idx[j]])
        for j in range(n_frame_samples)
    ])
    init_frame_imgs = initial_renders[frame_idx].astype(np.uint8).reshape(
        n_frame_samples, IMAGE_SIZE, IMAGE_SIZE, 4)
    early_frame_imgs = early_renders[frame_idx].astype(np.uint8).reshape(
        n_frame_samples, IMAGE_SIZE, IMAGE_SIZE, 4)
    final_frame_imgs = program_states[frame_idx][:, pixel_indices].astype(np.uint8).reshape(
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
            'final_frame_imgs': final_frame_imgs,
        },
    }
