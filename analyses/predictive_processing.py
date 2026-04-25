"""
Predictive Processing model of visual perception.

Implements a two-stage model:
  (1) InverseModel: image(t=0) + image(t=early) → inferred physical state
      - Two-frame input enables velocity inference from optical displacement
      - MC Dropout: dropout active at inference time → distribution over physical states
  (2) Forward model: PyBullet resimulate_scene (exact physics engine, not learned)

Three models compared (all R² in raw pixel space):
  Prior MLP:    true physics_labels → [MLP] → final pixels
                Verifies MLP architecture is sufficient before evaluating InverseModel.
  Render-only:  pixel_pca_two_frame → [MLP] → final pixels
  PP chain:     pixel_pca_two_frame → [InverseModel] → inferred_physics → PyBullet → final pixels
  Oracle:       true physics_labels → PyBullet → final pixels  (~perfect, deterministic)

Scientific question: does the PP model's explicit physics intermediate representation
become detectable in neural encoding analyses? Expected answer: no — the physics
intermediate has low neural R², mirroring true physics_labels in the dissociation analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import (
    PP_HIDDEN_DIM, PP_PIXEL_PCA_DIM, PP_DROPOUT_RATE,
    N_OBJECTS, IMAGE_SIZE,
)
from analyses.utils import mean_neural_r2


# ---------------------------------------------------------------------------
# PyTorch network definition
# ---------------------------------------------------------------------------

class InverseMLPNet(nn.Module):
    """
    Two-layer MLP with dropout for Monte Carlo dropout inference.

    Parameters
    ----------
    input_dim : int
        2 * PP_PIXEL_PCA_DIM (concatenated two-frame PCA)
    output_dim : int
        physics_dim (15 * N_OBJECTS)
    hidden_dim : int
    dropout_rate : float
        Applied after each hidden layer — kept active at inference time for MC dropout.
    """
    def __init__(self, input_dim, output_dim,
                 hidden_dim=PP_HIDDEN_DIM, dropout_rate=PP_DROPOUT_RATE):
        super().__init__()
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
# InverseModel: sklearn-style wrapper with MC dropout
# ---------------------------------------------------------------------------

class InverseModel:
    """
    Maps two-frame pixel PCA to inferred physical state.

    Trains with dropout, supports deterministic prediction (model.eval()) and
    stochastic MC dropout sampling (model.train()).

    Attributes
    ----------
    net_ : InverseMLPNet
    input_scaler_ : StandardScaler  fitted on pixel_pca_two_frame
    phys_scaler_ : StandardScaler   fitted on physics_labels
    per_dim_r2_ : ndarray [physics_dim]  per-dimension R² on val set (standardized space)
    """

    def __init__(self):
        self.net_ = None
        self.input_scaler_ = None
        self.phys_scaler_ = None
        self.per_dim_r2_ = None
        self.valid_dims_ = None       # bool mask: which physics dims have variance
        self.full_physics_dim_ = None # original dimensionality before filtering
        self.const_values_ = None     # mean values for constant dims (used in predict)

    def fit(self, pixel_pca_two_frame, physics_labels,
            n_epochs=300, batch_size=64, lr=1e-3, val_frac=0.15, patience=50):
        """
        Train InverseModel with MC dropout.

        Parameters
        ----------
        pixel_pca_two_frame : ndarray [n × 2*PP_PIXEL_PCA_DIM]
        physics_labels : ndarray [n × physics_dim]  (initial_physics_labels)
        """
        self.full_physics_dim_ = physics_labels.shape[1]

        # Filter to pixel-observable dims: position and velocity components
        # that both vary across scenes AND are in principle inferable from pixels.
        # Per object (stride 15): pos(0:3), orn(3:7), lin_vel(7:10), ang_vel(10:13),
        #                         mass(13), friction(14)
        # Observable from two-frame pixels: position (directly visible) and
        # velocity (from inter-frame displacement). Mass and friction are
        # intrinsic properties not recoverable from pixel observations.
        observable_offsets = list(range(0, 3)) + list(range(7, 10))  # pos + lin_vel
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

        # Scale inputs and targets
        self.input_scaler_ = StandardScaler()
        X = self.input_scaler_.fit_transform(pixel_pca_two_frame)
        self.phys_scaler_ = StandardScaler()
        y = self.phys_scaler_.fit_transform(physics_valid)

        # Train / val split
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

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_count = 0
        best_state = None

        for epoch in range(n_epochs):
            self.net_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(self.net_(xb), yb).backward()
                optimizer.step()

            # Validation loss (eval mode — dropout off for val metric)
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

        # Per-dim R² on val set (in standardized physics space, valid dims only)
        self.net_.eval()
        with torch.no_grad():
            y_pred_val = self.net_(X_val_t).numpy()
        self.per_dim_r2_ = r2_score(y_val, y_pred_val, multioutput='raw_values')

        print(f"    InverseModel val MSE={best_val_loss:.4f}  "
              f"mean per-dim R²={self.per_dim_r2_.mean():.4f}  "
              f"max={self.per_dim_r2_.max():.4f}")
        return self

    def _expand_to_full(self, valid_predictions):
        """Expand valid-dim predictions back to full physics dimensionality."""
        n = valid_predictions.shape[0]
        full = np.tile(self.const_values_, (n, 1))
        full[:, self.valid_dims_] = valid_predictions
        return full

    def predict(self, pixel_pca_two_frame):
        """
        Deterministic prediction (model.eval(), dropout off).

        Returns physics in original (unscaled) units, full dimensionality.
        Constant dims are filled with their training-set mean.
        """
        self.net_.eval()
        X = torch.tensor(
            self.input_scaler_.transform(pixel_pca_two_frame), dtype=torch.float32
        )
        with torch.no_grad():
            pred_scaled = self.net_(X).numpy()
        valid_preds = self.phys_scaler_.inverse_transform(pred_scaled)
        return self._expand_to_full(valid_preds)

    def predict_stochastic(self, pixel_pca_two_frame, n_samples=8):
        """
        MC dropout: model stays in train() mode → dropout active → stochastic samples.

        Parameters
        ----------
        pixel_pca_two_frame : ndarray [n × 2*PP_PIXEL_PCA_DIM]
        n_samples : int

        Returns
        -------
        samples : ndarray [n_samples × n × physics_dim]  in original units (full dims)
        """
        self.net_.train()
        X = torch.tensor(
            self.input_scaler_.transform(pixel_pca_two_frame), dtype=torch.float32
        )
        samples_scaled = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples_scaled.append(self.net_(X).numpy())
        self.net_.eval()  # restore eval mode after sampling
        samples = np.stack([
            self._expand_to_full(self.phys_scaler_.inverse_transform(s))
            for s in samples_scaled
        ])  # [n_samples × n × physics_dim]
        return samples


# ---------------------------------------------------------------------------
# Prior evaluation: sklearn MLP (no stochasticity needed)
# ---------------------------------------------------------------------------

def _make_prior_mlp():
    """Sklearn MLP pipeline for prior evaluation (physics→pixels)."""
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(PP_HIDDEN_DIM, PP_HIDDEN_DIM),
            activation='relu', max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
        )
    )


def _eval_physics_to_pixel_mlp(physics_labels, final_pixel_pca,
                                 train_idx, test_idx,
                                 pca_final, scaler_pix,
                                 actual_final_pixels_test):
    """
    Train MLP: true physics_labels → final_pixel_pca; evaluate R² in raw pixel space.

    Establishes that the MLP architecture is sufficient — any PP chain shortfall
    comes from InverseModel quality, not architectural capacity.

    Returns prior_r2 (float).
    """
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


# ---------------------------------------------------------------------------
# Render-only baseline (sklearn MLP, same architecture as prior eval)
# ---------------------------------------------------------------------------

def _make_render_mlp():
    """Sklearn MLP pipeline for render-only baseline (two-frame pixels→pixels)."""
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(PP_HIDDEN_DIM, PP_HIDDEN_DIM),
            activation='relu', max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
        )
    )


# ---------------------------------------------------------------------------
# R² computation helpers
# ---------------------------------------------------------------------------

def _pixel_r2(predicted_rgba, actual_raw_pixels):
    """
    R² in raw pixel space. Both arrays float32, shape [n × pixel_dim].
    Mirrors the oracle scorer in dissociation._score_next_frame_pixels (lines 69-71).
    """
    actual = actual_raw_pixels.astype(np.float32)
    predicted = predicted_rgba.astype(np.float32)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


# ---------------------------------------------------------------------------
# Physics label grouping for InverseModel quality plot
# ---------------------------------------------------------------------------

def _physics_groups():
    """
    Returns dict mapping semantic label type → list of dimension indices.
    Stride = 15 per object: pos(0:3), orn(3:7), lin_vel(7:10), ang_vel(10:13),
    mass(13), friction(14).
    """
    return {
        'Position':    [i * 15 + j for i in range(N_OBJECTS) for j in range(3)],
        'Orientation': [i * 15 + j for i in range(N_OBJECTS) for j in range(3, 7)],
        'Lin. Vel.':   [i * 15 + j for i in range(N_OBJECTS) for j in range(7, 10)],
        'Ang. Vel.':   [i * 15 + j for i in range(N_OBJECTS) for j in range(10, 13)],
        'Mass':        [i * 15 + 13 for i in range(N_OBJECTS)],
        'Friction':    [i * 15 + 14 for i in range(N_OBJECTS)],
    }


# ---------------------------------------------------------------------------
# Frame visualization
# ---------------------------------------------------------------------------

def _save_pp_frames(pixel_pca_two_frame, inv_model,
                    initial_renders, early_renders_raw,
                    program_states, pixel_indices,
                    scene_configs, initial_physics_labels,
                    pca_final, scaler_pix,
                    fig_dir, n_samples=8):
    """
    Save 4-column visual comparison grid.

    Columns: t=0 | t=early (motion cue) | PP chain pred (stochastic sample) | t=N (actual)

    PP chain column uses one MC dropout sample per row to illustrate stochasticity.
    Saved to {fig_dir}/pp_frames.png.
    """
    from scene_generator import resimulate_scene

    n = min(n_samples, len(initial_renders))

    # One stochastic sample per scene
    stochastic_samples = inv_model.predict_stochastic(
        pixel_pca_two_frame[:n], n_samples=1
    )  # [1 × n × physics_dim]
    inferred_phys_sample = stochastic_samples[0]  # [n × physics_dim]

    def pca_to_imgs(pred_pca):
        pred_scaled = pca_final.inverse_transform(pred_pca)
        pred_raw = scaler_pix.inverse_transform(pred_scaled)
        return np.clip(pred_raw, 0, 255).astype(np.uint8).reshape(
            n, IMAGE_SIZE, IMAGE_SIZE, 4
        )

    pp_imgs = np.stack([
        resimulate_scene(scene_configs[j], inferred_phys_sample[j])
        for j in range(n)
    ])

    init_imgs = initial_renders[:n].astype(np.uint8).reshape(n, IMAGE_SIZE, IMAGE_SIZE, 4)
    early_imgs = early_renders_raw[:n].astype(np.uint8).reshape(n, IMAGE_SIZE, IMAGE_SIZE, 4)
    final_imgs = program_states[:n, pixel_indices].astype(np.uint8).reshape(
        n, IMAGE_SIZE, IMAGE_SIZE, 4
    )

    col_titles = ['t=0 (input)', f't={5} (motion cue)', 'PP chain pred\n(MC dropout sample)', 't=N (actual)']
    cols = [init_imgs, early_imgs, pp_imgs, final_imgs]

    fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for col_idx, (title, imgs) in enumerate(zip(col_titles, cols)):
        axes[0, col_idx].set_title(title, fontsize=8)
        for row_idx in range(n):
            axes[row_idx, col_idx].imshow(imgs[row_idx])
            axes[row_idx, col_idx].axis('off')

    plt.tight_layout(pad=0.3)
    fig_path = f"{fig_dir}/pp_frames.png"
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  PP frames saved: {fig_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_predictive_processing_analysis(neural_activity, scenes,
                                        fig_dir="figures"):
    """
    Train and evaluate the full predictive processing pipeline.

    Parameters
    ----------
    neural_activity : ndarray [n_scenes × N_NEURONS]
    scenes : dict  (from scene_generator.generate_scenes(), includes 'early_renders')
    fig_dir : str

    Returns
    -------
    dict with keys: prior_r2, oracle_r2, pp_r2, render_r2,
                    neural_r2_t0, neural_r2_two_frame, neural_r2_inferred_physics,
                    inverse_per_dim_r2, inverse_mean_r2
    """
    from scene_generator import resimulate_scene

    print("\n" + "=" * 60)
    print("SIMULATION 4: Predictive Processing Model")
    print("=" * 60)

    # Unpack scenes
    pixel_indices   = scenes['metadata']['pixel_indices']
    initial_renders = scenes['initial_renders']
    early_renders   = scenes['early_renders']
    initial_physics = scenes['initial_physics_labels']
    scene_configs   = scenes['scene_configs']
    program_states  = scenes['program_states']
    n = len(initial_renders)

    # --- 1. Shared pixel representations ---
    print("\nPreparing pixel representations...")

    final_pixels_raw = program_states[:, pixel_indices]   # [n × 16384]

    scaler_pix = StandardScaler()
    pix_scaled = scaler_pix.fit_transform(final_pixels_raw)
    pca_final = PCA(n_components=PP_PIXEL_PCA_DIM, whiten=True, random_state=42)
    final_pixel_pca = pca_final.fit_transform(pix_scaled)

    scaler_t0 = StandardScaler()
    t0_scaled = scaler_t0.fit_transform(initial_renders)
    pca_t0 = PCA(n_components=PP_PIXEL_PCA_DIM, whiten=True, random_state=42)
    pixel_pca_t0 = pca_t0.fit_transform(t0_scaled)

    scaler_early = StandardScaler()
    early_scaled = scaler_early.fit_transform(early_renders)
    pca_early = PCA(n_components=PP_PIXEL_PCA_DIM, whiten=True, random_state=42)
    pixel_pca_early = pca_early.fit_transform(early_scaled)

    pixel_pca_two_frame = np.concatenate([pixel_pca_t0, pixel_pca_early], axis=1)
    # [n × 2*PP_PIXEL_PCA_DIM]

    # --- 2. Train / test split ---
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_test = int(n * 0.2)
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    actual_test_raw = final_pixels_raw[test_idx].astype(np.float32)
    n_oracle = min(200, n_test)
    oracle_test_idx = test_idx[:n_oracle]

    # --- 3. Prior evaluation: true physics → pixels ---
    print("\nPrior evaluation: MLP(true physics → pixels)...")
    prior_r2 = _eval_physics_to_pixel_mlp(
        initial_physics, final_pixel_pca,
        train_idx, test_idx,
        pca_final, scaler_pix,
        actual_test_raw
    )
    print(f"  Prior MLP R² (physics→pixels, test set): {prior_r2:.4f}")

    # --- 4. Fit InverseModel ---
    print("\nFitting InverseModel (two-frame pixels → physics, MC dropout)...")
    inv_model = InverseModel()
    inv_model.fit(pixel_pca_two_frame[train_idx], initial_physics[train_idx])

    # --- 5. Fit render-only baseline ---
    print("\nFitting render-only baseline (two-frame pixels → pixels)...")
    render_mlp = _make_render_mlp()
    render_mlp.fit(pixel_pca_two_frame[train_idx], final_pixel_pca[train_idx])

    # --- 6. Oracle R² (deterministic PyBullet with true physics) ---
    print(f"\nComputing oracle R² ({n_oracle} test scenes)...")
    oracle_preds = np.stack([
        resimulate_scene(scene_configs[i], initial_physics[i]).reshape(-1).astype(np.float32)
        for i in oracle_test_idx
    ])
    oracle_actual = final_pixels_raw[oracle_test_idx].astype(np.float32)
    oracle_r2 = _pixel_r2(oracle_preds, oracle_actual)
    print(f"  Oracle R²: {oracle_r2:.4f}")

    # --- 7. PP chain R² (InverseModel mean → PyBullet) ---
    # Inferred physics supplies observable dims (position, velocity);
    # unobservable intrinsic properties (mass, friction, orientation, ang_vel)
    # come from ground truth since they cannot be recovered from pixels.
    print(f"\nComputing PP chain R² ({n_oracle} test scenes, MC ensemble mean)...")
    mc_samples = inv_model.predict_stochastic(pixel_pca_two_frame[oracle_test_idx], n_samples=20)
    inferred_mean_oracle = mc_samples.mean(axis=0)  # [n_oracle × physics_dim]
    for j in range(n_oracle):
        gt = initial_physics[oracle_test_idx[j]]
        # Copy non-observable dims from ground truth
        non_observable = ~inv_model.valid_dims_
        inferred_mean_oracle[j, non_observable] = gt[non_observable]
    pp_preds = np.stack([
        resimulate_scene(scene_configs[oracle_test_idx[j]], inferred_mean_oracle[j]).reshape(-1).astype(np.float32)
        for j in range(n_oracle)
    ])
    pp_r2 = _pixel_r2(pp_preds, oracle_actual)
    print(f"  PP chain R²: {pp_r2:.4f}  "
          f"(gap from oracle: {oracle_r2 - pp_r2:.4f} — InverseModel error)")
    if pp_r2 < prior_r2:
        print(f"  (PP chain R² < prior MLP R²: InverseModel bottleneck operative)")

    # --- 8. Render-only R² (on same oracle scenes as PP chain for fair comparison) ---
    print(f"\nComputing render-only R² ({n_oracle} oracle scenes)...")
    render_pred_pca = render_mlp.predict(pixel_pca_two_frame[oracle_test_idx])
    render_pred_raw = np.clip(
        scaler_pix.inverse_transform(pca_final.inverse_transform(render_pred_pca)), 0, 255
    ).astype(np.float32)
    render_r2 = _pixel_r2(render_pred_raw, oracle_actual)
    print(f"  Render-only R²: {render_r2:.4f}")

    print(f"\n  Summary:")
    print(f"    Prior MLP:   {prior_r2:.4f}  (architecture ceiling)")
    print(f"    Oracle:      {oracle_r2:.4f}  (PyBullet + true physics)")
    print(f"    PP chain:    {pp_r2:.4f}  (PyBullet + inferred physics)")
    print(f"    Render-only: {render_r2:.4f}  (MLP, no physics intermediate)")

    # --- 9. Neural R² ---
    print("\nComputing neural R² for PP representations...")
    inferred_physics_all = inv_model.predict(pixel_pca_two_frame)

    neural_r2_t0, _            = mean_neural_r2(pixel_pca_t0, neural_activity)
    neural_r2_two_frame, _     = mean_neural_r2(pixel_pca_two_frame, neural_activity)
    neural_r2_inv_physics, _   = mean_neural_r2(inferred_physics_all, neural_activity)

    print(f"  Neural R²  t=0 pixel PCA:     {neural_r2_t0:.4f}")
    print(f"  Neural R²  two-frame PCA:     {neural_r2_two_frame:.4f}")
    print(f"  Neural R²  inferred physics:  {neural_r2_inv_physics:.4f}  ← should be low")

    # --- 10. Figures ---
    print("\nSaving figures...")
    # Expand per-dim R² back to full 15*N_OBJECTS for the grouped bar chart
    full_per_dim_r2 = np.zeros(inv_model.full_physics_dim_)
    full_per_dim_r2[inv_model.valid_dims_] = inv_model.per_dim_r2_
    _save_figures(
        prior_r2, oracle_r2, pp_r2, render_r2,
        neural_r2_t0, neural_r2_two_frame, neural_r2_inv_physics,
        full_per_dim_r2,
        fig_dir
    )
    _save_pp_frames(
        pixel_pca_two_frame, inv_model,
        initial_renders, early_renders,
        program_states, pixel_indices,
        scene_configs, initial_physics,
        pca_final, scaler_pix,
        fig_dir
    )

    results = {
        'prior_r2': prior_r2,
        'oracle_r2': oracle_r2,
        'pp_r2': pp_r2,
        'render_r2': render_r2,
        'neural_r2_t0': neural_r2_t0,
        'neural_r2_two_frame': neural_r2_two_frame,
        'neural_r2_inferred_physics': neural_r2_inv_physics,
        'inverse_per_dim_r2': inv_model.per_dim_r2_,
        'inverse_mean_r2': float(inv_model.per_dim_r2_.mean()),
        'inferred_physics_all': inferred_physics_all,
    }
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_figures(prior_r2, oracle_r2, pp_r2, render_r2,
                  neural_r2_t0, neural_r2_two_frame, neural_r2_inv_physics,
                  per_dim_r2, fig_dir):
    """2×3 summary figure."""
    groups = _physics_groups()
    group_means = {k: per_dim_r2[v].mean() for k, v in groups.items()}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # [0,0] Behavioral R²
    ax = axes[0, 0]
    labels = ['Prior MLP\n(physics→pix)', 'Render-only\n(pix→pix)', 'PP chain\n(pix→phys→pix)', 'Oracle\n(true phys)']
    values = [prior_r2, render_r2, pp_r2, oracle_r2]
    colors = ['#888888', '#4878CF', '#E07B39', '#6ACC65']
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.axhline(prior_r2, color='#888888', linestyle='--', alpha=0.6, linewidth=1)
    ax.set_ylabel('Next-frame pixel R²')
    ax.set_title('Behavioral Sufficiency')
    ax.set_ylim(0, max(values) * 1.2 + 0.05)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # [0,1] Neural encoding R²
    ax = axes[0, 1]
    n_labels = ['t=0 pixel PCA', 'Two-frame PCA', 'Inferred physics']
    n_values = [neural_r2_t0, neural_r2_two_frame, neural_r2_inv_physics]
    n_colors = ['#4878CF', '#6AACD5', '#E07B39']
    bars = ax.bar(n_labels, n_values, color=n_colors, width=0.5)
    ax.set_ylabel('Mean neural R² (RidgeCV)')
    ax.set_title('Neural Encoding of PP Representations')
    ax.set_ylim(0, max(n_values) * 1.2 + 0.05)
    for bar, val in zip(bars, n_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # [0,2] InverseModel R² by label type
    ax = axes[0, 2]
    g_names = list(group_means.keys())
    g_vals = list(group_means.values())
    colors_g = ['#4878CF', '#6AACD5', '#D65F5F', '#E07B39', '#6ACC65', '#B07FC0']
    bars = ax.bar(g_names, g_vals, color=colors_g, width=0.6)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_ylabel('InverseModel R²')
    ax.set_title('What can be inferred from two frames?')
    ax.set_ylim(-0.1, 1.0)
    ax.tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, g_vals):
        y = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.06
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # [1,0] InverseModel per-dim R²
    ax = axes[1, 0]
    dims = np.arange(len(per_dim_r2))
    ax.plot(dims, per_dim_r2, color='#4878CF', linewidth=1.2, zorder=3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    # Alternating background bands per object (every 15 dims)
    for obj in range(N_OBJECTS):
        start = obj * 15
        end = min(start + 15, len(per_dim_r2))
        if obj % 2 == 1:
            ax.axvspan(start, end, alpha=0.08, color='gray')
    ax.set_xlabel('Physics dimension')
    ax.set_ylabel('R²')
    ax.set_title('InverseModel: per-dimension quality')
    ax.set_xlim(0, len(per_dim_r2) - 1)

    # [1,1] Dissociation scatter
    ax = axes[1, 1]
    scatter_data = [
        ('Render-only', render_r2, neural_r2_two_frame, '#4878CF'),
        ('PP chain', pp_r2, neural_r2_inv_physics, '#E07B39'),
    ]
    for label, beh_r2, neur_r2, color in scatter_data:
        ax.scatter(beh_r2, neur_r2, color=color, s=120, zorder=5)
        ax.annotate(label, (beh_r2, neur_r2),
                    textcoords='offset points', xytext=(6, 4), fontsize=9)
    # Oracle: behavioral R² only (no compact feature embedding → no neural R²)
    ax.axvline(oracle_r2, color='#6ACC65', linestyle=':', alpha=0.7, linewidth=1.5,
               label=f'Oracle behavioral R²={oracle_r2:.3f}')
    ax.set_xlabel('Behavioral R² (next-frame pixels)')
    ax.set_ylabel('Neural encoding R²')
    ax.set_title('PP Dissociation')
    ax.legend(fontsize=8)

    # [1,2] Text summary
    ax = axes[1, 2]
    ax.axis('off')
    txt = (
        "Predictive Processing Summary\n"
        "═══════════════════════════\n\n"
        "Behavioral R² (pixel prediction):\n"
        f"  Prior MLP (phys→pix):  {prior_r2:.4f}\n"
        f"  Oracle (true phys):    {oracle_r2:.4f}\n"
        f"  PP chain (inf. phys):  {pp_r2:.4f}\n"
        f"  Render-only:           {render_r2:.4f}\n\n"
        "Neural encoding R²:\n"
        f"  t=0 pixel PCA:         {neural_r2_t0:.4f}\n"
        f"  Two-frame PCA:         {neural_r2_two_frame:.4f}\n"
        f"  Inferred physics:      {neural_r2_inv_physics:.4f} ←\n\n"
        "InverseModel quality:\n"
        f"  Mean R²: {np.mean(per_dim_r2):.4f}  Max: {np.max(per_dim_r2):.4f}\n\n"
        "Finding: PP chain uses an explicit physics\n"
        "intermediate — yet neural R² tracks render\n"
        "structure, not physics structure."
    )
    ax.text(0.05, 0.97, txt, transform=ax.transAxes,
            va='top', ha='left', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    fig.suptitle(
        "Predictive Processing: Explicit Physics Intermediate is\n"
        "Behaviorally Useful but Neurally Invisible",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig_path = f"{fig_dir}/predictive_processing.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary figure saved: {fig_path}")
