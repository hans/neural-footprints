"""
Standalone PP inverse-model evaluator (no snakemake).

Reuses cached `data/scenes.npz`, retrains the inverse model with overridable
hyperparameters, and reports:

  - per-dim physics R² (val split inside InverseModel.fit)
  - PP-chain pixel R² over a small oracle subset (PyBullet resimulation)
  - render-only baseline pixel R² on the same subset (sanity check)
  - a frame grid PDF (predicted vs actual) for qualitative inspection

Knobs exposed via CLI flags:
    --pixel-pca-dim   pixel PCA components per frame (default cfg.pp_pixel_pca_dim)
    --hidden-dim      MLP hidden size (default cfg.pp_hidden_dim)
    --dropout         dropout rate (default cfg.pp_dropout_rate)
    --epochs          max training epochs (default 300)
    --patience        early-stopping patience (default 50)
    --batch-size      batch size (default 64)
    --lr              learning rate (default 1e-3)
    --features        {concat,concat_diff,diff_only}  what to feed the MLP
    --n-oracle        number of test scenes for pixel-chain R² (default 100)
    --no-pixel-r2     skip PyBullet resimulation entirely
    --tag             label for outputs (default "default")

Outputs go to outputs/pp_eval/<tag>/{metrics.json, frames.pdf, log.txt}.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import IMAGE_SIZE, N_OBJECTS
from scripts.load_config import load_config


def load_scenes_any(path):
    """Load scenes file. Auto-detects 3-frame format (mid_renders + late_renders)
    and arbitrary FOV in metadata. Returns a dict with the union of fields.
    Also monkey-patches scene_generator._render_scene if FOV != 60 so that
    PyBullet resimulation matches the saved frames.
    """
    import json as _json

    data = np.load(path, allow_pickle=False)
    scenes = {}
    for key in [
        "program_states",
        "physics_labels",
        "initial_physics_labels",
        "initial_renders",
        "early_renders",
        "behavior_labels",
        "kinetic_energies",
    ]:
        if key in data.files:
            scenes[key] = data[key]
    if "mid_renders" in data.files:
        scenes["mid_renders"] = data["mid_renders"]
    if "late_renders" in data.files:
        scenes["late_renders"] = data["late_renders"]

    meta = _json.loads(str(data["metadata_json"]))
    pi = meta["pixel_indices"]
    meta["pixel_indices"] = slice(pi[0], pi[1])
    ri = meta["render_indices"]
    meta["render_indices"] = slice(ri[0], ri[1])
    scenes["metadata"] = meta

    scenes["scene_configs"] = _json.loads(str(data["scene_configs_json"]))
    scenes["pillar_grays"] = data["pillar_grays"].tolist()
    scenes["lightings"] = _json.loads(str(data["lightings_json"]))

    fov = meta.get("fov", 60.0)
    if abs(fov - 60.0) > 1e-3:
        _patch_render_scene_fov(fov)
    return scenes


def _patch_render_scene_fov(fov):
    """Monkey-patch scene_generator._render_scene with one that uses the given FOV.
    Required so resimulate_scene produces frames consistent with the saved scenes.
    """
    import pybullet as _p
    import scene_generator as _sg

    def render_with_fov(physics_client, lighting=None):
        if lighting is None:
            lighting = _sg._DEFAULT_LIGHTING
        view_matrix = _p.computeViewMatrix(
            cameraEyePosition=[0, -3, 2],
            cameraTargetPosition=[0, 0, 0.3],
            cameraUpVector=[0, 0, 1],
            physicsClientId=physics_client,
        )
        proj_matrix = _p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=1.0,
            nearVal=0.1,
            farVal=10.0,
            physicsClientId=physics_client,
        )
        _, _, rgba, depth, seg = _p.getCameraImage(
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            shadow=1,
            lightDirection=lighting["lightDirection"],
            lightColor=lighting["lightColor"],
            lightDistance=lighting["lightDistance"],
            physicsClientId=physics_client,
        )
        rgba_arr = np.array(rgba, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 4)
        depth_arr = np.array(depth, dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
        seg_arr = np.array(seg, dtype=np.int32).reshape(IMAGE_SIZE, IMAGE_SIZE)
        return rgba_arr.tobytes(), depth_arr.tobytes(), seg_arr.tobytes()

    _sg._render_scene = render_with_fov
    print(f"  [patched scene_generator._render_scene FOV → {fov}°]")


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


# ---------------------------------------------------------------------------
# Inverse model (parameterised; mirrors analyses/predictive_processing.py)
# ---------------------------------------------------------------------------


class InverseMLPNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_rate):
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


class InverseModel:
    def __init__(self, hidden_dim, dropout_rate):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.net_ = None
        self.input_scaler_ = None
        self.phys_scaler_ = None
        self.per_dim_r2_ = None
        self.valid_dims_ = None
        self.full_physics_dim_ = None
        self.const_values_ = None
        self.history_ = None

    def fit(
        self,
        X_in,
        physics_labels,
        n_epochs=300,
        batch_size=64,
        lr=1e-3,
        val_frac=0.15,
        patience=50,
        verbose=True,
    ):
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

        self.input_scaler_ = StandardScaler()
        X = self.input_scaler_.fit_transform(X_in)
        self.phys_scaler_ = StandardScaler()
        y = self.phys_scaler_.fit_transform(physics_valid)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_frac, random_state=42
        )

        self.net_ = InverseMLPNet(
            X.shape[1], y.shape[1], self.hidden_dim, self.dropout_rate
        )
        opt = torch.optim.Adam(self.net_.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
        loss_fn = nn.MSELoss()

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True
        )

        best_val = float("inf")
        best_state = None
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
            history.append({"epoch": epoch + 1, "train_loss": tl, "val_loss": vl})
            sch.step(vl)
            if vl < best_val - 1e-5:
                best_val = vl
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if verbose:
                        print(
                            f"    early stop @ epoch {epoch+1}, best val={best_val:.4f}"
                        )
                    break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        with torch.no_grad():
            y_pred_val = self.net_(X_val_t).numpy()
        self.per_dim_r2_ = r2_score(y_val, y_pred_val, multioutput="raw_values")
        self.history_ = history
        if verbose:
            print(
                f"    final val MSE={best_val:.4f}  mean per-dim R²={self.per_dim_r2_.mean():.4f}"
            )
        return self

    def _expand_to_full(self, valid_predictions):
        n = valid_predictions.shape[0]
        full = np.tile(self.const_values_, (n, 1))
        full[:, self.valid_dims_] = valid_predictions
        return full

    def predict(self, X_in):
        self.net_.eval()
        X = torch.tensor(self.input_scaler_.transform(X_in), dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = self.net_(X).numpy()
        return self._expand_to_full(self.phys_scaler_.inverse_transform(pred_scaled))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pixel_r2(predicted, actual):
    a = actual.astype(np.float32)
    p = predicted.astype(np.float32)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def _build_features(scenes, pca_dim, mode):
    """Build pixel-PCA features for the inverse model.

    2-frame modes (require initial_renders + early_renders):
        concat        : t0 || late
        concat_diff   : t0 || late || (late - t0)
        diff_only     : (late - t0)

    3-frame modes (require initial_renders + mid_renders + late_renders):
        3frame        : t0 || mid || late
        3frame_diff   : t0 || mid || late || (mid - t0) || (late - mid)
    """
    init = scenes["initial_renders"]

    def _pca(x):
        s = StandardScaler()
        xs = s.fit_transform(x)
        pca = PCA(n_components=pca_dim, whiten=True, random_state=42)
        return pca.fit_transform(xs)

    if mode.startswith("3frame"):
        if "mid_renders" not in scenes:
            raise ValueError(
                f"feature mode {mode} requires mid_renders/late_renders in scenes file"
            )
        mid = scenes["mid_renders"]
        late = scenes["late_renders"]
        parts = [_pca(init), _pca(mid), _pca(late)]
        if mode == "3frame_diff":
            parts.append(_pca(mid - init))
            parts.append(_pca(late - mid))
        return np.concatenate(parts, axis=1)

    early = scenes["early_renders"]
    parts = []
    if mode in ("concat", "concat_diff"):
        parts.append(_pca(init))
        parts.append(_pca(early))
    if mode in ("concat_diff", "diff_only"):
        parts.append(_pca(early - init))
    return np.concatenate(parts, axis=1)


def _resim_pixels(scenes, indices, physics_rows):
    """Resimulate scenes at given indices with given physics rows; return (n, H, W, 4) uint8."""
    from scene_generator import resimulate_scene

    out = []
    for k, idx in enumerate(indices):
        img = resimulate_scene(
            scenes["scene_configs"][idx],
            physics_rows[k],
            pillar_gray=scenes["pillar_grays"][idx],
            lighting=scenes["lightings"][idx],
        )
        out.append(img)
    return np.stack(out)


def _save_frame_grid(
    path, init_imgs, early_imgs, pp_imgs, actual_imgs, per_row_r2=None
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = init_imgs.shape[0]
    cols = 4 + (1 if per_row_r2 is not None else 0)
    fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
    titles = ["t=0 (input)", "t=early", "PP chain pred", "t=N (actual)"]
    for r in range(n):
        for c, im in enumerate(
            [init_imgs[r], early_imgs[r], pp_imgs[r], actual_imgs[r]]
        ):
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(im[..., :3])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
        if per_row_r2 is not None:
            ax = axes[r, 0] if n > 1 else axes[0]
            ax.set_ylabel(f"R²={per_row_r2[r]:+.2f}", fontsize=8)
    fig.suptitle(os.path.basename(os.path.dirname(path)), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixel-pca-dim", type=int, default=cfg["pp_pixel_pca_dim"])
    ap.add_argument("--hidden-dim", type=int, default=cfg["pp_hidden_dim"])
    ap.add_argument("--dropout", type=float, default=cfg["pp_dropout_rate"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--features",
        choices=["concat", "concat_diff", "diff_only", "3frame", "3frame_diff"],
        default="concat",
    )
    ap.add_argument("--n-oracle", type=int, default=100)
    ap.add_argument("--no-pixel-r2", action="store_true")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenes", default="data/scenes.npz")
    args = ap.parse_args()

    out_dir = os.path.join("outputs", "pp_eval", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"=== PP eval: tag={args.tag} ===")
    log(f"args: {vars(args)}")

    log(f"\nLoading scenes from {args.scenes}...")
    t0 = time.time()
    scenes = load_scenes_any(args.scenes)
    log(f"  loaded in {time.time()-t0:.1f}s; n={len(scenes['initial_renders'])}")

    log(f"\nBuilding features (mode={args.features}, pca_dim={args.pixel_pca_dim})...")
    t0 = time.time()
    X_feat = _build_features(scenes, args.pixel_pca_dim, args.features)
    log(f"  feature shape={X_feat.shape}  built in {time.time()-t0:.1f}s")

    initial_physics = scenes["initial_physics_labels"]
    n = X_feat.shape[0]

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.2)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    log("\nFitting InverseModel...")
    t0 = time.time()
    inv = InverseModel(hidden_dim=args.hidden_dim, dropout_rate=args.dropout)
    inv.fit(
        X_feat[train_idx],
        initial_physics[train_idx],
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    train_time = time.time() - t0
    log(
        f"  trained in {train_time:.1f}s  best val loss={inv.history_[-1]['val_loss']:.4f}"
    )

    full_per_dim_r2 = np.zeros(inv.full_physics_dim_)
    full_per_dim_r2[inv.valid_dims_] = inv.per_dim_r2_
    log("\nPer-dim physics R² (val split):")
    for i, name in enumerate(PHYSICS_LABELS):
        valid = inv.valid_dims_[i]
        marker = "  " if valid else " *"
        log(
            f"  {marker} {i:2d} {name:10s}  R²={full_per_dim_r2[i]:+.4f}{'  (constant)' if not valid else ''}"
        )
    log(f"  mean over valid dims: {inv.per_dim_r2_.mean():.4f}")

    metrics = {
        "tag": args.tag,
        "args": vars(args),
        "train_time_s": train_time,
        "val_loss": float(inv.history_[-1]["val_loss"]),
        "per_dim_r2": {
            PHYSICS_LABELS[i]: float(full_per_dim_r2[i])
            for i in range(inv.full_physics_dim_)
        },
        "mean_valid_dim_r2": float(inv.per_dim_r2_.mean()),
        "valid_dims": [
            PHYSICS_LABELS[i]
            for i in range(inv.full_physics_dim_)
            if inv.valid_dims_[i]
        ],
    }

    if not args.no_pixel_r2:
        n_oracle = min(args.n_oracle, n_test)
        oracle_idx = test_idx[:n_oracle]
        log(f"\nResimulating {n_oracle} oracle scenes (PyBullet)...")
        t0 = time.time()

        # PP chain: inverse model mean prediction → fill non-observable from gt → resim
        inferred = inv.predict(X_feat[oracle_idx])
        non_obs = ~inv.valid_dims_
        for j in range(n_oracle):
            inferred[j, non_obs] = initial_physics[oracle_idx[j], non_obs]
        pp_imgs = _resim_pixels(scenes, oracle_idx, inferred)

        # Oracle: true physics → resim
        oracle_imgs = _resim_pixels(scenes, oracle_idx, initial_physics[oracle_idx])

        # Actual final renders (pixels-only) from program_states
        meta = scenes["metadata"]
        actual_flat = scenes["program_states"][oracle_idx][:, meta["pixel_indices"]]
        actual_imgs = actual_flat.astype(np.uint8).reshape(
            n_oracle, IMAGE_SIZE, IMAGE_SIZE, 4
        )

        log(f"  resimulated in {time.time()-t0:.1f}s")

        pp_r2 = _pixel_r2(
            pp_imgs.reshape(n_oracle, -1).astype(np.float32),
            actual_flat.astype(np.float32),
        )
        oracle_r2 = _pixel_r2(
            oracle_imgs.reshape(n_oracle, -1).astype(np.float32),
            actual_flat.astype(np.float32),
        )

        # Per-scene PP R² for ranking the worst frames
        per_scene_r2 = []
        a = actual_flat.astype(np.float32)
        a_mean = a.mean(axis=0, keepdims=True)
        for j in range(n_oracle):
            pp_flat = pp_imgs[j].reshape(-1).astype(np.float32)
            ss_res = np.sum((a[j] - pp_flat) ** 2)
            ss_tot = np.sum((a[j] - a_mean) ** 2)
            per_scene_r2.append(float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0)
        per_scene_r2 = np.array(per_scene_r2)

        log(f"  PP chain pixel R²:   {pp_r2:.4f}")
        log(f"  Oracle pixel R²:     {oracle_r2:.4f}  (should be ~1.0)")
        log(
            f"  per-scene PP R² mean={per_scene_r2.mean():.3f}  median={np.median(per_scene_r2):.3f}  "
            f"min={per_scene_r2.min():.3f}  max={per_scene_r2.max():.3f}"
        )

        metrics["pp_r2"] = pp_r2
        metrics["oracle_r2"] = oracle_r2
        metrics["per_scene_r2"] = per_scene_r2.tolist()

        # Frame grid: 4 best + 4 worst PP scenes
        order = np.argsort(per_scene_r2)
        worst = order[:4]
        best = order[-4:][::-1]
        keep = np.concatenate([best, worst])

        init_pix = (
            scenes["initial_renders"][oracle_idx[keep]]
            .astype(np.uint8)
            .reshape(len(keep), IMAGE_SIZE, IMAGE_SIZE, 4)
        )
        early_pix = (
            scenes["early_renders"][oracle_idx[keep]]
            .astype(np.uint8)
            .reshape(len(keep), IMAGE_SIZE, IMAGE_SIZE, 4)
        )

        frames_path = os.path.join(out_dir, "frames.pdf")
        _save_frame_grid(
            frames_path,
            init_pix,
            early_pix,
            pp_imgs[keep],
            actual_imgs[keep],
            per_row_r2=per_scene_r2[keep],
        )
        log(f"\nSaved frame grid → {frames_path} (top 4 + bottom 4 by per-scene R²)")

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved metrics → {metrics_path}")
    log_f.close()


if __name__ == "__main__":
    main()
