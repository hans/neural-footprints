"""Base classes for inferential MLP models that map sensory input to a latent state.

Used as the architectural template for both ``InverseModel`` (predicts physics
from two-frame pixel PCA) and ``CardinalityModel`` (predicts numerosity N from
single-frame pixel PCA). Subclasses customize how the regression target is
preprocessed and how raw network outputs are turned back into native units.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class InferentialMLPNet(nn.Module):
    """Three-hidden-layer MLP with dropout, kept active at inference for MC sampling.

    Layers are split into named blocks so post-ReLU activations of each hidden
    layer are individually addressable via ``forward_with_activations``. Dropout
    is applied AFTER the activation tap, so a tapped activation reflects the
    deterministic representation when the net is in eval mode (and the
    stochastic representation when in train mode for MC sampling).
    """
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.h1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.d1 = nn.Dropout(dropout_rate)
        self.h2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.d2 = nn.Dropout(dropout_rate)
        self.h3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.d3 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        h1 = self.h1(x)
        h2 = self.h2(self.d1(h1))
        h3 = self.h3(self.d2(h2))
        return self.out(self.d3(h3))

    def forward_with_activations(self, x):
        """Returns (output, {'h1': ..., 'h2': ..., 'h3': ...}). Post-ReLU, pre-dropout."""
        h1 = self.h1(x)
        h2 = self.h2(self.d1(h1))
        h3 = self.h3(self.d2(h2))
        out = self.out(self.d3(h3))
        return out, {'h1': h1, 'h2': h2, 'h3': h3}


_LAYERS = ('h1', 'h2', 'h3')


class InferentialModel:
    """Train an InferentialMLPNet on (X, y). Subclasses override target hooks.

    Subclass contract:
        ``_preprocess_target(self, y_raw)`` — fit any subclass-specific scalers
            on the *first* call (during ``fit``); transform y_raw and return
            the scaled-and-filtered target the network actually regresses.
            Must also assign ``self.full_target_dim_`` if the subclass needs
            to remember the original target shape for ``_postprocess_prediction``.

        ``_postprocess_prediction(self, y_scaled)`` — invert the preprocessing
            so the returned array is in the original target's native shape and
            units.

    The base class handles: input scaler, train/val split, training loop,
    early stopping, per-dim val R², extracting hidden activations.
    """

    NET_CLASS = InferentialMLPNet

    def __init__(self, hidden_dim, dropout_rate):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.net_ = None
        self.input_scaler_ = None
        self.per_dim_r2_ = None

    # --- Subclass hooks --------------------------------------------------

    def _preprocess_target(self, y_raw):
        """Default: z-score every column. Subclasses can filter / mean-fill first."""
        self.target_scaler_ = StandardScaler()
        return self.target_scaler_.fit_transform(y_raw)

    def _postprocess_prediction(self, y_scaled):
        """Default: inverse z-score back to native units."""
        return self.target_scaler_.inverse_transform(y_scaled)

    # --- Training --------------------------------------------------------

    def fit(self, X, y, *, n_epochs=300, batch_size=64, lr=1e-3,
            val_frac=0.15, patience=50, verbose=True):
        self.input_scaler_ = StandardScaler()
        X_scaled = self.input_scaler_.fit_transform(X)
        y_scaled = self._preprocess_target(y)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y_scaled, test_size=val_frac, random_state=42
        )

        input_dim = X_scaled.shape[1]
        output_dim = y_scaled.shape[1]
        self.net_ = self.NET_CLASS(
            input_dim=input_dim, output_dim=output_dim,
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

        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                            batch_size=batch_size, shuffle=True)

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
                    if verbose:
                        print(f"    {self.__class__.__name__} early stop at "
                              f"epoch {epoch+1} (val loss={best_val_loss:.4f})")
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)

        self.net_.eval()
        with torch.no_grad():
            y_pred_val = self.net_(X_val_t).numpy()
        self.per_dim_r2_ = r2_score(y_val, y_pred_val, multioutput='raw_values')

        if verbose:
            print(f"    {self.__class__.__name__} val MSE={best_val_loss:.4f}  "
                  f"mean per-dim R²={self.per_dim_r2_.mean():.4f}  "
                  f"max={self.per_dim_r2_.max():.4f}")
        return self

    # --- Inference -------------------------------------------------------

    def _scaled_input(self, X):
        return torch.tensor(self.input_scaler_.transform(X), dtype=torch.float32)

    def predict(self, X):
        """Deterministic prediction (dropout off). Returns native-unit target."""
        self.net_.eval()
        X_t = self._scaled_input(X)
        with torch.no_grad():
            pred_scaled = self.net_(X_t).numpy()
        return self._postprocess_prediction(pred_scaled)

    def extract_activations(self, X, layer='h2'):
        """Deterministic post-ReLU activations of one hidden layer (dropout off)."""
        if layer not in _LAYERS:
            raise ValueError(f"layer must be one of {_LAYERS}; got {layer!r}")
        self.net_.eval()
        X_t = self._scaled_input(X)
        with torch.no_grad():
            _, acts = self.net_.forward_with_activations(X_t)
        return acts[layer].numpy()


def extract_activations(model: InferentialModel, X, layer='h2'):
    """Free-function form of ``InferentialModel.extract_activations``.

    Kept for symmetry with ``analyses.pp_io.extract_activations`` so io modules
    can re-export a single canonical name.
    """
    return model.extract_activations(X, layer=layer)


def net_dims(net: InferentialMLPNet):
    """Pull architecture dims out of a trained net for checkpoint serialization."""
    h1_linear = net.h1[0]
    return {
        'input_dim':    h1_linear.in_features,
        'hidden_dim':   h1_linear.out_features,
        'output_dim':   net.out.out_features,
        'dropout_rate': net.d1.p,
    }
