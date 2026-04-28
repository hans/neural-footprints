import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from load_config import load_config
from io_utils import load_scenes, load_neural, save_results
import analyses.predictive_processing as pp_mod
from analyses.pp_io import load_inverse_model

cfg = load_config()
scenes = load_scenes(snakemake.input.scenes)
neural, _ = load_neural(snakemake.input.neural)

# Load the same InverseModel that produced the activations baked into neural_input
# (see scripts/train_pp_for_neural.py). Reusing it here ensures the inferred-physics
# array fed to encoding/RSA matches what was projected into neural activity.
preloaded = load_inverse_model(snakemake.input.model)

# Monkey-patch InverseModel inside predictive_processing so its analysis function
# reuses the loaded checkpoint instead of training a fresh model. The analysis
# pipeline does `inv_model = InverseModel(); inv_model.fit(...)` — we replace the
# class with a subclass that copies the loaded state and makes fit() a no-op.
_OriginalInverseModel = pp_mod.InverseModel


class _ReusedInverseModel(_OriginalInverseModel):
    def __init__(self):
        super().__init__()
        self.net_              = preloaded.net_
        self.input_scaler_     = preloaded.input_scaler_
        self.phys_scaler_      = preloaded.phys_scaler_
        self.per_dim_r2_       = preloaded.per_dim_r2_
        self.valid_dims_       = preloaded.valid_dims_
        self.full_physics_dim_ = preloaded.full_physics_dim_
        self.const_values_     = preloaded.const_values_

    def fit(self, *args, **kwargs):
        print(f"    [reused checkpoint] InverseModel val per-dim R²: "
              f"mean={self.per_dim_r2_.mean():.4f}  max={self.per_dim_r2_.max():.4f}")
        return self


pp_mod.InverseModel = _ReusedInverseModel
try:
    results = pp_mod.run_predictive_processing_analysis(
        neural, scenes,
        pixel_pca_dim=cfg['pp_pixel_pca_dim'],
    )
finally:
    pp_mod.InverseModel = _OriginalInverseModel

inferred_physics_all = results.pop('inferred_physics_all')
plot_data = results.pop('plot_data')

save_results(results, snakemake.output.results)

np.savez_compressed(snakemake.output.inferred,
                    inferred_physics_all=inferred_physics_all)

np.savez_compressed(snakemake.output.plot_data, **plot_data)
