# Future Brain State Prediction — Reverse Dissociation

## Motivation

The encoding analysis (Simulation 1) shows that pixel features explain current
neural R² well while physics labels add essentially nothing (ΔR² ≈ 0.015).
A skeptic could conclude: "physics isn't in the brain data, so it doesn't
matter." The future brain state analysis provides the counter-argument by
flipping the dissociation direction.

## Logic

Neural activity at time T is generated from the full program state at T
(render bytes + physics blob) via a fixed random projection W. The program
state at T is **deterministically governed** by the physics at t=0: initial
positions, velocities, masses, and friction coefficients fully specify the
trajectory, and therefore the final render and bullet blob.

We compare two **forward models** — each takes t=0 information and attempts to
reconstruct the future program state, which is then used as encoding features
to predict future neural activity.

### Physics forward model (oracle resimulation)

1. Take initial physics state (pos, vel, mass, friction) + scene configs
2. Run PyBullet forward for N_TIMESTEPS → produces full program_state
   (all render buffers + bullet blob, built identically to original generation)
3. Use reconstructed program_state as encoding features → Ridge R² per neuron

Since PyBullet is deterministic, the reconstructed program_state is essentially
identical to the original. The encoding R² therefore matches the original
encoding analysis (~0.44). The physics model can predict future brain state
because it has access to the causal dynamics.

### Pixel forward model (learned MLP)

1. Take initial RGBA pixels, PCA-reduce (behavioral PCA dim, whitened)
2. Train MLP: initial pixel PCA → final pixel PCA (same architecture as
   dissociation analysis)
3. Inverse-transform predictions back to pixel space
4. PCA-reduce predicted pixels (encoding PCA dim) → encoding features
5. Ridge R² per neuron

The MLP cannot accurately predict future pixels because:
- Velocity is invisible in a single static frame
- Objects may move behind the occluding pillar — their future render is
  unpredictable from initial appearance alone
- The MLP produces blurry, averaged predictions that lose scene-specific detail

Result: encoding R² from predicted pixels is **low**.

## Expected Results

| Forward model | Encoding features | Future neural R² |
|---|---|---|
| Physics (PyBullet resimulation) | Reconstructed full program_state | HIGH (~0.44) |
| Pixel (MLP prediction) | MLP-predicted pixels only | LOW |

## Interpretation

Standard encoding models ask: "what predicts current neural variance?" and find
pixels. The future brain state analysis asks: "what predicts where this neural
population will be next?" and finds that only the physics forward model succeeds.

The key asymmetry: a physics engine + initial state can reconstruct the full
future program state (and thus future brain state). A pixel forward model cannot,
because pixels lack the causal variables (velocity, mass, friction) that
determine the trajectory.

This means the encoding model's failure to detect physics is not evidence that
physics is absent from the neural code. It is evidence that the encoding model
asks the wrong question. The variables that are invisible to cross-sectional
encoding are exactly the variables that govern temporal dynamics.
