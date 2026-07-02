# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Preparing the `0.2.0` release: a top-to-bottom production-quality overhaul delivered in
five phases. Because the crate is pre-1.0, breaking API changes were made deliberately;
they are listed explicitly below.

### Breaking Changes

- `LiquidNeuralNetwork::run_simulation` now returns `Result<Vec<Vec<f64>>>` and validates
  that every input row matches the number of input neurons.
- `LnnConfig`/`LnnBuilder`, `LtcConfig`/`LtcBuilder`, and `CfcConfig`/`CfcBuilder` gained
  `validate()` and `try_build()`; `build()` now panics on an invalid configuration.
- `LtcConfig` lost its `ode_step` field; `with_ode_params`/`ode_params` were replaced by
  `with_ode_unfolds`/`ode_unfolds`, and `LtcBuilder::ode_params` became
  `LtcBuilder::ode_unfolds`.
- `OdeSolver::integrate` and `integrate_with_trajectory` now return
  `Err(LnnError::OdeSolver(..))` when `max_steps` is exhausted before reaching `t1`, rather
  than silently truncating.
- `Neuron::fire` and the `NeuronEvent` type were removed (removed from the prelude).
  Threshold homeostasis is now opt-in via `ThresholdAdaptation` and a `bias` field.
- `LogicGate`'s inherent `from_str` was removed in favor of
  `impl std::str::FromStr for LogicGate` (`Err = LnnError`); call sites now use `.parse()`.
- Serialized JSON is not backwards-compatible with `0.1.0`: networks now store weights via
  the shared `layers::DenseLayer`. The sparse reservoir JSON format
  (`{ shape, synapses: [..] }`) is preserved.
- Dependencies `smartcore`, `nalgebra`, `ndarray-rand`, and `rayon` were removed;
  `rand` (0.9), `ndarray` (0.16), `thiserror` (2), and `criterion` (0.7, dev) were upgraded.

### Phase 1 - Correctness & semantics

- Implemented the real gated CfC update (Hasani et al. 2022): `ff1`, `ff2`, and a
  time-interpolation gate, distinct from the no-gate exponential-decay closed form.
- Made `ode_unfolds` the single source of truth for LTC integration: `forward` runs exactly
  `ode_unfolds` fixed RK4 steps over `t in [0, 1]`.
- Hardened the ODE solvers: `max_steps` exhaustion is now an error; `DormandPrince` stores
  `rtol`/`atol`, uses error-based step-size control, and implements FSAL.
- Redesigned reservoir semantics to be order-independent: neurons decoupled from synapses,
  synchronous propagation with a ring buffer honoring real per-synapse delays, opt-in
  bounded threshold adaptation, and a per-neuron `bias`.
- All six logic gates (AND, OR, NOT, NAND, NOR, XOR) now produce correct truth tables under
  the new dynamics, pinned by tests.
- Replaced the dead `train_readout` with a working ridge-regression readout
  (`ReadoutConfig`, `train_readout`, `predict`) solved by in-crate Gaussian elimination,
  removing the `smartcore` dependency.
- Made `mse_loss_with_grad` loss/gradient consistent and empty-input safe.

### Phase 2 - Reproducibility

- Added `seed: Option<u64>` to `LnnConfig`, `LtcConfig`, `CfcConfig`, and the trainable
  paths, with `.seed(u64)` builder methods.
- Every network is now constructed from a single `StdRng`, so identical seeds yield
  byte-for-byte identical serialized JSON across LNN, LTC, CfC, and `TrainableRnn`.

### Phase 3 - Architecture, dependencies & performance

- Added a shared `layers::DenseLayer` (ndarray-backed) replacing three duplicated dense
  layers and the inline LTC weight loops.
- Replaced the dense `Array2<Option<Synapse>>` reservoir with a CSR `SynapseMatrix`.
- Replaced `nalgebra` eigendecomposition with deterministic power iteration for the spectral
  radius, dropping `nalgebra`.
- Pruned and upgraded dependencies (see Breaking Changes).

### Phase 4 - Trainability

- Added an `optim` module: the `Optimizer` trait, `Sgd`, `Adam`, and a global-norm
  `clip_grad_norm` utility (gradient clipping moved from per-element clamping).
- Added `TrainableCfc`/`TrainableCfcBuilder`: analytic BPTT through the gated closed-form
  update, verified by a numerical gradient check (max relative error < 1e-4).
- CLI: added the `train` subcommand; `time-series` now trains a CfC by default, with
  `--no-train` for the untrained forward pass.

### Phase 5 - Hygiene, CI & docs

- Added a GitHub Actions CI workflow running `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings`, `cargo test`, and
  `cargo doc --no-deps` with `RUSTDOCFLAGS="-D warnings"`.
- Added `#![warn(missing_docs)]` and documented every public item.
- Added runnable examples: `logic_gates`, `train_cfc_sine`, and `ode_solvers`.
- Rewrote the README with re-measured benchmarks, documented training, reproducibility,
  gate semantics, and the readout; added a CI badge.
- Moved the sample CSV to `examples/data/input.csv`; added `output*.csv` to `.gitignore`.
- Set crate `keywords`, `categories`, `readme`, and `rust-version` (MSRV 1.95) in
  `Cargo.toml`.

## [0.1.0]

- Initial implementation of LNN, LTC, and CfC networks; ODE solvers; BPTT training for
  `TrainableRnn`; logic-gate demos; CSV I/O; and JSON serialization.
