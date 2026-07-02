# Production-Quality Overhaul Plan

Branch: `overhaul/production-quality`. Five sequential phases, each landing as its own
commit with behavior-locking tests. Baseline evidence for Phase 1: `cargo run -- demo`
on master produces **all-zero truth tables for every logic gate** (AND(1,1)=0,
OR(1,1)=0, NOT(0)=0) -- the current dynamics are broken end-to-end.

Ground rules for every phase (production standard):

- `cargo fmt` clean, `cargo clippy --all-targets` introduces **zero new warnings**
  (fix existing warnings in any file you touch).
- `cargo test` fully green. New behavior must be pinned by tests that would fail on
  the old code.
- Public items documented. Breaking API changes are acceptable (crate is 0.1.0) but
  must be deliberate, minimal, and listed in the final report.
- No new dependencies unless the plan says so. Do not commit -- the orchestrator
  reviews and commits.

---

## Phase 1 -- Correctness & semantics

### 1a. CfC gating (`src/cfc.rs`)

Today the `no_gate: false` branch of `CfcCell::forward` computes the *identical*
expression to the `no_gate: true` branch. Implement the real gated CfC
(Hasani et al. 2022, "Cfc" default mode as in the reference `ncps` implementation):

- Backbone unchanged: `z = backbone([state; input])`.
- Heads (all new `DenseLayer`s from `backbone_out_dim` to `units`):
  - `ff1 = tanh(W1 z + b1)`
  - `ff2 = tanh(W2 z + b2)`
  - `t_a = Wa z + ba`, `t_b = Wb z + bb`
  - `t_interp = sigmoid(t_a * dt + t_b)`  (elementwise)
- Gated update (elementwise): `x_new = ff1 * (1 - t_interp) + t_interp * ff2`
- `no_gate: true` keeps the **existing** exponential-decay closed form
  (`x * exp(-dt/tau) + (1 - exp(-dt/tau)) * f`) using `tau_layer` + `ff_projection` --
  this is the pure/decay variant and current behavior. The `tau_layer`/`ff_projection`
  fields are only used in no-gate mode; the new head layers only in gated mode.
  Construct all of them unconditionally so serialization stays simple.

Skeleton:

```rust
pub struct CfcCell {
    pub config: CfcConfig,
    backbone: Backbone,
    // no-gate (decay) mode:
    ff_projection: DenseLayer,
    tau_layer: DenseLayer,
    // gated mode:
    ff1: DenseLayer,
    ff2: DenseLayer,
    time_a: DenseLayer,
    time_b: DenseLayer,
    input_dim: usize,
}
```

Tests: (1) gated vs no-gate cells built from the same config produce different
sequences on the same input; (2) gated outputs bounded in [-1, 1] (convex
combination of two tanh heads); (3) serialization round-trip.

### 1b. LTC `ode_unfolds` (`src/ltc.rs`)

`ode_unfolds` is dead config. Make it the single source of truth: `forward` integrates
t in [0, 1] with exactly `ode_unfolds` fixed RK4 steps (`dt = 1.0 / ode_unfolds`),
calling `RungeKutta4::step` in a loop directly -- no `integrate()`/`max_steps`
interplay, no error path to swallow. Remove `ode_step` from `LtcConfig`;
`with_ode_params`/`ode_params` become `with_ode_unfolds(usize)` / `ode_unfolds(usize)`.
Validate `ode_unfolds >= 1` (see 1f). Test: a cell with `unfolds=1` differs from
`unfolds=32` on the same weights; results converge as unfolds grows
(err(unfolds=8) < err(unfolds=1), both measured against unfolds=32).

### 1c. ODE solvers (`src/ode_solver.rs`)

- `OdeSolver::integrate` (default impl) and `integrate_with_trajectory`: return
  `Err(LnnError::OdeSolver(...))` when `max_steps` is exhausted before reaching `t1`
  instead of silently truncating. Update `test_max_steps_limit` to expect `Err`.
- `DormandPrince`: store `rtol`/`atol` as struct fields (`Default`: 1e-6/1e-9;
  add `with_tolerances(rtol, atol)`). `step()` uses them. The overridden `integrate`
  uses `config.rtol`/`config.atol` (config wins over struct defaults).
- Step-size control: on accept, grow with the standard error-based factor
  `dt_next = dt * clamp(safety * err^(-1/5), 0.2, 5.0)` (clamped to `[min_dt, max_dt]`)
  instead of the blind `* 1.5`. Implement FSAL: k7 of an accepted step is k1 of the
  next (thread it through the integrate loop; `adaptive_step` may take/return the
  cached derivative, or restructure so integrate owns the stages -- your call, keep
  `adaptive_step`'s public API sensible).
- Floating-point endpoint: terminate when
  `t1 - state.time <= f64::EPSILON * t1.abs().max(1.0)`.

Tests: DP with loose vs tight tolerances takes measurably fewer derivative evaluations
(count via a wrapper `OdeSystem` with a `Cell<usize>` counter); existing accuracy
tests unchanged; FSAL: derivative-eval count per accepted step is ~6 (not 7) after the
first step.

### 1d. Reservoir simulation semantics (`src/network_components.rs`, `src/lnn.rs`)

Redesign the update loop with well-defined, order-independent semantics:

1. **Neuron decoupled from synapses.**
   `Neuron::update_state(&mut self, input_current: f64) -> bool` -- leaky integration
   exactly as now, returns `true` if it fired (crossed threshold and was reset to
   `rest_potential`). Delete `Neuron::fire`, delete `NeuronEvent` (dead machinery;
   remove from prelude). Neurons no longer compare `synapse.source` against the
   global-counter `id` -- the network owns topology by **index**.
2. **Synchronous propagation with real delays**, owned by `run_simulation`:
   - Signal emitted by neuron `i` at the end of step `t` arrives at target `j` during
     step `t + delay` (delay >= 1, validated at connection time).
   - Ring buffer: `arrivals: Vec<Vec<f64>>` with `max_delay + 1` slots of per-neuron
     accumulators, indexed `t % (max_delay + 1)`.
   - Per timestep `t`: (a) take the arrivals slot for `t` as each neuron's
     `input_current`; (b) `update_state` every neuron (reading only the snapshot --
     arrivals were computed from *previous* potentials, so updates are synchronous);
     (c) snapshot post-update potentials, and for every synapse schedule
     `weight * potential[source]` into slot `(t + delay) % ring`; (d) zero the
     consumed slot.
   - Emission value stays rate-based (`weight * membrane_potential`) as today.
3. **Bounded, opt-in threshold adaptation.** Replace `adjust_threshold`'s unbounded
   +0.1/step drift with an explicit, serde-defaulted config on `Neuron`:

```rust
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct ThresholdAdaptation {
    pub enabled: bool,        // default false
    pub increment: f64,       // default 0.1
    pub min: f64,             // default 0.5
    pub max: f64,             // default 2.0
}
```

   When enabled: threshold += increment after firing (not every step!), and decays by
   `increment` toward `min` on non-firing steps; always clamped to `[min, max]`.
   Homeostasis reacts to *firing activity*, not resting potential. Default **off** so
   configured thresholds (gates!) stay put. `LnnConfig` gets
   `threshold_adaptation: ThresholdAdaptation` applied to all neurons at build.
4. **`run_simulation` returns `Result<Vec<Vec<f64>>>`** and validates up front that
   every input row's length equals `input_indices.len()`. `encode_input` becomes
   infallible-by-construction (called only after validation) but should
   `debug_assert_eq!` lengths. Update all call sites/tests.
5. **Gate configurations must actually work.** Input neurons in gate configs must act
   as pure relays: give them a very high threshold (e.g. `1e9`) so they never
   fire/reset, rest potential 0. Re-derive the gate weights/thresholds under the new
   synchronous+delayed dynamics so that **all six truth tables are correct**, and pin
   them with tests like:

```rust
fn gate_output(gate: &str, input: &[f64]) -> bool {
    let mut lnn = LiquidNeuralNetwork::new(10);
    lnn.configure_for_logic_gate(gate).unwrap();
    let out = lnn.run_simulation(10, vec![input.to_vec()]).unwrap();
    out[0][0] > 0.5
}
// assert full truth tables for AND, OR, NOT, NAND, NOR, XOR
```

   Note the decode is `sigmoid(potential)`, so logical false must correspond to
   potential <= 0 and true to potential > 0 at read time. NOT/NAND/NOR need a bias
   mechanism that survives the loop: a `bias: f64` field on `Neuron` added to the
   potential each step (serde-default 0.0) is acceptable and honest -- design it,
   don't hack it.
6. Test for order-independence: build two networks with identical topology but
   synapses inserted in different orders; identical inputs -> identical outputs.
7. Test for delay semantics: two neurons, synapse with delay 3; stimulate source at
   t=0; assert target potential is unchanged until step 3 and changes at step 3.

### 1e. Functional readout (`src/lnn.rs`)

Replace the dead `train_readout`:

```rust
pub struct ReadoutConfig { pub timesteps: usize, pub ridge_lambda: f64 }
impl Default for ReadoutConfig { /* timesteps: 10, ridge_lambda: 1e-6 */ }

impl LiquidNeuralNetwork {
    /// Runs the reservoir for each input pattern (from reset state), collects the
    /// final full state vector [potentials..., 1.0 bias], and fits ridge regression
    /// (S^T S + lambda I) w = S^T y. Stores w in readout_weights.
    pub fn train_readout(&mut self, inputs: &[Vec<f64>], targets: &[f64],
                         cfg: &ReadoutConfig) -> Result<()>;

    /// Runs the reservoir on one pattern and applies the trained readout.
    pub fn predict(&mut self, input: &[f64], timesteps: usize) -> Result<f64>;
}
```

- Solve the ridge system with a private Gaussian-elimination-with-partial-pivoting
  helper (`fn solve_linear_system(a: Vec<Vec<f64>>, b: Vec<f64>) -> Option<Vec<f64>>`)
  -- **this removes the `smartcore` dependency entirely** (drop it from Cargo.toml).
- `predict` errors if readout was never trained.
- Tests: (i) the linear-solver helper against a known 3x3 system; (ii) build a small
  reservoir with *hand-set deterministic weights* (via `connect_neurons`), train on a
  linearly-solvable mapping (e.g. AND on {0,1}^2 with targets 0/1), assert train MSE
  < 1e-2 and predictions on all four patterns round to the right labels.

### 1f. Config validation & papercuts

- `LnnConfig` ranges: use inclusive sampling (`gen_range(a..=b)`) so degenerate ranges
  (min == max) are legal; validate `min <= max` for weight/threshold/delay ranges,
  `delay_range.0 >= 1`, `neuron_count >= 1`. Add
  `LnnConfig::validate(&self) -> Result<()>`; `build()` panics with the validation
  message (document `# Panics`), and add `try_build() -> Result<LiquidNeuralNetwork>`
  on both `LnnConfig` and `LnnBuilder`. Same pattern for `LtcConfig`/`CfcConfig`
  (`units >= 1`, `tau_min > 0`, `tau_min <= tau_max`, `ode_unfolds >= 1`,
  `backbone_layers >= 1`).
- `mse_loss_with_grad` (`src/training.rs`): make loss and gradient consistent --
  `loss = sum(diff^2) / (n_timesteps * n_dims)`,
  `grad = 2 * diff / (n_timesteps * n_dims)`. Return 0-loss/no-grad gracefully on
  empty input. Pin with a test computing both against hand-computed values.
- `src/main.rs`: `#[command(version)]` from Cargo (drop hardcoded "0.1.0"); adjust for
  `run_simulation` returning `Result`.
- `Cargo.toml`: `repository = "https://github.com/postrv/lnnrs"`; remove `smartcore`.

### Phase 1 acceptance

- All six gate truth tables correct in tests **and** in `cargo run -- demo` output.
- New tests listed above all present and green; full suite green.
- `smartcore` gone from Cargo.toml/Cargo.lock.
- Zero clippy warnings in touched files. `cargo fmt` clean.

---

## Phase 2 -- Reproducibility (seedable RNG)

Every random initialization must be reproducible.

- Add `seed: Option<u64>` to `LnnConfig`, `LtcConfig`, `CfcConfig`, and the
  `TrainableRnn::new` path, plus builder methods `.seed(u64)`.
- Internal constructors take `&mut StdRng` (or `&mut impl Rng`):
  `DenseLayer::new(input_dim, output_dim, rng)`, `LtcCell::new(input_dim, config, rng)`,
  etc. Public entry points create the rng once --
  `let mut rng = match config.seed { Some(s) => StdRng::seed_from_u64(s), None => StdRng::from_entropy() };`
  -- and thread it through **one** generator per network so unit ordering is stable.
- `LtcNetwork::with_config` / `CfcNetwork::with_config` / `TrainableRnn::new` must
  derive *all* their layers from that single rng.
- Tests: same seed -> identical serialized JSON for LNN, LTC, CfC, TrainableRnn;
  different seeds -> different JSON; no seed still works.
- Doc section in lib.rs about reproducibility.

Acceptance: determinism tests green; no public-API regressions beyond added methods.

---

## Phase 3 -- Architecture, dependencies & performance

### 3a. Shared layers module

New `src/layers.rs` (public): a single `DenseLayer` (weights, bias, dims, forward,
xavier init with rng) replacing the three copies in `cfc.rs`, `training.rs`
(`TrainableDense` becomes a wrapper holding a `DenseLayer` plus grad/momentum buffers,
or keeps its shape but reuses init/forward), and the inline `w_in`/`w_rec`/`w_out`
loops in `ltc.rs`. Serde back-compat is a non-goal (0.x); keep JSON round-trip tests
green against the *new* format.

### 3b. ndarray-backed math

Store weights as `ndarray::Array2<f64>`, states/vectors as `Array1<f64>` inside
`layers.rs`; `forward` uses `.dot()`. Sequence APIs keep `Vec<Vec<f64>>` at the public
boundary (convert at the edge). LTC derivative and CfC forward go through the shared
layer. Benchmarks must not regress (run `cargo bench "Model Comparison"` before/after
and report numbers).

### 3c. Sparse reservoir

Replace `Array2<Option<Synapse>>` with a CSR-style struct:

```rust
pub struct SynapseMatrix {
    n: usize,
    row_offsets: Vec<usize>,   // len n+1
    entries: Vec<Synapse>,     // sorted by (source, target)
}
```

Keep the **existing sparse JSON serialization format** (`{shape, synapses: [...]}`) by
hand-implementing Serialize/Deserialize as today. `connect_neurons`/
`disconnect_neurons`/`get_synapses_*` reimplemented on CSR (insertion may be O(row);
fine). `run_simulation`'s propagation loop iterates `entries` once per step.

### 3d. Spectral radius via power iteration

Replace `nalgebra` eigendecomposition with power iteration on the (sparse) weight
matrix: up to ~1000 iterations with early-exit on |lambda| change < 1e-10, seeded
deterministic start vector. Handles the zero-matrix case. Caveat to document: power
iteration estimates the dominant eigenvalue magnitude; for defective/nilpotent
matrices (e.g. [[0,2],[0,0]]) it returns ~0, which is the correct spectral radius
there anyway. Tests: diag(1,3) -> 3; a dense 4x4 with known dominant eigenvalue;
zero matrix -> 0. **Drop `nalgebra` from Cargo.toml.**

### 3e. Dependency & toolchain cleanup

- Remove: `nalgebra`, `ndarray-rand`, `rayon` (unused), `smartcore` (already gone).
- Upgrade: `rand` 0.9 (API renames: `thread_rng()` -> `rand::rng()`, `gen_range` ->
  `random_range`, `from_entropy` -> `from_os_rng` -- sweep all call sites),
  `ndarray` 0.16, `thiserror` 2, `criterion` 0.7 (dev), `env_logger` current,
  `rand_distr` to match rand 0.9 if still used (drop if unused).
- `cargo build` and full test suite green after upgrades.

Acceptance: dep tree shrinks (`cargo tree -d` clean of removed crates), benches within
noise or better vs the Phase 2 baseline (include numbers), tests green.

---

## Phase 4 -- Trainability

### 4a. Optimizer abstraction (new `src/optim.rs`)

```rust
pub trait Optimizer {
    /// Updates params in place given grads (same shapes).
    /// param_group identifies the tensor (registration order) so stateful
    /// optimizers keep per-tensor moment buffers.
    fn step(&mut self, params: &mut [f64], grads: &[f64], param_group: usize);
}
pub struct Sgd { /* lr, momentum, velocity per group */ }
pub struct Adam { /* lr, beta1, beta2, eps, t, m/v per group */ }
```

(Adapt the exact signature to the layer storage from Phase 3 -- flat `&mut [f64]`
views over `Array2` via `as_slice_mut()` are fine.) Gradient clipping moves to a
global-norm clip utility (`clip_grad_norm(grads, max_norm)`) applied before
`Optimizer::step` -- this replaces per-element clamping; keep the BpttConfig field but
document norm semantics.

### 4b. Trainable CfC (BPTT through the closed form)

New `TrainableCfc`: forward caches per-timestep intermediates (backbone activations
per layer, ff1/ff2 pre-tanh, t_a/t_b, t_interp, states); backward derives gradients
analytically for the gated update:

- `dL/dff1 = grad_x * (1 - t_interp)`, `dL/dff2 = grad_x * t_interp`  (elementwise)
- `dL/dt_interp = grad_x * (ff2 - ff1)`; then through the sigmoid;
  `dL/dt_a = ... * dt`, `dL/dt_b = ...`
- Heads backprop into backbone features (sum all four head input-grads), then through
  the backbone MLP layers (activation derivatives: implement
  `BackboneActivation::derivative(x)` for Tanh/ReLU/SiLU/GELU-approx), then split the
  combined input-grad into (state-grad, input-grad); state-grad flows to timestep t-1
  (BPTT), with truncation per config.

**Mandatory test: numerical gradient check.** Tiny net (e.g. 2 inputs -> 3 units,
backbone 1x4, seq len 3, fixed seed), central differences (eps = 1e-5) vs analytic
grads for every parameter tensor; max relative error < 1e-4. Write this test first.

Also: `train_step(&mut self, inputs, targets) -> f64` mirroring TrainableRnn, with
Adam default; `predict`; serialization (skip optimizer/caches).

### 4c. CLI & demo integration (`src/main.rs`)

- `lnn train --model cfc --task sine --epochs 200 --lr 0.01 --seed 42 --units 32`:
  trains, logs loss every N epochs, prints final MSE, optionally `--save model.json`.
- `time-series` subcommand gains training by default; the "Note: the network is not
  trained" apology is deleted; untrained mode behind `--no-train`.
- Integration test: seeded CfC on sine prediction reaches MSE below a threshold that
  passes robustly -- verify by running it repeatedly, don't guess.

Acceptance: gradient-check test green; sine task demonstrably learns (loss curve
decreasing, final MSE reported); suite green.

---

## Phase 5 -- Hygiene, CI, docs

- `.github/workflows/ci.yml`: jobs on push/PR -- `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo doc --no-deps`
  (with `RUSTDOCFLAGS="-D warnings"`). Cache with `Swatinem/rust-cache@v2`. Ubuntu
  stable.
- Zero clippy warnings across the whole workspace; add `#![warn(missing_docs)]` to
  lib.rs and fill gaps.
- `impl std::str::FromStr for LogicGate` (remove the inherent `from_str` in favor of
  the trait; update call sites).
- `examples/`: `logic_gates.rs`, `train_cfc_sine.rs` (trains + prints MSE),
  `ode_solvers.rs` -- all runnable via `cargo run --example ...` and referenced in
  README.
- README: re-measure the benchmark table on this machine (`cargo bench` numbers),
  correct or delete the 12x claim, document training, reproducibility (seeds), the
  new gate semantics, and update the test count. Fix install section paths. Add CI
  badge placeholder.
- `CHANGELOG.md`: one entry per phase, breaking changes called out.
- Move `test_input_data.csv` -> `examples/data/input.csv`; delete `output_data.csv`;
  add `output*.csv` to .gitignore.
- Cargo.toml: `keywords`, `categories`, `readme`, `rust-version` (verify MSRV builds),
  description tweak.

Acceptance: all commands in README verified to run; clippy/fmt/test/doc all clean
locally.
