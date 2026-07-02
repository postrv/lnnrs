# LNNrs

[![CI](https://github.com/postrv/lnnrs/actions/workflows/ci.yml/badge.svg)](https://github.com/postrv/lnnrs/actions/workflows/ci.yml)

A Rust implementation of Liquid Neural Networks (LNNs) based on MIT CSAIL research.

## Overview

Liquid Neural Networks are a class of neural networks with dynamic time constants that
adapt based on input, making them particularly suited for time-series processing tasks.
Unlike traditional RNNs, LNNs can model complex temporal dependencies with a continuous-time
formulation.

This implementation provides three main network architectures:

| Model | Description | Speed | Use case |
|-------|-------------|-------|----------|
| **LNN** | Classic reservoir computing with leaky integrate-and-fire neurons | Fast | Logic gates, reservoir readouts |
| **LTC** | Liquid Time-Constant with RK4 ODE integration | Slower | Accurate continuous-time modeling |
| **CfC** | Closed-form Continuous with an analytical/gated update | **Fastest** | Trainable, low-latency inference |

### Features

- **Three network architectures**: LNN, LTC, CfC (gated and no-gate variants)
- **Multiple ODE solvers**: Euler, RK4, Dormand-Prince (adaptive, FSAL, error-based step control)
- **Trainable CfC**: analytic backpropagation-through-time (BPTT) with the Adam optimizer
- **Optimizers**: SGD (+momentum) and Adam, plus a global-norm gradient-clip utility
- **Reservoir readout**: ridge-regression `train_readout` / `predict`
- **Reproducible initialization**: every network is seedable (`.seed(u64)`)
- **Config validation**: `validate()` / `try_build()` on every builder
- **Logic gate simulation**: correct truth tables for AND, OR, NOT, NAND, NOR, XOR
- **CSV I/O and full JSON serialization** via serde
- **Comprehensive tests and benchmarks**

## Installation

```bash
# Clone the repository
git clone https://github.com/postrv/lnnrs.git
cd lnnrs

# Build
cargo build --release

# Run the test suite
cargo test

# Run the benchmarks
cargo bench
```

Minimum supported Rust version (MSRV): **1.95**.

## Usage

### CLI Commands

The binary is named `lnn`.

```bash
# Run the demo (prints all logic-gate truth tables + network stats)
cargo run --release -- demo

# Test a single logic gate
cargo run --release -- gate --gate-type XOR --timesteps 10

# Test the LTC network on a sine/cosine sequence
cargo run --release -- ltc -u 32 -l 100

# Test the CfC network (gated by default; --no-gate for the decay variant)
cargo run --release -- cfc -u 32 -l 100
cargo run --release -- cfc -u 32 -l 100 --no-gate

# Train a CfC on a waveform task (BPTT + Adam); prints untrained vs trained MSE
cargo run --release -- train --model cfc --task sine --epochs 200 --lr 0.01 --seed 42 --units 32

# Optionally save the trained model
cargo run --release -- train --task sine --epochs 200 --save model.json

# Time-series prediction (trains a CfC by default; --no-train for a random forward pass)
cargo run --release -- time-series -m cfc -u 32 -t sine
cargo run --release -- time-series -m cfc -u 32 -t sine --no-train

# Run a reservoir simulation from a CSV file (a sample ships in examples/data)
cargo run --release -- run --input examples/data/input.csv --output output.csv --neurons 100

# Network statistics for a random reservoir
cargo run --release -- stats --neurons 50

# Version / help
cargo run --release -- --version
cargo run --release -- --help
```

The `run` subcommand reads comma-separated rows (with a header line) from `--input` and
writes the per-timestep outputs to `--output` (default `output.csv`, which is git-ignored).
`examples/data/input.csv` is a small 2-column sample.

### Examples

Three runnable examples live in `examples/`:

```bash
# Configure the reservoir as each logic gate and print truth tables
cargo run --example logic_gates

# Train a seeded CfC to predict a sine wave; prints untrained vs trained MSE
cargo run --example train_cfc_sine

# Integrate exponential decay and a harmonic oscillator with RK4 and Dormand-Prince,
# comparing against the analytic solutions
cargo run --example ode_solvers
```

### Library Usage

#### CfC Network (fastest; trainable)

```rust
use liquid_neural_network::prelude::*;

// Create a CfC network: 2 inputs, 1 output, 32 hidden units.
let network = CfcBuilder::new(2, 1)
    .hidden_units(32)
    .tau_range(0.1, 10.0)
    .backbone(2, 32) // 2 backbone layers of 32 units
    .seed(42)        // reproducible initialization
    .build();

// Process a sequence.
let inputs = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
let outputs = network.forward_sequence(&inputs);
```

#### Trainable CfC (BPTT + Adam)

```rust
use liquid_neural_network::prelude::*;

// A seeded, trainable CfC: 2 inputs -> 16 units -> 1 output.
let mut net = TrainableCfcBuilder::new(2, 16, 1)
    .learning_rate(0.01)
    .truncation_length(64)
    .seed(42)
    .build();

let inputs = vec![vec![0.0, 1.0], vec![0.5, 0.8], vec![0.9, 0.3]];
let targets = vec![vec![0.1], vec![0.4], vec![0.8]];

// Train for a few epochs; train_step returns the epoch loss.
for _ in 0..200 {
    let _loss = net.train_step(&inputs, &targets);
}

let predictions = net.predict(&inputs);
```

#### LTC Network (accurate continuous-time modeling)

```rust
use liquid_neural_network::prelude::*;

// Direct constructor: 2 inputs, 32 hidden, 1 output.
let network = LtcNetwork::new(2, 32, 1);

// Or use the builder. `ode_unfolds` sets the number of fixed RK4 steps per timestep.
let network = LtcBuilder::new(2, 1)
    .hidden_units(32)
    .tau_range(0.1, 10.0)
    .ode_unfolds(6)
    .seed(42)
    .build();

let inputs = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
let outputs = network.forward_sequence(&inputs);
```

#### Classic LNN (reservoir computing)

```rust
use liquid_neural_network::prelude::*;

// Builder pattern. `try_build` validates the configuration instead of panicking.
let mut lnn = LnnBuilder::new(50)
    .spectral_radius(0.95)
    .connection_probability(0.7)
    .weight_range(-0.5, 0.5)
    .threshold_range(0.3, 0.7)
    .seed(42)
    .try_build()
    .expect("valid configuration");

// Configure input/output neurons.
lnn.set_input_neurons(vec![0, 1, 2]);
lnn.set_output_neurons(vec![47, 48, 49]);

// Run the simulation. `run_simulation` returns a Result and validates row lengths.
let inputs = vec![vec![1.0, 0.5, 0.0], vec![0.0, 1.0, 0.5]];
let outputs = lnn.run_simulation(10, inputs).expect("valid inputs");
```

#### Reservoir Readout (ridge regression)

```rust
use liquid_neural_network::prelude::*;

let mut lnn = LnnBuilder::new(20).seed(42).build();
lnn.set_input_neurons(vec![0, 1]);

// Fit a linear readout over the reservoir's final state for each pattern.
let inputs = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0],
];
let targets = vec![0.0, 0.0, 0.0, 1.0]; // AND
let cfg = ReadoutConfig::default();     // timesteps: 10, ridge_lambda: 1e-6
lnn.train_readout(&inputs, &targets, &cfg).expect("trainable");

let y = lnn.predict(&[1.0, 1.0], cfg.timesteps).expect("readout trained");
```

#### Training an RNN with BPTT

```rust
use liquid_neural_network::prelude::*;

let mut rnn = TrainableRnnBuilder::new(2, 16, 1)
    .learning_rate(0.01)
    .truncation_length(50)
    .gradient_clip(1.0)
    .momentum(0.9)
    .seed(42)
    .build();

let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
let targets = vec![vec![0.5], vec![0.5]];

for _ in 0..100 {
    let _loss = rnn.train_step(&inputs, &targets);
}

let predictions = rnn.predict(&inputs);
```

#### ODE Solvers

```rust
use liquid_neural_network::prelude::*;

// Define a custom ODE system: dy/dt = -y + input.
struct MySystem;
impl OdeSystem for MySystem {
    fn derivative(&self, _t: f64, state: &[f64], input: &[f64]) -> Vec<f64> {
        state.iter().map(|y| -y + input[0]).collect()
    }
    fn dimension(&self) -> usize { 1 }
}

// Fixed-step RK4.
let solver = RungeKutta4;
let config = SolverConfig::new(0.01);
let result = solver
    .integrate(&MySystem, vec![0.0], &[1.0], 0.0, 10.0, &config)
    .unwrap();

// Adaptive Dormand-Prince with explicit tolerances.
let adaptive = DormandPrince::default();
let config = SolverConfig::new(0.1).with_rtol(1e-6).with_atol(1e-9);
let result = adaptive
    .integrate(&MySystem, vec![0.0], &[1.0], 0.0, 10.0, &config)
    .unwrap();
```

Note that `integrate` returns `Err(LnnError::OdeSolver(..))` if `max_steps` is exhausted
before reaching the endpoint, rather than silently truncating.

## Reproducibility

Every random initialization is reproducible. `LnnConfig`, `LtcConfig`, `CfcConfig`, and the
trainable network builders all carry an optional `seed: Option<u64>` and expose a
`.seed(u64)` method. When a seed is set, the whole network is constructed from a single
`StdRng::seed_from_u64(seed)`, threaded through every layer in a fixed order, so two networks
built from the same seed are byte-for-byte identical, including their serialized JSON. With
no seed, the RNG is seeded from the OS.

## Gate Semantics

Logic gates are realized inside a small reservoir with well-defined, order-independent
dynamics:

- **Synchronous updates**: every neuron integrates on the same snapshot each timestep, so
  results do not depend on the order synapses were inserted.
- **Real synaptic delays**: a signal emitted by neuron `i` at step `t` arrives at its target
  at step `t + delay` (`delay >= 1`), routed through a ring buffer.
- **Non-firing integrators**: gate neurons use a very high threshold and a constant `bias`,
  so the truth value is read from the sign of the output neuron's accumulated potential
  (`sigmoid(potential) > 0.5`, i.e. potential > 0).
- **Opt-in homeostasis**: threshold adaptation is off by default, so configured thresholds
  stay put.

All six gate truth tables are correct both in tests and in `cargo run -- demo`.

## Architecture

```
src/
  lib.rs                # Library entry point, prelude, reproducibility docs
  main.rs               # CLI binary (lnn)
  lnn.rs                # Classic LiquidNeuralNetwork + CSR SynapseMatrix + readout
  ltc.rs                # Liquid Time-Constant networks (RK4 ODE-based)
  cfc.rs                # Closed-form Continuous networks (gated + decay variants)
  ode_solver.rs         # ODE solvers (Euler, RK4, Dormand-Prince)
  layers.rs             # Shared ndarray-backed DenseLayer
  optim.rs              # Optimizer trait, SGD, Adam, clip_grad_norm
  train_cfc.rs          # TrainableCfc: analytic BPTT through the gated update
  training.rs           # TrainableRnn (BPTT) + MSE loss
  network_components.rs # Neuron, Synapse, ThresholdAdaptation
  activation.rs         # Activation functions
  error.rs              # LnnError / Result
  input.rs              # CSV input handling
  output.rs             # CSV output handling
```

### Key Components

- **LNN (Liquid Neural Network)**: reservoir computing with leaky integrate-and-fire
  neurons, spectral-radius scaling (via deterministic power iteration), synchronous delayed
  dynamics, logic-gate configurations, and a ridge-regression readout.

- **LTC (Liquid Time-Constant)**: continuous-time networks that integrate
  `tau(x, I) * dx/dt = -x + f(x, I)` with exactly `ode_unfolds` fixed RK4 steps per timestep,
  where `tau` is an input-dependent time constant.

- **CfC (Closed-form Continuous)**: an analytical update. The gated variant interpolates two
  `tanh` heads with a learned, time-dependent gate; the no-gate variant uses the exponential
  decay closed form `x(t+dt) = x(t) * exp(-dt/tau) + (1 - exp(-dt/tau)) * f(Wx + b)`.

- **ODE Solvers**: Euler (1st order), RK4 (4th order), and Dormand-Prince (adaptive step
  size with error-based control and FSAL).

- **Training**: `TrainableCfc` implements analytic BPTT through the gated update (verified by
  a numerical gradient check); `TrainableRnn` provides BPTT for a dense RNN. Both use the
  `optim` module (SGD/Adam) and global-norm gradient clipping.

## Benchmarks

Performance comparison at 32 hidden units and 100 timesteps, re-measured on this machine
(`cargo bench "Model Comparison"`, Criterion medians):

| Model | Time | Throughput |
|-------|------|------------|
| LNN | 463 us | 216 k steps/s |
| LTC (RK4 ODE) | 1.24 ms | 81 k steps/s |
| CfC (gated) | 161 us | 622 k steps/s |
| CfC (no-gate) | 132 us | 758 k steps/s |

The gated CfC is roughly **7.7x faster** than the LTC network with comparable modeling
capacity; the no-gate variant is faster still. Numbers vary with hardware; re-run locally to
reproduce.

```bash
cargo bench "Model Comparison"
cargo bench "ODE Solvers"
cargo bench "Scaling"
```

## Testing

The suite has 199 unit and integration tests plus doctests:

```bash
# Run everything
cargo test

# Run a specific module's tests
cargo test lnn::
cargo test ltc::
cargo test cfc::
cargo test ode_solver::
cargo test train_cfc::

# Show test output
cargo test -- --nocapture
```

## Research Background

This implementation is based on research from MIT CSAIL:

- [Liquid Time-constant Networks](https://arxiv.org/pdf/2006.04439.pdf) - the foundational LTC paper
- [Closed-form Continuous-Time Neural Networks](https://arxiv.org/pdf/2106.13898.pdf) - the CfC paper
- [Robust Flight Navigation with Liquid Neural Networks](https://dspace.mit.edu/handle/1721.1/144657) - applications
- [A Spectral Condition for Feature Learning](https://arxiv.org/pdf/2310.17813.pdf) - initialization theory

See also: [YouTube explainer on LNNs](https://www.youtube.com/watch?v=0FNkrjVIcuk)

## Roadmap

Future work under consideration:

- GPU acceleration (e.g. via `wgpu`)
- More optimizers for the LTC/RNN paths (e.g. RMSprop) and a trainable LTC
- Additional real-world time-series datasets and examples
- Batched sequence processing

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please feel free to open issues and pull requests.
