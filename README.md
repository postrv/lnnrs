# LNNrs

A comprehensive Rust implementation of Liquid Neural Networks (LNNs) based on MIT CSAIL research.

## Overview

Liquid Neural Networks are a class of neural networks with dynamic time constants that adapt based on input, making them particularly suited for time-series processing tasks. Unlike traditional RNNs, LNNs can model complex temporal dependencies with fewer neurons due to their continuous-time formulation.

This implementation provides three main network architectures:

| Model | Description | Speed | Use Case |
|-------|-------------|-------|----------|
| **LNN** | Classic reservoir computing with leaky integrate-and-fire neurons | Moderate | Logic gates, pattern recognition |
| **LTC** | Liquid Time-Constant with ODE-based dynamics | Slower | Accurate continuous-time modeling |
| **CfC** | Closed-form Continuous - analytical solution | **12x faster** | Production inference |

### Features

- **Multiple ODE Solvers**: Euler, RK4, Dormand-Prince (adaptive)
- **Three Network Architectures**: LNN, LTC, CfC
- **BPTT Training**: Backpropagation through time with gradient clipping
- **Configurable Architecture**: Builder pattern for all networks
- **Spectral Radius Initialization**: For stability control
- **Logic Gate Simulation**: AND, OR, NOT, NAND, NOR, XOR
- **CSV Input/Output**: Data processing support
- **Full Serialization**: JSON via serde
- **Comprehensive Benchmarks**: Performance comparison across models

## Installation

```bash
# Clone the repository
git clone https://github.com/postrv/lnnrs.git
cd lnnrs

# Build
cargo build --release

# Run tests (133 tests)
cargo test

# Run benchmarks
cargo bench
```

## Usage

### CLI Commands

```bash
# Run the demo (tests all logic gates)
cargo run --release -- demo

# Test LTC network on time series
cargo run --release -- ltc -u 32 -l 100

# Test CfC network (12x faster than LTC)
cargo run --release -- cfc -u 32 -l 100

# Test CfC with no-gate mode
cargo run --release -- cfc -u 32 -l 100 --no-gate

# Run time-series prediction example
cargo run --release -- time-series -m cfc -u 32 -t sine

# Test a specific logic gate
cargo run --release -- gate --gate-type AND --timesteps 10

# Run simulation with CSV input
cargo run --release -- run --input data.csv --output results.csv --neurons 100

# Display help
cargo run --release -- --help
```

### Library Usage

#### CfC Network (Recommended for Production)

```rust
use liquid_neural_network::prelude::*;

// Create a CfC network - fastest option
let network = CfcBuilder::new(2, 1)  // 2 inputs, 1 output
    .hidden_units(32)
    .tau_range(0.1, 10.0)
    .backbone(2, 32)
    .build();

// Process a sequence
let inputs = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
let outputs = network.forward_sequence(&inputs);
```

#### LTC Network (For Accurate Continuous-Time Modeling)

```rust
use liquid_neural_network::prelude::*;

// Create an LTC network with ODE integration
let network = LtcNetwork::new(2, 32, 1);  // 2 inputs, 32 hidden, 1 output

// Or use the builder
let network = LtcBuilder::new(2, 1)
    .hidden_units(32)
    .tau_range(0.1, 10.0)
    .ode_params(0.1, 6)  // step size, unfolds
    .build();

let outputs = network.forward_sequence(&inputs);
```

#### Classic LNN (Reservoir Computing)

```rust
use liquid_neural_network::prelude::*;

// Create using the builder pattern
let mut lnn = LnnBuilder::new(50)
    .spectral_radius(0.95)
    .connection_probability(0.7)
    .weight_range(-0.5, 0.5)
    .threshold_range(0.3, 0.7)
    .build();

// Configure input/output neurons
lnn.set_input_neurons(vec![0, 1, 2]);
lnn.set_output_neurons(vec![47, 48, 49]);

// Run simulation
let inputs = vec![vec![1.0, 0.5, 0.0], vec![0.0, 1.0, 0.5]];
let outputs = lnn.run_simulation(10, inputs);
```

#### Training with BPTT

```rust
use liquid_neural_network::prelude::*;

// Create a trainable RNN
let mut rnn = TrainableRnnBuilder::new(2, 16, 1)
    .learning_rate(0.01)
    .truncation_length(50)
    .gradient_clip(1.0)
    .momentum(0.9)
    .build();

// Training loop
let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
let targets = vec![vec![0.5], vec![0.5]];

for epoch in 0..100 {
    let loss = rnn.train_step(&inputs, &targets);
    println!("Epoch {}: loss = {:.6}", epoch, loss);
}

// Inference
let predictions = rnn.predict(&inputs);
```

#### ODE Solvers

```rust
use liquid_neural_network::prelude::*;

// Define a custom ODE system
struct MySystem;
impl OdeSystem for MySystem {
    fn derivative(&self, t: f64, state: &[f64], input: &[f64]) -> Vec<f64> {
        state.iter().map(|y| -y + input[0]).collect()
    }
    fn dimension(&self) -> usize { 1 }
}

// Use RK4 solver
let solver = RungeKutta4;
let config = SolverConfig::new(0.01);
let result = solver.integrate(&MySystem, vec![0.0], &[1.0], 0.0, 10.0, &config)?;

// Or use adaptive Dormand-Prince
let adaptive = DormandPrince::default();
let config = SolverConfig::new(0.1).with_rtol(1e-6).with_atol(1e-9);
let result = adaptive.integrate(&MySystem, vec![0.0], &[1.0], 0.0, 10.0, &config)?;
```

## Architecture

```
src/
├── lib.rs               # Library entry point and prelude
├── main.rs              # CLI binary
├── lnn.rs               # Classic LiquidNeuralNetwork
├── ltc.rs               # Liquid Time-Constant networks (ODE-based)
├── cfc.rs               # Closed-form Continuous networks (analytical)
├── ode_solver.rs        # ODE solvers (Euler, RK4, Dormand-Prince)
├── training.rs          # BPTT training with SGD+Momentum
├── network_components.rs # Neuron and Synapse definitions
├── activation.rs        # Activation functions
├── error.rs             # Custom error types
├── input.rs             # CSV input handling
└── output.rs            # CSV output handling
```

### Key Components

- **LNN (Liquid Neural Network)**: Classic reservoir computing with leaky integrate-and-fire neurons, spectral radius scaling, and logic gate configurations.

- **LTC (Liquid Time-Constant)**: Continuous-time networks using ODE integration. Implements the equation:
  ```
  τ(x,I) * dx/dt = -x + f(x,I)
  ```
  where τ is an input-dependent time constant.

- **CfC (Closed-form Continuous)**: Uses an analytical solution instead of numerical integration:
  ```
  x(t+Δt) = x(t) * exp(-Δt/τ) + (1 - exp(-Δt/τ)) * f(Wx + b)
  ```
  This provides ~12x speedup over LTC while maintaining similar dynamics.

- **ODE Solvers**:
  - Euler (1st order, fast)
  - RK4 (4th order, balanced)
  - Dormand-Prince (adaptive step size)

- **Training**: BPTT with truncation, gradient clipping, L2 regularization, and momentum.

## Benchmarks

Performance comparison (32 hidden units, 100 timesteps):

| Model | Time | Throughput |
|-------|------|------------|
| LTC (ODE) | 4.1 ms | 24k elem/s |
| CfC | 330 μs | 302k elem/s |
| CfC-NoGate | 330 μs | 302k elem/s |

CfC provides **12x speedup** over LTC with comparable modeling capabilities.

Run benchmarks:
```bash
cargo bench "Model Comparison"
cargo bench "ODE Solvers"
cargo bench "Scaling"
```

## Testing

The project includes 133 comprehensive tests:

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test lnn::
cargo test ltc::
cargo test cfc::
cargo test ode_solver::
cargo test training::

# Run with output
cargo test -- --nocapture
```

## Research Background

This implementation is based on research from MIT CSAIL:

- [Liquid Time-constant Networks](https://arxiv.org/pdf/2006.04439.pdf) - The foundational LTC paper
- [Closed-form Continuous-Time Neural Networks](https://arxiv.org/pdf/2106.13898.pdf) - CfC paper
- [Robust Flight Navigation with Liquid Neural Networks](https://dspace.mit.edu/handle/1721.1/144657) - Applications
- [A Spectral Condition for Feature Learning](https://arxiv.org/pdf/2310.17813.pdf) - Initialization theory

See also: [YouTube explainer on LNNs](https://www.youtube.com/watch?v=0FNkrjVIcuk)

## Current Status

**Implemented:**
- [x] Core neuron dynamics with leaky integrate-and-fire model
- [x] Synaptic connections with delays and weights
- [x] Dynamic time constants (liquid property)
- [x] Spectral radius-based weight initialization
- [x] Builder pattern for all network types
- [x] Logic gate demonstrations (AND, OR, NOT, NAND, NOR, XOR)
- [x] CSV input/output support
- [x] JSON serialization/deserialization
- [x] ODE solvers (Euler, RK4, Dormand-Prince)
- [x] LTC (Liquid Time-Constant) networks
- [x] CfC (Closed-form Continuous) networks
- [x] BPTT training with gradient clipping
- [x] Comprehensive test suite (133 tests)
- [x] Comprehensive benchmarks
- [x] Professional CLI with clap

**Next Steps:**
- [ ] Add CUDA/GPU acceleration via wgpu
- [ ] Implement more optimizer variants (Adam, RMSprop)
- [ ] Add real-world time-series examples
- [ ] API documentation improvements

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
