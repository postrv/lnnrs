# LNNrs

A Rust implementation of Liquid Neural Networks (LNNs) based on MIT CSAIL research.

## Overview

Liquid Neural Networks are a class of neural networks with dynamic time constants that adapt based on input, making them particularly suited for time-series processing tasks. Unlike traditional RNNs, LNNs can model complex temporal dependencies with fewer neurons due to their continuous-time formulation.

This implementation features:
- **Leaky integrate-and-fire neuron model** with dynamic time constants
- **Configurable network architecture** via builder pattern
- **Spectral radius-based weight initialization** for stability
- **Logic gate simulation** for testing and demonstration
- **CSV input/output support** for data processing
- **Readout layer training** using linear regression
- **Full serialization** support (JSON) for model persistence

## Installation

```bash
# Clone the repository
git clone https://github.com/postrv/lnnrs.git
cd lnnrs

# Build
cargo build --release

# Run tests
cargo test
```

## Usage

### CLI Commands

```bash
# Run the demo (tests all logic gates and shows network statistics)
cargo run --release -- demo

# Test a specific logic gate
cargo run --release -- gate --gate-type AND --timesteps 10

# Run simulation with CSV input
cargo run --release -- run --input data.csv --output results.csv --neurons 100 --timesteps 20

# Display network statistics
cargo run --release -- stats --neurons 50

# Show help
cargo run --release -- --help
```

### Library Usage

```rust
use liquid_neural_network::prelude::*;

// Create a network using the builder pattern
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

// Serialize/deserialize
let json = serde_json::to_string(&lnn)?;
let restored: LiquidNeuralNetwork = serde_json::from_str(&json)?;
```

### Logic Gate Configuration

```rust
let mut lnn = LiquidNeuralNetwork::new(10);
lnn.configure_for_logic_gate("AND")?;  // Also: OR, NOT, NAND, NOR, XOR

let output = lnn.run_simulation(10, vec![vec![1.0, 1.0]]);
```

## Architecture

```
src/
├── main.rs              # CLI entry point and integration tests
├── lnn.rs               # Core LiquidNeuralNetwork implementation
├── network_components.rs # Neuron and Synapse definitions
├── activation.rs        # Activation functions (sigmoid, tanh, ReLU, etc.)
├── error.rs             # Custom error types
├── input.rs             # CSV input handling
└── output.rs            # CSV output handling
```

### Key Components

- **Neuron**: Implements leaky integrate-and-fire dynamics with:
  - Membrane potential
  - Dynamic time constant (the "liquid" property)
  - Adaptive firing threshold
  - Configurable resting potential (biological or normalized)

- **Synapse**: Directed weighted connections with:
  - Configurable delay
  - Excitatory/inhibitory classification
  - Linear signal propagation

- **LiquidNeuralNetwork**: The main network container with:
  - Sparse synaptic matrix
  - Spectral radius scaling for stability
  - Input encoding and output decoding (sigmoid normalized)
  - Linear regression readout layer

## Testing

The project includes 80 comprehensive tests:

```bash
# Run all tests
cargo test

# Run specific test module
cargo test network_components

# Run with output
cargo test -- --nocapture
```

## Research Background

This implementation is based on research from MIT CSAIL:

- [Liquid Time-constant Networks](https://arxiv.org/pdf/2006.04439.pdf) - The foundational paper on LNNs
- [Robust Flight Navigation with Liquid Neural Networks](https://dspace.mit.edu/handle/1721.1/144657) - Applications in robotics
- [Liquid Structural State-Space Models](https://arxiv.org/pdf/2209.12951.pdf) - Extended architectures
- [A Spectral Condition for Feature Learning](https://arxiv.org/pdf/2310.17813.pdf) - Spectral radius initialization

See also: [YouTube explainer on LNNs](https://www.youtube.com/watch?v=0FNkrjVIcuk)

## Current Status

**Implemented:**
- [x] Core neuron dynamics with leaky integrate-and-fire model
- [x] Synaptic connections with delays and weights
- [x] Dynamic time constants (liquid property)
- [x] Spectral radius-based weight initialization
- [x] Builder pattern for network configuration
- [x] Logic gate demonstrations (AND, OR, NOT, NAND, NOR, XOR)
- [x] CSV input/output support
- [x] JSON serialization/deserialization
- [x] Comprehensive test suite (80 tests)
- [x] Professional CLI with clap
- [x] Custom error handling

**Next Steps:**
- [ ] Implement proper ODE solver for continuous-time dynamics
- [ ] Add BPTT (Backpropagation Through Time) training
- [ ] Benchmark against reference implementations
- [ ] Add support for different neuron models (LTC, CfC)
- [ ] GPU acceleration via wgpu or CUDA bindings
- [ ] Time-series prediction examples (weather, stock prices)
- [ ] Documentation and API docs

## Learning Goals

This project serves as an exercise in:
1. Learning Rust and its ownership model
2. Understanding neural network architectures
3. Exploring liquid neural networks and their properties
4. Working with the Rust ML ecosystem (ndarray, nalgebra, smartcore)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
