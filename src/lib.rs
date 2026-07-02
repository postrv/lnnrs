// src/lib.rs
//! Liquid Neural Network Library
//!
//! A Rust implementation of Liquid Neural Networks (LNNs) based on MIT CSAIL research.
//!
//! # Overview
//!
//! Liquid Neural Networks are a class of neural networks with dynamic time constants
//! that adapt based on input, making them particularly suited for time-series processing tasks.
//!
//! This library provides three main network architectures:
//!
//! - **LNN (Liquid Neural Network)**: Classic reservoir computing with leaky integrate-and-fire neurons
//! - **LTC (Liquid Time-Constant)**: Continuous-time networks with ODE-based dynamics
//! - **CfC (Closed-form Continuous)**: Analytical solution for faster inference
//!
//! # Quick Start
//!
//! ```rust
//! use liquid_neural_network::prelude::*;
//!
//! // Create a CfC network (fastest)
//! let network = CfcBuilder::new(2, 1)
//!     .hidden_units(32)
//!     .build();
//!
//! // Process a sequence
//! let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
//! let outputs = network.forward_sequence(&inputs);
//! ```
//!
//! # Features
//!
//! - Multiple ODE solvers (Euler, RK4, Dormand-Prince)
//! - Builder pattern for network configuration
//! - Full serialization support via serde
//! - Comprehensive test coverage
//!
//! # Reproducibility
//!
//! Every network's random initialization is reproducible. Each configuration
//! (`LnnConfig`, `LtcConfig`, `CfcConfig`, `BpttConfig`) carries an optional
//! `seed: Option<u64>`, and every builder exposes a `.seed(u64)` method. When a seed is set,
//! the whole network is constructed from a single [`rand::rngs::StdRng`] seeded with that
//! value: `StdRng::seed_from_u64(seed)`. When no seed is set, the generator is seeded from
//! the OS RNG (`StdRng::from_os_rng`). A single generator is threaded through every layer in a
//! fixed order, so two networks built from the same seed are byte-for-byte identical,
//! including their serialized JSON:
//!
//! ```rust
//! use liquid_neural_network::prelude::*;
//!
//! let a = CfcBuilder::new(2, 1).hidden_units(8).seed(42).build();
//! let b = CfcBuilder::new(2, 1).hidden_units(8).seed(42).build();
//! assert_eq!(
//!     serde_json::to_string(&a).unwrap(),
//!     serde_json::to_string(&b).unwrap(),
//! );
//! ```
//!
//! For `TrainableRnn`, use `TrainableRnnBuilder::new(..).seed(u64)` or the direct
//! constructor `TrainableRnn::with_seed(input, hidden, output, Some(seed))`.

pub mod activation;
pub mod cfc;
pub mod error;
pub mod input;
pub mod layers;
pub mod lnn;
pub mod ltc;
pub mod network_components;
pub mod ode_solver;
pub mod optim;
pub mod output;
pub mod train_cfc;
pub mod training;

/// Prelude module for convenient imports.
///
/// Import all commonly used types with:
/// ```rust
/// use liquid_neural_network::prelude::*;
/// ```
pub mod prelude {
    // Core LNN
    pub use crate::error::{LnnError, Result};
    pub use crate::input::{read_input_from_csv, validate_inputs};
    pub use crate::layers::DenseLayer;
    pub use crate::lnn::{
        LiquidNeuralNetwork, LnnBuilder, LnnConfig, LogicGate, NetworkStatistics, ReadoutConfig,
    };
    pub use crate::network_components::{Neuron, NeuronDerivatives, Synapse, ThresholdAdaptation};
    pub use crate::output::{write_output_to_csv, write_output_with_headers};

    // LTC (Liquid Time-Constant)
    pub use crate::ltc::{LtcBuilder, LtcCell, LtcConfig, LtcNetwork};

    // CfC (Closed-form Continuous)
    pub use crate::cfc::{BackboneActivation, CfcBuilder, CfcCell, CfcConfig, CfcNetwork};

    // ODE Solvers
    pub use crate::ode_solver::{
        DormandPrince, EulerSolver, OdeSolver, OdeState, OdeSystem, RungeKutta4, SolverConfig,
    };

    // Optimizers
    pub use crate::optim::{clip_grad_norm, Adam, Optimizer, Sgd};

    // Training
    pub use crate::train_cfc::{TrainableCfc, TrainableCfcBuilder};
    pub use crate::training::{
        mse_loss_with_grad, BpttConfig, TrainableDense, TrainableRnn, TrainableRnnBuilder,
        TrainableRnnCell,
    };
}

// Re-export main types at crate root for convenience
pub use cfc::{CfcBuilder, CfcCell, CfcConfig, CfcNetwork};
pub use error::{LnnError, Result};
pub use layers::DenseLayer;
pub use lnn::{LiquidNeuralNetwork, LnnBuilder, LnnConfig, LogicGate};
pub use ltc::{LtcBuilder, LtcCell, LtcConfig, LtcNetwork};
pub use network_components::{Neuron, Synapse};
