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

pub mod activation;
pub mod cfc;
pub mod error;
pub mod input;
pub mod lnn;
pub mod ltc;
pub mod network_components;
pub mod ode_solver;
pub mod output;
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
    pub use crate::lnn::{
        LiquidNeuralNetwork, LnnBuilder, LnnConfig, LogicGate, NetworkStatistics,
    };
    pub use crate::network_components::{Neuron, NeuronDerivatives, NeuronEvent, Synapse};
    pub use crate::output::{write_output_to_csv, write_output_with_headers};

    // LTC (Liquid Time-Constant)
    pub use crate::ltc::{LtcBuilder, LtcCell, LtcConfig, LtcNetwork};

    // CfC (Closed-form Continuous)
    pub use crate::cfc::{BackboneActivation, CfcBuilder, CfcCell, CfcConfig, CfcNetwork};

    // ODE Solvers
    pub use crate::ode_solver::{
        DormandPrince, EulerSolver, OdeSolver, OdeState, OdeSystem, RungeKutta4, SolverConfig,
    };

    // Training
    pub use crate::training::{
        mse_loss_with_grad, BpttConfig, TrainableDense, TrainableRnn, TrainableRnnBuilder,
        TrainableRnnCell,
    };
}

// Re-export main types at crate root for convenience
pub use cfc::{CfcBuilder, CfcCell, CfcConfig, CfcNetwork};
pub use error::{LnnError, Result};
pub use lnn::{LiquidNeuralNetwork, LnnBuilder, LnnConfig, LogicGate};
pub use ltc::{LtcBuilder, LtcCell, LtcConfig, LtcNetwork};
pub use network_components::{Neuron, Synapse};
