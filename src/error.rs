// src/error.rs
//! Custom error types for the Liquid Neural Network library.

use thiserror::Error;

/// Errors that can occur in the LNN library.
#[derive(Error, Debug)]
pub enum LnnError {
    /// Error during network initialization
    #[error("Network initialization error: {0}")]
    Initialization(String),

    /// Error during ODE solving
    #[error("ODE solver error: {0}")]
    OdeSolver(String),

    /// Error with neuron operations
    #[error("Neuron error: {0}")]
    Neuron(String),

    /// Error with synapse operations
    #[error("Synapse error: {0}")]
    Synapse(String),

    /// Error with network configuration
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Error during I/O operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error parsing CSV data
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// Error during serialization/deserialization
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid index access
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),

    /// Training error
    #[error("Training error: {0}")]
    Training(String),

    /// Invalid gate type specified
    #[error("Invalid gate type: {0}. Supported types: AND, OR, NOT, NAND, NOR, XOR")]
    InvalidGateType(String),
}

/// Type alias for Results with LnnError
pub type Result<T> = std::result::Result<T, LnnError>;

impl LnnError {
    /// Creates a new initialization error
    pub fn initialization<S: Into<String>>(msg: S) -> Self {
        LnnError::Initialization(msg.into())
    }

    /// Creates a new configuration error
    pub fn configuration<S: Into<String>>(msg: S) -> Self {
        LnnError::Configuration(msg.into())
    }

    /// Creates a new ODE solver error
    pub fn ode_solver<S: Into<String>>(msg: S) -> Self {
        LnnError::OdeSolver(msg.into())
    }

    /// Creates a new neuron error
    pub fn neuron<S: Into<String>>(msg: S) -> Self {
        LnnError::Neuron(msg.into())
    }

    /// Creates a new synapse error
    pub fn synapse<S: Into<String>>(msg: S) -> Self {
        LnnError::Synapse(msg.into())
    }

    /// Creates a new index out of bounds error
    pub fn index_out_of_bounds(index: usize, max: usize) -> Self {
        LnnError::IndexOutOfBounds(format!("Index {} is out of bounds (max: {})", index, max))
    }
}
