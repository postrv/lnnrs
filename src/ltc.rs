// src/ltc.rs
//! Liquid Time-Constant (LTC) neural network implementation.
//!
//! This module implements the continuous-time neural network model from the MIT CSAIL paper
//! "Liquid Time-constant Networks" (Hasani et al., 2020). LTC networks are a class of
//! recurrent neural networks with input-dependent time constants that adapt their dynamics
//! based on the current input and state.
//!
//! The key equation governing LTC neurons is:
//!
//! ```text
//! τ(x,I) * dx/dt = -x + f(x,I)
//! ```
//!
//! where τ(x,I) is an input-dependent time constant and f(x,I) is the activation function.

use crate::activation::{sigmoid, tanh};
use crate::ode_solver::{OdeSolver, OdeState, OdeSystem, RungeKutta4, SolverConfig};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

/// Configuration for LTC layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcConfig {
    /// Number of hidden units
    pub units: usize,
    /// Time constant base value
    pub tau_base: f64,
    /// Minimum time constant (for stability)
    pub tau_min: f64,
    /// Maximum time constant
    pub tau_max: f64,
    /// ODE solver step size
    pub ode_step: f64,
    /// Number of ODE steps per timestep
    pub ode_unfolds: usize,
}

impl Default for LtcConfig {
    fn default() -> Self {
        LtcConfig {
            units: 64,
            tau_base: 1.0,
            tau_min: 0.1,
            tau_max: 10.0,
            ode_step: 0.1,
            ode_unfolds: 6,
        }
    }
}

impl LtcConfig {
    /// Creates a new LTC config with the specified number of units.
    pub fn new(units: usize) -> Self {
        LtcConfig {
            units,
            ..Default::default()
        }
    }

    /// Sets the time constant range.
    pub fn with_tau_range(mut self, min: f64, max: f64) -> Self {
        self.tau_min = min;
        self.tau_max = max;
        self
    }

    /// Sets the base time constant.
    pub fn with_tau_base(mut self, base: f64) -> Self {
        self.tau_base = base;
        self
    }

    /// Sets the ODE solver parameters.
    pub fn with_ode_params(mut self, step: f64, unfolds: usize) -> Self {
        self.ode_step = step;
        self.ode_unfolds = unfolds;
        self
    }
}

/// Liquid Time-Constant Cell implementing the continuous-time dynamics.
///
/// This is the core computational unit of LTC networks, implementing the ODE:
/// ```text
/// dx/dt = (-x + f(W_in * I + W_rec * x + b)) / tau(x, I)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcCell {
    /// Configuration
    pub config: LtcConfig,
    /// Input weights [input_dim x units]
    pub w_in: Vec<Vec<f64>>,
    /// Recurrent weights [units x units]
    pub w_rec: Vec<Vec<f64>>,
    /// Bias terms [units]
    pub bias: Vec<f64>,
    /// Time constant weights (for input-dependent tau) [units]
    pub tau_weights: Vec<f64>,
    /// Time constant bias [units]
    pub tau_bias: Vec<f64>,
    /// Input dimension
    input_dim: usize,
}

impl LtcCell {
    /// Creates a new LTC cell with random initialization.
    pub fn new(input_dim: usize, config: LtcConfig) -> Self {
        let mut rng = thread_rng();
        let units = config.units;

        // Xavier/Glorot initialization
        let in_scale = (2.0 / (input_dim + units) as f64).sqrt();
        let rec_scale = (2.0 / (units + units) as f64).sqrt();

        let w_in: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| {
                (0..units)
                    .map(|_| rng.gen_range(-in_scale..in_scale))
                    .collect()
            })
            .collect();

        let w_rec: Vec<Vec<f64>> = (0..units)
            .map(|_| {
                (0..units)
                    .map(|_| rng.gen_range(-rec_scale..rec_scale))
                    .collect()
            })
            .collect();

        let bias = vec![0.0; units];
        let tau_weights = (0..units).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let tau_bias = vec![config.tau_base.ln(); units];

        LtcCell {
            config,
            w_in,
            w_rec,
            bias,
            tau_weights,
            tau_bias,
            input_dim,
        }
    }

    /// Returns the number of hidden units.
    pub fn units(&self) -> usize {
        self.config.units
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Computes the input-dependent time constant for each unit.
    ///
    /// τ(I) = tau_min + (tau_max - tau_min) * sigmoid(tau_weights * |I| + tau_bias)
    fn compute_tau(&self, input: &[f64]) -> Vec<f64> {
        let input_magnitude: f64 = input.iter().map(|x| x.abs()).sum();

        self.tau_weights
            .iter()
            .zip(self.tau_bias.iter())
            .map(|(&w, &b)| {
                let logit = w * input_magnitude + b;
                self.config.tau_min + (self.config.tau_max - self.config.tau_min) * sigmoid(logit)
            })
            .collect()
    }

    /// Computes the pre-activation: W_in * input + W_rec * state + bias
    fn pre_activation(&self, state: &[f64], input: &[f64]) -> Vec<f64> {
        let units = self.config.units;
        let mut result = self.bias.clone();

        // Input contribution: W_in * input
        for (i, &inp) in input.iter().enumerate() {
            if i < self.w_in.len() {
                for j in 0..units {
                    result[j] += self.w_in[i][j] * inp;
                }
            }
        }

        // Recurrent contribution: W_rec * state
        for (i, &s) in state.iter().enumerate() {
            if i < self.w_rec.len() {
                for j in 0..units {
                    result[j] += self.w_rec[i][j] * s;
                }
            }
        }

        result
    }

    /// Computes the derivative dx/dt for the LTC dynamics.
    ///
    /// dx/dt = (-x + tanh(pre_activation)) / tau
    pub fn derivative(&self, state: &[f64], input: &[f64]) -> Vec<f64> {
        let tau = self.compute_tau(input);
        let pre_act = self.pre_activation(state, input);

        state
            .iter()
            .enumerate()
            .map(|(i, &x)| (-x + tanh(pre_act[i])) / tau[i])
            .collect()
    }

    /// Performs one forward pass using ODE integration.
    ///
    /// Integrates the LTC ODE from t=0 to t=1 (one logical timestep).
    pub fn forward(&self, state: &[f64], input: &[f64]) -> Vec<f64> {
        let system = LtcOdeSystem {
            cell: self,
            input: input.to_vec(),
        };

        let solver = RungeKutta4;
        let config = SolverConfig::new(self.config.ode_step);

        let result = solver
            .integrate(&system, state.to_vec(), input, 0.0, 1.0, &config)
            .unwrap_or_else(|_| OdeState::new(state.to_vec()));

        result.values
    }

    /// Processes a sequence of inputs.
    ///
    /// Returns the sequence of hidden states.
    pub fn forward_sequence(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut state = vec![0.0; self.config.units];
        let mut states = Vec::with_capacity(inputs.len());

        for input in inputs {
            state = self.forward(&state, input);
            states.push(state.clone());
        }

        states
    }

    /// Initializes the hidden state to zeros.
    pub fn initial_state(&self) -> Vec<f64> {
        vec![0.0; self.config.units]
    }
}

/// ODE system wrapper for LTC cell integration.
struct LtcOdeSystem<'a> {
    cell: &'a LtcCell,
    input: Vec<f64>,
}

impl<'a> OdeSystem for LtcOdeSystem<'a> {
    fn derivative(&self, _t: f64, state: &[f64], _input: &[f64]) -> Vec<f64> {
        self.cell.derivative(state, &self.input)
    }

    fn dimension(&self) -> usize {
        self.cell.config.units
    }
}

/// A complete LTC network with input and output projections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcNetwork {
    /// The LTC cell
    pub cell: LtcCell,
    /// Output projection weights [units x output_dim]
    pub w_out: Vec<Vec<f64>>,
    /// Output bias [output_dim]
    pub b_out: Vec<f64>,
    /// Output dimension
    output_dim: usize,
}

impl LtcNetwork {
    /// Creates a new LTC network.
    pub fn new(input_dim: usize, hidden_units: usize, output_dim: usize) -> Self {
        let mut rng = thread_rng();
        let config = LtcConfig::new(hidden_units);
        let cell = LtcCell::new(input_dim, config);

        // Xavier initialization for output weights
        let out_scale = (2.0 / (hidden_units + output_dim) as f64).sqrt();
        let w_out: Vec<Vec<f64>> = (0..hidden_units)
            .map(|_| {
                (0..output_dim)
                    .map(|_| rng.gen_range(-out_scale..out_scale))
                    .collect()
            })
            .collect();

        let b_out = vec![0.0; output_dim];

        LtcNetwork {
            cell,
            w_out,
            b_out,
            output_dim,
        }
    }

    /// Creates a new LTC network with custom configuration.
    pub fn with_config(input_dim: usize, output_dim: usize, config: LtcConfig) -> Self {
        let mut rng = thread_rng();
        let hidden_units = config.units;
        let cell = LtcCell::new(input_dim, config);

        let out_scale = (2.0 / (hidden_units + output_dim) as f64).sqrt();
        let w_out: Vec<Vec<f64>> = (0..hidden_units)
            .map(|_| {
                (0..output_dim)
                    .map(|_| rng.gen_range(-out_scale..out_scale))
                    .collect()
            })
            .collect();

        let b_out = vec![0.0; output_dim];

        LtcNetwork {
            cell,
            w_out,
            b_out,
            output_dim,
        }
    }

    /// Projects hidden state to output.
    fn project_output(&self, hidden: &[f64]) -> Vec<f64> {
        let mut output = self.b_out.clone();
        for (i, &h) in hidden.iter().enumerate() {
            if i < self.w_out.len() {
                for j in 0..self.output_dim {
                    output[j] += self.w_out[i][j] * h;
                }
            }
        }
        output
    }

    /// Processes a single input and returns output.
    pub fn forward(&self, state: &[f64], input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let new_state = self.cell.forward(state, input);
        let output = self.project_output(&new_state);
        (new_state, output)
    }

    /// Processes a sequence and returns all outputs.
    pub fn forward_sequence(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let hidden_states = self.cell.forward_sequence(inputs);
        hidden_states
            .iter()
            .map(|h| self.project_output(h))
            .collect()
    }

    /// Returns the initial hidden state.
    pub fn initial_state(&self) -> Vec<f64> {
        self.cell.initial_state()
    }

    /// Returns the output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Returns the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.cell.units()
    }
}

/// Builder for LTC networks.
pub struct LtcBuilder {
    input_dim: usize,
    output_dim: usize,
    config: LtcConfig,
}

impl LtcBuilder {
    /// Creates a new LTC network builder.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        LtcBuilder {
            input_dim,
            output_dim,
            config: LtcConfig::default(),
        }
    }

    /// Sets the number of hidden units.
    pub fn hidden_units(mut self, units: usize) -> Self {
        self.config.units = units;
        self
    }

    /// Sets the time constant range.
    pub fn tau_range(mut self, min: f64, max: f64) -> Self {
        self.config.tau_min = min;
        self.config.tau_max = max;
        self
    }

    /// Sets the ODE integration parameters.
    pub fn ode_params(mut self, step: f64, unfolds: usize) -> Self {
        self.config.ode_step = step;
        self.config.ode_unfolds = unfolds;
        self
    }

    /// Builds the LTC network.
    pub fn build(self) -> LtcNetwork {
        LtcNetwork::with_config(self.input_dim, self.output_dim, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltc_config_default() {
        let config = LtcConfig::default();
        assert_eq!(config.units, 64);
        assert!(config.tau_min < config.tau_max);
    }

    #[test]
    fn test_ltc_config_builder() {
        let config = LtcConfig::new(32)
            .with_tau_range(0.5, 5.0)
            .with_tau_base(2.0)
            .with_ode_params(0.05, 10);

        assert_eq!(config.units, 32);
        assert_eq!(config.tau_min, 0.5);
        assert_eq!(config.tau_max, 5.0);
        assert_eq!(config.tau_base, 2.0);
        assert_eq!(config.ode_step, 0.05);
        assert_eq!(config.ode_unfolds, 10);
    }

    #[test]
    fn test_ltc_cell_creation() {
        let config = LtcConfig::new(16);
        let cell = LtcCell::new(4, config);

        assert_eq!(cell.units(), 16);
        assert_eq!(cell.input_dim(), 4);
        assert_eq!(cell.w_in.len(), 4);
        assert_eq!(cell.w_in[0].len(), 16);
        assert_eq!(cell.w_rec.len(), 16);
        assert_eq!(cell.w_rec[0].len(), 16);
    }

    #[test]
    fn test_ltc_cell_tau_computation() {
        let config = LtcConfig::new(8).with_tau_range(0.1, 10.0);
        let cell = LtcCell::new(2, config);

        let tau = cell.compute_tau(&[0.0, 0.0]);
        for &t in &tau {
            assert!(t >= 0.1 && t <= 10.0);
        }

        // Higher input should affect tau
        let tau_high = cell.compute_tau(&[10.0, 10.0]);
        // Just verify it's in range (actual direction depends on weights)
        for &t in &tau_high {
            assert!(t >= 0.1 && t <= 10.0);
        }
    }

    #[test]
    fn test_ltc_cell_forward() {
        let config = LtcConfig::new(8);
        let cell = LtcCell::new(2, config);

        let state = vec![0.0; 8];
        let input = vec![1.0, 0.5];

        let new_state = cell.forward(&state, &input);

        assert_eq!(new_state.len(), 8);
        // State should have changed
        assert!(new_state.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn test_ltc_cell_sequence() {
        let config = LtcConfig::new(4);
        let cell = LtcCell::new(2, config);

        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        let states = cell.forward_sequence(&inputs);

        assert_eq!(states.len(), 3);
        for state in &states {
            assert_eq!(state.len(), 4);
        }
    }

    #[test]
    fn test_ltc_network_creation() {
        let network = LtcNetwork::new(3, 16, 2);

        assert_eq!(network.hidden_dim(), 16);
        assert_eq!(network.output_dim(), 2);
    }

    #[test]
    fn test_ltc_network_forward() {
        let network = LtcNetwork::new(3, 8, 2);

        let state = network.initial_state();
        let input = vec![1.0, 0.5, 0.2];

        let (new_state, output) = network.forward(&state, &input);

        assert_eq!(new_state.len(), 8);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_ltc_network_sequence() {
        let network = LtcNetwork::new(2, 8, 1);

        let inputs = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
        ];

        let outputs = network.forward_sequence(&inputs);

        assert_eq!(outputs.len(), 4);
        for output in &outputs {
            assert_eq!(output.len(), 1);
        }
    }

    #[test]
    fn test_ltc_builder() {
        let network = LtcBuilder::new(4, 2)
            .hidden_units(32)
            .tau_range(0.5, 5.0)
            .ode_params(0.05, 8)
            .build();

        assert_eq!(network.hidden_dim(), 32);
        assert_eq!(network.output_dim(), 2);
    }

    #[test]
    fn test_ltc_derivative_bounded() {
        let config = LtcConfig::new(4);
        let cell = LtcCell::new(2, config);

        // State bounded in [-1, 1] due to tanh
        let state = vec![0.5, -0.3, 0.8, -0.2];
        let input = vec![1.0, 0.5];

        let derivative = cell.derivative(&state, &input);

        // Derivatives should be finite
        for &d in &derivative {
            assert!(d.is_finite());
        }
    }

    #[test]
    fn test_ltc_network_serialization() {
        let network = LtcNetwork::new(2, 8, 1);

        let json = serde_json::to_string(&network).unwrap();
        let restored: LtcNetwork = serde_json::from_str(&json).unwrap();

        assert_eq!(network.hidden_dim(), restored.hidden_dim());
        assert_eq!(network.output_dim(), restored.output_dim());
    }

    #[test]
    fn test_ltc_stability_long_sequence() {
        let network = LtcNetwork::new(2, 16, 1);

        // Generate a long sequence
        let inputs: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let t = i as f64 * 0.1;
                vec![t.sin(), t.cos()]
            })
            .collect();

        let outputs = network.forward_sequence(&inputs);

        // Check all outputs are finite
        for output in &outputs {
            for &o in output {
                assert!(o.is_finite(), "Output became non-finite");
            }
        }
    }
}
