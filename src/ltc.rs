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
//! tau(x,I) * dx/dt = -x + f(x,I)
//! ```
//!
//! where tau(x,I) is an input-dependent time constant and f(x,I) is the activation function.

use crate::activation::{sigmoid, tanh};
use crate::error::{LnnError, Result};
use crate::layers::DenseLayer;
use crate::ode_solver::{OdeSolver, OdeState, OdeSystem, RungeKutta4};
use ndarray::ArrayView1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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
    /// Number of fixed RK4 sub-steps used to integrate one logical timestep over `[0, 1]`
    /// (the integration step size is `1.0 / ode_unfolds`). Must be `>= 1`.
    pub ode_unfolds: usize,
    /// Optional seed for reproducible random initialization. `Some(seed)` seeds a `StdRng`
    /// with `seed`; `None` seeds from entropy.
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Default for LtcConfig {
    fn default() -> Self {
        LtcConfig {
            units: 64,
            tau_base: 1.0,
            tau_min: 0.1,
            tau_max: 10.0,
            ode_unfolds: 6,
            seed: None,
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

    /// Sets the number of RK4 sub-steps used to integrate one logical timestep.
    ///
    /// Larger values give a more accurate integration of the LTC ODE at higher cost.
    pub fn with_ode_unfolds(mut self, unfolds: usize) -> Self {
        self.ode_unfolds = unfolds;
        self
    }

    /// Sets the seed for reproducible random initialization.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    /// Returns [`LnnError::Configuration`] if `units < 1`, `tau_min <= 0`,
    /// `tau_min > tau_max`, or `ode_unfolds < 1`.
    pub fn validate(&self) -> Result<()> {
        if self.units < 1 {
            return Err(LnnError::configuration("units must be >= 1"));
        }
        if self.tau_min <= 0.0 {
            return Err(LnnError::configuration("tau_min must be > 0"));
        }
        if self.tau_min > self.tau_max {
            return Err(LnnError::configuration("tau_min must be <= tau_max"));
        }
        if self.ode_unfolds < 1 {
            return Err(LnnError::configuration("ode_unfolds must be >= 1"));
        }
        Ok(())
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
    /// Input projection `W_in` (`input_dim -> units`). Its bias doubles as the cell bias
    /// (added once to the pre-activation), replacing the former separate `bias` field.
    pub w_in: DenseLayer,
    /// Recurrent projection `W_rec` (`units -> units`); its bias is zero.
    pub w_rec: DenseLayer,
    /// Time constant weights for the input-dependent tau (length `units`).
    pub tau_weights: Vec<f64>,
    /// Time constant bias (length `units`).
    pub tau_bias: Vec<f64>,
    /// Input dimension
    input_dim: usize,
}

impl LtcCell {
    /// Creates a new LTC cell with random initialization.
    ///
    /// This is a self-contained convenience constructor: it creates its own [`StdRng`] from
    /// `config.seed` (or from entropy when unseeded). When building a full [`LtcNetwork`],
    /// [`LtcCell::with_rng`] is used instead so the cell and the output projection share a
    /// single generator.
    pub fn new(input_dim: usize, config: LtcConfig) -> Self {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        Self::with_rng(input_dim, config, &mut rng)
    }

    /// Creates a new LTC cell drawing all weights from the supplied generator.
    ///
    /// Threading a single [`StdRng`] through every layer of a network is what makes
    /// construction reproducible; higher-level constructors ([`LtcNetwork::with_config`])
    /// call this so unit ordering is stable.
    pub fn with_rng(input_dim: usize, config: LtcConfig, rng: &mut StdRng) -> Self {
        let units = config.units;

        // Xavier/Glorot initialization via the shared layer. `DenseLayer::new` uses
        // `scale = sqrt(2 / (in + out))`, which reproduces the previous `in_scale`/`rec_scale`
        // exactly. The cell bias is `w_in`'s (zero-initialised) bias.
        let w_in = DenseLayer::new(input_dim, units, rng);
        let w_rec = DenseLayer::new(units, units, rng);

        let tau_weights = (0..units).map(|_| rng.random_range(-0.5..0.5)).collect();
        let tau_bias = vec![config.tau_base.ln(); units];

        LtcCell {
            config,
            w_in,
            w_rec,
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
    /// tau(I) = tau_min + (tau_max - tau_min) * sigmoid(tau_weights * |I| + tau_bias)
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

    /// Computes the pre-activation: `W_in * input + W_rec * state + bias`.
    ///
    /// The cell bias is carried by `w_in` (`w_rec`'s bias is zero), so it is added exactly
    /// once. `W_in * input + bias` comes from the shared layer's `.dot()`; the recurrent
    /// `W_rec * state` term is accumulated in place to avoid an extra allocation (and the
    /// no-op zero-bias add) in the RK4 hot path.
    fn pre_activation(&self, state: &[f64], input: &[f64]) -> ndarray::Array1<f64> {
        let mut pre = self.w_in.forward_arr(ArrayView1::from(input));
        pre += &self.w_rec.weights.dot(&ArrayView1::from(state));
        pre
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
    /// Integrates the LTC ODE over `t` in `[0, 1]` (one logical timestep) with exactly
    /// `config.ode_unfolds` fixed RK4 sub-steps of size `dt = 1.0 / ode_unfolds`, applying
    /// [`RungeKutta4::step`] directly. There is no adaptive stepping and no error path to
    /// swallow: `ode_unfolds` is the single source of truth for integration effort.
    pub fn forward(&self, state: &[f64], input: &[f64]) -> Vec<f64> {
        let system = LtcOdeSystem {
            cell: self,
            input: input.to_vec(),
        };

        let solver = RungeKutta4;
        // `ode_unfolds` must be >= 1 (validated at config-build time in section 1f); guard
        // defensively here so a degenerate config can never divide by zero.
        let unfolds = self.config.ode_unfolds.max(1);
        let dt = 1.0 / unfolds as f64;

        let mut ode_state = OdeState::new(state.to_vec());
        for _ in 0..unfolds {
            solver.step(&system, &mut ode_state, input, dt);
            ode_state.time += dt;
        }

        ode_state.values
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
    /// Output projection (`units -> output_dim`), replacing the former `w_out`/`b_out` fields.
    output_layer: DenseLayer,
}

impl LtcNetwork {
    /// Creates a new LTC network.
    pub fn new(input_dim: usize, hidden_units: usize, output_dim: usize) -> Self {
        Self::with_config(input_dim, output_dim, LtcConfig::new(hidden_units))
    }

    /// Creates a new LTC network with custom configuration.
    ///
    /// A single [`StdRng`] (seeded from `config.seed`, or from entropy when unseeded) is
    /// threaded through the cell and then the output projection, so a given seed yields a
    /// byte-identical network.
    pub fn with_config(input_dim: usize, output_dim: usize, config: LtcConfig) -> Self {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        let hidden_units = config.units;
        let cell = LtcCell::with_rng(input_dim, config, &mut rng);
        let output_layer = DenseLayer::new(hidden_units, output_dim, &mut rng);

        LtcNetwork { cell, output_layer }
    }

    /// Projects hidden state to output.
    fn project_output(&self, hidden: &[f64]) -> Vec<f64> {
        self.output_layer.forward(hidden)
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
        self.output_layer.output_dim()
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

    /// Sets the number of RK4 sub-steps used to integrate one logical timestep.
    pub fn ode_unfolds(mut self, unfolds: usize) -> Self {
        self.config.ode_unfolds = unfolds;
        self
    }

    /// Sets the seed for reproducible random initialization.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Builds the LTC network.
    ///
    /// # Panics
    /// Panics with the validation message if the configuration is invalid. Use
    /// [`try_build`](Self::try_build) for a non-panicking variant.
    pub fn build(self) -> LtcNetwork {
        self.try_build()
            .unwrap_or_else(|e| panic!("invalid LtcConfig: {e}"))
    }

    /// Builds the LTC network, returning an error instead of panicking on an invalid
    /// configuration.
    ///
    /// # Errors
    /// Returns the [`LtcConfig::validate`] error if the configuration is invalid.
    pub fn try_build(self) -> Result<LtcNetwork> {
        self.config.validate()?;
        Ok(LtcNetwork::with_config(
            self.input_dim,
            self.output_dim,
            self.config,
        ))
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
            .with_ode_unfolds(10);

        assert_eq!(config.units, 32);
        assert_eq!(config.tau_min, 0.5);
        assert_eq!(config.tau_max, 5.0);
        assert_eq!(config.tau_base, 2.0);
        assert_eq!(config.ode_unfolds, 10);
    }

    #[test]
    fn test_ltc_config_validate() {
        assert!(LtcConfig::new(16).validate().is_ok());
        assert!(LtcConfig::new(0).validate().is_err());
        assert!(LtcConfig::new(16)
            .with_tau_range(0.0, 1.0)
            .validate()
            .is_err());
        assert!(LtcConfig::new(16)
            .with_tau_range(5.0, 1.0)
            .validate()
            .is_err());
        assert!(LtcConfig::new(16).with_ode_unfolds(0).validate().is_err());
    }

    #[test]
    fn test_ltc_builder_try_build() {
        assert!(LtcBuilder::new(2, 1).hidden_units(8).try_build().is_ok());
        assert!(LtcBuilder::new(2, 1).ode_unfolds(0).try_build().is_err());
    }

    #[test]
    fn test_ltc_cell_creation() {
        let config = LtcConfig::new(16);
        let cell = LtcCell::new(4, config);

        assert_eq!(cell.units(), 16);
        assert_eq!(cell.input_dim(), 4);
        assert_eq!(cell.w_in.input_dim(), 4);
        assert_eq!(cell.w_in.output_dim(), 16);
        assert_eq!(cell.w_rec.input_dim(), 16);
        assert_eq!(cell.w_rec.output_dim(), 16);
    }

    #[test]
    fn test_ltc_cell_tau_computation() {
        let config = LtcConfig::new(8).with_tau_range(0.1, 10.0);
        let cell = LtcCell::new(2, config);

        let tau = cell.compute_tau(&[0.0, 0.0]);
        for &t in &tau {
            assert!((0.1..=10.0).contains(&t));
        }

        // Higher input should affect tau
        let tau_high = cell.compute_tau(&[10.0, 10.0]);
        // Just verify it's in range (actual direction depends on weights)
        for &t in &tau_high {
            assert!((0.1..=10.0).contains(&t));
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
            .ode_unfolds(8)
            .build();

        assert_eq!(network.hidden_dim(), 32);
        assert_eq!(network.output_dim(), 2);
    }

    #[test]
    fn test_ltc_ode_unfolds_effect_and_convergence() {
        // Build one cell, then clone it and vary only `ode_unfolds` so all three cells
        // share identical weights. A coarser integration (fewer unfolds) must give a
        // measurably different result, and refining the integration must converge toward
        // the finest reference (unfolds = 32). Small time constants are used so that the
        // dynamics are fast enough for the step count to matter over the [0, 1] window.
        let config = LtcConfig::new(16)
            .with_tau_range(0.2, 0.6)
            .with_tau_base(0.3);
        let cell1 = LtcCell::new(2, config.with_ode_unfolds(1));
        let mut cell8 = cell1.clone();
        cell8.config.ode_unfolds = 8;
        let mut cell32 = cell1.clone();
        cell32.config.ode_unfolds = 32;

        let state = cell1.initial_state();
        let input = vec![2.0, -2.0];

        let out1 = cell1.forward(&state, &input);
        let out8 = cell8.forward(&state, &input);
        let out32 = cell32.forward(&state, &input);

        let sq_err = |a: &[f64], b: &[f64]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
        };

        let err_1 = sq_err(&out1, &out32);
        let err_8 = sq_err(&out8, &out32);

        // unfolds = 1 must differ from the fine reference.
        assert!(
            err_1 > 1e-9,
            "unfolds=1 should differ from unfolds=32 (err = {err_1})"
        );
        // Refining the integration converges: unfolds = 8 is closer than unfolds = 1.
        assert!(
            err_8 < err_1,
            "unfolds=8 should be closer to unfolds=32 than unfolds=1 ({err_8} !< {err_1})"
        );
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
    fn test_ltc_seed_reproducible() {
        // Same seed → byte-identical serialized network (single RNG threaded through the
        // cell and the output projection).
        let a = LtcBuilder::new(3, 2).hidden_units(16).seed(7).build();
        let b = LtcBuilder::new(3, 2).hidden_units(16).seed(7).build();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
        );
    }

    #[test]
    fn test_ltc_seed_differs() {
        let a = LtcBuilder::new(3, 2).hidden_units(16).seed(7).build();
        let b = LtcBuilder::new(3, 2).hidden_units(16).seed(8).build();
        assert_ne!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
        );
    }

    #[test]
    fn test_ltc_unseeded_still_works() {
        let net = LtcNetwork::new(2, 8, 1);
        assert_eq!(net.hidden_dim(), 8);
        assert_eq!(net.output_dim(), 1);
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
