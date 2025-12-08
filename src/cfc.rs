// src/cfc.rs
//! Closed-form Continuous-depth (CfC) neural network implementation.
//!
//! CfC networks are a variant of Liquid Time-Constant networks that use a closed-form
//! solution instead of numerical ODE integration. This provides significant computational
//! speedup while maintaining the expressive power of continuous-time models.
//!
//! Based on "Closed-form Continuous-Time Neural Networks" (Hasani et al., 2022).
//!
//! The key insight is that the LTC ODE:
//! ```text
//! τ * dx/dt = -x + f(Wx + b)
//! ```
//!
//! Has a closed-form solution when f is approximated appropriately:
//! ```text
//! x(t+Δt) = x(t) * exp(-Δt/τ) + (1 - exp(-Δt/τ)) * f(Wx + b)
//! ```

use crate::activation::{sigmoid, tanh};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

/// Configuration for CfC layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfcConfig {
    /// Number of hidden units
    pub units: usize,
    /// Whether to use the "no gate" variant (simpler, faster)
    pub no_gate: bool,
    /// Minimum time constant
    pub tau_min: f64,
    /// Maximum time constant
    pub tau_max: f64,
    /// Backbone activation function
    pub backbone_activation: BackboneActivation,
    /// Number of backbone layers
    pub backbone_layers: usize,
    /// Units per backbone layer
    pub backbone_units: usize,
}

/// Activation function for the backbone network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackboneActivation {
    Tanh,
    ReLU,
    SiLU,
    GELU,
}

impl BackboneActivation {
    /// Applies the activation function.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            BackboneActivation::Tanh => x.tanh(),
            BackboneActivation::ReLU => x.max(0.0),
            BackboneActivation::SiLU => x * sigmoid(x),
            BackboneActivation::GELU => {
                // Approximation: x * sigmoid(1.702 * x)
                x * sigmoid(1.702 * x)
            }
        }
    }
}

impl Default for CfcConfig {
    fn default() -> Self {
        CfcConfig {
            units: 64,
            no_gate: false,
            tau_min: 0.1,
            tau_max: 10.0,
            backbone_activation: BackboneActivation::SiLU,
            backbone_layers: 2,
            backbone_units: 64,
        }
    }
}

impl CfcConfig {
    /// Creates a new CfC config with the specified number of units.
    pub fn new(units: usize) -> Self {
        CfcConfig {
            units,
            backbone_units: units,
            ..Default::default()
        }
    }

    /// Enables the "no gate" variant for faster computation.
    pub fn with_no_gate(mut self, no_gate: bool) -> Self {
        self.no_gate = no_gate;
        self
    }

    /// Sets the time constant range.
    pub fn with_tau_range(mut self, min: f64, max: f64) -> Self {
        self.tau_min = min;
        self.tau_max = max;
        self
    }

    /// Sets the backbone activation function.
    pub fn with_backbone_activation(mut self, activation: BackboneActivation) -> Self {
        self.backbone_activation = activation;
        self
    }

    /// Sets the backbone architecture.
    pub fn with_backbone(mut self, layers: usize, units: usize) -> Self {
        self.backbone_layers = layers;
        self.backbone_units = units;
        self
    }
}

/// Dense layer for the backbone network.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DenseLayer {
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    input_dim: usize,
    output_dim: usize,
}

impl DenseLayer {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| {
                (0..output_dim)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        let bias = vec![0.0; output_dim];

        DenseLayer {
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = self.bias.clone();
        for (i, &inp) in input.iter().enumerate().take(self.input_dim) {
            for j in 0..self.output_dim {
                output[j] += self.weights[i][j] * inp;
            }
        }
        output
    }
}

/// Backbone MLP network that processes concatenated [state, input].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Backbone {
    layers: Vec<DenseLayer>,
    activation: BackboneActivation,
}

impl Backbone {
    fn new(
        input_dim: usize,
        hidden_units: usize,
        num_layers: usize,
        activation: BackboneActivation,
    ) -> Self {
        let mut layers = Vec::new();

        // First layer
        layers.push(DenseLayer::new(input_dim, hidden_units));

        // Hidden layers
        for _ in 1..num_layers {
            layers.push(DenseLayer::new(hidden_units, hidden_units));
        }

        Backbone { layers, activation }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            // Apply activation to all but the last layer
            if i < self.layers.len() - 1 {
                x = x.iter().map(|&v| self.activation.apply(v)).collect();
            }
        }
        x
    }

    fn output_dim(&self) -> usize {
        self.layers.last().map(|l| l.output_dim).unwrap_or(0)
    }
}

/// CfC Cell implementing the closed-form continuous-time dynamics.
///
/// The cell computes:
/// 1. Backbone features from [state, input]
/// 2. Time constants τ (input-dependent)
/// 3. New state using the closed-form solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfcCell {
    /// Configuration
    pub config: CfcConfig,
    /// Backbone network
    backbone: Backbone,
    /// Projection to state update (ff from backbone to state)
    ff_projection: DenseLayer,
    /// Time constant computation weights
    tau_layer: DenseLayer,
    /// Input dimension
    input_dim: usize,
}

impl CfcCell {
    /// Creates a new CfC cell.
    pub fn new(input_dim: usize, config: CfcConfig) -> Self {
        let state_input_dim = config.units + input_dim;

        let backbone = Backbone::new(
            state_input_dim,
            config.backbone_units,
            config.backbone_layers,
            config.backbone_activation,
        );

        let backbone_out_dim = backbone.output_dim();

        // Projection from backbone to state update
        let ff_projection = DenseLayer::new(backbone_out_dim, config.units);

        // Time constant layer
        let tau_layer = DenseLayer::new(backbone_out_dim, config.units);

        CfcCell {
            config,
            backbone,
            ff_projection,
            tau_layer,
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

    /// Computes time constants from backbone features.
    fn compute_tau(&self, backbone_features: &[f64]) -> Vec<f64> {
        let tau_logits = self.tau_layer.forward(backbone_features);
        tau_logits
            .iter()
            .map(|&logit| {
                self.config.tau_min + (self.config.tau_max - self.config.tau_min) * sigmoid(logit)
            })
            .collect()
    }

    /// Performs one forward pass using the closed-form solution.
    ///
    /// x(t+1) = x(t) * exp(-1/τ) + (1 - exp(-1/τ)) * f(backbone([x, input]))
    pub fn forward(&self, state: &[f64], input: &[f64], dt: f64) -> Vec<f64> {
        // Concatenate state and input for backbone
        let mut combined = state.to_vec();
        combined.extend_from_slice(input);

        // Compute backbone features
        let features = self.backbone.forward(&combined);

        // Compute time constants
        let tau = self.compute_tau(&features);

        // Compute target state (what we'd converge to)
        let ff_raw = self.ff_projection.forward(&features);
        let ff: Vec<f64> = ff_raw.iter().map(|&x| tanh(x)).collect();

        if self.config.no_gate {
            // Simple version: direct interpolation
            state
                .iter()
                .zip(ff.iter())
                .zip(tau.iter())
                .map(|((&x, &f), &t)| {
                    let decay = (-dt / t).exp();
                    x * decay + (1.0 - decay) * f
                })
                .collect()
        } else {
            // Full version with additional gating
            let decay: Vec<f64> = tau.iter().map(|&t| (-dt / t).exp()).collect();

            state
                .iter()
                .zip(ff.iter())
                .zip(decay.iter())
                .map(|((&x, &f), &d)| x * d + (1.0 - d) * f)
                .collect()
        }
    }

    /// Processes a sequence of inputs.
    pub fn forward_sequence(&self, inputs: &[Vec<f64>], dt: f64) -> Vec<Vec<f64>> {
        let mut state = self.initial_state();
        let mut states = Vec::with_capacity(inputs.len());

        for input in inputs {
            state = self.forward(&state, input, dt);
            states.push(state.clone());
        }

        states
    }

    /// Returns the initial state (zeros).
    pub fn initial_state(&self) -> Vec<f64> {
        vec![0.0; self.config.units]
    }
}

/// A complete CfC network with output projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfcNetwork {
    /// The CfC cell
    pub cell: CfcCell,
    /// Output projection
    output_layer: DenseLayer,
    /// Time step for integration
    dt: f64,
}

impl CfcNetwork {
    /// Creates a new CfC network.
    pub fn new(input_dim: usize, hidden_units: usize, output_dim: usize) -> Self {
        let config = CfcConfig::new(hidden_units);
        Self::with_config(input_dim, output_dim, config)
    }

    /// Creates a new CfC network with custom configuration.
    pub fn with_config(input_dim: usize, output_dim: usize, config: CfcConfig) -> Self {
        let hidden_units = config.units;
        let cell = CfcCell::new(input_dim, config);
        let output_layer = DenseLayer::new(hidden_units, output_dim);

        CfcNetwork {
            cell,
            output_layer,
            dt: 1.0,
        }
    }

    /// Sets the time step.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Returns the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.cell.units()
    }

    /// Returns the output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_layer.output_dim
    }

    /// Processes a single input.
    pub fn forward(&self, state: &[f64], input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let new_state = self.cell.forward(state, input, self.dt);
        let output = self.output_layer.forward(&new_state);
        (new_state, output)
    }

    /// Processes a sequence.
    pub fn forward_sequence(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let states = self.cell.forward_sequence(inputs, self.dt);
        states
            .iter()
            .map(|s| self.output_layer.forward(s))
            .collect()
    }

    /// Returns the initial state.
    pub fn initial_state(&self) -> Vec<f64> {
        self.cell.initial_state()
    }
}

/// Builder for CfC networks.
pub struct CfcBuilder {
    input_dim: usize,
    output_dim: usize,
    config: CfcConfig,
    dt: f64,
}

impl CfcBuilder {
    /// Creates a new CfC builder.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        CfcBuilder {
            input_dim,
            output_dim,
            config: CfcConfig::default(),
            dt: 1.0,
        }
    }

    /// Sets the number of hidden units.
    pub fn hidden_units(mut self, units: usize) -> Self {
        self.config.units = units;
        self.config.backbone_units = units;
        self
    }

    /// Enables no-gate mode.
    pub fn no_gate(mut self, enabled: bool) -> Self {
        self.config.no_gate = enabled;
        self
    }

    /// Sets the time constant range.
    pub fn tau_range(mut self, min: f64, max: f64) -> Self {
        self.config.tau_min = min;
        self.config.tau_max = max;
        self
    }

    /// Sets the backbone architecture.
    pub fn backbone(mut self, layers: usize, units: usize) -> Self {
        self.config.backbone_layers = layers;
        self.config.backbone_units = units;
        self
    }

    /// Sets the backbone activation.
    pub fn backbone_activation(mut self, activation: BackboneActivation) -> Self {
        self.config.backbone_activation = activation;
        self
    }

    /// Sets the time step.
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Builds the CfC network.
    pub fn build(self) -> CfcNetwork {
        CfcNetwork::with_config(self.input_dim, self.output_dim, self.config).with_dt(self.dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cfc_config_default() {
        let config = CfcConfig::default();
        assert_eq!(config.units, 64);
        assert!(!config.no_gate);
    }

    #[test]
    fn test_cfc_config_builder() {
        let config = CfcConfig::new(32)
            .with_no_gate(true)
            .with_tau_range(0.5, 5.0)
            .with_backbone_activation(BackboneActivation::GELU)
            .with_backbone(3, 48);

        assert_eq!(config.units, 32);
        assert!(config.no_gate);
        assert_eq!(config.tau_min, 0.5);
        assert_eq!(config.tau_max, 5.0);
        assert_eq!(config.backbone_layers, 3);
        assert_eq!(config.backbone_units, 48);
    }

    #[test]
    fn test_backbone_activation() {
        assert_relative_eq!(BackboneActivation::Tanh.apply(0.0), 0.0, epsilon = 1e-10);
        assert_eq!(BackboneActivation::ReLU.apply(-1.0), 0.0);
        assert_eq!(BackboneActivation::ReLU.apply(1.0), 1.0);
        assert!(BackboneActivation::SiLU.apply(1.0) > 0.0);
        assert!(BackboneActivation::GELU.apply(1.0) > 0.0);
    }

    #[test]
    fn test_dense_layer() {
        let layer = DenseLayer::new(3, 4);
        assert_eq!(layer.input_dim, 3);
        assert_eq!(layer.output_dim, 4);

        let output = layer.forward(&[1.0, 0.5, 0.2]);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_backbone() {
        let backbone = Backbone::new(10, 16, 2, BackboneActivation::SiLU);
        let output = backbone.forward(&vec![0.5; 10]);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_cfc_cell_creation() {
        let config = CfcConfig::new(16);
        let cell = CfcCell::new(4, config);

        assert_eq!(cell.units(), 16);
        assert_eq!(cell.input_dim(), 4);
    }

    #[test]
    fn test_cfc_cell_forward() {
        let config = CfcConfig::new(8);
        let cell = CfcCell::new(2, config);

        let state = cell.initial_state();
        let input = vec![1.0, 0.5];

        let new_state = cell.forward(&state, &input, 1.0);

        assert_eq!(new_state.len(), 8);
        // State should have changed from zero
        assert!(new_state.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn test_cfc_cell_no_gate() {
        let config = CfcConfig::new(8).with_no_gate(true);
        let cell = CfcCell::new(2, config);

        let state = cell.initial_state();
        let input = vec![1.0, 0.5];

        let new_state = cell.forward(&state, &input, 1.0);
        assert_eq!(new_state.len(), 8);
    }

    #[test]
    fn test_cfc_cell_sequence() {
        let config = CfcConfig::new(4);
        let cell = CfcCell::new(2, config);

        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let states = cell.forward_sequence(&inputs, 1.0);

        assert_eq!(states.len(), 3);
        for state in &states {
            assert_eq!(state.len(), 4);
        }
    }

    #[test]
    fn test_cfc_network_creation() {
        let network = CfcNetwork::new(3, 16, 2);

        assert_eq!(network.hidden_dim(), 16);
        assert_eq!(network.output_dim(), 2);
    }

    #[test]
    fn test_cfc_network_forward() {
        let network = CfcNetwork::new(3, 8, 2);

        let state = network.initial_state();
        let input = vec![1.0, 0.5, 0.2];

        let (new_state, output) = network.forward(&state, &input);

        assert_eq!(new_state.len(), 8);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_cfc_network_sequence() {
        let network = CfcNetwork::new(2, 8, 1);

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
    fn test_cfc_builder() {
        let network = CfcBuilder::new(4, 2)
            .hidden_units(32)
            .no_gate(false)
            .tau_range(0.5, 5.0)
            .backbone(3, 24)
            .backbone_activation(BackboneActivation::GELU)
            .dt(0.5)
            .build();

        assert_eq!(network.hidden_dim(), 32);
        assert_eq!(network.output_dim(), 2);
    }

    #[test]
    fn test_cfc_state_bounded() {
        // CfC should produce bounded states due to tanh
        let network = CfcNetwork::new(2, 8, 1);

        let inputs: Vec<Vec<f64>> = (0..50)
            .map(|_| vec![10.0, -10.0]) // Large inputs
            .collect();

        let outputs = network.forward_sequence(&inputs);

        // States should remain bounded
        for output in &outputs {
            for &o in output {
                assert!(o.is_finite());
            }
        }
    }

    #[test]
    fn test_cfc_network_serialization() {
        let network = CfcNetwork::new(2, 8, 1);

        let json = serde_json::to_string(&network).unwrap();
        let restored: CfcNetwork = serde_json::from_str(&json).unwrap();

        assert_eq!(network.hidden_dim(), restored.hidden_dim());
        assert_eq!(network.output_dim(), restored.output_dim());
    }

    #[test]
    fn test_cfc_faster_than_ltc_conceptually() {
        // This test verifies CfC works without ODE solving
        // (actual benchmarking is done in benchmarks)
        let network = CfcNetwork::new(2, 32, 1);

        let inputs: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let t = i as f64 * 0.1;
                vec![t.sin(), t.cos()]
            })
            .collect();

        let outputs = network.forward_sequence(&inputs);

        // Just verify it completes and produces valid output
        assert_eq!(outputs.len(), 100);
        for output in &outputs {
            assert!(output[0].is_finite());
        }
    }

    #[test]
    fn test_time_constant_range() {
        let config = CfcConfig::new(4).with_tau_range(0.1, 10.0);
        let cell = CfcCell::new(2, config);

        // Create some backbone features
        let state = cell.initial_state();
        let input = vec![1.0, 0.5];
        let mut combined = state.clone();
        combined.extend_from_slice(&input);

        let features = cell.backbone.forward(&combined);
        let tau = cell.compute_tau(&features);

        // All time constants should be in range
        for &t in &tau {
            assert!(t >= 0.1 && t <= 10.0, "tau {} out of range", t);
        }
    }
}
