// src/cfc.rs
//! Closed-form Continuous-depth (CfC) neural network implementation.
//!
//! CfC networks are a variant of Liquid Time-Constant networks that replace numerical
//! ODE integration with a closed-form update, providing significant computational speedup
//! while retaining the expressive power of continuous-time models.
//!
//! Based on "Closed-form Continuous-Time Neural Networks" (Hasani et al., 2022).
//!
//! This module supports two update modes, selected by [`CfcConfig::no_gate`]:
//!
//! * **Gated (default, `no_gate = false`).** The reference CfC formulation. A shared
//!   backbone produces features `z = backbone([state, input])`, from which two candidate
//!   heads and a time-dependent interpolation gate are derived:
//!   ```text
//!   ff1      = tanh(W1 z + b1)
//!   ff2      = tanh(W2 z + b2)
//!   t_a      = Wa z + ba,   t_b = Wb z + bb
//!   t_interp = sigmoid(t_a * dt + t_b)                    (elementwise)
//!   x_new    = ff1 * (1 - t_interp) + t_interp * ff2      (elementwise)
//!   ```
//!   Because `x_new` is a convex combination of two `tanh` heads, it is bounded in
//!   `[-1, 1]`.
//!
//! * **Decay (`no_gate = true`).** The pure exponential-decay closed form, equivalent to
//!   an LTC neuron solved in closed form:
//!   ```text
//!   x(t+dt) = x(t) * exp(-dt/tau) + (1 - exp(-dt/tau)) * f(z)
//!   ```
//!   using an input-dependent time constant `tau` and `f(z) = tanh(W z + b)`.
//!
//! The decay-mode layers (`ff_projection`, `tau_layer`) and the gated-mode heads
//! (`ff1`, `ff2`, `time_a`, `time_b`) are all constructed unconditionally so that
//! serialization is mode-independent; only the active mode's layers are evaluated.

use crate::activation::{sigmoid, tanh};
use crate::error::{LnnError, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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
    /// Optional seed for reproducible random initialization. `Some(seed)` seeds a `StdRng`
    /// with `seed`; `None` seeds from entropy.
    #[serde(default)]
    pub seed: Option<u64>,
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
            seed: None,
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

    /// Sets the seed for reproducible random initialization.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    /// Returns [`LnnError::Configuration`] if `units < 1`, `tau_min <= 0`,
    /// `tau_min > tau_max`, or `backbone_layers < 1`.
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
        if self.backbone_layers < 1 {
            return Err(LnnError::configuration("backbone_layers must be >= 1"));
        }
        Ok(())
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
    /// Creates a dense layer with Xavier-initialised weights drawn from `rng`.
    ///
    /// The generator is borrowed (never created here) so that every layer of a network is
    /// derived from the single seeded generator owned by the network constructor.
    fn new(input_dim: usize, output_dim: usize, rng: &mut StdRng) -> Self {
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
            for (out, &w) in output.iter_mut().zip(self.weights[i].iter()) {
                *out += w * inp;
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
        rng: &mut StdRng,
    ) -> Self {
        let mut layers = Vec::new();

        // First layer
        layers.push(DenseLayer::new(input_dim, hidden_units, rng));

        // Hidden layers
        for _ in 1..num_layers {
            layers.push(DenseLayer::new(hidden_units, hidden_units, rng));
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
/// The cell computes backbone features `z` from the concatenated `[state, input]` vector
/// and then updates the state using one of two modes (see the [module docs](crate::cfc)):
///
/// * gated (`no_gate = false`): `x_new = ff1 * (1 - t_interp) + t_interp * ff2`,
/// * decay (`no_gate = true`): the exponential-decay closed form using an input-dependent
///   time constant.
///
/// Both modes' layers are constructed unconditionally so serialization is mode-independent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfcCell {
    /// Configuration
    pub config: CfcConfig,
    /// Backbone network
    backbone: Backbone,
    /// Decay-mode: projection from backbone features to the target state `f(z)`
    ff_projection: DenseLayer,
    /// Decay-mode: input-dependent time-constant logits
    tau_layer: DenseLayer,
    /// Gated-mode: first candidate head (pre-`tanh`)
    ff1: DenseLayer,
    /// Gated-mode: second candidate head (pre-`tanh`)
    ff2: DenseLayer,
    /// Gated-mode: time-gate slope term `t_a`
    time_a: DenseLayer,
    /// Gated-mode: time-gate offset term `t_b`
    time_b: DenseLayer,
    /// Input dimension
    input_dim: usize,
}

impl CfcCell {
    /// Creates a new CfC cell.
    ///
    /// All layers for both the gated and decay modes are constructed; only the layers of
    /// the mode selected by [`CfcConfig::no_gate`] are evaluated during [`CfcCell::forward`].
    ///
    /// This self-contained constructor creates its own [`StdRng`] from `config.seed` (or
    /// from entropy when unseeded). Full networks use [`CfcCell::with_rng`] so the cell and
    /// output projection share one generator.
    pub fn new(input_dim: usize, config: CfcConfig) -> Self {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self::with_rng(input_dim, config, &mut rng)
    }

    /// Creates a new CfC cell drawing all weights from the supplied generator.
    ///
    /// The layer construction order (backbone, then decay-mode layers, then the gated-mode
    /// heads) is fixed, so a given generator state produces a deterministic cell.
    pub fn with_rng(input_dim: usize, config: CfcConfig, rng: &mut StdRng) -> Self {
        let state_input_dim = config.units + input_dim;

        let backbone = Backbone::new(
            state_input_dim,
            config.backbone_units,
            config.backbone_layers,
            config.backbone_activation,
            rng,
        );

        let backbone_out_dim = backbone.output_dim();
        let units = config.units;

        // Decay-mode layers.
        let ff_projection = DenseLayer::new(backbone_out_dim, units, rng);
        let tau_layer = DenseLayer::new(backbone_out_dim, units, rng);

        // Gated-mode heads.
        let ff1 = DenseLayer::new(backbone_out_dim, units, rng);
        let ff2 = DenseLayer::new(backbone_out_dim, units, rng);
        let time_a = DenseLayer::new(backbone_out_dim, units, rng);
        let time_b = DenseLayer::new(backbone_out_dim, units, rng);

        CfcCell {
            config,
            backbone,
            ff_projection,
            tau_layer,
            ff1,
            ff2,
            time_a,
            time_b,
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

    /// Performs one forward pass.
    ///
    /// In gated mode (`no_gate = false`) computes
    /// `x_new = ff1 * (1 - t_interp) + t_interp * ff2`, where
    /// `t_interp = sigmoid(t_a * dt + t_b)` and `ff1`/`ff2` are `tanh` heads.
    ///
    /// In decay mode (`no_gate = true`) computes the exponential-decay closed form
    /// `x_new = x * exp(-dt/tau) + (1 - exp(-dt/tau)) * f(z)`.
    pub fn forward(&self, state: &[f64], input: &[f64], dt: f64) -> Vec<f64> {
        // Concatenate state and input for the backbone.
        let mut combined = state.to_vec();
        combined.extend_from_slice(input);

        // Compute backbone features z = backbone([state, input]).
        let features = self.backbone.forward(&combined);

        if self.config.no_gate {
            // Decay (pure) mode: closed-form exponential decay toward f(z) = tanh(W z + b).
            let tau = self.compute_tau(&features);
            let ff: Vec<f64> = self
                .ff_projection
                .forward(&features)
                .iter()
                .map(|&x| tanh(x))
                .collect();

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
            // Gated (default) mode: interpolate two tanh heads with a time-dependent gate.
            let ff1: Vec<f64> = self
                .ff1
                .forward(&features)
                .iter()
                .map(|&x| tanh(x))
                .collect();
            let ff2: Vec<f64> = self
                .ff2
                .forward(&features)
                .iter()
                .map(|&x| tanh(x))
                .collect();
            let t_a = self.time_a.forward(&features);
            let t_b = self.time_b.forward(&features);

            (0..self.config.units)
                .map(|i| {
                    let t_interp = sigmoid(t_a[i] * dt + t_b[i]);
                    ff1[i] * (1.0 - t_interp) + t_interp * ff2[i]
                })
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
    ///
    /// A single [`StdRng`] (seeded from `config.seed`, or from entropy when unseeded) is
    /// threaded through the cell and then the output projection, so a given seed yields a
    /// byte-identical network.
    pub fn with_config(input_dim: usize, output_dim: usize, config: CfcConfig) -> Self {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let hidden_units = config.units;
        let cell = CfcCell::with_rng(input_dim, config, &mut rng);
        let output_layer = DenseLayer::new(hidden_units, output_dim, &mut rng);

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

    /// Sets the seed for reproducible random initialization.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Builds the CfC network.
    ///
    /// # Panics
    /// Panics with the validation message if the configuration is invalid. Use
    /// [`try_build`](Self::try_build) for a non-panicking variant.
    pub fn build(self) -> CfcNetwork {
        self.try_build()
            .unwrap_or_else(|e| panic!("invalid CfcConfig: {e}"))
    }

    /// Builds the CfC network, returning an error instead of panicking on an invalid
    /// configuration.
    ///
    /// # Errors
    /// Returns the [`CfcConfig::validate`] error if the configuration is invalid.
    pub fn try_build(self) -> Result<CfcNetwork> {
        self.config.validate()?;
        Ok(CfcNetwork::with_config(self.input_dim, self.output_dim, self.config).with_dt(self.dt))
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
    fn test_cfc_config_validate() {
        assert!(CfcConfig::new(16).validate().is_ok());
        assert!(CfcConfig::new(0).validate().is_err());
        assert!(CfcConfig::new(16)
            .with_tau_range(0.0, 1.0)
            .validate()
            .is_err());
        assert!(CfcConfig::new(16)
            .with_tau_range(5.0, 1.0)
            .validate()
            .is_err());
        assert!(CfcConfig::new(16).with_backbone(0, 16).validate().is_err());
    }

    #[test]
    fn test_cfc_builder_try_build() {
        assert!(CfcBuilder::new(2, 1).hidden_units(8).try_build().is_ok());
        assert!(CfcBuilder::new(2, 1).backbone(0, 8).try_build().is_err());
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
        let mut rng = StdRng::seed_from_u64(0);
        let layer = DenseLayer::new(3, 4, &mut rng);
        assert_eq!(layer.input_dim, 3);
        assert_eq!(layer.output_dim, 4);

        let output = layer.forward(&[1.0, 0.5, 0.2]);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_backbone() {
        let mut rng = StdRng::seed_from_u64(0);
        let backbone = Backbone::new(10, 16, 2, BackboneActivation::SiLU, &mut rng);
        let output = backbone.forward(&[0.5; 10]);
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
            assert!((0.1..=10.0).contains(&t), "tau {} out of range", t);
        }
    }

    #[test]
    fn test_cfc_gated_vs_no_gate_differ() {
        // Build one gated cell, then clone it and flip `no_gate` so both modes share
        // identical weights (constructors are randomly initialized, so cloning is the
        // only way to guarantee a fair comparison). The gated and decay updates read
        // disjoint layers, so the sequences must differ.
        let gated = CfcCell::new(2, CfcConfig::new(8));
        assert!(!gated.config.no_gate);

        let mut ungated = gated.clone();
        ungated.config.no_gate = true;

        let inputs = vec![vec![1.0, -0.5], vec![0.3, 0.7], vec![-1.0, 0.2]];
        let gated_seq = gated.forward_sequence(&inputs, 1.0);
        let ungated_seq = ungated.forward_sequence(&inputs, 1.0);

        let differ = gated_seq
            .iter()
            .zip(ungated_seq.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .any(|(&x, &y)| (x - y).abs() > 1e-9);

        assert!(
            differ,
            "gated and no-gate modes should produce different sequences"
        );
    }

    #[test]
    fn test_cfc_gated_bounded() {
        // The gated update is a convex combination of two tanh heads, so every state
        // component must stay within [-1, 1] even for large, sustained inputs.
        let cell = CfcCell::new(2, CfcConfig::new(8));
        assert!(!cell.config.no_gate);

        let inputs: Vec<Vec<f64>> = (0..50).map(|_| vec![10.0, -10.0]).collect();
        let seq = cell.forward_sequence(&inputs, 1.0);

        for state in &seq {
            for &v in state {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "gated state component {} out of [-1, 1]",
                    v
                );
            }
        }
    }

    #[test]
    fn test_cfc_seed_reproducible() {
        // Same seed → byte-identical serialized network (one RNG threaded through the
        // backbone, both mode's layers, and the output projection).
        let a = CfcBuilder::new(2, 1).hidden_units(8).seed(99).build();
        let b = CfcBuilder::new(2, 1).hidden_units(8).seed(99).build();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
        );
    }

    #[test]
    fn test_cfc_seed_differs() {
        let a = CfcBuilder::new(2, 1).hidden_units(8).seed(99).build();
        let b = CfcBuilder::new(2, 1).hidden_units(8).seed(100).build();
        assert_ne!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
        );
    }

    #[test]
    fn test_cfc_unseeded_still_works() {
        let net = CfcNetwork::new(2, 8, 1);
        assert_eq!(net.hidden_dim(), 8);
        assert_eq!(net.output_dim(), 1);
    }

    #[test]
    fn test_cfc_cell_serialization() {
        // Round-trips must preserve every layer for both modes; JSON float encoding can
        // shift the final ULP, so compare within a tight tolerance.
        let cell = CfcCell::new(3, CfcConfig::new(8));

        let json = serde_json::to_string(&cell).unwrap();
        let restored: CfcCell = serde_json::from_str(&json).unwrap();

        assert_eq!(cell.units(), restored.units());
        assert_eq!(cell.input_dim(), restored.input_dim());

        let state = cell.initial_state();
        let input = vec![0.5, -0.2, 0.1];
        let before = cell.forward(&state, &input, 1.0);
        let after = restored.forward(&state, &input, 1.0);
        for (a, b) in before.iter().zip(after.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }
}
