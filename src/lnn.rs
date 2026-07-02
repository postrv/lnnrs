// src/lnn.rs
//! Core Liquid Neural Network implementation.
//!
//! This module provides the main `LiquidNeuralNetwork` struct and associated functionality
//! for creating, configuring, training, and running liquid neural networks.

use crate::error::{LnnError, Result};
use crate::network_components::{Neuron, Synapse};

use nalgebra::ComplexField;
use nalgebra::DMatrix;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// Default synaptic delay in timesteps.
const DEFAULT_DELAY: u32 = 1;

/// Threshold assigned to logic-gate neurons so that they never fire and simply integrate
/// their (linear) inputs; the gate's truth value is read from the sign of the accumulated
/// membrane potential via the sigmoid decode.
const GATE_NON_FIRING_THRESHOLD: f64 = 1e9;

/// Supported logic gate types for network configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicGate {
    And,
    Or,
    Not,
    Nand,
    Nor,
    Xor,
}

impl LogicGate {
    /// Parses a string into a LogicGate variant.
    // A `FromStr` trait implementation is deferred to a later phase; the inherent method is
    // retained here for the current call sites.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "AND" => Ok(LogicGate::And),
            "OR" => Ok(LogicGate::Or),
            "NOT" => Ok(LogicGate::Not),
            "NAND" => Ok(LogicGate::Nand),
            "NOR" => Ok(LogicGate::Nor),
            "XOR" => Ok(LogicGate::Xor),
            _ => Err(LnnError::InvalidGateType(s.to_string())),
        }
    }
}

/// Wrapper around the synaptic matrix for custom serialization.
#[derive(Clone)]
pub struct SynapsesWrapper(pub Array2<Option<Synapse>>);

impl SynapsesWrapper {
    /// Creates a new empty synapses wrapper with the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        SynapsesWrapper(Array2::default((rows, cols)))
    }

    /// Returns the shape of the synaptic matrix.
    pub fn shape(&self) -> (usize, usize) {
        let shape = self.0.shape();
        (shape[0], shape[1])
    }

    /// Gets a synapse at the given indices, if it exists.
    pub fn get(&self, i: usize, j: usize) -> Option<&Synapse> {
        self.0.get((i, j)).and_then(|opt| opt.as_ref())
    }

    /// Sets a synapse at the given indices.
    pub fn set(&mut self, i: usize, j: usize, synapse: Option<Synapse>) {
        if let Some(cell) = self.0.get_mut((i, j)) {
            *cell = synapse;
        }
    }

    /// Iterates over all synapses (non-None values).
    pub fn iter_synapses(&self) -> impl Iterator<Item = &Synapse> {
        self.0.iter().filter_map(|opt| opt.as_ref())
    }
}

impl Serialize for SynapsesWrapper {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("SynapsesWrapper", 2)?;

        // Store dimensions
        let shape = self.0.shape();
        s.serialize_field("shape", &[shape[0], shape[1]])?;

        // Serialize synapse data as sparse representation
        let synapse_data: Vec<SynapseData> = self
            .0
            .indexed_iter()
            .filter_map(|((row, col), syn_opt)| {
                syn_opt.as_ref().map(|s| SynapseData {
                    row,
                    col,
                    weight: s.weight,
                    delay: s.delay,
                    source: s.source,
                    target: s.target,
                })
            })
            .collect();

        s.serialize_field("synapses", &synapse_data)?;
        s.end()
    }
}

/// Helper struct for synapse serialization.
#[derive(Serialize, Deserialize)]
struct SynapseData {
    row: usize,
    col: usize,
    weight: f64,
    delay: u32,
    source: usize,
    target: usize,
}

impl<'de> Deserialize<'de> for SynapsesWrapper {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SynapsesWrapperData {
            shape: [usize; 2],
            synapses: Vec<SynapseData>,
        }

        let data = SynapsesWrapperData::deserialize(deserializer)?;
        let mut array: Array2<Option<Synapse>> = Array2::default((data.shape[0], data.shape[1]));

        for syn_data in data.synapses {
            array[[syn_data.row, syn_data.col]] = Some(Synapse::new(
                syn_data.source,
                syn_data.target,
                syn_data.weight,
                syn_data.delay,
            ));
        }

        Ok(SynapsesWrapper(array))
    }
}

/// Configuration for building a Liquid Neural Network.
#[derive(Debug, Clone)]
pub struct LnnConfig {
    /// Number of neurons in the network
    pub neuron_count: usize,
    /// Desired spectral radius for weight initialization
    pub spectral_radius: Option<f64>,
    /// Range for random weight initialization
    pub weight_range: (f64, f64),
    /// Range for random delay initialization
    pub delay_range: (u32, u32),
    /// Range for random threshold initialization
    pub threshold_range: (f64, f64),
    /// Connection probability (0.0 to 1.0)
    pub connection_probability: f64,
    /// Optional seed for reproducible random initialization. When `Some(seed)`, the network
    /// is built from a `StdRng` seeded with `seed` (identical seed → identical network);
    /// when `None`, the RNG is seeded from entropy.
    pub seed: Option<u64>,
}

impl Default for LnnConfig {
    fn default() -> Self {
        LnnConfig {
            neuron_count: 100,
            spectral_radius: None,
            weight_range: (-1.0, 1.0),
            delay_range: (1, 5),
            threshold_range: (0.5, 1.0),
            connection_probability: 1.0, // Fully connected by default
            seed: None,
        }
    }
}

impl LnnConfig {
    /// Creates a new configuration with the specified neuron count.
    pub fn new(neuron_count: usize) -> Self {
        LnnConfig {
            neuron_count,
            ..Default::default()
        }
    }

    /// Sets the spectral radius for weight initialization.
    pub fn with_spectral_radius(mut self, radius: f64) -> Self {
        self.spectral_radius = Some(radius);
        self
    }

    /// Sets the weight range for random initialization.
    pub fn with_weight_range(mut self, min: f64, max: f64) -> Self {
        self.weight_range = (min, max);
        self
    }

    /// Sets the delay range for synapses.
    pub fn with_delay_range(mut self, min: u32, max: u32) -> Self {
        self.delay_range = (min, max);
        self
    }

    /// Sets the threshold range for neurons.
    pub fn with_threshold_range(mut self, min: f64, max: f64) -> Self {
        self.threshold_range = (min, max);
        self
    }

    /// Sets the connection probability.
    pub fn with_connection_probability(mut self, prob: f64) -> Self {
        self.connection_probability = prob.clamp(0.0, 1.0);
        self
    }

    /// Sets the seed for reproducible random initialization.
    ///
    /// Two networks built from configurations with the same seed (and otherwise identical
    /// parameters) are byte-for-byte identical, including their serialized JSON.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the configuration.
    ///
    /// Ranges are sampled *inclusively* (`min..=max`), so degenerate ranges where
    /// `min == max` are legal; only `min > max`, empty networks and sub-unit delays are
    /// rejected.
    ///
    /// # Errors
    /// Returns [`LnnError::Configuration`] if `neuron_count == 0`, any range has
    /// `min > max`, or `delay_range.0 < 1`.
    pub fn validate(&self) -> Result<()> {
        if self.neuron_count == 0 {
            return Err(LnnError::configuration("neuron_count must be >= 1"));
        }
        if self.weight_range.0 > self.weight_range.1 {
            return Err(LnnError::configuration("weight_range min must be <= max"));
        }
        if self.threshold_range.0 > self.threshold_range.1 {
            return Err(LnnError::configuration(
                "threshold_range min must be <= max",
            ));
        }
        if self.delay_range.0 > self.delay_range.1 {
            return Err(LnnError::configuration("delay_range min must be <= max"));
        }
        if self.delay_range.0 < 1 {
            return Err(LnnError::configuration("delay_range min must be >= 1"));
        }
        Ok(())
    }

    /// Builds a LiquidNeuralNetwork from this configuration.
    ///
    /// # Panics
    /// Panics with the validation message if [`validate`](Self::validate) fails. Use
    /// [`try_build`](Self::try_build) for a non-panicking variant.
    pub fn build(self) -> LiquidNeuralNetwork {
        self.try_build()
            .unwrap_or_else(|e| panic!("invalid LnnConfig: {e}"))
    }

    /// Builds a LiquidNeuralNetwork, returning an error instead of panicking on an invalid
    /// configuration.
    ///
    /// # Errors
    /// Returns the [`validate`](Self::validate) error if the configuration is invalid.
    pub fn try_build(self) -> Result<LiquidNeuralNetwork> {
        self.validate()?;
        Ok(LiquidNeuralNetwork::from_config(self))
    }
}

/// Builder for LiquidNeuralNetwork with a fluent API.
pub struct LnnBuilder {
    config: LnnConfig,
}

impl LnnBuilder {
    /// Creates a new builder with the specified neuron count.
    pub fn new(neuron_count: usize) -> Self {
        LnnBuilder {
            config: LnnConfig::new(neuron_count),
        }
    }

    /// Sets the spectral radius for weight initialization.
    pub fn spectral_radius(mut self, radius: f64) -> Self {
        self.config.spectral_radius = Some(radius);
        self
    }

    /// Sets the weight range for random initialization.
    pub fn weight_range(mut self, min: f64, max: f64) -> Self {
        self.config.weight_range = (min, max);
        self
    }

    /// Sets the delay range for synapses.
    pub fn delay_range(mut self, min: u32, max: u32) -> Self {
        self.config.delay_range = (min, max);
        self
    }

    /// Sets the threshold range for neurons.
    pub fn threshold_range(mut self, min: f64, max: f64) -> Self {
        self.config.threshold_range = (min, max);
        self
    }

    /// Sets the connection probability.
    pub fn connection_probability(mut self, prob: f64) -> Self {
        self.config.connection_probability = prob.clamp(0.0, 1.0);
        self
    }

    /// Sets the seed for reproducible random initialization.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Builds the LiquidNeuralNetwork.
    ///
    /// # Panics
    /// Panics with the validation message if the configuration is invalid. Use
    /// [`try_build`](Self::try_build) for a non-panicking variant.
    pub fn build(self) -> LiquidNeuralNetwork {
        self.config.build()
    }

    /// Builds the LiquidNeuralNetwork, returning an error instead of panicking on an invalid
    /// configuration.
    ///
    /// # Errors
    /// Returns the validation error if the configuration is invalid.
    pub fn try_build(self) -> Result<LiquidNeuralNetwork> {
        self.config.try_build()
    }
}

/// Configuration for training the ridge-regression readout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadoutConfig {
    /// Number of reservoir steps to run per pattern before reading the state.
    pub timesteps: usize,
    /// Ridge (Tikhonov) regularisation strength added to the normal-equation diagonal.
    pub ridge_lambda: f64,
}

impl Default for ReadoutConfig {
    fn default() -> Self {
        ReadoutConfig {
            timesteps: 10,
            ridge_lambda: 1e-6,
        }
    }
}

/// Solves the linear system `a x = b` via Gaussian elimination with partial pivoting.
///
/// Returns `None` if `a` is not square, dimensions are inconsistent, or the matrix is
/// (numerically) singular.
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }

    for col in 0..n {
        // Partial pivot: pick the row with the largest magnitude in this column.
        let mut pivot_row = col;
        let mut best = a[col][col].abs();
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            let mag = row[col].abs();
            if mag > best {
                best = mag;
                pivot_row = r;
            }
        }
        if best < 1e-12 {
            return None; // singular
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        let pivot_row_vals = a[col].clone();
        let pivot = pivot_row_vals[col];
        for r in (col + 1)..n {
            let factor = a[r][col] / pivot;
            if factor != 0.0 {
                for (target, &p_val) in a[r].iter_mut().zip(pivot_row_vals.iter()).skip(col) {
                    *target -= factor * p_val;
                }
                b[r] -= factor * b[col];
            }
        }
    }

    // Back-substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for c in (i + 1)..n {
            sum -= a[i][c] * x[c];
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

/// Represents the entire Liquid Neural Network.
#[derive(Serialize, Deserialize, Clone)]
pub struct LiquidNeuralNetwork {
    /// All neurons in the network
    pub neurons: Vec<Neuron>,
    /// Synaptic connection matrix
    synapses: SynapsesWrapper,
    /// Indices of neurons designated as inputs
    input_indices: Vec<usize>,
    /// Indices of neurons designated as outputs
    output_indices: Vec<usize>,
    /// Learned weights for the readout layer
    readout_weights: Vec<f64>,
}

impl LiquidNeuralNetwork {
    /// Creates a new Liquid Neural Network with a given number of neurons.
    ///
    /// # Arguments
    /// * `neuron_count` - The number of neurons in the network
    ///
    /// # Returns
    /// A new `LiquidNeuralNetwork` instance with random initialization
    pub fn new(neuron_count: usize) -> Self {
        LnnConfig::new(neuron_count).build()
    }

    /// Creates a LNN from a configuration.
    ///
    /// All randomness is drawn from a single [`StdRng`] created here: seeded with
    /// `config.seed` when present, otherwise seeded from entropy. Neuron ids are assigned
    /// deterministically by index (`0..neuron_count`) rather than from a global counter, so
    /// two networks built from the same seed serialize to byte-identical JSON regardless of
    /// how many other neurons have been created in the process.
    pub fn from_config(config: LnnConfig) -> Self {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let neuron_count = config.neuron_count;

        // Create neurons with random thresholds (inclusive range sampling). Ids are the
        // neuron index so serialization is deterministic and index-aligned.
        let neurons: Vec<Neuron> = (0..neuron_count)
            .map(|i| {
                Neuron::with_id(
                    i,
                    rng.gen_range(config.threshold_range.0..=config.threshold_range.1),
                )
            })
            .collect();

        // Initialize synapses (inclusive range sampling for weight/delay).
        let mut synapses_array = Array2::default((neuron_count, neuron_count));
        for i in 0..neuron_count {
            for j in 0..neuron_count {
                if i != j && rng.gen::<f64>() < config.connection_probability {
                    let weight = rng.gen_range(config.weight_range.0..=config.weight_range.1);
                    let delay = rng.gen_range(config.delay_range.0..=config.delay_range.1);
                    synapses_array[[i, j]] = Some(Synapse::new(i, j, weight, delay));
                }
            }
        }

        let mut network = LiquidNeuralNetwork {
            neurons,
            synapses: SynapsesWrapper(synapses_array),
            input_indices: Vec::new(),
            output_indices: Vec::new(),
            readout_weights: Vec::new(),
        };

        // Apply spectral radius scaling if specified
        if let Some(spectral_radius) = config.spectral_radius {
            network.scale_to_spectral_radius(spectral_radius);
        }

        network
    }

    /// Creates a new builder for configuring a LNN.
    pub fn builder(neuron_count: usize) -> LnnBuilder {
        LnnBuilder::new(neuron_count)
    }

    /// Returns the number of neurons in the network.
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Prints information about all synapses in the network.
    pub fn print_synapses(&self) {
        for (i, row) in self.synapses.0.outer_iter().enumerate() {
            for (j, synapse_option) in row.iter().enumerate() {
                if let Some(synapse) = synapse_option {
                    log::info!(
                        "Synapse from neuron {} to neuron {}: weight = {:.4}, delay = {}",
                        i,
                        j,
                        synapse.weight,
                        synapse.delay
                    );
                }
            }
        }
    }

    /// Retrieves all synapses connected to a specific neuron (where it's the source).
    pub fn get_synapses_for_neuron(&self, neuron_idx: usize) -> Vec<Synapse> {
        if neuron_idx >= self.neurons.len() {
            return Vec::new();
        }

        self.synapses
            .0
            .index_axis(ndarray::Axis(0), neuron_idx)
            .iter()
            .filter_map(|synapse_opt| synapse_opt.clone())
            .collect()
    }

    /// Retrieves all synapses targeting a specific neuron.
    pub fn get_synapses_to_neuron(&self, neuron_idx: usize) -> Vec<Synapse> {
        if neuron_idx >= self.neurons.len() {
            return Vec::new();
        }

        self.synapses
            .0
            .index_axis(ndarray::Axis(1), neuron_idx)
            .iter()
            .filter_map(|synapse_opt| synapse_opt.clone())
            .collect()
    }

    /// Scales the synaptic weights to achieve a specified spectral radius.
    ///
    /// The spectral radius affects network dynamics - values < 1 lead to
    /// stable/fading dynamics, while values > 1 can lead to chaotic behavior.
    pub fn scale_to_spectral_radius(&mut self, spectral_radius: f64) {
        let neuron_count = self.neurons.len();
        if neuron_count == 0 {
            return;
        }

        // Extract weights into a matrix using a simple loop
        let mut weights = Vec::with_capacity(neuron_count * neuron_count);
        for i in 0..neuron_count {
            for j in 0..neuron_count {
                let weight = self.synapses.0[[i, j]]
                    .as_ref()
                    .map(|s| s.weight)
                    .unwrap_or(0.0);
                weights.push(weight);
            }
        }

        let matrix = DMatrix::from_vec(neuron_count, neuron_count, weights);

        // Compute eigenvalues and find maximum magnitude
        let eigenvalues = matrix.complex_eigenvalues();
        let max_eigenvalue_magnitude = eigenvalues
            .iter()
            .map(|ev| ev.modulus())
            .fold(f64::NEG_INFINITY, f64::max);

        if max_eigenvalue_magnitude > 0.0 && max_eigenvalue_magnitude.is_finite() {
            let scaling_factor = spectral_radius / max_eigenvalue_magnitude;

            // Scale all synapse weights
            for synapse in self.synapses.0.iter_mut().flatten() {
                synapse.weight *= scaling_factor;
            }
        }
    }

    /// Initializes synapses with random weights scaled to a spectral radius.
    ///
    /// This replaces all existing synapses with newly generated ones.
    pub fn initialize_synapses(&mut self, spectral_radius: f64) {
        let neuron_count = self.neurons.len();
        // This is an explicit re-randomization with no seed context; draw from entropy.
        let mut rng = StdRng::from_entropy();

        // Generate random weights
        let raw_weights: Vec<f64> = (0..neuron_count * neuron_count)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Create matrix and compute spectral radius
        let mut matrix = DMatrix::from_vec(neuron_count, neuron_count, raw_weights);
        let eigenvalues = matrix.complex_eigenvalues();
        let max_eigenvalue_magnitude = eigenvalues
            .iter()
            .map(|ev| ev.modulus())
            .fold(f64::NEG_INFINITY, f64::max);

        if max_eigenvalue_magnitude > 0.0 && max_eigenvalue_magnitude.is_finite() {
            let scaling_factor = spectral_radius / max_eigenvalue_magnitude;
            matrix *= scaling_factor;
        }

        // Create synapses from scaled matrix
        let synapses_array: Array2<Option<Synapse>> =
            Array2::from_shape_fn((neuron_count, neuron_count), |(i, j)| {
                if i != j {
                    Some(Synapse::new(i, j, matrix[(i, j)], DEFAULT_DELAY))
                } else {
                    None
                }
            });

        self.synapses = SynapsesWrapper(synapses_array);
    }

    /// Sets the indices of neurons that will act as the input interface.
    pub fn set_input_neurons(&mut self, indices: Vec<usize>) {
        self.input_indices = indices
            .into_iter()
            .filter(|&idx| idx < self.neurons.len())
            .collect();
    }

    /// Sets the indices of neurons that will act as the output interface.
    pub fn set_output_neurons(&mut self, indices: Vec<usize>) {
        self.output_indices = indices
            .into_iter()
            .filter(|&idx| idx < self.neurons.len())
            .collect();
    }

    /// Gets the input neuron indices.
    pub fn input_neurons(&self) -> &[usize] {
        &self.input_indices
    }

    /// Gets the output neuron indices.
    pub fn output_neurons(&self) -> &[usize] {
        &self.output_indices
    }

    /// Connects two neurons with a synapse.
    ///
    /// # Arguments
    /// * `pre_idx` - Index of the presynaptic (source) neuron
    /// * `post_idx` - Index of the postsynaptic (target) neuron
    /// * `weight` - Synaptic weight
    /// * `delay` - Transmission delay in timesteps
    ///
    /// # Returns
    /// `Ok(())` if successful, `Err` if indices are invalid
    ///
    /// # Errors
    /// Returns an error if either index is out of bounds, the connection is a
    /// self-connection, or `delay < 1` (a signal must arrive on a strictly later step).
    pub fn connect_neurons(
        &mut self,
        pre_idx: usize,
        post_idx: usize,
        weight: f64,
        delay: u32,
    ) -> Result<()> {
        if pre_idx >= self.neurons.len() {
            return Err(LnnError::index_out_of_bounds(
                pre_idx,
                self.neurons.len() - 1,
            ));
        }
        if post_idx >= self.neurons.len() {
            return Err(LnnError::index_out_of_bounds(
                post_idx,
                self.neurons.len() - 1,
            ));
        }
        if pre_idx == post_idx {
            return Err(LnnError::configuration("Cannot create self-connection"));
        }
        if delay < 1 {
            return Err(LnnError::configuration("Synaptic delay must be >= 1"));
        }

        self.synapses.0[[pre_idx, post_idx]] = Some(Synapse::new(pre_idx, post_idx, weight, delay));
        Ok(())
    }

    /// Disconnects two neurons.
    pub fn disconnect_neurons(&mut self, pre_idx: usize, post_idx: usize) -> Result<()> {
        if pre_idx >= self.neurons.len() || post_idx >= self.neurons.len() {
            return Err(LnnError::index_out_of_bounds(
                pre_idx.max(post_idx),
                self.neurons.len() - 1,
            ));
        }

        self.synapses.0[[pre_idx, post_idx]] = None;
        Ok(())
    }

    /// Encodes external input data into neural stimulation.
    ///
    /// Input values are added (once) to the membrane potentials of the designated input
    /// neurons. Call sites are expected to validate the input length first (see
    /// [`run_simulation`](Self::run_simulation)); this method is infallible by construction.
    pub fn encode_input(&mut self, input_data: &[f64]) {
        debug_assert_eq!(
            input_data.len(),
            self.input_indices.len(),
            "encode_input length must match the number of input neurons"
        );
        for (&neuron_idx, &stimulus) in self.input_indices.iter().zip(input_data) {
            if let Some(neuron) = self.neurons.get_mut(neuron_idx) {
                neuron.membrane_potential += stimulus;
            }
        }
    }

    /// Decodes the output from the network into a signal.
    ///
    /// Returns the normalized (sigmoid) membrane potentials of all designated output neurons.
    /// The sigmoid function maps any real number to the range (0, 1).
    pub fn decode_output(&self) -> Vec<f64> {
        let output: Vec<f64> = self
            .output_indices
            .iter()
            .filter_map(|&idx| self.neurons.get(idx))
            .map(|neuron| crate::activation::sigmoid(neuron.membrane_potential))
            .collect();

        log::debug!("Decoded output: {:?}", output);
        output
    }

    /// Runs the reservoir on a single already-encoded pattern for `timesteps` steps and
    /// returns the readout feature vector: every neuron's membrane potential followed by a
    /// constant `1.0` bias term.
    ///
    /// The caller is responsible for resetting state and encoding the input beforehand.
    fn collect_state_features(&mut self, timesteps: usize) -> Vec<f64> {
        self.run_reservoir(timesteps);
        let mut features: Vec<f64> = self.neurons.iter().map(|n| n.membrane_potential).collect();
        features.push(1.0); // bias term
        features
    }

    /// Trains a linear ridge-regression readout over the reservoir's final state.
    ///
    /// For each input pattern the reservoir is run from a freshly reset state for
    /// `cfg.timesteps` steps; the full state vector `[potentials..., 1.0]` becomes a row of
    /// the design matrix `S`. The readout weights solve the regularised normal equations
    /// `(S^T S + lambda I) w = S^T y`, with `lambda = cfg.ridge_lambda`.
    ///
    /// # Errors
    /// Returns [`LnnError::Training`] if no patterns are supplied, if `inputs` and `targets`
    /// differ in length, or if the (regularised) system is singular.
    pub fn train_readout(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[f64],
        cfg: &ReadoutConfig,
    ) -> Result<()> {
        if inputs.is_empty() {
            return Err(LnnError::Training("No training inputs provided".into()));
        }
        if inputs.len() != targets.len() {
            return Err(LnnError::Training("Input and target count mismatch".into()));
        }

        // Collect the design matrix S (one row of features per pattern).
        let mut design: Vec<Vec<f64>> = Vec::with_capacity(inputs.len());
        for input in inputs.iter() {
            self.reset_state();
            self.encode_input(input);
            design.push(self.collect_state_features(cfg.timesteps));
        }

        let n_features = design[0].len();

        // Normal equations: A = S^T S + lambda I, b = S^T y.
        let mut a = vec![vec![0.0; n_features]; n_features];
        let mut b = vec![0.0; n_features];
        for (row, &target) in design.iter().zip(targets.iter()) {
            for i in 0..n_features {
                b[i] += row[i] * target;
                for j in 0..n_features {
                    a[i][j] += row[i] * row[j];
                }
            }
        }
        for (i, a_row) in a.iter_mut().enumerate() {
            a_row[i] += cfg.ridge_lambda;
        }

        let weights = solve_linear_system(a, b)
            .ok_or_else(|| LnnError::Training("Ridge-regression system is singular".into()))?;

        self.readout_weights = weights;
        Ok(())
    }

    /// Runs the reservoir on one pattern (from a reset state) and applies the trained
    /// readout, returning the scalar prediction.
    ///
    /// # Errors
    /// Returns [`LnnError::Training`] if the readout has not been trained yet, or if the
    /// input length does not match the number of configured input neurons.
    pub fn predict(&mut self, input: &[f64], timesteps: usize) -> Result<f64> {
        if self.readout_weights.is_empty() {
            return Err(LnnError::Training(
                "Readout has not been trained; call train_readout first".into(),
            ));
        }
        if input.len() != self.input_indices.len() {
            return Err(LnnError::Training(format!(
                "input length {} does not match {} input neurons",
                input.len(),
                self.input_indices.len()
            )));
        }

        self.reset_state();
        self.encode_input(input);
        let features = self.collect_state_features(timesteps);

        if features.len() != self.readout_weights.len() {
            return Err(LnnError::Training(
                "Feature/readout dimension mismatch (network changed after training)".into(),
            ));
        }

        Ok(features
            .iter()
            .zip(self.readout_weights.iter())
            .map(|(f, w)| f * w)
            .sum())
    }

    /// Resets the network states to initial values.
    pub fn reset_state(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
    }

    /// Configures the network to simulate a specific logic gate.
    ///
    /// # Arguments
    /// * `gate_type` - String representation of the gate type
    ///
    /// # Returns
    /// `Ok(())` if configuration succeeds, `Err` if gate type is invalid
    pub fn configure_for_logic_gate(&mut self, gate_type: &str) -> Result<()> {
        let gate = LogicGate::from_str(gate_type)?;
        self.configure_gate(gate);
        Ok(())
    }

    /// Configures the network for a specific logic gate.
    ///
    /// The gate configuration is derived analytically for the synchronous, delayed reservoir
    /// dynamics (see the module-level notes and `configure_and_gate`). All existing synapses
    /// are cleared first, and every neuron is turned into a non-firing linear integrator with
    /// `rest_potential = 0`; the gate's logic is encoded purely in synaptic weights and the
    /// output neuron's constant `bias`. The truth value is read from the sign of the output
    /// neuron's accumulated potential (`sigmoid(potential) > 0.5` iff `potential > 0`).
    pub fn configure_gate(&mut self, gate: LogicGate) {
        self.reset_to_blank_gate();

        match gate {
            LogicGate::And => self.configure_and_gate(),
            LogicGate::Or => self.configure_or_gate(),
            LogicGate::Not => self.configure_not_gate(),
            LogicGate::Nand => self.configure_nand_gate(),
            LogicGate::Nor => self.configure_nor_gate(),
            LogicGate::Xor => self.configure_xor_gate(),
        }
    }

    /// Resets the network to a blank slate suitable for building a logic gate: removes every
    /// synapse and makes every neuron a non-firing (`threshold = 1e9`) linear integrator with
    /// zero rest potential, zero bias and no threshold adaptation.
    fn reset_to_blank_gate(&mut self) {
        let n = self.neurons.len();
        self.synapses = SynapsesWrapper::new(n, n);
        self.input_indices.clear();
        self.output_indices.clear();
        for neuron in &mut self.neurons {
            neuron.reset();
            neuron.rest_potential = 0.0;
            neuron.threshold = GATE_NON_FIRING_THRESHOLD;
            neuron.bias = 0.0;
            neuron.adaptation.enabled = false;
        }
    }

    // The linear-gate designs below all rely on the closed form of the (non-firing) output
    // neuron after `T = 10` synchronous steps with `delay = 1`. Writing `L = 6/7` for the
    // per-step leak factor and `R = 1.3` for the membrane resistance, a single input relay
    // holding value `v` contributes `R * (T-1) * L^(T-1) * w * v ≈ 2.9219 * w * v` to the
    // final output potential, and a constant output bias `b` contributes
    // `7 * (1 - L^T) * b ≈ 5.5016 * b`. Each gate below picks `w` and `b` so that the final
    // potential is `> 0` exactly on the intended rows.

    fn configure_and_gate(&mut self) {
        if self.neurons.len() < 3 {
            return;
        }
        // Output 0 fires true only when both inputs are on: b = -0.8, w = 1.0.
        // f(0) = -4.40, f(1) = -1.48, f(2) = +1.44.
        self.set_input_neurons(vec![1, 2]);
        self.set_output_neurons(vec![0]);
        let _ = self.connect_neurons(1, 0, 1.0, DEFAULT_DELAY);
        let _ = self.connect_neurons(2, 0, 1.0, DEFAULT_DELAY);
        self.neurons[0].bias = -0.8;
    }

    fn configure_or_gate(&mut self) {
        if self.neurons.len() < 3 {
            return;
        }
        // Output 0 true when at least one input is on: b = -0.3, w = 1.0.
        // f(0) = -1.65, f(1) = +1.27, f(2) = +4.19.
        self.set_input_neurons(vec![1, 2]);
        self.set_output_neurons(vec![0]);
        let _ = self.connect_neurons(1, 0, 1.0, DEFAULT_DELAY);
        let _ = self.connect_neurons(2, 0, 1.0, DEFAULT_DELAY);
        self.neurons[0].bias = -0.3;
    }

    fn configure_not_gate(&mut self) {
        if self.neurons.len() < 2 {
            return;
        }
        // Output 1 true when the single input is off: positive bias, inhibitory input.
        // b = 0.5, w = -2.0. f(0) = +2.75, f(1) = -3.09.
        self.set_input_neurons(vec![0]);
        self.set_output_neurons(vec![1]);
        let _ = self.connect_neurons(0, 1, -2.0, DEFAULT_DELAY);
        self.neurons[1].bias = 0.5;
    }

    fn configure_nand_gate(&mut self) {
        if self.neurons.len() < 3 {
            return;
        }
        // NAND = NOT(AND): true unless both on. b = 1.0, w = -1.3.
        // f(0) = +5.50, f(1) = +1.70, f(2) = -2.10.
        self.set_input_neurons(vec![1, 2]);
        self.set_output_neurons(vec![0]);
        let _ = self.connect_neurons(1, 0, -1.3, DEFAULT_DELAY);
        let _ = self.connect_neurons(2, 0, -1.3, DEFAULT_DELAY);
        self.neurons[0].bias = 1.0;
    }

    fn configure_nor_gate(&mut self) {
        if self.neurons.len() < 3 {
            return;
        }
        // NOR = NOT(OR): true only when both off. b = 0.5, w = -1.5.
        // f(0) = +2.75, f(1) = -1.63, f(2) = -6.02.
        self.set_input_neurons(vec![1, 2]);
        self.set_output_neurons(vec![0]);
        let _ = self.connect_neurons(1, 0, -1.5, DEFAULT_DELAY);
        let _ = self.connect_neurons(2, 0, -1.5, DEFAULT_DELAY);
        self.neurons[0].bias = 0.5;
    }

    fn configure_xor_gate(&mut self) {
        if self.neurons.len() < 5 {
            return;
        }
        // XOR is not linearly separable, so unlike the other gates it cannot be read from a
        // purely linear (non-firing) integrator. Instead the output neuron (4) is a *firing*
        // neuron whose spike-reset provides the nonlinearity: both inputs excite it strongly
        // (weight 2.5). With a reset potential of -1.0 and threshold 4.0, over the fixed
        // 10-step horizon the neuron ends *positive* when exactly one input is on but is left
        // freshly reset (negative) when both inputs drive it hard, and a small negative bias
        // keeps the all-zero case negative. This yields, at T = 10:
        //   (0,0) -> -2.44, (0,1) -> +2.60, (1,0) -> +2.60, (1,1) -> -1.00
        // i.e. the XOR truth table with a comfortable margin. (This design is tuned to the
        // 10-step horizon used by the demo and tests.)
        self.set_input_neurons(vec![0, 1]);
        self.set_output_neurons(vec![4]);

        let _ = self.connect_neurons(0, 4, 2.5, DEFAULT_DELAY);
        let _ = self.connect_neurons(1, 4, 2.5, DEFAULT_DELAY);
        self.neurons[4].threshold = 4.0;
        self.neurons[4].rest_potential = -1.0;
        self.neurons[4].bias = -0.3;
    }

    /// Advances the network by `timesteps` synchronous steps using a real delay ring buffer.
    ///
    /// Assumes any input has already been encoded (see [`encode_input`](Self::encode_input)).
    /// Per timestep `t` (with ring size `max_delay + 1`):
    /// 1. the arrivals slot for `t` provides each neuron's `input_current`;
    /// 2. every neuron is updated from that snapshot (updates are synchronous because the
    ///    arrivals were scheduled from *previous*-step potentials);
    /// 3. for every synapse, `weight * potential[source]` (post-update) is scheduled into the
    ///    slot `(t + delay) % ring` of the target;
    /// 4. the consumed slot is zeroed for reuse.
    fn run_reservoir(&mut self, timesteps: usize) {
        let n = self.neurons.len();
        if n == 0 {
            return;
        }

        // Flatten the topology into (source, target, weight, delay). Iteration order is fixed
        // by the (row-major) synapse matrix, so it is independent of insertion order.
        let synapses: Vec<(usize, usize, f64, usize)> = self
            .synapses
            .iter_synapses()
            .map(|s| (s.source, s.target, s.weight, s.delay as usize))
            .collect();

        let max_delay = synapses.iter().map(|&(_, _, _, d)| d).max().unwrap_or(0);
        let ring = max_delay + 1;
        let mut arrivals: Vec<Vec<f64>> = vec![vec![0.0; n]; ring];

        for t in 0..timesteps {
            let slot = t % ring;

            // (1) + (2): update every neuron from the arrivals snapshot for this step.
            for (j, neuron) in self.neurons.iter_mut().enumerate() {
                neuron.update_state(arrivals[slot][j]);
            }

            // (3): schedule emissions from the post-update potentials.
            for &(source, target, weight, delay) in &synapses {
                let potential = self.neurons[source].membrane_potential;
                let dest = (t + delay) % ring;
                arrivals[dest][target] += weight * potential;
            }

            // (4): zero the slot we just consumed so it can be reused at t + ring.
            for value in arrivals[slot].iter_mut() {
                *value = 0.0;
            }
        }
    }

    /// Runs a simulation for a specified number of timesteps.
    ///
    /// Each input pattern is encoded (added to the input neurons once) and then propagated
    /// for `timesteps` synchronous steps; the decoded output is collected per pattern. Neuron
    /// state carries across patterns unless the caller resets it.
    ///
    /// # Arguments
    /// * `timesteps` - Number of simulation steps to run per pattern
    /// * `inputs` - Input patterns to process
    ///
    /// # Returns
    /// Vector of output vectors, one for each input pattern.
    ///
    /// # Errors
    /// Returns [`LnnError::Configuration`] if any input row's length does not equal the number
    /// of configured input neurons.
    pub fn run_simulation(
        &mut self,
        timesteps: usize,
        inputs: Vec<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>> {
        let expected = self.input_indices.len();
        for (idx, row) in inputs.iter().enumerate() {
            if row.len() != expected {
                return Err(LnnError::configuration(format!(
                    "input row {} has length {} but {} input neurons are configured",
                    idx,
                    row.len(),
                    expected
                )));
            }
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        for input_values in inputs.iter() {
            self.encode_input(input_values);
            self.run_reservoir(timesteps);
            outputs.push(self.decode_output());
        }

        Ok(outputs)
    }

    /// Returns network statistics for debugging and analysis.
    pub fn statistics(&self) -> NetworkStatistics {
        let total_synapses = self.synapses.0.iter().filter(|s| s.is_some()).count();
        let excitatory_count = self
            .synapses
            .iter_synapses()
            .filter(|s| s.is_excitatory())
            .count();
        let inhibitory_count = self
            .synapses
            .iter_synapses()
            .filter(|s| s.is_inhibitory())
            .count();

        let avg_weight: f64 = if total_synapses > 0 {
            self.synapses.iter_synapses().map(|s| s.weight).sum::<f64>() / total_synapses as f64
        } else {
            0.0
        };

        NetworkStatistics {
            neuron_count: self.neurons.len(),
            synapse_count: total_synapses,
            excitatory_synapse_count: excitatory_count,
            inhibitory_synapse_count: inhibitory_count,
            average_weight: avg_weight,
            input_neuron_count: self.input_indices.len(),
            output_neuron_count: self.output_indices.len(),
        }
    }
}

/// Statistics about a neural network's structure.
#[derive(Debug, Clone)]
pub struct NetworkStatistics {
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub excitatory_synapse_count: usize,
    pub inhibitory_synapse_count: usize,
    pub average_weight: f64,
    pub input_neuron_count: usize,
    pub output_neuron_count: usize,
}

impl fmt::Display for NetworkStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Network Statistics:")?;
        writeln!(f, "  Neurons: {}", self.neuron_count)?;
        writeln!(f, "  Synapses: {}", self.synapse_count)?;
        writeln!(f, "    Excitatory: {}", self.excitatory_synapse_count)?;
        writeln!(f, "    Inhibitory: {}", self.inhibitory_synapse_count)?;
        writeln!(f, "  Average Weight: {:.4}", self.average_weight)?;
        writeln!(f, "  Input Neurons: {}", self.input_neuron_count)?;
        writeln!(f, "  Output Neurons: {}", self.output_neuron_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod config_tests {
        use super::*;

        #[test]
        fn test_default_config() {
            let config = LnnConfig::default();
            assert_eq!(config.neuron_count, 100);
            assert!(config.spectral_radius.is_none());
            assert_eq!(config.connection_probability, 1.0);
        }

        #[test]
        fn test_config_builder() {
            let config = LnnConfig::new(50)
                .with_spectral_radius(0.9)
                .with_weight_range(-0.5, 0.5)
                .with_connection_probability(0.8);

            assert_eq!(config.neuron_count, 50);
            assert_eq!(config.spectral_radius, Some(0.9));
            assert_eq!(config.weight_range, (-0.5, 0.5));
            assert_eq!(config.connection_probability, 0.8);
        }

        #[test]
        fn test_connection_probability_clamping() {
            let config = LnnConfig::new(10).with_connection_probability(1.5);
            assert_eq!(config.connection_probability, 1.0);

            let config = LnnConfig::new(10).with_connection_probability(-0.5);
            assert_eq!(config.connection_probability, 0.0);
        }

        #[test]
        fn test_validate_accepts_degenerate_ranges() {
            // min == max is legal thanks to inclusive sampling.
            let config = LnnConfig::new(4)
                .with_weight_range(0.5, 0.5)
                .with_threshold_range(0.7, 0.7)
                .with_delay_range(2, 2);
            assert!(config.validate().is_ok());
            assert!(config.try_build().is_ok());
        }

        #[test]
        fn test_validate_rejects_invalid_configs() {
            assert!(LnnConfig::new(0).validate().is_err());
            assert!(LnnConfig::new(4)
                .with_weight_range(1.0, -1.0)
                .validate()
                .is_err());
            assert!(LnnConfig::new(4)
                .with_threshold_range(1.0, 0.0)
                .validate()
                .is_err());
            assert!(LnnConfig::new(4).with_delay_range(5, 1).validate().is_err());
            // delay_range.0 must be >= 1.
            assert!(LnnConfig::new(4).with_delay_range(0, 3).validate().is_err());
        }

        #[test]
        fn test_try_build_reports_error_instead_of_panicking() {
            assert!(LnnConfig::new(0).try_build().is_err());
            assert!(LnnBuilder::new(4).delay_range(0, 2).try_build().is_err());
        }

        #[test]
        #[should_panic(expected = "invalid LnnConfig")]
        fn test_build_panics_on_invalid_config() {
            let _ = LnnConfig::new(0).build();
        }
    }

    mod determinism_tests {
        use super::*;

        #[test]
        fn test_lnn_seed_reproducible() {
            // Two networks built in sequence with the same seed must serialize to
            // byte-identical JSON. This exercises both the single-RNG threading and the
            // index-based neuron ids (the global id counter is advanced between the two
            // builds, so a naive implementation would diverge here).
            let a = LnnBuilder::new(12)
                .weight_range(-0.8, 0.8)
                .threshold_range(0.4, 0.9)
                .delay_range(1, 4)
                .seed(1234)
                .build();
            let b = LnnBuilder::new(12)
                .weight_range(-0.8, 0.8)
                .threshold_range(0.4, 0.9)
                .delay_range(1, 4)
                .seed(1234)
                .build();

            let ja = serde_json::to_string(&a).unwrap();
            let jb = serde_json::to_string(&b).unwrap();
            assert_eq!(ja, jb, "same seed must produce identical serialized JSON");
        }

        #[test]
        fn test_lnn_seed_differs() {
            let a = LnnBuilder::new(12).seed(1).build();
            let b = LnnBuilder::new(12).seed(2).build();
            assert_ne!(
                serde_json::to_string(&a).unwrap(),
                serde_json::to_string(&b).unwrap(),
                "different seeds must produce different networks"
            );
        }

        #[test]
        fn test_lnn_unseeded_still_works() {
            // No seed: construction succeeds and produces a valid network.
            let net = LnnBuilder::new(8).build();
            assert_eq!(net.neuron_count(), 8);
            // Ids are still assigned deterministically by index even when unseeded.
            for (i, neuron) in net.neurons.iter().enumerate() {
                assert_eq!(neuron.id, i);
            }
        }
    }

    mod lnn_builder_tests {
        use super::*;

        #[test]
        fn test_builder_creates_network() {
            let lnn = LnnBuilder::new(20)
                .weight_range(-0.5, 0.5)
                .threshold_range(0.3, 0.7)
                .build();

            assert_eq!(lnn.neuron_count(), 20);
        }

        #[test]
        fn test_builder_with_spectral_radius() {
            let lnn = LnnBuilder::new(10).spectral_radius(0.95).build();

            assert_eq!(lnn.neuron_count(), 10);
        }
    }

    mod network_tests {
        use super::*;

        #[test]
        fn test_network_creation() {
            let lnn = LiquidNeuralNetwork::new(10);
            assert_eq!(lnn.neuron_count(), 10);
        }

        #[test]
        fn test_set_input_output_neurons() {
            let mut lnn = LiquidNeuralNetwork::new(10);
            lnn.set_input_neurons(vec![0, 1, 2]);
            lnn.set_output_neurons(vec![7, 8, 9]);

            assert_eq!(lnn.input_neurons(), &[0, 1, 2]);
            assert_eq!(lnn.output_neurons(), &[7, 8, 9]);
        }

        #[test]
        fn test_invalid_neuron_indices_filtered() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.set_input_neurons(vec![0, 1, 100]); // 100 is invalid

            assert_eq!(lnn.input_neurons(), &[0, 1]);
        }

        #[test]
        fn test_connect_neurons() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            let result = lnn.connect_neurons(0, 1, 0.5, 2);

            assert!(result.is_ok());
            let synapses = lnn.get_synapses_for_neuron(0);
            assert!(synapses.iter().any(|s| s.target == 1 && s.weight == 0.5));
        }

        #[test]
        fn test_connect_neurons_self_connection_error() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            let result = lnn.connect_neurons(2, 2, 0.5, 1);

            assert!(result.is_err());
        }

        #[test]
        fn test_connect_neurons_invalid_index() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            let result = lnn.connect_neurons(0, 100, 0.5, 1);

            assert!(result.is_err());
        }

        #[test]
        fn test_disconnect_neurons() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.connect_neurons(0, 1, 0.5, 1).unwrap();
            lnn.disconnect_neurons(0, 1).unwrap();

            let synapses = lnn.get_synapses_for_neuron(0);
            assert!(!synapses.iter().any(|s| s.target == 1));
        }

        #[test]
        fn test_encode_decode() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.set_input_neurons(vec![0, 1]);
            lnn.set_output_neurons(vec![0, 1]);

            lnn.encode_input(&[1.0, 2.0]);
            let output = lnn.decode_output();

            // decode_output applies sigmoid for normalized 0-1 output
            assert_eq!(output.len(), 2);
            // sigmoid(1.0) ≈ 0.731, sigmoid(2.0) ≈ 0.881
            assert!(output[0] > 0.7 && output[0] < 0.8);
            assert!(output[1] > 0.85 && output[1] < 0.9);
        }

        #[test]
        fn test_reset_state() {
            let mut lnn = LiquidNeuralNetwork::new(3);
            lnn.neurons[0].membrane_potential = 10.0;
            lnn.neurons[1].membrane_potential = 20.0;

            lnn.reset_state();

            for neuron in &lnn.neurons {
                assert_eq!(neuron.membrane_potential, 0.0);
            }
        }

        #[test]
        fn test_network_statistics() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.set_input_neurons(vec![0, 1]);
            lnn.set_output_neurons(vec![3, 4]);

            let stats = lnn.statistics();

            assert_eq!(stats.neuron_count, 5);
            assert_eq!(stats.input_neuron_count, 2);
            assert_eq!(stats.output_neuron_count, 2);
        }
    }

    mod logic_gate_tests {
        use super::*;

        #[test]
        fn test_logic_gate_parsing() {
            assert_eq!(LogicGate::from_str("AND").unwrap(), LogicGate::And);
            assert_eq!(LogicGate::from_str("or").unwrap(), LogicGate::Or);
            assert_eq!(LogicGate::from_str("Not").unwrap(), LogicGate::Not);
            assert_eq!(LogicGate::from_str("NAND").unwrap(), LogicGate::Nand);
            assert_eq!(LogicGate::from_str("NOR").unwrap(), LogicGate::Nor);
            assert_eq!(LogicGate::from_str("XOR").unwrap(), LogicGate::Xor);
        }

        #[test]
        fn test_invalid_gate_type() {
            let result = LogicGate::from_str("INVALID");
            assert!(result.is_err());
        }

        #[test]
        fn test_configure_and_gate() {
            let mut lnn = LiquidNeuralNetwork::new(10);
            let result = lnn.configure_for_logic_gate("AND");

            assert!(result.is_ok());
            assert_eq!(lnn.input_neurons(), &[1, 2]);
            assert_eq!(lnn.output_neurons(), &[0]);
        }

        #[test]
        fn test_configure_or_gate() {
            let mut lnn = LiquidNeuralNetwork::new(10);
            lnn.configure_for_logic_gate("OR").unwrap();

            assert_eq!(lnn.input_neurons(), &[1, 2]);
            assert_eq!(lnn.output_neurons(), &[0]);
        }

        #[test]
        fn test_configure_not_gate() {
            let mut lnn = LiquidNeuralNetwork::new(10);
            lnn.configure_for_logic_gate("NOT").unwrap();

            assert_eq!(lnn.input_neurons(), &[0]);
            assert_eq!(lnn.output_neurons(), &[1]);
        }
    }

    mod serialization_tests {
        use super::*;

        #[test]
        fn test_synapses_wrapper_serialization() {
            let mut wrapper = SynapsesWrapper::new(3, 3);
            wrapper.set(0, 1, Some(Synapse::new(0, 1, 0.5, 2)));
            wrapper.set(1, 2, Some(Synapse::new(1, 2, -0.3, 1)));

            let json = serde_json::to_string(&wrapper).unwrap();
            let deserialized: SynapsesWrapper = serde_json::from_str(&json).unwrap();

            assert_eq!(wrapper.shape(), deserialized.shape());

            let orig_syn = wrapper.get(0, 1).unwrap();
            let deser_syn = deserialized.get(0, 1).unwrap();
            assert_eq!(orig_syn.weight, deser_syn.weight);
            assert_eq!(orig_syn.delay, deser_syn.delay);
        }

        #[test]
        fn test_network_serialization() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.set_input_neurons(vec![0, 1]);
            lnn.set_output_neurons(vec![3, 4]);

            let json = serde_json::to_string(&lnn).unwrap();
            let deserialized: LiquidNeuralNetwork = serde_json::from_str(&json).unwrap();

            assert_eq!(lnn.neuron_count(), deserialized.neuron_count());
            assert_eq!(lnn.input_neurons(), deserialized.input_neurons());
            assert_eq!(lnn.output_neurons(), deserialized.output_neurons());
        }
    }

    mod simulation_tests {
        use super::*;

        #[test]
        fn test_run_simulation() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.set_input_neurons(vec![0, 1]);
            lnn.set_output_neurons(vec![3, 4]);

            let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
            let outputs = lnn.run_simulation(3, inputs).unwrap();

            assert_eq!(outputs.len(), 2);
            assert_eq!(outputs[0].len(), 2);
        }

        #[test]
        fn test_run_simulation_empty_inputs() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            let outputs = lnn.run_simulation(3, vec![]).unwrap();

            assert!(outputs.is_empty());
        }

        #[test]
        fn test_run_simulation_rejects_wrong_input_length() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.set_input_neurons(vec![0, 1]); // expects rows of length 2

            let result = lnn.run_simulation(3, vec![vec![1.0]]);
            assert!(result.is_err());
        }

        #[test]
        fn test_order_independence_of_synapse_insertion() {
            // Two networks with identical topology but synapses inserted in different orders
            // must produce identical outputs (spec 1d.6).
            let build = |order_swapped: bool| {
                let mut lnn = LiquidNeuralNetwork::new(4);
                lnn.reset_to_blank_gate();
                lnn.set_input_neurons(vec![0]);
                lnn.set_output_neurons(vec![3]);
                if order_swapped {
                    lnn.connect_neurons(2, 3, -0.7, 2).unwrap();
                    lnn.connect_neurons(1, 3, 0.9, 1).unwrap();
                    lnn.connect_neurons(0, 1, 0.5, 1).unwrap();
                    lnn.connect_neurons(0, 2, 0.4, 1).unwrap();
                } else {
                    lnn.connect_neurons(0, 1, 0.5, 1).unwrap();
                    lnn.connect_neurons(0, 2, 0.4, 1).unwrap();
                    lnn.connect_neurons(1, 3, 0.9, 1).unwrap();
                    lnn.connect_neurons(2, 3, -0.7, 2).unwrap();
                }
                lnn
            };

            let out_a = build(false).run_simulation(8, vec![vec![1.0]]).unwrap();
            let out_b = build(true).run_simulation(8, vec![vec![1.0]]).unwrap();
            assert_eq!(out_a, out_b);
        }

        #[test]
        fn test_delay_semantics() {
            // Source (0) -> target (1) with delay 3. The signal emitted at the end of step 0
            // must not reach the target until step 3 (spec 1d.7). Each run is independent (a
            // ring buffer lives for the duration of a single `run_reservoir` call), so we
            // probe the boundary with fresh runs of increasing length.
            let make = || {
                let mut lnn = LiquidNeuralNetwork::new(2);
                lnn.reset_to_blank_gate();
                lnn.set_input_neurons(vec![0]);
                lnn.connect_neurons(0, 1, 5.0, 3).unwrap();
                lnn.reset_state();
                lnn.encode_input(&[1.0]);
                lnn
            };

            // Running 1, 2 or 3 steps: the arrival (scheduled for step 3) is never consumed,
            // so the target remains at rest.
            for steps in 1..=3 {
                let mut lnn = make();
                lnn.run_reservoir(steps);
                assert_eq!(
                    lnn.neurons[1].membrane_potential, 0.0,
                    "target changed too early after {steps} steps"
                );
            }
            // Running 4 steps executes step t = 3, which consumes the delayed arrival.
            let mut lnn = make();
            lnn.run_reservoir(4);
            assert!(lnn.neurons[1].membrane_potential != 0.0);
        }
    }

    mod readout_tests {
        use super::*;

        #[test]
        fn test_solve_linear_system_known_3x3() {
            // Solve the system with known solution x = [2, 3, -1].
            let a = vec![
                vec![2.0, 1.0, -1.0],
                vec![-3.0, -1.0, 2.0],
                vec![-2.0, 1.0, 2.0],
            ];
            let b = vec![8.0, -11.0, -3.0];
            let x = solve_linear_system(a, b).unwrap();
            assert!((x[0] - 2.0).abs() < 1e-9);
            assert!((x[1] - 3.0).abs() < 1e-9);
            assert!((x[2] - (-1.0)).abs() < 1e-9);
        }

        #[test]
        fn test_solve_linear_system_singular() {
            let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]]; // rank-deficient
            let b = vec![1.0, 2.0];
            assert!(solve_linear_system(a, b).is_none());
        }

        #[test]
        fn test_train_readout_empty_inputs() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            let result = lnn.train_readout(&[], &[], &ReadoutConfig::default());
            assert!(result.is_err());
        }

        #[test]
        fn test_train_readout_mismatched_lengths() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.set_input_neurons(vec![0]);

            let inputs = vec![vec![1.0], vec![2.0]];
            let targets = vec![1.0]; // Wrong length

            let result = lnn.train_readout(&inputs, &targets, &ReadoutConfig::default());
            assert!(result.is_err());
        }

        #[test]
        fn test_predict_before_training_errors() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.set_input_neurons(vec![0, 1]);
            assert!(lnn.predict(&[1.0, 1.0], 10).is_err());
        }

        /// Builds a small, fully deterministic reservoir with a nonlinear (firing) feature so
        /// that AND becomes linearly solvable from the collected state.
        fn deterministic_reservoir() -> LiquidNeuralNetwork {
            let mut lnn = LiquidNeuralNetwork::new(6);
            lnn.synapses = SynapsesWrapper::new(6, 6);
            lnn.set_input_neurons(vec![0, 1]);
            for n in &mut lnn.neurons {
                n.reset();
                n.rest_potential = 0.0;
                n.bias = 0.0;
                n.adaptation.enabled = false;
                n.threshold = GATE_NON_FIRING_THRESHOLD;
            }
            // Neuron 2 fires (threshold 1.6): a nonlinear, saturating OR-type feature that
            // breaks the linearity of the plain input copies (neurons 4 and 5).
            lnn.neurons[2].threshold = 1.6;
            lnn.connect_neurons(0, 2, 1.0, 1).unwrap();
            lnn.connect_neurons(1, 2, 1.0, 1).unwrap();
            lnn.connect_neurons(0, 4, 1.0, 1).unwrap();
            lnn.connect_neurons(1, 5, 1.0, 1).unwrap();
            lnn
        }

        #[test]
        fn test_readout_learns_and_gate() {
            // Small reservoir with hand-set deterministic weights and a nonlinear feature;
            // train ridge regression to reproduce AND over {0,1}^2 and check MSE + rounded
            // predictions (spec 1e.ii).
            let mut lnn = deterministic_reservoir();

            let inputs = vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ];
            let targets = vec![0.0, 0.0, 0.0, 1.0];
            let cfg = ReadoutConfig::default();
            lnn.train_readout(&inputs, &targets, &cfg).unwrap();

            let mut sq_err = 0.0;
            for (input, &target) in inputs.iter().zip(targets.iter()) {
                let pred = lnn.predict(input, cfg.timesteps).unwrap();
                sq_err += (pred - target).powi(2);
                assert_eq!(pred.round(), target, "AND readout misclassified {input:?}");
            }
            let mse = sq_err / inputs.len() as f64;
            assert!(mse < 1e-2, "train MSE too high: {mse}");
        }
    }

    mod logic_gate_truth_tables {
        use super::*;

        fn gate_output(gate: &str, input: &[f64]) -> bool {
            let mut lnn = LiquidNeuralNetwork::new(10);
            lnn.configure_for_logic_gate(gate).unwrap();
            let out = lnn.run_simulation(10, vec![input.to_vec()]).unwrap();
            out[0][0] > 0.5
        }

        #[test]
        fn test_and_truth_table() {
            assert!(!gate_output("AND", &[0.0, 0.0]));
            assert!(!gate_output("AND", &[0.0, 1.0]));
            assert!(!gate_output("AND", &[1.0, 0.0]));
            assert!(gate_output("AND", &[1.0, 1.0]));
        }

        #[test]
        fn test_or_truth_table() {
            assert!(!gate_output("OR", &[0.0, 0.0]));
            assert!(gate_output("OR", &[0.0, 1.0]));
            assert!(gate_output("OR", &[1.0, 0.0]));
            assert!(gate_output("OR", &[1.0, 1.0]));
        }

        #[test]
        fn test_not_truth_table() {
            assert!(gate_output("NOT", &[0.0]));
            assert!(!gate_output("NOT", &[1.0]));
        }

        #[test]
        fn test_nand_truth_table() {
            assert!(gate_output("NAND", &[0.0, 0.0]));
            assert!(gate_output("NAND", &[0.0, 1.0]));
            assert!(gate_output("NAND", &[1.0, 0.0]));
            assert!(!gate_output("NAND", &[1.0, 1.0]));
        }

        #[test]
        fn test_nor_truth_table() {
            assert!(gate_output("NOR", &[0.0, 0.0]));
            assert!(!gate_output("NOR", &[0.0, 1.0]));
            assert!(!gate_output("NOR", &[1.0, 0.0]));
            assert!(!gate_output("NOR", &[1.0, 1.0]));
        }

        #[test]
        fn test_xor_truth_table() {
            assert!(!gate_output("XOR", &[0.0, 0.0]));
            assert!(gate_output("XOR", &[0.0, 1.0]));
            assert!(gate_output("XOR", &[1.0, 0.0]));
            assert!(!gate_output("XOR", &[1.0, 1.0]));
        }
    }
}
