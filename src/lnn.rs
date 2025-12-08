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
use rand::{thread_rng, Rng};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::*;
use std::fmt;

/// Default synaptic delay in timesteps.
const DEFAULT_DELAY: u32 = 1;

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

    /// Builds a LiquidNeuralNetwork from this configuration.
    pub fn build(self) -> LiquidNeuralNetwork {
        LiquidNeuralNetwork::from_config(self)
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

    /// Builds the LiquidNeuralNetwork.
    pub fn build(self) -> LiquidNeuralNetwork {
        LiquidNeuralNetwork::from_config(self.config)
    }
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
    pub fn from_config(config: LnnConfig) -> Self {
        let mut rng = thread_rng();
        let neuron_count = config.neuron_count;

        // Create neurons with random thresholds
        let neurons: Vec<Neuron> = (0..neuron_count)
            .map(|_| Neuron::new(rng.gen_range(config.threshold_range.0..config.threshold_range.1)))
            .collect();

        // Initialize synapses
        let mut synapses_array = Array2::default((neuron_count, neuron_count));
        for i in 0..neuron_count {
            for j in 0..neuron_count {
                if i != j && rng.gen::<f64>() < config.connection_probability {
                    let weight = rng.gen_range(config.weight_range.0..config.weight_range.1);
                    let delay = rng.gen_range(config.delay_range.0..config.delay_range.1);
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
            for synapse_opt in self.synapses.0.iter_mut() {
                if let Some(synapse) = synapse_opt {
                    synapse.weight *= scaling_factor;
                }
            }
        }
    }

    /// Initializes synapses with random weights scaled to a spectral radius.
    ///
    /// This replaces all existing synapses with newly generated ones.
    pub fn initialize_synapses(&mut self, spectral_radius: f64) {
        let neuron_count = self.neurons.len();
        let mut rng = thread_rng();

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
    pub fn connect_neurons(
        &mut self,
        pre_idx: usize,
        post_idx: usize,
        weight: f64,
        delay: u32,
    ) -> Result<()> {
        if pre_idx >= self.neurons.len() {
            return Err(LnnError::index_out_of_bounds(pre_idx, self.neurons.len() - 1));
        }
        if post_idx >= self.neurons.len() {
            return Err(LnnError::index_out_of_bounds(post_idx, self.neurons.len() - 1));
        }
        if pre_idx == post_idx {
            return Err(LnnError::configuration("Cannot create self-connection"));
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
    /// Input values are added to the membrane potentials of designated input neurons.
    pub fn encode_input(&mut self, input_data: &[f64]) {
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

    /// Trains a Linear Regression model as the readout layer.
    ///
    /// # Arguments
    /// * `inputs` - Training input patterns
    /// * `expected_outputs` - Expected output values
    ///
    /// # Returns
    /// `Ok(())` if training succeeds, `Err` otherwise
    pub fn train_readout(&mut self, inputs: &[Vec<f64>], expected_outputs: &[f64]) -> Result<()> {
        if inputs.is_empty() {
            return Err(LnnError::Training("No training inputs provided".into()));
        }
        if inputs.len() != expected_outputs.len() {
            return Err(LnnError::Training(
                "Input and output count mismatch".into(),
            ));
        }

        let mut readout_inputs: Vec<Vec<f64>> = Vec::with_capacity(inputs.len());

        for input in inputs.iter() {
            self.reset_state();
            self.encode_input(input);
            let output_response = self.decode_output();
            readout_inputs.push(output_response);
        }

        // Check if we have valid readout inputs
        if readout_inputs.is_empty() || readout_inputs[0].is_empty() {
            return Err(LnnError::Training(
                "No output neurons configured for readout".into(),
            ));
        }

        let x = DenseMatrix::from_2d_vec(&readout_inputs);
        let y = expected_outputs.to_vec();

        let lr = LinearRegression::fit(
            &x,
            &y,
            LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::SVD),
        )
        .map_err(|e| LnnError::Training(format!("Linear regression failed: {:?}", e)))?;

        // Store the learned weights
        self.readout_weights = lr.coefficients().iter().copied().collect();
        Ok(())
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
    pub fn configure_gate(&mut self, gate: LogicGate) {
        // Set all neurons to use normalized mode (rest_potential = 0) for logic gates
        for neuron in &mut self.neurons {
            neuron.rest_potential = 0.0;
        }

        match gate {
            LogicGate::And => self.configure_and_gate(),
            LogicGate::Or => self.configure_or_gate(),
            LogicGate::Not => self.configure_not_gate(),
            LogicGate::Nand => self.configure_nand_gate(),
            LogicGate::Nor => self.configure_nor_gate(),
            LogicGate::Xor => self.configure_xor_gate(),
        }
    }

    fn configure_and_gate(&mut self) {
        if self.neurons.len() < 3 {
            return;
        }

        // Input neurons: 1, 2; Output neuron: 0
        self.set_input_neurons(vec![1, 2]);
        self.set_output_neurons(vec![0]);

        let _ = self.connect_neurons(1, 0, 0.5, 1);
        let _ = self.connect_neurons(2, 0, 0.5, 1);

        if let Some(neuron) = self.neurons.get_mut(0) {
            neuron.threshold = 0.9; // High threshold for AND
        }
    }

    fn configure_or_gate(&mut self) {
        if self.neurons.len() < 3 {
            return;
        }

        self.set_input_neurons(vec![1, 2]);
        self.set_output_neurons(vec![0]);

        let _ = self.connect_neurons(1, 0, 0.5, 1);
        let _ = self.connect_neurons(2, 0, 0.5, 1);

        if let Some(neuron) = self.neurons.get_mut(0) {
            neuron.threshold = 0.4; // Low threshold for OR
        }
    }

    fn configure_not_gate(&mut self) {
        if self.neurons.len() < 2 {
            return;
        }

        self.set_input_neurons(vec![0]);
        self.set_output_neurons(vec![1]);

        // Configure neurons
        self.neurons[0] = Neuron::with_id(0, 0.5);
        self.neurons[1] = Neuron::with_id(1, 1.5);

        // Inhibitory connection
        let _ = self.connect_neurons(0, 1, -1.0, DEFAULT_DELAY);
    }

    fn configure_nand_gate(&mut self) {
        if self.neurons.len() < 4 {
            return;
        }

        // NAND = NOT(AND)
        self.set_input_neurons(vec![1, 2]);
        self.set_output_neurons(vec![3]);

        // AND part (neuron 0 as intermediate)
        let _ = self.connect_neurons(1, 0, 0.5, 1);
        let _ = self.connect_neurons(2, 0, 0.5, 1);
        if let Some(neuron) = self.neurons.get_mut(0) {
            neuron.threshold = 0.9;
        }

        // NOT part (invert the AND result)
        let _ = self.connect_neurons(0, 3, -1.0, 1);
        if let Some(neuron) = self.neurons.get_mut(3) {
            neuron.threshold = 0.5;
            neuron.membrane_potential = 1.0; // Bias to be high by default
        }
    }

    fn configure_nor_gate(&mut self) {
        if self.neurons.len() < 4 {
            return;
        }

        // NOR = NOT(OR)
        self.set_input_neurons(vec![1, 2]);
        self.set_output_neurons(vec![3]);

        // OR part
        let _ = self.connect_neurons(1, 0, 0.5, 1);
        let _ = self.connect_neurons(2, 0, 0.5, 1);
        if let Some(neuron) = self.neurons.get_mut(0) {
            neuron.threshold = 0.4;
        }

        // NOT part
        let _ = self.connect_neurons(0, 3, -1.0, 1);
        if let Some(neuron) = self.neurons.get_mut(3) {
            neuron.threshold = 0.5;
            neuron.membrane_potential = 1.0;
        }
    }

    fn configure_xor_gate(&mut self) {
        if self.neurons.len() < 5 {
            return;
        }

        // XOR needs multiple layers
        self.set_input_neurons(vec![0, 1]);
        self.set_output_neurons(vec![4]);

        // Layer 1: OR (neuron 2) and NAND-like (neuron 3)
        let _ = self.connect_neurons(0, 2, 0.6, 1);
        let _ = self.connect_neurons(1, 2, 0.6, 1);
        if let Some(n) = self.neurons.get_mut(2) {
            n.threshold = 0.5;
        }

        // NAND-like behavior through inhibition
        let _ = self.connect_neurons(0, 3, 0.6, 1);
        let _ = self.connect_neurons(1, 3, 0.6, 1);
        if let Some(n) = self.neurons.get_mut(3) {
            n.threshold = 1.1; // Only fires if both inputs are high
        }

        // Output: OR output AND NOT(AND output)
        let _ = self.connect_neurons(2, 4, 0.6, 1);
        let _ = self.connect_neurons(3, 4, -1.0, 1); // Inhibit when both inputs high
        if let Some(n) = self.neurons.get_mut(4) {
            n.threshold = 0.5;
        }
    }

    /// Runs a simulation for a specified number of timesteps.
    ///
    /// # Arguments
    /// * `timesteps` - Number of simulation steps to run
    /// * `inputs` - Input patterns to process
    ///
    /// # Returns
    /// Vector of output vectors, one for each input pattern
    pub fn run_simulation(&mut self, timesteps: usize, inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut outputs = Vec::with_capacity(inputs.len());

        // Pre-collect synapses for each neuron
        let synapses_for_neurons: Vec<Vec<Synapse>> = (0..self.neurons.len())
            .map(|idx| self.get_synapses_for_neuron(idx))
            .collect();

        for input_values in inputs.iter() {
            self.encode_input(input_values);

            for timestep in 0..timesteps {
                log::debug!("Running timestep {}", timestep);

                // Update all neurons
                for (neuron, synapses) in self.neurons.iter_mut().zip(synapses_for_neurons.iter()) {
                    neuron.update_state(0.0, synapses);
                }

                // Propagate signals through synapses
                for synapse_option in self.synapses.0.iter() {
                    if let Some(synapse) = synapse_option {
                        if let Some(source_neuron) = self.neurons.get(synapse.source) {
                            let source_activity = source_neuron.membrane_potential;
                            synapse.propagate_signal(
                                source_activity,
                                &mut self.neurons,
                                timestep as u32,
                            );
                        }
                    }
                }

                log::debug!("Output at timestep {}: {:?}", timestep, self.decode_output());
            }

            outputs.push(self.decode_output());
        }

        outputs
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
    use approx::assert_relative_eq;

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
            let lnn = LnnBuilder::new(10)
                .spectral_radius(0.95)
                .build();

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
            let outputs = lnn.run_simulation(3, inputs);

            assert_eq!(outputs.len(), 2);
            assert_eq!(outputs[0].len(), 2);
        }

        #[test]
        fn test_run_simulation_empty_inputs() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            let outputs = lnn.run_simulation(3, vec![]);

            assert!(outputs.is_empty());
        }
    }

    mod training_tests {
        use super::*;

        #[test]
        fn test_train_readout_empty_inputs() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.set_output_neurons(vec![3, 4]);

            let result = lnn.train_readout(&[], &[]);
            assert!(result.is_err());
        }

        #[test]
        fn test_train_readout_mismatched_lengths() {
            let mut lnn = LiquidNeuralNetwork::new(5);
            lnn.set_output_neurons(vec![3, 4]);

            let inputs = vec![vec![1.0], vec![2.0]];
            let outputs = vec![1.0]; // Wrong length

            let result = lnn.train_readout(&inputs, &outputs);
            assert!(result.is_err());
        }
    }
}
