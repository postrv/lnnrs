// lnn.rs

use crate::network_components::{Neuron, Synapse};
use ndarray::Array;
use ndarray::Array2;
use rand::{thread_rng, Rng};
use serde::de::{self, Deserializer as _deDeserializer, SeqAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer};
use serde::{Serialize, Serializer};
use smartcore::{linalg::basic::matrix::DenseMatrix, linear::linear_regression::*};
use std::fmt;



/// Determines the size of the synaptic matrix based on given connections.
/// Assumes `connections` is non-empty and that index tuples are 0-based.
fn determine_size_of_matrix_based_on_connections(
    connections: &Vec<((usize, usize), f64)>,
) -> (usize, usize) {
    let max_row_index = connections.iter().map(|((row, _), _)| row).max().unwrap();
    let max_col_index = connections.iter().map(|((_, col), _)| col).max().unwrap();

    // Add 1 because indices are 0-based, and we need the count for size.
    (max_row_index + 1, max_col_index + 1)
}

/// A default synaptic delay, modify this value as as required.
const DEFAULT_DELAY: u32 = 1;

impl<'de> Deserialize<'de> for SynapsesWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: _deDeserializer<'de>,
    {
        // Define a visitor struct that will traverse the serialized data
        struct SynapsesWrapperVisitor;

        // Implement the Visitor trait for visitor struct
        impl<'de> Visitor<'de> for SynapsesWrapperVisitor {
            type Value = SynapsesWrapper;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of synapse connections")
            }
            // Define how to visit a sequence, we expect to find a Vec of tuples for the connections
            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let mut connections = Vec::new();

                while let Some((index, weight)) = seq.next_element()? {
                    connections.push((index, weight));
                }

                let size = determine_size_of_matrix_based_on_connections(&connections);
                let mut synapses_array: Array2<Option<Synapse>> = Array2::default(size);
                synapses_array.fill(None);
                for elem in synapses_array.iter_mut() {
                    *elem = None; // This sets each element without requiring Clone
                }
                for ((row, col), weight) in connections {
                    synapses_array[[row, col]] = Some(Synapse::new(weight, DEFAULT_DELAY));
                }

                Ok(SynapsesWrapper(synapses_array))
            }
        }
        Err(de::Error::custom("Failed to deserialize synapses"))
    }
}
pub struct SynapsesWrapper(pub Array2<Option<Synapse>>);
// Implement `Serialize` for `SynapsesWrapper` instead of for `Array2`
impl Serialize for SynapsesWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("SynapsesWrapper", 1)?;
        // Serialize the data contained in `Array2<Option<Synapse>>` as necessary
        // For example, consider serializing only the index and weight of each Synapse
        // and turning them into a Vec of tuples:
        let synapse_data: Vec<((usize, usize), f64)> = self
            .0
            .indexed_iter()
            .filter_map(|((row, col), &syn)| syn.as_ref().map(|s| ((row, col), s.weight)))
            .collect();

        s.serialize_field("synapses", &synapse_data)?;
        s.end()
    }
}

/// Represents the entire Liquid Neural Network.
#[derive(Serialize, Deserialize)]
pub struct LiquidNeuralNetwork {
    neurons: Vec<Neuron>,
    synapses: SynapsesWrapper,
    input_indices: Vec<usize>,  // Indices of input neurons
    output_indices: Vec<usize>, // Indices of output neurons
    readout_weights: Vec<f64>,  // Learned weights for the readout layer
}

impl LiquidNeuralNetwork {
    /// Creates a new Liquid Neural Network with a given number of neurons, and undefined input/output neurons.
    pub fn new(neuron_count: usize) -> Self {
        let mut rng = thread_rng();
        let neurons = (0..neuron_count)
            .map(|_| Neuron::new(rng.gen_range(0.5..1.0))) // Randomly initialize neuron thresholds
            .collect();

        let mut synapses_array = Array::default((neuron_count, neuron_count));
        for mut synapse_option in synapses_array.iter_mut() {
            *synapse_option = None;
        }
        let synapses = SynapsesWrapper(synapses_array);

        LiquidNeuralNetwork {
            neurons,
            synapses,
            input_indices: Vec::new(),
            output_indices: Vec::new(),
            readout_weights: vec![],
        }
    }

    /// Sets the indices of neurons that will act as the input interface.
    pub fn set_input_neurons(&mut self, indices: Vec<usize>) {
        self.input_indices = indices;
    }

    /// Sets the indices of neurons that will act as the output interface.
    pub fn set_output_neurons(&mut self, indices: Vec<usize>) {
        self.output_indices = indices;
    }

    /// A method for connecting two neurons.
    pub fn connect_neurons(&mut self, pre_idx: usize, post_idx: usize, weight: f64, delay: u32) {
        // Check for valid indices and prevent self-connections
        if pre_idx != post_idx && pre_idx < self.neurons.len() && post_idx < self.neurons.len() {
            self.synapses.0[[pre_idx, post_idx]] = Some(Synapse::new(weight, delay));
        }
    }

    /// Encodes external input data into neural stimulation.
    pub fn encode_input(&mut self, input_data: &[f64]) {
        for (&neuron_idx, &stimulus) in self.input_indices.iter().zip(input_data) {
            if let Some(neuron) = self.neurons.get_mut(neuron_idx) {
                neuron.potential += stimulus; // Simple direct current stimulation
            }
        }
    }

    /// Decodes the output from the network into a meaningful signal.
    pub fn decode_output(&self) -> Vec<f64> {
        self.output_indices
            .iter()
            .filter_map(|&idx| self.neurons.get(idx))
            .map(|neuron| neuron.potential) // Read out the potentials as the output signal
            .collect()
    }

    /// Trains a Linear Regression model as the readout layer for the LNN.
    pub fn train_readout(&mut self, inputs: &[Vec<f64>], expected_outputs: &[f64]) {
        // Collect the encoded inputs and expected outputs for training
        let mut readout_inputs: Vec<Vec<f64>> = Vec::new();

        for input in inputs.iter() {
            self.reset_state(); // Reset network state before input encoding
            self.encode_input(input); // Encode the current input pattern

            // TODO: Simulate network dynamics and collect output response
            let output_response = self.decode_output();
            readout_inputs.push(output_response);
        }

        // Convert readout_inputs to a DenseMatrix for Smartcore's Linear Regression fit
        let x = DenseMatrix::from_2d_vec(&readout_inputs);
        let y = DenseMatrix::from_2d_vec(&vec![expected_outputs.clone()]);

        let lr = LinearRegression::fit(&x, &y, Default::default())
            .expect("Failed to fit Linear Regression model");

        // Store the learned weights
        self.readout_weights = lr.coefficients().to_vec();
    }

    /// Resets the network states, useful before processing each input during readout training.
    pub fn reset_state(&mut self) {
        for neuron in &mut self.neurons {
            neuron.potential = 0.0; // Reset the potential to the resting state
                                    // Reset any other neuron states if necessary
        }
        // Add synapses reset if required by the model
    }

    pub fn run_simulation(&mut self, timesteps: usize, inputs: Vec<Vec<f64>>) {
        assert_eq!(inputs.len(), self.input_indices.len(), "Input data must match the number of input neurons.");

        for _ in 0..timesteps {
            for (input_values, &idx) in inputs.iter().zip(self.input_indices.iter()) {
                self.encode_input(input_values);
            }

            // Update the state of each neuron here; may involve complex dynamics
            for neuron in &mut self.neurons {
                neuron.update_state();
            }


            // Propagate signals through the synapses
            // TODO: define additional logic to handle synaptic transmission and potential delay
            // for synapse in &self.synapses {
            //     synapse.transmit_signal();
            // }

            // Decode output here if needed, or store state for post-simulation analysis
            let output = self.decode_output();

            // Placeholder: process the output as necessary, e.g., storing it, or using it in some way
            //
        }
    }
}
