// network_components.rs

use serde::{Deserialize, Serialize, Serializer};

/// Represents the state of a neuron within the LNN.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Neuron {
    pub potential: f64, // Membrane potential of the neuron
    pub recovery: f64,  // Recovery variable, used as an example here
    pub threshold: f64, // Threshold for firing
                        // Add more properties if necessary
}

impl Neuron {
    pub fn new(threshold: f64) -> Self {
        Neuron {
            potential: 0.0,
            recovery: 0.0,
            threshold,
        }
    }

    /// Updates the state of the neuron based on its inputs and current state.
    pub fn update_state(&mut self, input_current: f64) {
        // Constants for a simple neural model.
        // Update these as per specific model.
        const MEMBRANE_RESISTANCE: f64 = 1.0; // Example
        const MEMBRANE_TIME_CONSTANT: f64 = 20.0; // Example
        const MEMBRANE_POTENTIAL_REST: f64 = -65.0; // Example

        // Update the neuron's potential based on the model.
        let leakage = (MEMBRANE_POTENTIAL_REST - self.potential) / MEMBRANE_TIME_CONSTANT;
        self.potential += (input_current * MEMBRANE_RESISTANCE + leakage) * MEMBRANE_TIME_CONSTANT;

        // Check for threshold crossing and potentially fire the neuron.
        if self.potential > self.threshold {
            self.fire();
            self.potential = MEMBRANE_POTENTIAL_REST; // Reset potential after firing
        }

        // Include logic for recovery variable, if used in model.
        // Recovery dynamics would need to be defined here.
    }

    /// Simulate the neuron firing.
    fn fire(&self) {
        // Implement the logic for firing the neuron.
        // This would typically involve interacting with other neurons via synapses in a network.
    }
}

/// Represents a directional synapse between two neurons within the LNN.
#[derive(Debug, Clone)]
pub struct Synapse {
    pub weight: f64,
    pub delay: u32,
    pub source: usize,
    pub target: usize,
    // Add more properties if necessary
}

impl Synapse {
    pub fn new(weight: f64, delay: u32, source: usize, target: usize) -> Self {
        Synapse { weight, delay, source, target}
    }
}

impl Serialize for Synapse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // If Synapse only has types that automatically serialize, could consider using a helper method of Serializer
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Synapse", 4)?;
        s.serialize_field("weight", &self.weight)?;
        s.serialize_field("delay", &self.delay)?;
        s.serialize_field("weight", &self.source)?;
        s.serialize_field("delay", &self.target)?;
        // Add serialization for more properties as needed
        s.end()
    }
}

// below are the contents of previous synapses.rs and neurons.rs files retained for reference
//TODO - update the above to make use of the below gating etc and then delete the below.

// synapses.rs -- Synapse module


/// Represents a directional connection (synapse) between two neurons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseState {
    pub weight: f64,
    pub delay: u32,
    pub source: usize,
    pub target: usize,
}

impl SynapseState {
    /// Creates a new synapse with specified properties.
    pub fn new(source: usize, target: usize, weight: f64, delay: u32) -> Self {
        SynapseState { source, target, weight, delay }
    }

    /// Stimulates the target neuron based on this synapse's properties and the source neuron's activity.
    pub fn stimulate(&self, source_activity: f64) -> f64 {
        // Simple model: the input is multiplied by synaptic weight.
        // More complex models could also factor in delay and the time since the last pre-synaptic neuron fired.
        source_activity * self.weight
    }
}

// File: src/neuron.rs

/// Defines the state of a neuron, including its gating and time constant.
pub struct NeuronState {
    pub membrane_potential: f64,
    pub gating_variable: f64,   // Represents gating dynamics (e.g., for ion channels)
    pub time_constant: f64,     // The time constant modulated by input (making it 'liquid')
    // Additional state variables can be added here
}

impl NeuronState {
    // Assuming the `new` constructor is already defined above.

    /// Calculate the gating dynamics based on current state and input.
    fn gating_dynamics(&self, _input: f64) -> f64 {
        // Apply a nonlinear function to calculate the gating dynamics, such as a sigmoid or tanh.
        // The sigmoid function can represent the gating as a probability function.
        // Replace the implementation with the appropriate model from the LNN theory.
        1.0 / (1.0 + (-self.membrane_potential).exp()) // Example sigmoid function
    }

    /// Calculate the dynamic time constant based on the current state and input.
    fn dynamic_time_constant(&self, _input: f64) -> f64 {
        // The time constant could be modulated by input and/or state.
        // This function needs to be in line with the LNN model specifics.
        20.0 // Placeholder, utilize a proper function here
    }

    /// The derivatives function that describes how the neuron state changes over time.
    /// This function returns the time derivatives of the neuron state variables.
    pub fn derivatives(&self, _t: f64, input: f64) -> Self {
        // Update the gating dynamics and time constants based on inputs/state
        let gating_dynamics = self.gating_dynamics(input);
        let dynamic_time_constant = self.dynamic_time_constant(input);

        // Depending on the specifics of the model, calculate the differential equations for neuron state
        // For example, here we use a leaky integrate-and-fire model as a placeholder
        let d_membrane_potential = (-(self.membrane_potential - gating_dynamics) / dynamic_time_constant) + input;

        // Other state derivatives would follow similar logic
        let d_gating_variable = 0.0; // replace with actual gating dynamics model

        NeuronState {
            membrane_potential: d_membrane_potential,
            gating_variable: d_gating_variable,
            time_constant: dynamic_time_constant, // may want to include dynamics for the time constant itself
        }
    }
}