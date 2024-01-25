// network_components.rs

use serde::{Deserialize, Serialize, Serializer};

use crate::activation::sigmoid;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEURON_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);
static SYNAPSE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Represents the state of a neuron within the LNN.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Neuron {
    pub id: usize,               // Unique identifier for the neuron
    pub membrane_potential: f64, // Membrane potential of the neuron
    pub gating_variable: f64,    // Represents gating dynamics (e.g., for ion channels)
    pub time_constant: f64,      // The time constant modulated by input (making it 'liquid')
    pub recovery: f64,           // Recovery variable, used as an example here
    pub threshold: f64,          // Threshold for firing
                                 // Add more properties if necessary
}

impl Neuron {
    pub fn new(threshold: f64) -> Self {
        Neuron {
            id: NEURON_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            membrane_potential: 0.0,
            gating_variable: 0.0,
            time_constant: 20.0, // Example value, should be based on model specifics
            recovery: 0.0,
            threshold,
        }
    }

    /// Dynamically adjusts the neuron's firing threshold.
    fn adjust_threshold(&mut self) {
        // Example: Increase threshold if membrane potential is high
        if self.membrane_potential > -2.0 {
            self.threshold += 0.1;
        }
        // Decrease threshold otherwise, with a lower limit
        else {
            self.threshold = (self.threshold - 0.1).max(0.5);
        }
    }

    /// Updates the state of the neuron based on its inputs and current state.
    pub fn update_state(&mut self, input_current: f64, synapses: &[Synapse]) {
        // Constants for a simple neural model.
        // Update these as per specific model.
        log::debug!("Neuron {} state before update: {:?}", self.id, self);
        const MEMBRANE_RESISTANCE: f64 = 1.3;
        const MEMBRANE_TIME_CONSTANT: f64 = 7.0; // Example
        const MEMBRANE_POTENTIAL_REST: f64 = -69.0; // Example

        // Update the neuron's potential based on the model.
        let leakage = (MEMBRANE_POTENTIAL_REST - self.membrane_potential) / MEMBRANE_TIME_CONSTANT;
        self.membrane_potential +=
            (input_current * MEMBRANE_RESISTANCE + leakage) * MEMBRANE_TIME_CONSTANT;

        // Check for threshold crossing and potentially fire the neuron.
        if self.membrane_potential > self.threshold {
            let events = self.fire(synapses);
            // Process the events as needed
            self.membrane_potential = MEMBRANE_POTENTIAL_REST; // Reset potential after firing
        }
        self.adjust_threshold(); // Adjust the threshold based on the neuron's state

        log::debug!("Neuron {} state after update: {:?}", self.id, self);
    }

    /// Simulate the neuron firing.
    fn fire(&self, synapses: &[Synapse]) -> Vec<NeuronEvent> {
        let mut events = Vec::new();
        for synapse in synapses {
            if synapse.source == self.id {
                events.push(NeuronEvent::new(synapse.target, synapse.weight));
            }
        }
        events
    }
}

/// Represents a directional synapse between two neurons within the LNN.
#[derive(Debug, Clone)]
pub struct Synapse {
    pub id: usize,     // Unique identifier for the synapse
    pub weight: f64,   // Synaptic weight
    pub delay: u32,    // Synaptic delay
    pub source: usize, // Index of the source neuron
    pub target: usize, // Index of the target neuron
                       // Add more properties if necessary
}

impl Synapse {
    /// Propagates a signal from the source neuron to the target neuron.
    /// Propagates a signal from the source neuron to the target neuron.
    pub fn propagate_signal(
        &self,
        source_activity: f64,
        neurons: &mut [Neuron],
        current_time: u32,
    ) {
        if let Some(target_neuron) = neurons.get_mut(self.target) {
            log::debug!(
                "Propagating signal from neuron {} to neuron {} at time {}",
                self.source,
                self.target,
                current_time
            );
            // Implementing synaptic delay
            if current_time >= self.delay {
                // Non-linear integration of the signal
                let signal = self.weight * source_activity;
                target_neuron.membrane_potential += sigmoid(signal); // Using a sigmoid function for non-linear integration
                log::debug!(
                    "Neuron {} state after receiving signal: {:?}",
                    self.target,
                    target_neuron
                );
            }
        }
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

impl Synapse {
    /// Creates a new synapse with specified properties.
    pub fn new(source: usize, target: usize, weight: f64, delay: u32) -> Self {
        Synapse {
            id: SYNAPSE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            source,
            target,
            weight,
            delay,
        }
    }

    /// Stimulates the target neuron based on this synapse's properties and the source neuron's activity.
    pub fn stimulate(&self, source_activity: f64) -> f64 {
        // Simple model: the input is multiplied by synaptic weight.
        // More complex models could also factor in delay and the time since the last pre-synaptic neuron fired.
        source_activity * self.weight
    }
}

// Neuron module previously in neuron.rs

/// Defines the state of a neuron, including its gating and time constant.

impl Neuron {
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
        20.0 // Placeholder, TODO: utilize a proper function here
    }

    /// The derivatives function that describes how the neuron state changes over time.
    /// This function returns the time derivatives of the neuron state variables.
    pub fn derivatives(&self, _t: f64, input: f64) -> Self {
        // Update the gating dynamics and time constants based on inputs/state
        let gating_dynamics = self.gating_dynamics(input);
        let dynamic_time_constant = self.dynamic_time_constant(input);

        // Depending on the specifics of the model, calculate the differential equations for neuron state
        // For example, here we use a leaky integrate-and-fire model as a placeholder
        let d_membrane_potential =
            (-(self.membrane_potential - gating_dynamics) / dynamic_time_constant) + input;

        // Other state derivatives would follow similar logic
        let d_gating_variable = 0.0; // replace with actual gating dynamics model

        Neuron {
            id: self.id,
            membrane_potential: d_membrane_potential,
            gating_variable: d_gating_variable,
            time_constant: dynamic_time_constant, // may want to include dynamics for the time constant itself
            recovery: 0.0,
            threshold: self.threshold,
        }
    }
}

// In network_components.rs

/// Represents an event generated by a neuron when it fires.
pub struct NeuronEvent {
    pub target_index: usize,  // Index of the target neuron
    pub signal_strength: f64, // Strength of the signal
}

impl NeuronEvent {
    pub fn new(target_index: usize, signal_strength: f64) -> Self {
        NeuronEvent {
            target_index,
            signal_strength,
        }
    }
}
