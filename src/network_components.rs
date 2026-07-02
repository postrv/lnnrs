// src/network_components.rs
//! Core neural network components: Neurons and Synapses.
//!
//! This module implements the fundamental building blocks of the Liquid Neural Network:
//! - `Neuron`: Represents a single neuron with membrane potential, gating dynamics, etc.
//! - `Synapse`: Represents a directed connection between two neurons.
//! - `ThresholdAdaptation`: Optional homeostatic threshold plasticity for a neuron.
//!
//! Neurons are decoupled from the network topology: [`Neuron::update_state`] performs a
//! single leaky-integrate-and-fire step given a scalar input current, and returns whether
//! the neuron fired. Signal routing, delays and propagation are owned by the network
//! (`LiquidNeuralNetwork::run_simulation`), which indexes neurons positionally.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::activation::sigmoid;

// Global atomic counters for unique IDs
static NEURON_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);
static SYNAPSE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Resets the ID counters (useful for testing)
#[cfg(test)]
pub fn reset_id_counters() {
    NEURON_ID_COUNTER.store(0, Ordering::SeqCst);
    SYNAPSE_ID_COUNTER.store(0, Ordering::SeqCst);
}

/// Configuration constants for the neuron model
pub mod config {
    /// Membrane resistance in MΩ
    pub const MEMBRANE_RESISTANCE: f64 = 1.3;
    /// Membrane time constant in ms
    pub const MEMBRANE_TIME_CONSTANT: f64 = 7.0;
    /// Resting membrane potential in mV
    pub const MEMBRANE_POTENTIAL_REST: f64 = -69.0;
    /// Base time constant for dynamic time constant calculation
    pub const BASE_TIME_CONSTANT: f64 = 20.0;
    /// Scaling factor for input in time constant calculation
    pub const INPUT_SCALING_FACTOR: f64 = 0.5;
    /// Scaling factor for membrane potential in time constant calculation
    pub const POTENTIAL_SCALING_FACTOR: f64 = 0.3;
    /// Default threshold adjustment increment
    pub const THRESHOLD_ADJUSTMENT_INCREMENT: f64 = 0.1;
    /// Minimum threshold value
    pub const MIN_THRESHOLD: f64 = 0.5;
    /// Threshold adjustment membrane potential boundary
    pub const THRESHOLD_ADJUSTMENT_BOUNDARY: f64 = -2.0;
}

/// Represents the state of a neuron within the LNN.
///
/// The neuron implements a leaky integrate-and-fire model with dynamic time constants,
/// which is the core mechanism of Liquid Neural Networks.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Neuron {
    /// Unique identifier for the neuron
    pub id: usize,
    /// Membrane potential of the neuron (in mV conceptually)
    pub membrane_potential: f64,
    /// Gating variable for ion channel dynamics
    pub gating_variable: f64,
    /// Dynamic time constant (what makes it "liquid")
    pub time_constant: f64,
    /// Recovery variable for refractory period modeling
    pub recovery: f64,
    /// Firing threshold - potential must exceed this to fire
    pub threshold: f64,
    /// Resting membrane potential (defaults to biological -69mV, use 0.0 for logic gates)
    #[serde(default = "default_rest_potential")]
    pub rest_potential: f64,
    /// Constant bias current added to the membrane potential on every update step.
    ///
    /// This is used to make a neuron read "true" by default (e.g. the output neuron of a
    /// NOT/NAND/NOR gate) without a one-shot potential hack: because it is applied every
    /// step it survives the leaky decay. Defaults to `0.0`.
    #[serde(default)]
    pub bias: f64,
    /// Optional, bounded homeostatic threshold adaptation. Disabled by default so that
    /// explicitly configured thresholds (logic gates!) stay put.
    #[serde(default)]
    pub adaptation: ThresholdAdaptation,
}

fn default_rest_potential() -> f64 {
    config::MEMBRANE_POTENTIAL_REST
}

/// Bounded, opt-in homeostatic threshold adaptation for a [`Neuron`].
///
/// When [`enabled`](Self::enabled), the neuron's firing threshold increases by
/// [`increment`](Self::increment) after every step on which it fired and decays by the same
/// amount toward [`min`](Self::min) on steps where it did not fire, always clamped to
/// `[min, max]`. Homeostasis therefore reacts to *firing activity*, not to the resting
/// potential. Disabled by default.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct ThresholdAdaptation {
    /// Whether adaptation is active. Defaults to `false`.
    pub enabled: bool,
    /// Amount the threshold moves per step. Defaults to `0.1`.
    pub increment: f64,
    /// Lower clamp for the threshold. Defaults to `0.5`.
    pub min: f64,
    /// Upper clamp for the threshold. Defaults to `2.0`.
    pub max: f64,
}

impl Default for ThresholdAdaptation {
    fn default() -> Self {
        ThresholdAdaptation {
            enabled: false,
            increment: config::THRESHOLD_ADJUSTMENT_INCREMENT,
            min: config::MIN_THRESHOLD,
            max: 2.0,
        }
    }
}

impl Default for Neuron {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Neuron {
    /// Creates a new neuron with the specified firing threshold.
    ///
    /// # Arguments
    /// * `threshold` - The membrane potential threshold for firing
    ///
    /// # Returns
    /// A new `Neuron` instance with initialized state variables
    pub fn new(threshold: f64) -> Self {
        Neuron {
            id: NEURON_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            membrane_potential: 0.0,
            gating_variable: 0.0,
            time_constant: config::BASE_TIME_CONSTANT,
            recovery: 0.0,
            threshold,
            rest_potential: config::MEMBRANE_POTENTIAL_REST,
            bias: 0.0,
            adaptation: ThresholdAdaptation::default(),
        }
    }

    /// Creates a new neuron with a specific ID (useful for testing/deserialization).
    ///
    /// # Arguments
    /// * `id` - The unique identifier for this neuron
    /// * `threshold` - The membrane potential threshold for firing
    pub fn with_id(id: usize, threshold: f64) -> Self {
        Neuron {
            id,
            membrane_potential: 0.0,
            gating_variable: 0.0,
            time_constant: config::BASE_TIME_CONSTANT,
            recovery: 0.0,
            threshold,
            rest_potential: config::MEMBRANE_POTENTIAL_REST,
            bias: 0.0,
            adaptation: ThresholdAdaptation::default(),
        }
    }

    /// Creates a new neuron configured for logic gate simulations (normalized 0-1 range).
    ///
    /// # Arguments
    /// * `id` - The unique identifier for this neuron
    /// * `threshold` - The membrane potential threshold for firing
    pub fn for_logic_gate(id: usize, threshold: f64) -> Self {
        Neuron {
            id,
            membrane_potential: 0.0,
            gating_variable: 0.0,
            time_constant: config::BASE_TIME_CONSTANT,
            recovery: 0.0,
            threshold,
            rest_potential: 0.0, // Use 0 for normalized logic gate behavior
            bias: 0.0,
            adaptation: ThresholdAdaptation::default(),
        }
    }

    /// Advances the neuron by one leaky-integrate-and-fire step.
    ///
    /// The update is, in order:
    /// 1. Leaky integration of the scalar `input_current` (through the membrane resistance)
    ///    toward `rest_potential` with time constant `MEMBRANE_TIME_CONSTANT`.
    /// 2. Injection of the constant [`bias`](Self::bias) current.
    /// 3. Threshold check: if the potential exceeds `threshold` the neuron *fires* and is
    ///    reset to `rest_potential`.
    /// 4. Optional bounded [`ThresholdAdaptation`].
    ///
    /// The neuron is intentionally unaware of synapses: the network snapshots potentials and
    /// routes signals through its delay buffers. `input_current` is the total current
    /// arriving at this neuron on this timestep.
    ///
    /// # Arguments
    /// * `input_current` - Total external/synaptic current arriving this timestep
    ///
    /// # Returns
    /// `true` if the neuron fired (crossed threshold and was reset), `false` otherwise.
    pub fn update_state(&mut self, input_current: f64) -> bool {
        // Leaky integrate-and-fire step. `rest_potential` is 0.0 for logic gates and
        // -69.0 for the biological default; dt is the simulation time step (1.0 ms).
        const DT: f64 = 1.0;
        let leakage_rate =
            (self.rest_potential - self.membrane_potential) / config::MEMBRANE_TIME_CONSTANT;
        self.membrane_potential +=
            (input_current * config::MEMBRANE_RESISTANCE + leakage_rate) * DT;

        // Constant bias current (persists across steps, unlike a one-shot potential offset).
        self.membrane_potential += self.bias;

        // Threshold crossing: fire and reset.
        let fired = self.membrane_potential > self.threshold;
        if fired {
            self.membrane_potential = self.rest_potential;
        }

        // Bounded, activity-driven threshold homeostasis (opt-in).
        if self.adaptation.enabled {
            let delta = if fired {
                self.adaptation.increment
            } else {
                -self.adaptation.increment
            };
            self.threshold =
                (self.threshold + delta).clamp(self.adaptation.min, self.adaptation.max);
        }

        fired
    }

    /// Computes the derivative of the neuron state for ODE integration.
    ///
    /// # Arguments
    /// * `_t` - Current time (unused in this simple model)
    /// * `input` - Current input value
    ///
    /// # Returns
    /// The rate of change of the neuron's state
    pub fn compute_derivative(&self, _t: f64, input: f64) -> f64 {
        let gating = self.gating_dynamics(input);
        let tau = self.dynamic_time_constant(input);
        (-(self.membrane_potential - gating) / tau) + input
    }

    /// Calculates gating dynamics based on current state.
    ///
    /// Uses a sigmoid function to model ion channel gating probability.
    fn gating_dynamics(&self, _input: f64) -> f64 {
        sigmoid(self.membrane_potential)
    }

    /// Calculates the dynamic time constant based on current state and input.
    ///
    /// This is the key feature of Liquid Neural Networks - the time constant
    /// adapts based on the neuron's current state and input.
    fn dynamic_time_constant(&self, input: f64) -> f64 {
        let weighted_sum = (input * config::INPUT_SCALING_FACTOR)
            + (self.membrane_potential * config::POTENTIAL_SCALING_FACTOR);
        let sigmoid_response = sigmoid(weighted_sum);
        config::BASE_TIME_CONSTANT * sigmoid_response.max(0.1) // Prevent division by zero
    }

    /// Computes the full state derivatives for ODE integration.
    ///
    /// Returns a new `Neuron` struct containing the derivatives of all state variables.
    ///
    /// # Arguments
    /// * `_t` - Current time
    /// * `input` - Current input value
    pub fn derivatives(&self, _t: f64, input: f64) -> NeuronDerivatives {
        let gating_dynamics = self.gating_dynamics(input);
        let dynamic_time_constant = self.dynamic_time_constant(input);

        let d_membrane_potential =
            (-(self.membrane_potential - gating_dynamics) / dynamic_time_constant) + input;
        let d_gating_variable = 0.0; // Placeholder for more complex gating dynamics

        NeuronDerivatives {
            d_membrane_potential,
            d_gating_variable,
            d_time_constant: 0.0,
            d_recovery: 0.0,
        }
    }

    /// Resets the neuron to its initial state.
    pub fn reset(&mut self) {
        self.membrane_potential = 0.0;
        self.gating_variable = 0.0;
        self.recovery = 0.0;
        self.time_constant = config::BASE_TIME_CONSTANT;
    }

    /// Checks if the neuron is currently firing (above threshold).
    pub fn is_firing(&self) -> bool {
        self.membrane_potential > self.threshold
    }
}

/// Struct to hold neuron state derivatives for ODE integration.
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronDerivatives {
    /// Time derivative of the membrane potential.
    pub d_membrane_potential: f64,
    /// Time derivative of the gating variable.
    pub d_gating_variable: f64,
    /// Time derivative of the time constant.
    pub d_time_constant: f64,
    /// Time derivative of the recovery variable.
    pub d_recovery: f64,
}

/// Represents a directional synapse between two neurons within the LNN.
///
/// Synapses transmit signals from a source neuron to a target neuron,
/// modulated by weight and subject to transmission delay.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Synapse {
    /// Unique identifier for the synapse
    pub id: usize,
    /// Synaptic weight (positive = excitatory, negative = inhibitory)
    pub weight: f64,
    /// Synaptic transmission delay in timesteps
    pub delay: u32,
    /// Index of the source (presynaptic) neuron
    pub source: usize,
    /// Index of the target (postsynaptic) neuron
    pub target: usize,
}

impl Synapse {
    /// Creates a new synapse with specified properties.
    ///
    /// # Arguments
    /// * `source` - Index of the source neuron
    /// * `target` - Index of the target neuron
    /// * `weight` - Synaptic weight
    /// * `delay` - Transmission delay in timesteps
    pub fn new(source: usize, target: usize, weight: f64, delay: u32) -> Self {
        Synapse {
            id: SYNAPSE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            source,
            target,
            weight,
            delay,
        }
    }

    /// Creates a new synapse with a specific ID.
    pub fn with_id(id: usize, source: usize, target: usize, weight: f64, delay: u32) -> Self {
        Synapse {
            id,
            source,
            target,
            weight,
            delay,
        }
    }

    /// Calculates the stimulation effect without applying it.
    ///
    /// # Arguments
    /// * `source_activity` - The activity level of the source neuron
    ///
    /// # Returns
    /// The weighted stimulation value
    pub fn stimulate(&self, source_activity: f64) -> f64 {
        source_activity * self.weight
    }

    /// Checks if this synapse is excitatory (positive weight).
    pub fn is_excitatory(&self) -> bool {
        self.weight > 0.0
    }

    /// Checks if this synapse is inhibitory (negative weight).
    pub fn is_inhibitory(&self) -> bool {
        self.weight < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn setup() {
        reset_id_counters();
    }

    mod neuron_tests {
        use super::*;

        #[test]
        fn test_neuron_creation() {
            setup();
            let neuron = Neuron::new(0.5);

            assert_eq!(neuron.id, 0);
            assert_eq!(neuron.threshold, 0.5);
            assert_eq!(neuron.membrane_potential, 0.0);
            assert_eq!(neuron.gating_variable, 0.0);
            assert_eq!(neuron.recovery, 0.0);
        }

        #[test]
        fn test_neuron_with_id() {
            let neuron = Neuron::with_id(42, 0.75);

            assert_eq!(neuron.id, 42);
            assert_eq!(neuron.threshold, 0.75);
        }

        #[test]
        fn test_neuron_default() {
            setup();
            let neuron = Neuron::default();

            assert_eq!(neuron.threshold, 0.5);
        }

        #[test]
        fn test_neuron_reset() {
            setup();
            let mut neuron = Neuron::new(0.5);
            neuron.membrane_potential = 10.0;
            neuron.gating_variable = 5.0;
            neuron.recovery = 2.0;

            neuron.reset();

            assert_eq!(neuron.membrane_potential, 0.0);
            assert_eq!(neuron.gating_variable, 0.0);
            assert_eq!(neuron.recovery, 0.0);
        }

        #[test]
        fn test_neuron_is_firing() {
            setup();
            let mut neuron = Neuron::new(0.5);

            assert!(!neuron.is_firing());

            neuron.membrane_potential = 1.0;
            assert!(neuron.is_firing());

            neuron.membrane_potential = 0.5;
            assert!(!neuron.is_firing());
        }

        #[test]
        fn test_neuron_gating_dynamics() {
            setup();
            let neuron = Neuron::new(0.5);

            let gating = neuron.gating_dynamics(0.0);
            // sigmoid(0) = 0.5
            assert_relative_eq!(gating, 0.5, epsilon = 1e-10);
        }

        #[test]
        fn test_neuron_dynamic_time_constant() {
            setup();
            let neuron = Neuron::new(0.5);

            let tau = neuron.dynamic_time_constant(0.0);
            // Should be BASE_TIME_CONSTANT * sigmoid(0) = 20 * 0.5 = 10
            assert_relative_eq!(tau, 10.0, epsilon = 1e-10);
        }

        #[test]
        fn test_neuron_compute_derivative() {
            setup();
            let mut neuron = Neuron::new(0.5);
            neuron.membrane_potential = 0.0;

            let derivative = neuron.compute_derivative(0.0, 1.0);

            // Should be non-zero with input
            assert!(derivative != 0.0);
        }

        #[test]
        fn test_neuron_update_state_no_fire() {
            setup();
            let mut neuron = Neuron::new(100.0); // Very high threshold

            let fired = neuron.update_state(0.01);

            assert!(!fired);
            assert!(neuron.membrane_potential != 0.0);
        }

        #[test]
        fn test_neuron_update_state_fires_and_resets() {
            setup();
            let mut neuron = Neuron::new(0.5);
            neuron.rest_potential = 0.0;

            // A large input current drives the potential over threshold.
            let fired = neuron.update_state(10.0);

            assert!(fired);
            // On firing the potential is reset to the rest potential.
            assert_eq!(neuron.membrane_potential, neuron.rest_potential);
        }

        #[test]
        fn test_neuron_bias_persists_each_step() {
            setup();
            let mut neuron = Neuron::new(1e9); // never fires
            neuron.rest_potential = 0.0;
            neuron.bias = 0.5;

            // With zero input the potential converges toward bias / (1 - leak) = 7 * bias.
            for _ in 0..200 {
                neuron.update_state(0.0);
            }
            assert_relative_eq!(neuron.membrane_potential, 3.5, epsilon = 1e-6);
        }

        #[test]
        fn test_threshold_adaptation_disabled_by_default() {
            setup();
            let mut neuron = Neuron::new(0.5);
            neuron.rest_potential = 0.0;
            let threshold_before = neuron.threshold;

            // Fire it a few times; threshold must not move while adaptation is off.
            for _ in 0..5 {
                neuron.update_state(10.0);
            }
            assert_eq!(neuron.threshold, threshold_before);
        }

        #[test]
        fn test_threshold_adaptation_enabled_is_bounded() {
            setup();
            let mut neuron = Neuron::new(0.5);
            neuron.rest_potential = 0.0;
            neuron.adaptation = ThresholdAdaptation {
                enabled: true,
                increment: 0.1,
                min: 0.5,
                max: 1.0,
            };

            // Repeated firing raises the threshold but it is clamped to `max`.
            for _ in 0..100 {
                neuron.update_state(100.0);
            }
            assert!(neuron.threshold <= 1.0);
            assert_relative_eq!(neuron.threshold, 1.0, epsilon = 1e-9);

            // Repeated non-firing lowers it, clamped to `min`.
            neuron.threshold = 1.0;
            for _ in 0..100 {
                neuron.update_state(0.0);
            }
            assert!(neuron.threshold >= 0.5);
            assert_relative_eq!(neuron.threshold, 0.5, epsilon = 1e-9);
        }

        #[test]
        fn test_neuron_serialization() {
            setup();
            let neuron = Neuron::new(0.75);

            let json = serde_json::to_string(&neuron).unwrap();
            let deserialized: Neuron = serde_json::from_str(&json).unwrap();

            assert_eq!(neuron.id, deserialized.id);
            assert_eq!(neuron.threshold, deserialized.threshold);
            assert_eq!(neuron.membrane_potential, deserialized.membrane_potential);
        }

        #[test]
        fn test_neuron_derivatives() {
            setup();
            let neuron = Neuron::new(0.5);

            let derivs = neuron.derivatives(0.0, 1.0);

            // With positive input, membrane potential should increase
            assert!(derivs.d_membrane_potential > 0.0);
        }
    }

    mod synapse_tests {
        use super::*;

        #[test]
        fn test_synapse_creation() {
            setup();
            let synapse = Synapse::new(0, 1, 0.5, 2);

            assert_eq!(synapse.source, 0);
            assert_eq!(synapse.target, 1);
            assert_eq!(synapse.weight, 0.5);
            assert_eq!(synapse.delay, 2);
        }

        #[test]
        fn test_synapse_with_id() {
            let synapse = Synapse::with_id(100, 0, 1, 0.5, 2);

            assert_eq!(synapse.id, 100);
            assert_eq!(synapse.source, 0);
            assert_eq!(synapse.target, 1);
        }

        #[test]
        fn test_synapse_excitatory_inhibitory() {
            setup();
            let excitatory = Synapse::new(0, 1, 0.5, 1);
            let inhibitory = Synapse::new(1, 2, -0.5, 1);
            let neutral = Synapse::new(2, 3, 0.0, 1);

            assert!(excitatory.is_excitatory());
            assert!(!excitatory.is_inhibitory());

            assert!(!inhibitory.is_excitatory());
            assert!(inhibitory.is_inhibitory());

            assert!(!neutral.is_excitatory());
            assert!(!neutral.is_inhibitory());
        }

        #[test]
        fn test_synapse_stimulate() {
            setup();
            let synapse = Synapse::new(0, 1, 0.5, 1);

            let stim = synapse.stimulate(2.0);
            assert_relative_eq!(stim, 1.0, epsilon = 1e-10);
        }

        #[test]
        fn test_synapse_serialization() {
            setup();
            let synapse = Synapse::new(0, 1, 0.75, 3);

            let json = serde_json::to_string(&synapse).unwrap();
            let deserialized: Synapse = serde_json::from_str(&json).unwrap();

            assert_eq!(synapse.id, deserialized.id);
            assert_eq!(synapse.source, deserialized.source);
            assert_eq!(synapse.target, deserialized.target);
            assert_eq!(synapse.weight, deserialized.weight);
            assert_eq!(synapse.delay, deserialized.delay);
        }
    }
}
