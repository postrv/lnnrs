// src/training.rs
//! Training algorithms for Liquid Neural Networks.
//!
//! This module provides training capabilities including:
//! - Backpropagation Through Time (BPTT) for sequence learning
//! - Gradient computation with automatic differentiation
//! - Trainable RNN cells with SGD+Momentum optimization
//!
//! # Example
//!
//! ```rust
//! use liquid_neural_network::training::*;
//!
//! // Create a trainable RNN
//! let mut rnn = TrainableRnnBuilder::new(2, 8, 1)
//!     .learning_rate(0.01)
//!     .truncation_length(50)
//!     .build();
//!
//! // Train on a sequence
//! let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
//! let targets = vec![vec![0.5], vec![0.5]];
//! let loss = rnn.train_step(&inputs, &targets);
//! ```

use crate::activation::tanh;
use crate::layers::DenseLayer;
use crate::optim::{clip_grad_norm, Optimizer, Sgd};
use ndarray::{Array1, Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Configuration for BPTT training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpttConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Truncation length (how many timesteps to backprop through)
    pub truncation_length: usize,
    /// Global-norm gradient-clipping threshold.
    ///
    /// When `Some(max_norm)`, all of a model's gradient tensors are rescaled jointly (see
    /// [`clip_grad_norm`]) so their combined L2 norm does not
    /// exceed `max_norm`, applied once before the optimizer step. `None` disables clipping.
    /// (This replaces the earlier per-element gradient clamping; the standalone
    /// [`TrainableDense::update`] still clamps per element for backward compatibility.)
    pub gradient_clip: Option<f64>,
    /// L2 regularization coefficient
    pub l2_lambda: f64,
    /// Momentum coefficient
    pub momentum: f64,
    /// Optional seed for reproducible weight initialization. `Some(seed)` seeds a `StdRng`
    /// with `seed`; `None` seeds from entropy. Consumed by [`TrainableRnnBuilder::build`]
    /// and [`TrainableRnn::with_seed`].
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Default for BpttConfig {
    fn default() -> Self {
        BpttConfig {
            learning_rate: 0.001,
            truncation_length: 50,
            gradient_clip: Some(1.0),
            l2_lambda: 1e-4,
            momentum: 0.9,
            seed: None,
        }
    }
}

impl BpttConfig {
    /// Creates a new BPTT config with specified learning rate.
    pub fn new(learning_rate: f64) -> Self {
        BpttConfig {
            learning_rate,
            ..Default::default()
        }
    }

    /// Sets the truncation length.
    pub fn with_truncation(mut self, length: usize) -> Self {
        self.truncation_length = length;
        self
    }

    /// Sets gradient clipping threshold.
    pub fn with_gradient_clip(mut self, clip: f64) -> Self {
        self.gradient_clip = Some(clip);
        self
    }

    /// Disables gradient clipping.
    pub fn without_gradient_clip(mut self) -> Self {
        self.gradient_clip = None;
        self
    }

    /// Sets L2 regularization coefficient.
    pub fn with_l2_lambda(mut self, lambda: f64) -> Self {
        self.l2_lambda = lambda;
        self
    }

    /// Sets momentum coefficient.
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the seed for reproducible weight initialization.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Simple dense layer with gradients for training.
///
/// Wraps the shared [`DenseLayer`] (parameters + Xavier init + `.dot()` forward) and adds the
/// gradient and SGD-momentum buffers needed for BPTT. Weights are stored as
/// `[output_dim, input_dim]` (the `DenseLayer` convention), so the accumulated gradient and
/// momentum tensors share that shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableDense {
    /// Parameters (weights `[output_dim, input_dim]`, bias `[output_dim]`).
    layer: DenseLayer,
    /// Weight gradients (accumulated), shape `[output_dim, input_dim]`.
    grad_weights: Array2<f64>,
    /// Bias gradients (accumulated), shape `[output_dim]`.
    grad_bias: Array1<f64>,
    /// Weight momentum, shape `[output_dim, input_dim]`.
    momentum_weights: Array2<f64>,
    /// Bias momentum, shape `[output_dim]`.
    momentum_bias: Array1<f64>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
}

impl TrainableDense {
    /// Creates a new trainable dense layer with random initialization.
    ///
    /// Self-contained convenience constructor: it seeds its own [`StdRng`] from entropy.
    /// Seeded/reproducible construction goes through [`TrainableDense::with_rng`] so a whole
    /// network can be derived from a single generator.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = StdRng::from_os_rng();
        Self::with_rng(input_dim, output_dim, &mut rng)
    }

    /// Creates a new trainable dense layer drawing weights from the supplied generator.
    pub fn with_rng(input_dim: usize, output_dim: usize, rng: &mut StdRng) -> Self {
        TrainableDense {
            layer: DenseLayer::new(input_dim, output_dim, rng),
            grad_weights: Array2::zeros((output_dim, input_dim)),
            grad_bias: Array1::zeros(output_dim),
            momentum_weights: Array2::zeros((output_dim, input_dim)),
            momentum_bias: Array1::zeros(output_dim),
            input_dim,
            output_dim,
        }
    }

    /// Returns a reference to the weight matrix (`[output_dim, input_dim]`).
    pub fn weights(&self) -> &Array2<f64> {
        &self.layer.weights
    }

    /// Returns a reference to the bias vector (`[output_dim]`).
    pub fn bias(&self) -> &Array1<f64> {
        &self.layer.bias
    }

    /// Forward pass: `output = W x + b`.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.layer.forward(input)
    }

    /// Backward pass: computes gradients and returns input gradient.
    ///
    /// # Arguments
    /// * `input` - Input from forward pass
    /// * `grad_output` - Gradient w.r.t. output
    ///
    /// # Returns
    /// Gradient w.r.t. input
    pub fn backward(&mut self, input: &[f64], grad_output: &[f64]) -> Vec<f64> {
        // Accumulate parameter gradients: dL/dW[j][i] = grad_output[j] * input[i].
        for (j, &go) in grad_output.iter().enumerate() {
            self.grad_bias[j] += go;
            for (i, &xi) in input.iter().enumerate() {
                self.grad_weights[[j, i]] += xi * go;
            }
        }

        // Input gradient: dL/dx = W^T @ grad_output.
        self.layer
            .weights
            .t()
            .dot(&ArrayView1::from(grad_output))
            .to_vec()
    }

    /// Applies accumulated gradients using SGD with momentum.
    pub fn update(&mut self, config: &BpttConfig) {
        for j in 0..self.output_dim {
            for i in 0..self.input_dim {
                // Add L2 regularization.
                let grad =
                    self.grad_weights[[j, i]] + config.l2_lambda * self.layer.weights[[j, i]];

                // Clip gradient.
                let clipped_grad = match config.gradient_clip {
                    Some(clip) => grad.clamp(-clip, clip),
                    None => grad,
                };

                // Apply momentum.
                self.momentum_weights[[j, i]] = config.momentum * self.momentum_weights[[j, i]]
                    + config.learning_rate * clipped_grad;

                // Update weight.
                self.layer.weights[[j, i]] -= self.momentum_weights[[j, i]];
            }
        }

        for j in 0..self.output_dim {
            let grad = match config.gradient_clip {
                Some(clip) => self.grad_bias[j].clamp(-clip, clip),
                None => self.grad_bias[j],
            };

            self.momentum_bias[j] =
                config.momentum * self.momentum_bias[j] + config.learning_rate * grad;

            self.layer.bias[j] -= self.momentum_bias[j];
        }
    }

    /// Zeros accumulated gradients.
    pub fn zero_grad(&mut self) {
        self.grad_weights.fill(0.0);
        self.grad_bias.fill(0.0);
    }

    /// Folds L2 regularization into the accumulated weight gradient
    /// (`grad_weights += lambda * weights`); biases are left unregularized.
    ///
    /// Used by optimizer-driven training paths that clip and step outside of
    /// [`Self::update`], so the L2 term is applied consistently before global-norm clipping.
    pub fn fold_l2(&mut self, lambda: f64) {
        if lambda != 0.0 {
            self.grad_weights.scaled_add(lambda, &self.layer.weights);
        }
    }
}

/// A trainable simple RNN cell for demonstrating BPTT.
///
/// This is a basic tanh-RNN that can be trained using backpropagation through time.
/// State equation: h_t = tanh(W_in * x_t + W_rec * h_{t-1} + b)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableRnnCell {
    /// Input weights
    pub w_in: TrainableDense,
    /// Recurrent weights
    pub w_rec: TrainableDense,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl TrainableRnnCell {
    /// Creates a new trainable RNN cell (RNG seeded from entropy).
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = StdRng::from_os_rng();
        Self::with_rng(input_dim, hidden_dim, &mut rng)
    }

    /// Creates a new trainable RNN cell drawing all weights from the supplied generator.
    ///
    /// The input and recurrent weight matrices are drawn in that fixed order from `rng`.
    pub fn with_rng(input_dim: usize, hidden_dim: usize, rng: &mut StdRng) -> Self {
        TrainableRnnCell {
            w_in: TrainableDense::with_rng(input_dim, hidden_dim, rng),
            w_rec: TrainableDense::with_rng(hidden_dim, hidden_dim, rng),
            hidden_dim,
        }
    }

    /// Forward pass for one timestep.
    ///
    /// Returns (new_state, pre_activation) for use in backward pass.
    pub fn forward(&self, input: &[f64], state: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let in_contrib = self.w_in.forward(input);
        let rec_contrib = self.w_rec.forward(state);

        let pre_activation: Vec<f64> = in_contrib
            .iter()
            .zip(rec_contrib.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let new_state: Vec<f64> = pre_activation.iter().map(|&x| tanh(x)).collect();

        (new_state, pre_activation)
    }

    /// Backward pass for one timestep.
    ///
    /// # Arguments
    /// * `grad_state` - Gradient w.r.t. output state
    /// * `input` - Input from forward pass
    /// * `prev_state` - Previous state from forward pass
    /// * `pre_activation` - Pre-activation from forward pass
    ///
    /// # Returns
    /// (grad_input, grad_prev_state)
    pub fn backward(
        &mut self,
        grad_state: &[f64],
        input: &[f64],
        prev_state: &[f64],
        pre_activation: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        // Gradient through tanh: d_tanh/dx = 1 - tanh^2(x)
        let grad_pre_act: Vec<f64> = grad_state
            .iter()
            .zip(pre_activation.iter())
            .map(|(&g, &z)| {
                let t = tanh(z);
                g * (1.0 - t * t)
            })
            .collect();

        // Backward through w_in and w_rec
        let grad_input = self.w_in.backward(input, &grad_pre_act);
        let grad_prev_state = self.w_rec.backward(prev_state, &grad_pre_act);

        (grad_input, grad_prev_state)
    }

    /// Updates weights.
    pub fn update(&mut self, config: &BpttConfig) {
        self.w_in.update(config);
        self.w_rec.update(config);
    }

    /// Zeros gradients.
    pub fn zero_grad(&mut self) {
        self.w_in.zero_grad();
        self.w_rec.zero_grad();
    }

    /// Returns initial state (zeros).
    pub fn initial_state(&self) -> Vec<f64> {
        vec![0.0; self.hidden_dim]
    }
}

/// A complete trainable RNN for sequence-to-sequence learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableRnn {
    /// RNN cell
    pub cell: TrainableRnnCell,
    /// Output projection
    pub output: TrainableDense,
    /// Training configuration
    pub config: BpttConfig,
    /// SGD+momentum optimizer (state is not serialized).
    #[serde(skip)]
    optimizer: Sgd,
    /// Cached states for backward pass
    #[serde(skip)]
    cached_states: Vec<Vec<f64>>,
    /// Cached pre-activations for backward pass
    #[serde(skip)]
    cached_pre_acts: Vec<Vec<f64>>,
    /// Cached inputs for backward pass
    #[serde(skip)]
    cached_inputs: Vec<Vec<f64>>,
}

impl TrainableRnn {
    /// Creates a new trainable RNN with weights seeded from entropy.
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self::with_seed(input_dim, hidden_dim, output_dim, None)
    }

    /// Creates a new trainable RNN with reproducible weight initialization.
    ///
    /// A single [`StdRng`] (seeded from `seed`, or from entropy when `None`) is threaded
    /// through the recurrent cell and then the output projection, so a given seed produces a
    /// byte-identical network. The seed is also recorded in [`Self::config`] (`config.seed`)
    /// so the resulting network is self-describing.
    pub fn with_seed(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        let config = BpttConfig {
            seed,
            ..BpttConfig::default()
        };
        let optimizer = Sgd::new(config.learning_rate, config.momentum);
        TrainableRnn {
            cell: TrainableRnnCell::with_rng(input_dim, hidden_dim, &mut rng),
            output: TrainableDense::with_rng(hidden_dim, output_dim, &mut rng),
            config,
            optimizer,
            cached_states: Vec::new(),
            cached_pre_acts: Vec::new(),
            cached_inputs: Vec::new(),
        }
    }

    /// Sets the training configuration.
    pub fn with_config(mut self, config: BpttConfig) -> Self {
        self.config = config;
        self
    }

    /// Forward pass over a sequence.
    ///
    /// Returns outputs and caches states for backward pass.
    pub fn forward(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.cached_states.clear();
        self.cached_pre_acts.clear();
        self.cached_inputs.clear();

        let mut state = self.cell.initial_state();
        self.cached_states.push(state.clone());

        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            self.cached_inputs.push(input.clone());

            let (new_state, pre_act) = self.cell.forward(input, &state);
            self.cached_states.push(new_state.clone());
            self.cached_pre_acts.push(pre_act);

            let output = self.output.forward(&new_state);
            outputs.push(output);

            state = new_state;
        }

        outputs
    }

    /// Backward pass using BPTT.
    ///
    /// # Arguments
    /// * `grad_outputs` - Gradients w.r.t. outputs (from loss)
    ///
    /// Accumulates gradients in the layer parameters.
    pub fn backward(&mut self, grad_outputs: &[Vec<f64>]) {
        let seq_len = grad_outputs.len();
        let truncation = self.config.truncation_length.min(seq_len);

        // Start from the last timestep
        let mut grad_state = vec![0.0; self.cell.hidden_dim];

        // Backward through time (truncated)
        for t in (seq_len.saturating_sub(truncation)..seq_len).rev() {
            // Gradient from output
            let grad_from_output = self
                .output
                .backward(&self.cached_states[t + 1], &grad_outputs[t]);

            // Add to accumulated state gradient
            for (gs, gfo) in grad_state.iter_mut().zip(grad_from_output.iter()) {
                *gs += *gfo;
            }

            // Backward through RNN cell
            let (_, grad_prev_state) = self.cell.backward(
                &grad_state,
                &self.cached_inputs[t],
                &self.cached_states[t],
                &self.cached_pre_acts[t],
            );

            grad_state = grad_prev_state;
        }
    }

    /// Performs one training step.
    ///
    /// # Arguments
    /// * `inputs` - Input sequence
    /// * `targets` - Target outputs
    ///
    /// # Returns
    /// Mean squared error loss
    pub fn train_step(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
        // Zero gradients
        self.cell.zero_grad();
        self.output.zero_grad();

        // Forward pass
        let outputs = self.forward(inputs);

        // Compute loss and gradients
        let (loss, grad_outputs) = mse_loss_with_grad(&outputs, targets);

        // Backward pass
        self.backward(&grad_outputs);

        // Fold L2 regularization into weight gradients (matches the standalone update()).
        let l2 = self.config.l2_lambda;
        self.cell.w_in.fold_l2(l2);
        self.cell.w_rec.fold_l2(l2);
        self.output.fold_l2(l2);

        // Global-norm gradient clipping across every tensor before the optimizer step.
        if let Some(max_norm) = self.config.gradient_clip {
            let mut grads: Vec<&mut [f64]> = vec![
                self.cell.w_in.grad_weights.as_slice_mut().unwrap(),
                self.cell.w_in.grad_bias.as_slice_mut().unwrap(),
                self.cell.w_rec.grad_weights.as_slice_mut().unwrap(),
                self.cell.w_rec.grad_bias.as_slice_mut().unwrap(),
                self.output.grad_weights.as_slice_mut().unwrap(),
                self.output.grad_bias.as_slice_mut().unwrap(),
            ];
            clip_grad_norm(&mut grads, max_norm);
        }

        // Update weights through the Optimizer trait (SGD + momentum by default).
        self.optimizer.set_learning_rate(self.config.learning_rate);
        self.optimizer.set_momentum(self.config.momentum);
        let mut group = 0usize;
        sgd_step_dense(&mut self.optimizer, &mut self.cell.w_in, &mut group);
        sgd_step_dense(&mut self.optimizer, &mut self.cell.w_rec, &mut group);
        sgd_step_dense(&mut self.optimizer, &mut self.output, &mut group);

        loss
    }

    /// Runs inference on a sequence (no caching).
    pub fn predict(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut state = self.cell.initial_state();
        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            let (new_state, _) = self.cell.forward(input, &state);
            let output = self.output.forward(&new_state);
            outputs.push(output);
            state = new_state;
        }

        outputs
    }

    /// Returns the output dimension.
    pub fn output_dim(&self) -> usize {
        self.output.output_dim
    }

    /// Returns the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.cell.hidden_dim
    }
}

/// Applies one SGD step to a trainable dense layer's weight and bias tensors via the
/// [`Optimizer`] trait, advancing the `param_group` counter for each tensor.
fn sgd_step_dense(opt: &mut Sgd, dense: &mut TrainableDense, group: &mut usize) {
    opt.step(
        dense
            .layer
            .weights
            .as_slice_mut()
            .expect("contiguous weights"),
        dense.grad_weights.as_slice().expect("contiguous grad"),
        *group,
    );
    *group += 1;
    opt.step(
        dense.layer.bias.as_slice_mut().expect("contiguous bias"),
        dense.grad_bias.as_slice().expect("contiguous grad"),
        *group,
    );
    *group += 1;
}

/// Computes MSE loss and its gradient w.r.t. the predictions.
///
/// The loss and gradient share a single normalization factor `n_timesteps * n_dims` so
/// that the returned gradient is exactly the derivative of the returned loss:
///
/// ```text
/// loss = sum(diff^2) / (n_timesteps * n_dims)
/// grad = 2 * diff    / (n_timesteps * n_dims)
/// ```
///
/// where `diff = prediction - target`, `n_timesteps = predictions.len()`, and `n_dims` is
/// the width of a timestep. Empty input (no timesteps or zero-width timesteps) yields a
/// zero loss and per-timestep zero gradients rather than dividing by zero.
///
/// # Arguments
/// * `predictions` - Network outputs
/// * `targets` - Ground truth
///
/// # Returns
/// (loss, gradients w.r.t. predictions)
pub fn mse_loss_with_grad(predictions: &[Vec<f64>], targets: &[Vec<f64>]) -> (f64, Vec<Vec<f64>>) {
    let n_timesteps = predictions.len();
    let n_dims = predictions.first().map_or(0, |p| p.len());
    let denom = (n_timesteps * n_dims) as f64;

    // Graceful empty input: no loss, and a zero gradient matching each timestep's shape.
    if denom == 0.0 {
        return (
            0.0,
            predictions.iter().map(|p| vec![0.0; p.len()]).collect(),
        );
    }

    let mut total_loss = 0.0;
    let mut grad_outputs = Vec::with_capacity(n_timesteps);

    for (pred, target) in predictions.iter().zip(targets.iter()) {
        let mut grad = Vec::with_capacity(pred.len());

        for (&p, &t) in pred.iter().zip(target.iter()) {
            let diff = p - t;
            total_loss += diff * diff;
            grad.push(2.0 * diff / denom);
        }

        grad_outputs.push(grad);
    }

    (total_loss / denom, grad_outputs)
}

/// Builder for trainable RNN networks.
pub struct TrainableRnnBuilder {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    config: BpttConfig,
}

impl TrainableRnnBuilder {
    /// Creates a new builder.
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        TrainableRnnBuilder {
            input_dim,
            hidden_dim,
            output_dim,
            config: BpttConfig::default(),
        }
    }

    /// Sets the learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Sets the truncation length.
    pub fn truncation_length(mut self, length: usize) -> Self {
        self.config.truncation_length = length;
        self
    }

    /// Sets gradient clipping.
    pub fn gradient_clip(mut self, clip: f64) -> Self {
        self.config.gradient_clip = Some(clip);
        self
    }

    /// Sets momentum.
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.config.momentum = momentum;
        self
    }

    /// Sets the seed for reproducible weight initialization.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Builds the trainable RNN.
    ///
    /// Weights are initialized from the builder's [`BpttConfig::seed`] (via a single
    /// generator), so two builders with the same seed and shape produce identical networks.
    pub fn build(self) -> TrainableRnn {
        TrainableRnn::with_seed(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            self.config.seed,
        )
        .with_config(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bptt_config_default() {
        let config = BpttConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.truncation_length, 50);
        assert!(config.gradient_clip.is_some());
    }

    #[test]
    fn test_bptt_config_builder() {
        let config = BpttConfig::new(0.01)
            .with_truncation(100)
            .with_gradient_clip(0.5)
            .with_momentum(0.95);

        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.truncation_length, 100);
        assert_eq!(config.gradient_clip, Some(0.5));
        assert_eq!(config.momentum, 0.95);
    }

    #[test]
    fn test_trainable_dense_forward() {
        let layer = TrainableDense::new(3, 2);
        let output = layer.forward(&[1.0, 0.5, 0.2]);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_trainable_dense_backward() {
        let mut layer = TrainableDense::new(3, 2);
        let input = vec![1.0, 0.5, 0.2];
        let grad_output = vec![0.1, 0.2];

        let grad_input = layer.backward(&input, &grad_output);

        assert_eq!(grad_input.len(), 3);
    }

    #[test]
    fn test_trainable_rnn_cell() {
        let cell = TrainableRnnCell::new(2, 4);
        let state = cell.initial_state();
        let input = vec![1.0, 0.5];

        let (new_state, pre_act) = cell.forward(&input, &state);

        assert_eq!(new_state.len(), 4);
        assert_eq!(pre_act.len(), 4);
    }

    #[test]
    fn test_trainable_rnn_forward() {
        let mut rnn = TrainableRnn::new(2, 4, 1);
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        let outputs = rnn.forward(&inputs);

        assert_eq!(outputs.len(), 3);
        for output in &outputs {
            assert_eq!(output.len(), 1);
        }
    }

    #[test]
    fn test_trainable_rnn_train_step() {
        let mut rnn = TrainableRnn::new(2, 8, 1).with_config(BpttConfig::new(0.01));

        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let targets = vec![vec![0.5], vec![0.5], vec![1.0]];

        let loss1 = rnn.train_step(&inputs, &targets);
        let loss2 = rnn.train_step(&inputs, &targets);
        let loss3 = rnn.train_step(&inputs, &targets);

        // Loss should generally decrease (though not guaranteed with small steps)
        // At least verify it's finite
        assert!(loss1.is_finite());
        assert!(loss2.is_finite());
        assert!(loss3.is_finite());
    }

    #[test]
    fn test_mse_loss_with_grad() {
        let predictions = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let targets = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let (loss, grads) = mse_loss_with_grad(&predictions, &targets);

        // Perfect predictions should have zero loss
        assert_relative_eq!(loss, 0.0, epsilon = 1e-10);
        for grad in &grads {
            for &g in grad {
                assert_relative_eq!(g, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_mse_loss_with_grad_hand_computed() {
        // 2 timesteps x 2 dims => denom = n_timesteps * n_dims = 4.
        let predictions = vec![vec![1.0, 3.0], vec![5.0, 2.0]];
        let targets = vec![vec![0.0, 1.0], vec![2.0, 2.0]];
        // diffs: [1, 2], [3, 0]; sum(diff^2) = 1 + 4 + 9 + 0 = 14 => loss = 14 / 4 = 3.5.
        let (loss, grads) = mse_loss_with_grad(&predictions, &targets);
        assert_relative_eq!(loss, 3.5, epsilon = 1e-12);

        // grad = 2 * diff / 4 = diff / 2.
        let expected = [[0.5, 1.0], [1.5, 0.0]];
        for (grad, exp) in grads.iter().zip(expected.iter()) {
            for (&g, &e) in grad.iter().zip(exp.iter()) {
                assert_relative_eq!(g, e, epsilon = 1e-12);
            }
        }

        // Loss must be exactly the mean of grad-weighted diffs: gradient is the analytic
        // derivative of the loss (both divided by the same denom).
        let recomputed: f64 = predictions
            .iter()
            .zip(targets.iter())
            .flat_map(|(p, t)| p.iter().zip(t.iter()))
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f64>()
            / 4.0;
        assert_relative_eq!(loss, recomputed, epsilon = 1e-12);

        // Empty input is handled gracefully: zero loss, no gradients.
        let (empty_loss, empty_grads) = mse_loss_with_grad(&[], &[]);
        assert_eq!(empty_loss, 0.0);
        assert!(empty_grads.is_empty());
    }

    #[test]
    fn test_trainable_rnn_builder() {
        let rnn = TrainableRnnBuilder::new(2, 8, 1)
            .learning_rate(0.001)
            .truncation_length(30)
            .gradient_clip(0.5)
            .momentum(0.9)
            .build();

        assert_eq!(rnn.hidden_dim(), 8);
        assert_eq!(rnn.output_dim(), 1);
        assert_eq!(rnn.config.truncation_length, 30);
    }

    #[test]
    fn test_training_reduces_loss_over_epochs() {
        let mut rnn = TrainableRnnBuilder::new(2, 16, 1)
            .learning_rate(0.1)
            .truncation_length(10)
            .build();

        // Simple task: output the sum of inputs
        let inputs: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![(i as f64 % 3.0) / 3.0, ((i + 1) as f64 % 3.0) / 3.0])
            .collect();
        let targets: Vec<Vec<f64>> = inputs
            .iter()
            .map(|inp| vec![(inp[0] + inp[1]) / 2.0])
            .collect();

        let initial_loss = rnn.train_step(&inputs, &targets);

        // Train for multiple epochs
        let mut final_loss = initial_loss;
        for _ in 0..50 {
            final_loss = rnn.train_step(&inputs, &targets);
        }

        // Loss should decrease
        assert!(
            final_loss < initial_loss,
            "Loss didn't decrease: {} -> {}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_trainable_rnn_seed_reproducible() {
        // Same seed → byte-identical serialized network (one RNG threaded through the cell's
        // two weight matrices and the output projection).
        let a = TrainableRnnBuilder::new(2, 8, 1).seed(2024).build();
        let b = TrainableRnnBuilder::new(2, 8, 1).seed(2024).build();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
        );
    }

    #[test]
    fn test_trainable_rnn_seed_differs() {
        let a = TrainableRnnBuilder::new(2, 8, 1).seed(2024).build();
        let b = TrainableRnnBuilder::new(2, 8, 1).seed(2025).build();
        assert_ne!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
        );
    }

    #[test]
    fn test_trainable_rnn_with_seed_records_seed() {
        // The seeded constructor records the seed in its config so the network is
        // self-describing.
        let rnn = TrainableRnn::with_seed(2, 4, 1, Some(11));
        assert_eq!(rnn.config.seed, Some(11));
        // Unseeded construction leaves the seed unset.
        assert_eq!(TrainableRnn::new(2, 4, 1).config.seed, None);
    }

    #[test]
    fn test_trainable_rnn_serialization() {
        let rnn = TrainableRnn::new(2, 4, 1);

        let json = serde_json::to_string(&rnn).unwrap();
        let restored: TrainableRnn = serde_json::from_str(&json).unwrap();

        assert_eq!(rnn.hidden_dim(), restored.hidden_dim());
        assert_eq!(rnn.output_dim(), restored.output_dim());
    }

    #[test]
    fn test_gradient_clipping() {
        let mut layer = TrainableDense::new(2, 2);
        let input = vec![100.0, 100.0]; // Large inputs
        let grad_output = vec![100.0, 100.0]; // Large gradients

        layer.backward(&input, &grad_output);

        let config = BpttConfig::new(0.1).with_gradient_clip(1.0);
        let original_weights = layer.weights().clone();
        layer.update(&config);

        // Weights should have changed but not by huge amounts due to clipping
        for (w, orig_w) in layer.weights().iter().zip(original_weights.iter()) {
            let diff = (w - orig_w).abs();
            assert!(
                diff < 1.0,
                "Weight change {} exceeds expected with clipping",
                diff
            );
        }
    }
}
