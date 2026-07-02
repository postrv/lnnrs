// src/train_cfc.rs
//! Trainable Closed-form Continuous-time (CfC) network with Backpropagation Through Time.
//!
//! [`TrainableCfc`] mirrors the architecture of [`crate::cfc::CfcCell`] in its **gated**
//! mode -- a shared backbone MLP feeding four heads (`ff1`, `ff2`, `time_a`, `time_b`) that
//! drive the closed-form gated update -- plus a trainable output projection. The forward
//! pass caches every per-timestep intermediate needed to differentiate the update
//! analytically, and [`TrainableCfc::backward`] implements truncated BPTT through the
//! closed-form dynamics. Parameters are updated with the [`Adam`] optimizer after
//! global-norm gradient clipping.
//!
//! Only the gated CfC is trainable here; the decay (`no_gate`) variant is rejected at
//! construction time.
//!
//! # Gated update and its gradients
//!
//! With backbone features `z = backbone([state, input])` (the backbone's final layer is
//! linear), the per-unit update is
//!
//! ```text
//! ff1      = tanh(W1 z + b1)
//! ff2      = tanh(W2 z + b2)
//! t_a      = Wa z + ba,   t_b = Wb z + bb
//! t_interp = sigmoid(t_a * dt + t_b)
//! state'   = ff1 * (1 - t_interp) + t_interp * ff2
//! output   = W_out state' + b_out
//! ```
//!
//! Given `g = dL/dstate'` (the sum of the output-projection gradient at this step and the
//! state-slice of the next step's backbone-input gradient), the backward pass computes
//!
//! ```text
//! dL/dff1      = g * (1 - t_interp)              dL/dff2 = g * t_interp
//! dL/dt_interp = g * (ff2 - ff1)
//! dL/d(ff1_pre) = dL/dff1 * (1 - ff1^2)          (tanh)
//! dL/d(ff2_pre) = dL/dff2 * (1 - ff2^2)          (tanh)
//! dL/ds        = dL/dt_interp * t_interp * (1 - t_interp)   (sigmoid, s = t_a*dt + t_b)
//! dL/dt_a      = dL/ds * dt                       dL/dt_b = dL/ds
//! ```
//!
//! These flow through the four heads into a summed feature gradient `dL/dz`, then back
//! through the backbone MLP (using [`BackboneActivation::derivative`]), and finally the
//! combined-input gradient is split into a state-gradient (propagated to `t-1`) and an
//! input-gradient (discarded).
//!
//! [`BackboneActivation::derivative`]: crate::cfc::BackboneActivation::derivative

use crate::activation::{sigmoid, tanh};
use crate::cfc::CfcConfig;
use crate::error::{LnnError, Result};
use crate::layers::DenseLayer;
use crate::optim::{clip_grad_norm, Adam, Optimizer};
use crate::training::{mse_loss_with_grad, BpttConfig};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Per-timestep forward cache used by [`TrainableCfc::backward`].
///
/// Stores exactly the intermediates the analytic gradients require. `tanh`/`sigmoid`
/// derivatives are expressed in terms of the cached *post-activation* values
/// (`1 - ff^2`, `t_interp * (1 - t_interp)`), so the pre-activation head logits do not need
/// to be cached.
#[derive(Debug, Clone)]
struct StepCache {
    /// Input to each backbone layer; `backbone_inputs[0]` is the combined `[state, input]`.
    backbone_inputs: Vec<Array1<f64>>,
    /// Pre-activation of each backbone layer; the last entry is the linear features `z`.
    backbone_pre: Vec<Array1<f64>>,
    /// `ff1 = tanh(W1 z + b1)`.
    ff1: Array1<f64>,
    /// `ff2 = tanh(W2 z + b2)`.
    ff2: Array1<f64>,
    /// `t_interp = sigmoid(t_a * dt + t_b)`.
    t_interp: Array1<f64>,
    /// The new state (output of the gated update).
    state: Array1<f64>,
}

/// Accumulated gradients for every parameter tensor of a [`TrainableCfc`].
///
/// Shapes mirror the corresponding parameters. Reconstructed to zeros before each backward
/// pass, so this is deliberately excluded from serialization.
#[derive(Debug, Clone)]
struct CfcGrads {
    backbone_w: Vec<Array2<f64>>,
    backbone_b: Vec<Array1<f64>>,
    ff1_w: Array2<f64>,
    ff1_b: Array1<f64>,
    ff2_w: Array2<f64>,
    ff2_b: Array1<f64>,
    time_a_w: Array2<f64>,
    time_a_b: Array1<f64>,
    time_b_w: Array2<f64>,
    time_b_b: Array1<f64>,
    output_w: Array2<f64>,
    output_b: Array1<f64>,
}

impl Default for CfcGrads {
    fn default() -> Self {
        // Empty placeholder used only after deserialization; the real buffers are rebuilt
        // to the correct shapes by [`TrainableCfc::zero_grad`].
        CfcGrads {
            backbone_w: Vec::new(),
            backbone_b: Vec::new(),
            ff1_w: Array2::zeros((0, 0)),
            ff1_b: Array1::zeros(0),
            ff2_w: Array2::zeros((0, 0)),
            ff2_b: Array1::zeros(0),
            time_a_w: Array2::zeros((0, 0)),
            time_a_b: Array1::zeros(0),
            time_b_w: Array2::zeros((0, 0)),
            time_b_b: Array1::zeros(0),
            output_w: Array2::zeros((0, 0)),
            output_b: Array1::zeros(0),
        }
    }
}

impl CfcGrads {
    /// Allocates zero gradient buffers matching the shapes of `net`'s parameters.
    fn zeros_for(net: &TrainableCfc) -> Self {
        CfcGrads {
            backbone_w: net
                .backbone
                .iter()
                .map(|l| Array2::zeros(l.weights.dim()))
                .collect(),
            backbone_b: net
                .backbone
                .iter()
                .map(|l| Array1::zeros(l.bias.len()))
                .collect(),
            ff1_w: Array2::zeros(net.ff1.weights.dim()),
            ff1_b: Array1::zeros(net.ff1.bias.len()),
            ff2_w: Array2::zeros(net.ff2.weights.dim()),
            ff2_b: Array1::zeros(net.ff2.bias.len()),
            time_a_w: Array2::zeros(net.time_a.weights.dim()),
            time_a_b: Array1::zeros(net.time_a.bias.len()),
            time_b_w: Array2::zeros(net.time_b.weights.dim()),
            time_b_b: Array1::zeros(net.time_b.bias.len()),
            output_w: Array2::zeros(net.output_layer.weights.dim()),
            output_b: Array1::zeros(net.output_layer.bias.len()),
        }
    }

    /// Mutable flat views over every gradient tensor (for global-norm clipping).
    fn slices_mut(&mut self) -> Vec<&mut [f64]> {
        let mut v: Vec<&mut [f64]> = Vec::new();
        for w in self.backbone_w.iter_mut() {
            v.push(w.as_slice_mut().expect("contiguous grad"));
        }
        for b in self.backbone_b.iter_mut() {
            v.push(b.as_slice_mut().expect("contiguous grad"));
        }
        v.push(self.ff1_w.as_slice_mut().expect("contiguous grad"));
        v.push(self.ff1_b.as_slice_mut().expect("contiguous grad"));
        v.push(self.ff2_w.as_slice_mut().expect("contiguous grad"));
        v.push(self.ff2_b.as_slice_mut().expect("contiguous grad"));
        v.push(self.time_a_w.as_slice_mut().expect("contiguous grad"));
        v.push(self.time_a_b.as_slice_mut().expect("contiguous grad"));
        v.push(self.time_b_w.as_slice_mut().expect("contiguous grad"));
        v.push(self.time_b_b.as_slice_mut().expect("contiguous grad"));
        v.push(self.output_w.as_slice_mut().expect("contiguous grad"));
        v.push(self.output_b.as_slice_mut().expect("contiguous grad"));
        v
    }
}

/// A trainable gated CfC network for sequence-to-sequence learning.
///
/// Construct via [`TrainableCfc::new`], [`TrainableCfc::with_seed`], or
/// [`TrainableCfcBuilder`]. Only the gated CfC is supported; passing a `no_gate` config to
/// [`TrainableCfc::try_with_config`] returns an error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainableCfc {
    /// CfC architecture configuration (must be gated).
    pub config: CfcConfig,
    /// BPTT / optimization configuration (learning rate, truncation, gradient clip).
    pub bptt: BpttConfig,
    input_dim: usize,
    output_dim: usize,
    dt: f64,
    backbone: Vec<DenseLayer>,
    ff1: DenseLayer,
    ff2: DenseLayer,
    time_a: DenseLayer,
    time_b: DenseLayer,
    output_layer: DenseLayer,
    #[serde(skip)]
    grads: CfcGrads,
    #[serde(skip)]
    optimizer: Adam,
    #[serde(skip)]
    cache: Vec<StepCache>,
}

impl TrainableCfc {
    /// Creates a trainable CfC with default gated configuration (RNG seeded from entropy).
    pub fn new(input_dim: usize, hidden_units: usize, output_dim: usize) -> Self {
        Self::with_seed(input_dim, hidden_units, output_dim, None)
    }

    /// Creates a trainable CfC with reproducible weight initialization.
    ///
    /// A single [`StdRng`] (seeded from `seed`, or from entropy when `None`) is threaded
    /// through the backbone, the four heads, and the output projection in that fixed order,
    /// so a given seed yields a byte-identical network.
    pub fn with_seed(
        input_dim: usize,
        hidden_units: usize,
        output_dim: usize,
        seed: Option<u64>,
    ) -> Self {
        let mut config = CfcConfig::new(hidden_units);
        config.seed = seed;
        let bptt = BpttConfig {
            seed,
            ..BpttConfig::default()
        };
        Self::try_with_config(input_dim, output_dim, config, bptt)
            .expect("default CfcConfig is gated and valid")
    }

    /// Creates a trainable CfC from explicit configurations.
    ///
    /// # Errors
    /// Returns [`LnnError::Configuration`] if `config` fails [`CfcConfig::validate`] or if
    /// `config.no_gate` is `true` (only the gated CfC is trainable).
    pub fn try_with_config(
        input_dim: usize,
        output_dim: usize,
        config: CfcConfig,
        bptt: BpttConfig,
    ) -> Result<Self> {
        config.validate()?;
        if config.no_gate {
            return Err(LnnError::configuration(
                "TrainableCfc supports only the gated CfC (no_gate = false)",
            ));
        }

        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        let units = config.units;
        let state_input_dim = units + input_dim;

        // Backbone (same construction order as `cfc::Backbone`).
        let mut backbone = Vec::with_capacity(config.backbone_layers);
        backbone.push(DenseLayer::new(
            state_input_dim,
            config.backbone_units,
            &mut rng,
        ));
        for _ in 1..config.backbone_layers {
            backbone.push(DenseLayer::new(
                config.backbone_units,
                config.backbone_units,
                &mut rng,
            ));
        }
        let backbone_out = config.backbone_units;

        let ff1 = DenseLayer::new(backbone_out, units, &mut rng);
        let ff2 = DenseLayer::new(backbone_out, units, &mut rng);
        let time_a = DenseLayer::new(backbone_out, units, &mut rng);
        let time_b = DenseLayer::new(backbone_out, units, &mut rng);
        let output_layer = DenseLayer::new(units, output_dim, &mut rng);

        let optimizer = Adam::new(bptt.learning_rate);

        let mut net = TrainableCfc {
            config,
            bptt,
            input_dim,
            output_dim,
            dt: 1.0,
            backbone,
            ff1,
            ff2,
            time_a,
            time_b,
            output_layer,
            grads: CfcGrads::default(),
            optimizer,
            cache: Vec::new(),
        };
        net.grads = CfcGrads::zeros_for(&net);
        Ok(net)
    }

    /// Sets the integration time step `dt` used by the interpolation gate.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the number of hidden units.
    pub fn hidden_dim(&self) -> usize {
        self.config.units
    }

    /// Returns the output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Returns the integration time step.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Runs a single forward step from `state`/`input`, returning the full cache and output.
    fn forward_step(&self, state: &Array1<f64>, input: &[f64]) -> (StepCache, Array1<f64>) {
        let units = self.config.units;
        let activation = self.config.backbone_activation;
        let num_layers = self.backbone.len();
        let last = num_layers - 1;

        // combined = [state; input].
        let mut combined = Array1::<f64>::zeros(units + input.len());
        for i in 0..units {
            combined[i] = state[i];
        }
        for (k, &v) in input.iter().enumerate() {
            combined[units + k] = v;
        }

        // Backbone: activation on every layer but the last (which is linear -> features z).
        let mut backbone_inputs = Vec::with_capacity(num_layers);
        let mut backbone_pre = Vec::with_capacity(num_layers);
        let mut x = combined;
        for (l, layer) in self.backbone.iter().enumerate() {
            backbone_inputs.push(x.clone());
            let pre = layer.forward_arr(x.view());
            if l < last {
                x = pre.mapv(|v| activation.apply(v));
            } else {
                x = pre.clone();
            }
            backbone_pre.push(pre);
        }
        let z = &backbone_pre[last];

        // Heads.
        let ff1 = self.ff1.forward_arr(z.view()).mapv(tanh);
        let ff2 = self.ff2.forward_arr(z.view()).mapv(tanh);
        let t_a = self.time_a.forward_arr(z.view());
        let t_b = self.time_b.forward_arr(z.view());

        let mut t_interp = Array1::<f64>::zeros(units);
        let mut new_state = Array1::<f64>::zeros(units);
        for i in 0..units {
            let ti = sigmoid(t_a[i] * self.dt + t_b[i]);
            t_interp[i] = ti;
            new_state[i] = ff1[i] * (1.0 - ti) + ti * ff2[i];
        }

        let output = self.output_layer.forward_arr(new_state.view());

        let cache = StepCache {
            backbone_inputs,
            backbone_pre,
            ff1,
            ff2,
            t_interp,
            state: new_state,
        };
        (cache, output)
    }

    /// Forward pass over a sequence, caching intermediates for [`Self::backward`].
    pub fn forward(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.cache.clear();
        self.cache.reserve(inputs.len());
        let mut state = Array1::<f64>::zeros(self.config.units);
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let (cache, output) = self.forward_step(&state, input);
            outputs.push(output.to_vec());
            state = cache.state.clone();
            self.cache.push(cache);
        }
        outputs
    }

    /// Runs inference on a sequence without caching (usable on a `&self` model).
    pub fn predict(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut state = Array1::<f64>::zeros(self.config.units);
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let (cache, output) = self.forward_step(&state, input);
            state = cache.state;
            outputs.push(output.to_vec());
        }
        outputs
    }

    /// Resets all gradient accumulators to zero (rebuilding buffers to the current shapes).
    pub fn zero_grad(&mut self) {
        self.grads = CfcGrads::zeros_for(self);
    }

    /// Backward pass (truncated BPTT) accumulating gradients from `grad_outputs`.
    ///
    /// `grad_outputs[t]` is `dL/doutput_t`. Gradients are accumulated for the last
    /// `min(truncation_length, seq_len)` timesteps.
    pub fn backward(&mut self, grad_outputs: &[Vec<f64>]) {
        let units = self.config.units;
        let num_layers = self.backbone.len();
        let last = num_layers - 1;
        let dt = self.dt;
        let activation = self.config.backbone_activation;

        let seq_len = self.cache.len();
        if seq_len == 0 {
            return;
        }
        let truncation = self.bptt.truncation_length.min(seq_len);
        let start = seq_len - truncation;

        // grad_state carries dL/dstate_t coming from step t+1 (the state slice of the next
        // step's backbone-input gradient). Starts at zero for the final timestep.
        let mut grad_state = Array1::<f64>::zeros(units);

        for t in (start..seq_len).rev() {
            let cache = &self.cache[t];
            let z = &cache.backbone_pre[last];

            // ---- Output projection ----
            let go = Array1::from(grad_outputs[t].clone());
            accumulate_grad(
                &mut self.grads.output_w,
                &mut self.grads.output_b,
                &go,
                &cache.state,
            );
            let grad_from_output = self.output_layer.weights.t().dot(&go);

            // Total gradient w.r.t. state_t.
            let mut g = Array1::<f64>::zeros(units);
            for i in 0..units {
                g[i] = grad_state[i] + grad_from_output[i];
            }

            // ---- Gated update backward ----
            let mut dff1_pre = Array1::<f64>::zeros(units);
            let mut dff2_pre = Array1::<f64>::zeros(units);
            let mut dt_a = Array1::<f64>::zeros(units);
            let mut dt_b = Array1::<f64>::zeros(units);
            for i in 0..units {
                let ti = cache.t_interp[i];
                let f1 = cache.ff1[i];
                let f2 = cache.ff2[i];
                let dff1 = g[i] * (1.0 - ti);
                let dff2 = g[i] * ti;
                let dinterp = g[i] * (f2 - f1);
                // tanh': 1 - tanh^2.
                dff1_pre[i] = dff1 * (1.0 - f1 * f1);
                dff2_pre[i] = dff2 * (1.0 - f2 * f2);
                // sigmoid': t_interp * (1 - t_interp); s = t_a * dt + t_b.
                let dsig = dinterp * ti * (1.0 - ti);
                dt_a[i] = dsig * dt;
                dt_b[i] = dsig;
            }

            // ---- Head weight gradients ----
            accumulate_grad(&mut self.grads.ff1_w, &mut self.grads.ff1_b, &dff1_pre, z);
            accumulate_grad(&mut self.grads.ff2_w, &mut self.grads.ff2_b, &dff2_pre, z);
            accumulate_grad(&mut self.grads.time_a_w, &mut self.grads.time_a_b, &dt_a, z);
            accumulate_grad(&mut self.grads.time_b_w, &mut self.grads.time_b_b, &dt_b, z);

            // ---- Grad w.r.t. backbone features z (sum over the four heads) ----
            let mut grad_z = self.ff1.weights.t().dot(&dff1_pre);
            grad_z += &self.ff2.weights.t().dot(&dff2_pre);
            grad_z += &self.time_a.weights.t().dot(&dt_a);
            grad_z += &self.time_b.weights.t().dot(&dt_b);

            // ---- Backbone backward ----
            let mut grad_pre = grad_z; // grad w.r.t. the last (linear) layer's pre-activation
            for l in (0..num_layers).rev() {
                let a_l = &cache.backbone_inputs[l];
                accumulate_grad(
                    &mut self.grads.backbone_w[l],
                    &mut self.grads.backbone_b[l],
                    &grad_pre,
                    a_l,
                );
                let grad_a = self.backbone[l].weights.t().dot(&grad_pre);
                if l > 0 {
                    // a_l = activation(pre_{l-1}); back through the activation.
                    let pre_prev = &cache.backbone_pre[l - 1];
                    let mut next = Array1::<f64>::zeros(pre_prev.len());
                    for k in 0..pre_prev.len() {
                        next[k] = grad_a[k] * activation.derivative(pre_prev[k]);
                    }
                    grad_pre = next;
                } else {
                    // l == 0: grad_a is w.r.t. combined [state_prev; input].
                    // State slice flows to t-1; input slice is discarded.
                    let mut next = Array1::<f64>::zeros(units);
                    for i in 0..units {
                        next[i] = grad_a[i];
                    }
                    grad_state = next;
                }
            }
        }
    }

    /// Zeroes gradients, runs forward + MSE loss + backward, and returns the loss.
    ///
    /// Leaves the analytic gradients populated (no optimizer step); used by
    /// [`Self::train_step`] and by the numerical gradient check.
    pub fn compute_gradients(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
        self.zero_grad();
        let outputs = self.forward(inputs);
        let (loss, grad_outputs) = mse_loss_with_grad(&outputs, targets);
        self.backward(&grad_outputs);
        loss
    }

    /// Performs one training step: forward, backward, global-norm clip, Adam update.
    ///
    /// Returns the MSE loss computed before the parameter update.
    pub fn train_step(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
        let loss = self.compute_gradients(inputs, targets);

        if let Some(max_norm) = self.bptt.gradient_clip {
            let mut slices = self.grads.slices_mut();
            clip_grad_norm(&mut slices, max_norm);
        }

        self.optimizer.set_learning_rate(self.bptt.learning_rate);
        self.adam_step();
        loss
    }

    /// Convenience: MSE loss of the current parameters on a sequence (no gradients).
    pub fn loss(&self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
        let outputs = self.predict(inputs);
        mse_loss_with_grad(&outputs, targets).0
    }

    /// Applies the Adam step to every parameter tensor in canonical registration order.
    fn adam_step(&mut self) {
        let mut group = 0usize;
        for l in 0..self.backbone.len() {
            step_dense(
                &mut self.optimizer,
                &mut self.backbone[l],
                &self.grads.backbone_w[l],
                &self.grads.backbone_b[l],
                &mut group,
            );
        }
        step_dense(
            &mut self.optimizer,
            &mut self.ff1,
            &self.grads.ff1_w,
            &self.grads.ff1_b,
            &mut group,
        );
        step_dense(
            &mut self.optimizer,
            &mut self.ff2,
            &self.grads.ff2_w,
            &self.grads.ff2_b,
            &mut group,
        );
        step_dense(
            &mut self.optimizer,
            &mut self.time_a,
            &self.grads.time_a_w,
            &self.grads.time_a_b,
            &mut group,
        );
        step_dense(
            &mut self.optimizer,
            &mut self.time_b,
            &self.grads.time_b_w,
            &self.grads.time_b_b,
            &mut group,
        );
        step_dense(
            &mut self.optimizer,
            &mut self.output_layer,
            &self.grads.output_w,
            &self.grads.output_b,
            &mut group,
        );
    }

    /// Test-only: mutable flat views over every parameter tensor in canonical order
    /// (backbone weight/bias per layer, then ff1, ff2, time_a, time_b, output).
    #[cfg(test)]
    fn parameter_slices_mut(&mut self) -> Vec<&mut [f64]> {
        let mut v: Vec<&mut [f64]> = Vec::new();
        for layer in self.backbone.iter_mut() {
            v.push(layer.weights.as_slice_mut().unwrap());
            v.push(layer.bias.as_slice_mut().unwrap());
        }
        v.push(self.ff1.weights.as_slice_mut().unwrap());
        v.push(self.ff1.bias.as_slice_mut().unwrap());
        v.push(self.ff2.weights.as_slice_mut().unwrap());
        v.push(self.ff2.bias.as_slice_mut().unwrap());
        v.push(self.time_a.weights.as_slice_mut().unwrap());
        v.push(self.time_a.bias.as_slice_mut().unwrap());
        v.push(self.time_b.weights.as_slice_mut().unwrap());
        v.push(self.time_b.bias.as_slice_mut().unwrap());
        v.push(self.output_layer.weights.as_slice_mut().unwrap());
        v.push(self.output_layer.bias.as_slice_mut().unwrap());
        v
    }

    /// Test-only: flattened copies of every gradient tensor, in the same canonical order
    /// as [`Self::parameter_slices_mut`].
    #[cfg(test)]
    fn gradient_tensors(&self) -> Vec<Vec<f64>> {
        let mut v: Vec<Vec<f64>> = Vec::new();
        for l in 0..self.backbone.len() {
            v.push(self.grads.backbone_w[l].iter().copied().collect());
            v.push(self.grads.backbone_b[l].iter().copied().collect());
        }
        v.push(self.grads.ff1_w.iter().copied().collect());
        v.push(self.grads.ff1_b.iter().copied().collect());
        v.push(self.grads.ff2_w.iter().copied().collect());
        v.push(self.grads.ff2_b.iter().copied().collect());
        v.push(self.grads.time_a_w.iter().copied().collect());
        v.push(self.grads.time_a_b.iter().copied().collect());
        v.push(self.grads.time_b_w.iter().copied().collect());
        v.push(self.grads.time_b_b.iter().copied().collect());
        v.push(self.grads.output_w.iter().copied().collect());
        v.push(self.grads.output_b.iter().copied().collect());
        v
    }
}

/// Accumulates `grad_w[j,i] += grad_out[j] * input[i]` and `grad_b[j] += grad_out[j]`.
fn accumulate_grad(
    grad_w: &mut Array2<f64>,
    grad_b: &mut Array1<f64>,
    grad_out: &Array1<f64>,
    input: &Array1<f64>,
) {
    for j in 0..grad_out.len() {
        let go = grad_out[j];
        grad_b[j] += go;
        let mut row = grad_w.row_mut(j);
        for (i, &xi) in input.iter().enumerate() {
            row[i] += go * xi;
        }
    }
}

/// Applies one Adam step to a dense layer's weight and bias tensors, advancing `group`.
fn step_dense(
    opt: &mut Adam,
    layer: &mut DenseLayer,
    grad_w: &Array2<f64>,
    grad_b: &Array1<f64>,
    group: &mut usize,
) {
    opt.step(
        layer.weights.as_slice_mut().expect("contiguous weights"),
        grad_w.as_slice().expect("contiguous grad"),
        *group,
    );
    *group += 1;
    opt.step(
        layer.bias.as_slice_mut().expect("contiguous bias"),
        grad_b.as_slice().expect("contiguous grad"),
        *group,
    );
    *group += 1;
}

/// Builder for [`TrainableCfc`], mirroring [`crate::training::TrainableRnnBuilder`].
pub struct TrainableCfcBuilder {
    input_dim: usize,
    output_dim: usize,
    config: CfcConfig,
    bptt: BpttConfig,
    dt: f64,
}

impl TrainableCfcBuilder {
    /// Creates a builder for a gated CfC with the given dimensions.
    pub fn new(input_dim: usize, hidden_units: usize, output_dim: usize) -> Self {
        let config = CfcConfig::new(hidden_units);
        TrainableCfcBuilder {
            input_dim,
            output_dim,
            config,
            bptt: BpttConfig::default(),
            dt: 1.0,
        }
    }

    /// Sets the learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.bptt.learning_rate = lr;
        self
    }

    /// Sets the truncation length for BPTT.
    pub fn truncation_length(mut self, length: usize) -> Self {
        self.bptt.truncation_length = length;
        self
    }

    /// Sets the global-norm gradient clipping threshold.
    pub fn gradient_clip(mut self, clip: f64) -> Self {
        self.bptt.gradient_clip = Some(clip);
        self
    }

    /// Disables gradient clipping.
    pub fn without_gradient_clip(mut self) -> Self {
        self.bptt.gradient_clip = None;
        self
    }

    /// Sets the backbone architecture (number of layers and units per layer).
    pub fn backbone(mut self, layers: usize, units: usize) -> Self {
        self.config.backbone_layers = layers;
        self.config.backbone_units = units;
        self
    }

    /// Sets the backbone activation function.
    pub fn backbone_activation(mut self, activation: crate::cfc::BackboneActivation) -> Self {
        self.config.backbone_activation = activation;
        self
    }

    /// Sets the integration time step.
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Sets the seed for reproducible weight initialization.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self.bptt.seed = Some(seed);
        self
    }

    /// Builds the trainable CfC.
    ///
    /// # Panics
    /// Panics with the validation message if the configuration is invalid (e.g. `no_gate`
    /// is set). Use [`try_build`](Self::try_build) for a non-panicking variant.
    pub fn build(self) -> TrainableCfc {
        self.try_build()
            .unwrap_or_else(|e| panic!("invalid TrainableCfc configuration: {e}"))
    }

    /// Builds the trainable CfC, returning an error on an invalid configuration.
    ///
    /// # Errors
    /// Returns the [`CfcConfig::validate`] error, or a configuration error if `no_gate` is
    /// set (only the gated CfC is trainable).
    pub fn try_build(self) -> Result<TrainableCfc> {
        Ok(
            TrainableCfc::try_with_config(self.input_dim, self.output_dim, self.config, self.bptt)?
                .with_dt(self.dt),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfc::BackboneActivation;

    /// Runs the central-difference gradient check for a given architecture and returns the
    /// maximum relative error observed across all parameter tensors.
    fn run_gradient_check(config: CfcConfig, seq_len: usize) -> f64 {
        let input_dim = 2;
        let output_dim = 1;
        // Full BPTT: truncation is larger than the sequence.
        let bptt = BpttConfig::default().with_truncation(seq_len + 10);

        let mut net = TrainableCfc::try_with_config(input_dim, output_dim, config, bptt).unwrap();

        // Deterministic, non-trivial inputs and targets.
        let inputs: Vec<Vec<f64>> = (0..seq_len)
            .map(|i| {
                let t = i as f64;
                vec![(0.3 * t).sin(), (0.2 * t + 0.5).cos()]
            })
            .collect();
        let targets: Vec<Vec<f64>> = (0..seq_len)
            .map(|i| vec![((i as f64) * 0.4).sin() * 0.5])
            .collect();

        // Analytic gradients.
        net.compute_gradients(&inputs, &targets);
        let analytic = net.gradient_tensors();

        let eps = 1e-5;
        let mut max_rel = 0.0f64;
        for ti in 0..analytic.len() {
            for k in 0..analytic[ti].len() {
                let orig = {
                    let p = net.parameter_slices_mut();
                    p[ti][k]
                };
                {
                    let mut p = net.parameter_slices_mut();
                    p[ti][k] = orig + eps;
                }
                let loss_plus = net.loss(&inputs, &targets);
                {
                    let mut p = net.parameter_slices_mut();
                    p[ti][k] = orig - eps;
                }
                let loss_minus = net.loss(&inputs, &targets);
                {
                    let mut p = net.parameter_slices_mut();
                    p[ti][k] = orig;
                }

                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let a = analytic[ti][k];
                let rel = (a - numeric).abs() / (a.abs() + numeric.abs() + 1e-8);
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 1e-4,
                    "gradient mismatch: tensor {} elem {} rel_err {} (analytic {}, numeric {})",
                    ti,
                    k,
                    rel,
                    a,
                    numeric
                );
            }
        }
        max_rel
    }

    #[test]
    fn test_gradient_check_single_backbone_layer() {
        // Tiny net: 2 inputs, 3 units, backbone 1 layer x 4 units, output 1, seq len 3.
        let config = CfcConfig::new(3)
            .with_backbone(1, 4)
            .with_backbone_activation(BackboneActivation::SiLU)
            .with_seed(1234);
        let max_rel = run_gradient_check(config, 3);
        assert!(max_rel < 1e-4, "max relative error {} too large", max_rel);
    }

    #[test]
    fn test_gradient_check_two_backbone_layers() {
        // Two backbone layers exercise the inter-layer activation derivative path.
        let config = CfcConfig::new(3)
            .with_backbone(2, 4)
            .with_backbone_activation(BackboneActivation::SiLU)
            .with_seed(77);
        let max_rel = run_gradient_check(config, 3);
        assert!(max_rel < 1e-4, "max relative error {} too large", max_rel);
    }

    #[test]
    fn test_gradient_check_gelu_backbone() {
        // Also verify the GELU-approx derivative path end-to-end.
        let config = CfcConfig::new(3)
            .with_backbone(2, 4)
            .with_backbone_activation(BackboneActivation::GELU)
            .with_seed(9);
        let max_rel = run_gradient_check(config, 4);
        assert!(max_rel < 1e-4, "max relative error {} too large", max_rel);
    }

    #[test]
    fn test_trainable_cfc_rejects_no_gate() {
        let config = CfcConfig::new(4).with_no_gate(true);
        let bptt = BpttConfig::default();
        let result = TrainableCfc::try_with_config(2, 1, config, bptt);
        assert!(result.is_err());
    }

    #[test]
    fn test_trainable_cfc_seed_reproducible() {
        let a = TrainableCfcBuilder::new(2, 8, 1).seed(42).build();
        let b = TrainableCfcBuilder::new(2, 8, 1).seed(42).build();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
        );
    }

    #[test]
    fn test_trainable_cfc_seed_differs() {
        let a = TrainableCfcBuilder::new(2, 8, 1).seed(42).build();
        let b = TrainableCfcBuilder::new(2, 8, 1).seed(43).build();
        assert_ne!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
        );
    }

    #[test]
    fn test_trainable_cfc_serialization_roundtrip() {
        // Optimizer state and caches are skipped; parameters and configs round-trip and
        // predictions match.
        let mut net = TrainableCfcBuilder::new(2, 6, 1).seed(5).build();
        let inputs = vec![vec![0.5, -0.2], vec![0.1, 0.3], vec![-0.4, 0.7]];
        let targets = vec![vec![0.2], vec![0.1], vec![-0.3]];
        // Take a step so the (skipped) optimizer state is non-trivial.
        net.train_step(&inputs, &targets);

        let json = serde_json::to_string(&net).unwrap();
        let restored: TrainableCfc = serde_json::from_str(&json).unwrap();

        assert_eq!(net.hidden_dim(), restored.hidden_dim());
        assert_eq!(net.output_dim(), restored.output_dim());
        let before = net.predict(&inputs);
        let after = restored.predict(&inputs);
        for (b, a) in before.iter().zip(after.iter()) {
            for (x, y) in b.iter().zip(a.iter()) {
                assert!((x - y).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_trainable_cfc_serialized_then_trained() {
        // After deserialization, gradient buffers are rebuilt lazily so train_step works.
        let net = TrainableCfcBuilder::new(2, 6, 1).seed(5).build();
        let json = serde_json::to_string(&net).unwrap();
        let mut restored: TrainableCfc = serde_json::from_str(&json).unwrap();
        let inputs = vec![vec![0.5, -0.2], vec![0.1, 0.3]];
        let targets = vec![vec![0.2], vec![0.1]];
        let loss = restored.train_step(&inputs, &targets);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_trainable_cfc_learns_sine_next_step() {
        // Seeded end-to-end learning test on the main.rs sine next-step task. The trained
        // MSE must beat the untrained MSE by more than 10x.
        let length = 200;
        let targets_seq: Vec<f64> = (0..length).map(|i| (i as f64 * 0.05).sin()).collect();
        let inputs: Vec<Vec<f64>> = (0..length)
            .map(|i| {
                let prev = if i > 0 { targets_seq[i - 1] } else { 0.0 };
                vec![prev, (i as f64 * 0.05).cos()]
            })
            .collect();
        let targets: Vec<Vec<f64>> = targets_seq.iter().map(|&t| vec![t]).collect();

        let epochs = 200;
        let mut net = TrainableCfcBuilder::new(2, 16, 1)
            .learning_rate(0.01)
            .truncation_length(40)
            .seed(42)
            .build();

        let untrained_mse = net.loss(&inputs, &targets);

        let mut last = untrained_mse;
        for _ in 0..epochs {
            last = net.train_step(&inputs, &targets);
        }
        let trained_mse = net.loss(&inputs, &targets);

        // With seed 42 this reaches roughly untrained 0.48 -> trained 0.020 (about 24x),
        // comfortably beating the required 10x margin.
        assert!(last.is_finite());
        assert!(
            trained_mse < untrained_mse / 10.0,
            "insufficient learning: untrained {} trained {}",
            untrained_mse,
            trained_mse
        );
    }
}
