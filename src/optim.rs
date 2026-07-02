// src/optim.rs
//! Gradient-based optimizers and gradient utilities.
//!
//! This module provides a small [`Optimizer`] abstraction plus two concrete
//! implementations used across the crate's trainable models:
//!
//! * [`Sgd`] -- stochastic gradient descent with (heavy-ball) momentum.
//! * [`Adam`] -- the bias-corrected Adam optimizer (Kingma & Ba, 2015) with the standard
//!   defaults `lr = 1e-3`, `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`.
//!
//! Both optimizers operate on flat `&mut [f64]` parameter views (e.g. an
//! [`ndarray::Array2`] weight matrix exposed via `as_slice_mut()`), and keep their
//! per-tensor state (momentum / moment estimates) keyed by a `param_group` index. The
//! caller is responsible for assigning each parameter tensor a stable `param_group` in
//! registration order and stepping every group exactly once per update, so that stateful
//! optimizers keep a consistent per-tensor history.
//!
//! Gradient clipping is provided as a separate [`clip_grad_norm`] utility that rescales a
//! model's gradient tensors jointly (global-norm clipping) *before* the optimizer step,
//! rather than per-element clamping inside the update.

use std::collections::HashMap;

/// A gradient-descent optimizer that updates parameters in place.
pub trait Optimizer {
    /// Updates `params` in place using `grads` (which must have the same length).
    ///
    /// `param_group` identifies the tensor by its stable registration order so that
    /// stateful optimizers can keep independent per-tensor buffers. Every group should be
    /// stepped exactly once per optimization step.
    fn step(&mut self, params: &mut [f64], grads: &[f64], param_group: usize);
}

/// Stochastic gradient descent with heavy-ball momentum.
///
/// Update rule per element, with velocity `v` initialised to zero per `param_group`:
///
/// ```text
/// v <- momentum * v + lr * grad
/// param <- param - v
/// ```
///
/// This matches the classic SGD+momentum update used by the crate's trainable dense
/// layers; regularization and gradient clipping are applied by the caller before
/// [`Optimizer::step`].
#[derive(Debug, Clone)]
pub struct Sgd {
    lr: f64,
    momentum: f64,
    velocity: HashMap<usize, Vec<f64>>,
}

impl Sgd {
    /// Creates an SGD optimizer with the given learning rate and momentum.
    pub fn new(lr: f64, momentum: f64) -> Self {
        Sgd {
            lr,
            momentum,
            velocity: HashMap::new(),
        }
    }

    /// Returns the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.lr
    }

    /// Sets the learning rate (keeps existing momentum buffers).
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Sets the momentum coefficient (keeps existing momentum buffers).
    pub fn set_momentum(&mut self, momentum: f64) {
        self.momentum = momentum;
    }
}

impl Default for Sgd {
    fn default() -> Self {
        Sgd::new(0.01, 0.9)
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut [f64], grads: &[f64], param_group: usize) {
        debug_assert_eq!(params.len(), grads.len());
        let v = self
            .velocity
            .entry(param_group)
            .or_insert_with(|| vec![0.0; params.len()]);
        if v.len() != params.len() {
            *v = vec![0.0; params.len()];
        }
        for ((p, g), vel) in params.iter_mut().zip(grads.iter()).zip(v.iter_mut()) {
            *vel = self.momentum * *vel + self.lr * *g;
            *p -= *vel;
        }
    }
}

/// Per-tensor state for [`Adam`]: first/second moment estimates and a local step counter.
#[derive(Debug, Clone)]
struct AdamState {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

/// The Adam optimizer with bias-corrected moment estimates.
///
/// Update rule per element (per `param_group`), with `t` the number of updates applied to
/// that group:
///
/// ```text
/// m <- beta1 * m + (1 - beta1) * grad
/// v <- beta2 * v + (1 - beta2) * grad^2
/// m_hat <- m / (1 - beta1^t)
/// v_hat <- v / (1 - beta2^t)
/// param <- param - lr * m_hat / (sqrt(v_hat) + eps)
/// ```
///
/// The step counter is tracked per group; because every group is stepped once per
/// optimization step, each group's `t` equals the number of optimization steps taken,
/// giving consistent bias correction across tensors.
#[derive(Debug, Clone)]
pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    state: HashMap<usize, AdamState>,
}

impl Adam {
    /// Creates an Adam optimizer with the given learning rate and standard defaults
    /// (`beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`).
    pub fn new(lr: f64) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            state: HashMap::new(),
        }
    }

    /// Creates an Adam optimizer with fully specified hyperparameters.
    pub fn with_params(lr: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        Adam {
            lr,
            beta1,
            beta2,
            eps,
            state: HashMap::new(),
        }
    }

    /// Returns the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.lr
    }

    /// Sets the learning rate (keeps existing moment buffers).
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl Default for Adam {
    fn default() -> Self {
        Adam::new(1e-3)
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [f64], grads: &[f64], param_group: usize) {
        debug_assert_eq!(params.len(), grads.len());
        let st = self.state.entry(param_group).or_insert_with(|| AdamState {
            m: vec![0.0; params.len()],
            v: vec![0.0; params.len()],
            t: 0,
        });
        if st.m.len() != params.len() {
            st.m = vec![0.0; params.len()];
            st.v = vec![0.0; params.len()];
            st.t = 0;
        }

        st.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(st.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(st.t as i32);

        for (i, (p, &g)) in params.iter_mut().zip(grads.iter()).enumerate() {
            st.m[i] = self.beta1 * st.m[i] + (1.0 - self.beta1) * g;
            st.v[i] = self.beta2 * st.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = st.m[i] / bias_correction1;
            let v_hat = st.v[i] / bias_correction2;
            *p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

/// Clips a model's gradient tensors jointly by their global L2 norm.
///
/// The global norm is `sqrt(sum over every element of every tensor of grad^2)`. If it
/// exceeds `max_norm` (and `max_norm > 0`), every gradient element is scaled by
/// `max_norm / norm` so the clipped tensors have global norm `max_norm`; otherwise the
/// tensors are left unchanged. The pre-clip global norm is always returned.
///
/// This is the global-norm counterpart to per-element gradient clamping and should be
/// applied across *all* of a model's gradient tensors before calling [`Optimizer::step`].
pub fn clip_grad_norm(grads: &mut [&mut [f64]], max_norm: f64) -> f64 {
    let sum_sq: f64 = grads.iter().flat_map(|g| g.iter()).map(|&x| x * x).sum();
    let norm = sum_sq.sqrt();

    if max_norm > 0.0 && norm > max_norm && norm.is_finite() {
        let scale = max_norm / norm;
        for tensor in grads.iter_mut() {
            for x in tensor.iter_mut() {
                *x *= scale;
            }
        }
    }

    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_no_momentum_matches_plain_gd() {
        let mut opt = Sgd::new(0.1, 0.0);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![1.0, 1.0, 1.0];
        opt.step(&mut params, &grads, 0);
        // param - lr * grad
        assert!((params[0] - 0.9).abs() < 1e-12);
        assert!((params[1] - 1.9).abs() < 1e-12);
        assert!((params[2] - 2.9).abs() < 1e-12);
    }

    #[test]
    fn test_sgd_momentum_accumulates() {
        let mut opt = Sgd::new(0.1, 0.9);
        let mut params = vec![0.0];
        let grads = vec![1.0];
        opt.step(&mut params, &grads, 0);
        // v = 0.1, param = -0.1
        assert!((params[0] + 0.1).abs() < 1e-12);
        opt.step(&mut params, &grads, 0);
        // v = 0.9*0.1 + 0.1 = 0.19, param = -0.1 - 0.19 = -0.29
        assert!((params[0] + 0.29).abs() < 1e-12);
    }

    #[test]
    fn test_sgd_groups_are_independent() {
        let mut opt = Sgd::new(0.1, 0.9);
        let mut a = vec![0.0];
        let mut b = vec![0.0];
        opt.step(&mut a, &[1.0], 0);
        opt.step(&mut b, &[1.0], 1);
        // Both groups saw one step with identical grads, so their velocities match.
        assert!((a[0] - b[0]).abs() < 1e-12);
    }

    #[test]
    fn test_adam_first_step_is_signed_lr() {
        // On the very first step Adam's bias-corrected update magnitude equals lr
        // (m_hat = grad, v_hat = grad^2, so update = lr * grad / (|grad| + eps) ~ lr).
        let mut opt = Adam::new(0.01);
        let mut params = vec![0.0, 0.0];
        let grads = vec![5.0, -5.0];
        opt.step(&mut params, &grads, 0);
        assert!((params[0] + 0.01).abs() < 1e-5);
        assert!((params[1] - 0.01).abs() < 1e-5);
    }

    #[test]
    fn test_adam_reduces_quadratic_loss() {
        // Minimize f(x) = sum((x - target)^2); grad = 2 (x - target). Adam should converge.
        let target = [3.0, -2.0, 0.5];
        let mut x = [0.0f64; 3];
        let mut opt = Adam::new(0.1);
        for _ in 0..500 {
            let grads: Vec<f64> = x
                .iter()
                .zip(target.iter())
                .map(|(&xi, &ti)| 2.0 * (xi - ti))
                .collect();
            opt.step(&mut x, &grads, 0);
        }
        for (&xi, &ti) in x.iter().zip(target.iter()) {
            assert!((xi - ti).abs() < 1e-2, "x={} target={}", xi, ti);
        }
    }

    #[test]
    fn test_clip_grad_norm_below_threshold_is_noop() {
        let mut a = [0.3, 0.4];
        let mut b = [0.0];
        let mut grads: Vec<&mut [f64]> = vec![&mut a, &mut b];
        // global norm = sqrt(0.09 + 0.16) = 0.5.
        let norm = clip_grad_norm(&mut grads, 1.0);
        assert!((norm - 0.5).abs() < 1e-12);
        assert_eq!(a, [0.3, 0.4]);
        assert_eq!(b, [0.0]);
    }

    #[test]
    fn test_clip_grad_norm_scales_to_max() {
        let mut a = [3.0, 4.0];
        let mut grads: Vec<&mut [f64]> = vec![&mut a];
        // norm = 5.0, clip to 1.0 => scale 0.2.
        let norm = clip_grad_norm(&mut grads, 1.0);
        assert!((norm - 5.0).abs() < 1e-12);
        assert!((a[0] - 0.6).abs() < 1e-12);
        assert!((a[1] - 0.8).abs() < 1e-12);
        // Post-clip global norm is exactly max_norm.
        let clipped_norm = (a[0] * a[0] + a[1] * a[1]).sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_clip_grad_norm_multiple_tensors() {
        let mut a = [3.0];
        let mut b = [4.0];
        let mut grads: Vec<&mut [f64]> = vec![&mut a, &mut b];
        let norm = clip_grad_norm(&mut grads, 5.0);
        // norm exactly 5 -> no scaling (not strictly greater).
        assert!((norm - 5.0).abs() < 1e-12);
        assert_eq!(a, [3.0]);
        assert_eq!(b, [4.0]);
    }
}
