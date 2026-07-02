// src/layers.rs
//! Shared neural-network layers.
//!
//! This module provides a single [`DenseLayer`] used across the crate (CfC backbone and
//! heads, LTC input/recurrent/output projections, and the trainable dense layer in
//! [`crate::training`]). Weights are stored as an [`ndarray::Array2`] and biases as an
//! [`ndarray::Array1`] so the forward pass is a single `.dot()` matrix-vector product; the
//! public sequence APIs continue to exchange `Vec<Vec<f64>>` and convert at the boundary.

use ndarray::{Array1, Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A fully-connected (affine) layer: `y = W x + b`.
///
/// The weight matrix has shape `[output_dim, input_dim]` so that the forward pass is the
/// natural matrix-vector product `weights.dot(x)`; the bias has shape `[output_dim]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    /// Weight matrix, shape `[output_dim, input_dim]`.
    pub weights: Array2<f64>,
    /// Bias vector, shape `[output_dim]`.
    pub bias: Array1<f64>,
    input_dim: usize,
    output_dim: usize,
}

impl DenseLayer {
    /// Creates a dense layer with Xavier/Glorot-initialised weights drawn from `rng`.
    ///
    /// Each weight is sampled uniformly from `[-scale, scale)` with
    /// `scale = sqrt(2 / (input_dim + output_dim))`; the bias is initialised to zero. The
    /// generator is borrowed (never created here) so that every layer of a network is derived
    /// from the single seeded generator owned by the network constructor, keeping construction
    /// reproducible.
    pub fn new(input_dim: usize, output_dim: usize, rng: &mut StdRng) -> Self {
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let mut weights = Array2::zeros((output_dim, input_dim));
        // Row-major (C-order) traversal keeps the draw order deterministic for a given rng.
        for w in weights.iter_mut() {
            *w = rng.random_range(-scale..scale);
        }
        DenseLayer {
            weights,
            bias: Array1::zeros(output_dim),
            input_dim,
            output_dim,
        }
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Forward pass on an [`ArrayView1`] input: `W x + b`.
    ///
    /// Taking a view lets slice-based callers avoid an allocation (via
    /// [`ArrayView1::from`]) while `ndarray` callers pass `array.view()`.
    ///
    /// # Panics
    /// Panics if `input.len() != input_dim` (the underlying `.dot()` requires matching dims).
    pub fn forward_arr(&self, input: ArrayView1<f64>) -> Array1<f64> {
        self.weights.dot(&input) + &self.bias
    }

    /// Forward pass on a slice, returning a `Vec` (convenience at the public boundary).
    ///
    /// # Panics
    /// Panics if `input.len() != input_dim`.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.forward_arr(ArrayView1::from(input)).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_dense_layer_dims() {
        let mut rng = StdRng::seed_from_u64(0);
        let layer = DenseLayer::new(3, 4, &mut rng);
        assert_eq!(layer.input_dim(), 3);
        assert_eq!(layer.output_dim(), 4);
        assert_eq!(layer.weights.shape(), &[4, 3]);
        assert_eq!(layer.bias.len(), 4);
    }

    #[test]
    fn test_dense_layer_forward_matches_manual() {
        let mut rng = StdRng::seed_from_u64(7);
        let layer = DenseLayer::new(3, 2, &mut rng);
        let input = [1.0, 0.5, 0.2];

        let out = layer.forward(&input);
        assert_eq!(out.len(), 2);

        // Manual reference: y[j] = bias[j] + sum_i W[j][i] * x[i].
        for (j, &o) in out.iter().enumerate() {
            let mut expected = layer.bias[j];
            for (i, &xi) in input.iter().enumerate() {
                expected += layer.weights[[j, i]] * xi;
            }
            assert!((o - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_dense_layer_forward_arr_matches_forward() {
        let mut rng = StdRng::seed_from_u64(11);
        let layer = DenseLayer::new(4, 3, &mut rng);
        let input = vec![0.1, -0.2, 0.3, 0.4];
        let via_slice = layer.forward(&input);
        let via_arr = layer.forward_arr(Array1::from(input).view());
        for (a, b) in via_slice.iter().zip(via_arr.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_dense_layer_serialization_roundtrip() {
        let mut rng = StdRng::seed_from_u64(3);
        let layer = DenseLayer::new(5, 3, &mut rng);
        let json = serde_json::to_string(&layer).unwrap();
        let restored: DenseLayer = serde_json::from_str(&json).unwrap();
        let input = [0.2, 0.4, -0.1, 0.0, 0.9];
        assert_eq!(layer.forward(&input), restored.forward(&input));
    }
}
