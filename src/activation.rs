// src/activation.rs
//! Activation functions and their derivatives for neural network computations.

/// Applies the sigmoid activation function to a value.
///
/// The sigmoid function maps any real number to the range (0, 1).
/// Formula: σ(x) = 1 / (1 + e^(-x))
///
/// # Arguments
/// * `x` - The input value
///
/// # Returns
/// A value in the range (0, 1)
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Computes the derivative of the sigmoid function.
///
/// Formula: σ'(x) = σ(x) * (1 - σ(x))
///
/// # Arguments
/// * `x` - The input value (pre-activation)
///
/// # Returns
/// The derivative value
#[inline]
pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Applies the hyperbolic tangent activation function to a value.
///
/// Maps any real number to the range (-1, 1).
/// Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
///
/// # Arguments
/// * `x` - The input value
///
/// # Returns
/// A value in the range (-1, 1)
#[inline]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Computes the derivative of the hyperbolic tangent function.
///
/// Formula: tanh'(x) = 1 - tanh²(x)
///
/// # Arguments
/// * `x` - The input value
///
/// # Returns
/// The derivative value
#[inline]
pub fn tanh_derivative(x: f64) -> f64 {
    let t = x.tanh();
    1.0 - t * t
}

/// Applies the ReLU (Rectified Linear Unit) activation function.
///
/// Formula: ReLU(x) = max(0, x)
///
/// # Arguments
/// * `x` - The input value
///
/// # Returns
/// The input if positive, otherwise 0
#[inline]
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Computes the derivative of the ReLU function.
///
/// Formula: ReLU'(x) = 1 if x > 0, else 0
///
/// # Arguments
/// * `x` - The input value
///
/// # Returns
/// 1.0 if x > 0, otherwise 0.0
#[inline]
pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Applies the Leaky ReLU activation function.
///
/// Formula: LeakyReLU(x) = max(αx, x) where α is typically 0.01
///
/// # Arguments
/// * `x` - The input value
/// * `alpha` - The slope for negative values (default: 0.01)
///
/// # Returns
/// The input if positive, otherwise alpha * input
#[inline]
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

/// Computes the derivative of the Leaky ReLU function.
///
/// # Arguments
/// * `x` - The input value
/// * `alpha` - The slope for negative values
///
/// # Returns
/// 1.0 if x > 0, otherwise alpha
#[inline]
pub fn leaky_relu_derivative(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        alpha
    }
}

/// Applies the softplus activation function.
///
/// A smooth approximation of ReLU.
/// Formula: softplus(x) = ln(1 + e^x)
///
/// # Arguments
/// * `x` - The input value
///
/// # Returns
/// A smooth positive value
#[inline]
pub fn softplus(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

/// Computes the derivative of the softplus function.
///
/// Formula: softplus'(x) = sigmoid(x)
///
/// # Arguments
/// * `x` - The input value
///
/// # Returns
/// The derivative value (equals sigmoid)
#[inline]
pub fn softplus_derivative(x: f64) -> f64 {
    sigmoid(x)
}

/// Enum representing available activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Softplus,
    Linear,
}

impl ActivationFunction {
    /// Applies the activation function to a value
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => sigmoid(x),
            ActivationFunction::Tanh => tanh(x),
            ActivationFunction::ReLU => relu(x),
            ActivationFunction::LeakyReLU => leaky_relu(x, 0.01),
            ActivationFunction::Softplus => softplus(x),
            ActivationFunction::Linear => x,
        }
    }

    /// Applies the derivative of the activation function
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => sigmoid_derivative(x),
            ActivationFunction::Tanh => tanh_derivative(x),
            ActivationFunction::ReLU => relu_derivative(x),
            ActivationFunction::LeakyReLU => leaky_relu_derivative(x, 0.01),
            ActivationFunction::Softplus => softplus_derivative(x),
            ActivationFunction::Linear => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid_bounds() {
        // Sigmoid should always be between 0 and 1
        // Note: For very large/small inputs, floating point may saturate to 0 or 1
        assert!(sigmoid(-10.0) > 0.0);
        assert!(sigmoid(-10.0) < 1.0);
        assert!(sigmoid(10.0) > 0.0);
        assert!(sigmoid(10.0) < 1.0);
        // Verify extreme values are still valid (within [0, 1])
        assert!(sigmoid(-100.0) >= 0.0);
        assert!(sigmoid(100.0) <= 1.0);
    }

    #[test]
    fn test_sigmoid_center() {
        // sigmoid(0) = 0.5
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // sigmoid(-x) = 1 - sigmoid(x)
        for x in [-5.0, -1.0, 0.5, 2.0, 10.0] {
            assert_relative_eq!(sigmoid(-x), 1.0 - sigmoid(x), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tanh_bounds() {
        // tanh should always be between -1 and 1
        // Note: For very large/small inputs, floating point may saturate to -1 or 1
        assert!(tanh(-10.0) > -1.0);
        assert!(tanh(-10.0) < 1.0);
        assert!(tanh(10.0) > -1.0);
        assert!(tanh(10.0) < 1.0);
        // Verify extreme values are still valid (within [-1, 1])
        assert!(tanh(-100.0) >= -1.0);
        assert!(tanh(100.0) <= 1.0);
    }

    #[test]
    fn test_tanh_center() {
        // tanh(0) = 0
        assert_relative_eq!(tanh(0.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(-5.0), 0.0);
        assert_eq!(relu(-0.1), 0.0);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(0.1), 0.1);
        assert_eq!(relu(5.0), 5.0);
    }

    #[test]
    fn test_relu_derivative() {
        assert_eq!(relu_derivative(-5.0), 0.0);
        assert_eq!(relu_derivative(-0.1), 0.0);
        assert_eq!(relu_derivative(0.1), 1.0);
        assert_eq!(relu_derivative(5.0), 1.0);
    }

    #[test]
    fn test_leaky_relu() {
        let alpha = 0.01;
        assert_relative_eq!(leaky_relu(-5.0, alpha), -0.05, epsilon = 1e-10);
        assert_eq!(leaky_relu(0.0, alpha), 0.0);
        assert_eq!(leaky_relu(5.0, alpha), 5.0);
    }

    #[test]
    fn test_softplus_positive() {
        // softplus(x) should always be positive
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            assert!(softplus(x) > 0.0);
        }
    }

    #[test]
    fn test_softplus_approximates_relu() {
        // For large positive x, softplus ≈ x
        assert_relative_eq!(softplus(10.0), 10.0, epsilon = 0.001);
        // For large negative x, softplus ≈ 0
        assert!(softplus(-10.0) < 0.001);
    }

    #[test]
    fn test_activation_enum() {
        let x = 0.5;

        assert_relative_eq!(
            ActivationFunction::Sigmoid.apply(x),
            sigmoid(x),
            epsilon = 1e-10
        );
        assert_relative_eq!(ActivationFunction::Tanh.apply(x), tanh(x), epsilon = 1e-10);
        assert_relative_eq!(ActivationFunction::ReLU.apply(x), relu(x), epsilon = 1e-10);
        assert_relative_eq!(ActivationFunction::Linear.apply(x), x, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative_at_zero() {
        // sigmoid'(0) = 0.25
        assert_relative_eq!(sigmoid_derivative(0.0), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_tanh_derivative_at_zero() {
        // tanh'(0) = 1
        assert_relative_eq!(tanh_derivative(0.0), 1.0, epsilon = 1e-10);
    }
}
