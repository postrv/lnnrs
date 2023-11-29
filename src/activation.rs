// activation.rs

/// Applies the sigmoid activation function to a value.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Applies the hyperbolic tangent activation function to a value.
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Applies the ReLU activation function to a value.
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}
