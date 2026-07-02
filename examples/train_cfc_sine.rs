//! Trains a seeded CfC network to predict a sine wave and reports untrained vs trained MSE.
//!
//! Run with:
//! ```bash
//! cargo run --example train_cfc_sine
//! ```
//!
//! The task is one-step-ahead prediction: at step `i` the network sees
//! `[value(i - 1), cos(t_i)]` and must output `value(i) = sin(t_i)`. Training is BPTT
//! through the closed-form gated update with the Adam optimizer. A fixed seed makes the
//! run reproducible.

use liquid_neural_network::prelude::*;

/// Length of the training sequence.
const LENGTH: usize = 200;
/// Number of hidden units in the CfC cell.
const UNITS: usize = 32;
/// Number of training epochs.
const EPOCHS: usize = 200;
/// Adam learning rate.
const LR: f64 = 0.01;
/// RNG seed for reproducible weight initialization.
const SEED: u64 = 42;

/// Builds the sine prediction task, returning `(inputs, targets)`.
fn sine_task() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let values: Vec<f64> = (0..LENGTH).map(|i| (i as f64 * 0.05).sin()).collect();
    let inputs: Vec<Vec<f64>> = (0..LENGTH)
        .map(|i| {
            let prev = if i > 0 { values[i - 1] } else { 0.0 };
            vec![prev, (i as f64 * 0.05).cos()]
        })
        .collect();
    // Targets are single-element rows to match the 1-D output.
    let targets: Vec<Vec<f64>> = values.iter().map(|&v| vec![v]).collect();
    (inputs, targets)
}

/// Mean squared error between predictions and targets.
fn mse(predictions: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
    let n = targets.len().max(1) as f64;
    predictions
        .iter()
        .zip(targets)
        .map(|(p, t)| (p[0] - t[0]).powi(2))
        .sum::<f64>()
        / n
}

fn main() {
    let (inputs, targets) = sine_task();

    // Seeded builder -> identical initial weights on every run.
    let mut net = TrainableCfcBuilder::new(2, UNITS, 1)
        .learning_rate(LR)
        .truncation_length(inputs.len())
        .seed(SEED)
        .build();

    let untrained_mse = mse(&net.predict(&inputs), &targets);
    println!("Untrained MSE: {untrained_mse:.6}");

    println!("Training for {EPOCHS} epochs (Adam, lr = {LR}):");
    let log_every = EPOCHS / 10;
    for epoch in 1..=EPOCHS {
        let loss = net.train_step(&inputs, &targets);
        if epoch % log_every == 0 {
            println!("  epoch {epoch:>4}: loss = {loss:.6}");
        }
    }

    let trained_mse = mse(&net.predict(&inputs), &targets);
    println!("Trained MSE:   {trained_mse:.6}");
    println!(
        "Improvement:   {:.1}x lower error",
        untrained_mse / trained_mse
    );
}
