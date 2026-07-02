// src/main.rs
//! Liquid Neural Network (LNN) CLI - A Rust Implementation
//!
//! This binary provides a command-line interface for the Liquid Neural Network library.
//!
//! # Usage
//! ```bash
//! # Run demo
//! lnn demo
//!
//! # Test LTC network
//! lnn ltc -u 32 -l 100
//!
//! # Test CfC network
//! lnn cfc -u 32 -l 100
//!
//! # Train a CfC network on a waveform prediction task
//! lnn train --model cfc --task sine --epochs 200 --lr 0.01 --seed 42 --units 32
//! ```

use clap::{Parser, Subcommand};
use env_logger::Builder;
use log::{info, LevelFilter};
use std::error::Error;

use liquid_neural_network::prelude::*;
use liquid_neural_network::{cfc, input, ltc, output};

/// Liquid Neural Network CLI
#[derive(Parser)]
#[command(name = "lnn")]
#[command(author = "LNN Contributors")]
#[command(version)]
#[command(about = "Liquid Neural Network implementation in Rust", long_about = None)]
struct Cli {
    /// Set the logging level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a simulation with input from a CSV file
    Run {
        /// Input CSV file path
        #[arg(short, long)]
        input: String,

        /// Output CSV file path
        #[arg(short, long, default_value = "output.csv")]
        output: String,

        /// Number of neurons in the network
        #[arg(short, long, default_value = "100")]
        neurons: usize,

        /// Number of simulation timesteps
        #[arg(short, long, default_value = "10")]
        timesteps: usize,

        /// Spectral radius for weight initialization
        #[arg(short, long)]
        spectral_radius: Option<f64>,
    },

    /// Test logic gate configurations
    Gate {
        /// Gate type to test (AND, OR, NOT, NAND, NOR, XOR)
        #[arg(short, long, default_value = "AND")]
        gate_type: String,

        /// Number of simulation timesteps
        #[arg(short, long, default_value = "10")]
        timesteps: usize,
    },

    /// Display network statistics
    Stats {
        /// Number of neurons for the test network
        #[arg(short, long, default_value = "50")]
        neurons: usize,
    },

    /// Run demo with all logic gates
    Demo,

    /// Test LTC (Liquid Time-Constant) network on a time series
    Ltc {
        /// Number of hidden units
        #[arg(short = 'u', long, default_value = "32")]
        units: usize,

        /// Length of the sequence
        #[arg(short, long, default_value = "100")]
        length: usize,
    },

    /// Test CfC (Closed-form Continuous) network on a time series
    Cfc {
        /// Number of hidden units
        #[arg(short = 'u', long, default_value = "32")]
        units: usize,

        /// Length of the sequence
        #[arg(short, long, default_value = "100")]
        length: usize,

        /// Use no-gate mode (faster)
        #[arg(long)]
        no_gate: bool,
    },

    /// Run time-series prediction example (trains a CfC by default)
    TimeSeries {
        /// Model type: ltc, cfc, or lnn (only cfc supports training)
        #[arg(short, long, default_value = "cfc")]
        model: String,

        /// Number of hidden units
        #[arg(short = 'u', long, default_value = "32")]
        units: usize,

        /// Prediction task: sine, square, or sawtooth
        #[arg(short, long, default_value = "sine")]
        task: String,

        /// Skip training and run the untrained forward pass (random weights)
        #[arg(long)]
        no_train: bool,
    },

    /// Train a model on a waveform prediction task using BPTT
    Train {
        /// Model type (only cfc is currently trainable)
        #[arg(short, long, default_value = "cfc")]
        model: String,

        /// Prediction task: sine, square, or sawtooth
        #[arg(short, long, default_value = "sine")]
        task: String,

        /// Number of training epochs
        #[arg(short, long, default_value = "200")]
        epochs: usize,

        /// Learning rate (Adam)
        #[arg(short, long, default_value = "0.01")]
        lr: f64,

        /// Seed for reproducible weight initialization
        #[arg(short, long, default_value = "42")]
        seed: u64,

        /// Number of hidden units
        #[arg(short = 'u', long, default_value = "32")]
        units: usize,

        /// Save the trained model as JSON to this path
        #[arg(long)]
        save: Option<String>,
    },
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // Initialize logger
    let log_level = match cli.log_level.to_lowercase().as_str() {
        "trace" => LevelFilter::Trace,
        "debug" => LevelFilter::Debug,
        "info" => LevelFilter::Info,
        "warn" => LevelFilter::Warn,
        "error" => LevelFilter::Error,
        _ => LevelFilter::Info,
    };

    Builder::new().filter(None, log_level).init();

    match cli.command {
        Some(Commands::Run {
            input,
            output,
            neurons,
            timesteps,
            spectral_radius,
        }) => run_simulation(&input, &output, neurons, timesteps, spectral_radius)?,

        Some(Commands::Gate {
            gate_type,
            timesteps,
        }) => {
            test_single_gate(&gate_type, timesteps)?;
        }

        Some(Commands::Stats { neurons }) => {
            display_network_stats(neurons);
        }

        Some(Commands::Demo) => {
            run_demo()?;
        }

        Some(Commands::Ltc { units, length }) => {
            test_ltc_network(units, length)?;
        }

        Some(Commands::Cfc {
            units,
            length,
            no_gate,
        }) => {
            test_cfc_network(units, length, no_gate)?;
        }

        Some(Commands::TimeSeries {
            model,
            units,
            task,
            no_train,
        }) => {
            run_time_series_example(&model, units, &task, no_train)?;
        }

        Some(Commands::Train {
            model,
            task,
            epochs,
            lr,
            seed,
            units,
            save,
        }) => {
            run_training(&model, &task, epochs, lr, seed, units, save.as_deref())?;
        }

        None => {
            // Default behavior: run demo
            run_demo()?;
        }
    }

    Ok(())
}

/// Runs a simulation with the given parameters.
fn run_simulation(
    input_file: &str,
    output_file: &str,
    neuron_count: usize,
    timesteps: usize,
    spectral_radius: Option<f64>,
) -> std::result::Result<(), Box<dyn Error>> {
    info!(
        "Initializing Liquid Neural Network with {} neurons",
        neuron_count
    );

    // Read input data
    let inputs = input::read_input_from_csv(input_file)?;
    info!("Read {} input patterns from '{}'", inputs.len(), input_file);

    // Create network
    let mut builder = LnnBuilder::new(neuron_count);
    if let Some(sr) = spectral_radius {
        builder = builder.spectral_radius(sr);
        info!("Using spectral radius: {}", sr);
    }
    let mut lnn = builder.build();

    // Configure input/output neurons based on input dimensions
    if let Some(first_input) = inputs.first() {
        let input_size = first_input.len();
        let input_indices: Vec<usize> = (0..input_size.min(neuron_count)).collect();
        let output_indices: Vec<usize> =
            ((neuron_count - input_size.min(neuron_count))..neuron_count).collect();

        lnn.set_input_neurons(input_indices);
        lnn.set_output_neurons(output_indices);
    }

    // Run simulation
    info!("Running simulation for {} timesteps", timesteps);
    let outputs = lnn.run_simulation(timesteps, inputs)?;

    // Write outputs
    output::write_output_to_csv(output_file, &outputs)?;
    info!(
        "Wrote {} output patterns to '{}'",
        outputs.len(),
        output_file
    );

    // Display statistics
    let stats = lnn.statistics();
    info!("Network statistics:\n{}", stats);

    Ok(())
}

/// Tests a single logic gate configuration.
fn test_single_gate(gate_type: &str, timesteps: usize) -> std::result::Result<(), Box<dyn Error>> {
    info!("Testing {} gate", gate_type);

    let mut lnn = LiquidNeuralNetwork::new(10);
    lnn.configure_for_logic_gate(gate_type)?;

    let test_inputs = get_test_inputs_for_gate(gate_type);

    println!("\n{} Gate Truth Table:", gate_type.to_uppercase());
    println!("{}", "-".repeat(40));

    for input in test_inputs {
        let output = lnn.run_simulation(timesteps, vec![input.clone()])?;
        let output_val = output.last().and_then(|v| v.first()).unwrap_or(&0.0);
        let binary_output = if *output_val > 0.5 { 1 } else { 0 };

        println!(
            "Input: {:?} -> Output: {} (raw: {:.4})",
            input.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            binary_output,
            output_val
        );

        // Reset network state between tests
        lnn.reset_state();
    }

    Ok(())
}

/// Returns appropriate test inputs for the given gate type.
fn get_test_inputs_for_gate(gate_type: &str) -> Vec<Vec<f64>> {
    match gate_type.to_uppercase().as_str() {
        "NOT" => vec![vec![0.0], vec![1.0]],
        _ => vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
    }
}

/// Displays network statistics.
fn display_network_stats(neuron_count: usize) {
    info!(
        "Creating network with {} neurons for statistics",
        neuron_count
    );

    let lnn = LnnBuilder::new(neuron_count)
        .spectral_radius(0.9)
        .connection_probability(0.8)
        .build();

    let stats = lnn.statistics();
    println!("\n{}", stats);
}

/// Runs a demonstration of all logic gates.
fn run_demo() -> std::result::Result<(), Box<dyn Error>> {
    println!("\n========================================");
    println!("   Liquid Neural Network Demo");
    println!("========================================\n");

    println!("This demo tests the LNN's ability to simulate");
    println!("various logic gates through network configuration.\n");

    let gates = ["AND", "OR", "NOT", "NAND", "NOR", "XOR"];
    let timesteps = 10;

    for gate in &gates {
        test_single_gate(gate, timesteps)?;
        println!();
    }

    // Show network statistics
    println!("\n========================================");
    println!("   Network Statistics");
    println!("========================================\n");

    let lnn = LnnBuilder::new(50)
        .spectral_radius(0.95)
        .connection_probability(0.7)
        .weight_range(-0.5, 0.5)
        .build();

    println!("{}", lnn.statistics());

    println!("\n========================================");
    println!("   Demo Complete");
    println!("========================================\n");

    Ok(())
}

/// Tests the LTC (Liquid Time-Constant) network.
fn test_ltc_network(units: usize, length: usize) -> std::result::Result<(), Box<dyn Error>> {
    use ltc::LtcNetwork;

    println!("\n========================================");
    println!("   LTC Network Test");
    println!("========================================\n");

    println!("Configuration:");
    println!("  Hidden units: {}", units);
    println!("  Sequence length: {}", length);

    // Create network: 2 inputs (sin, cos), hidden units, 1 output
    let network = LtcNetwork::new(2, units, 1);

    // Generate sine wave input
    let inputs: Vec<Vec<f64>> = (0..length)
        .map(|i| {
            let t = i as f64 * 0.1;
            vec![t.sin(), t.cos()]
        })
        .collect();

    // Process sequence
    let start = std::time::Instant::now();
    let outputs = network.forward_sequence(&inputs);
    let elapsed = start.elapsed();

    println!("\nResults:");
    println!("  Processing time: {:?}", elapsed);
    println!("  Output samples (first 5):");
    for (i, output) in outputs.iter().take(5).enumerate() {
        let t = i as f64 * 0.1;
        println!(
            "    t={:.1}: input=({:.3}, {:.3}) -> output={:.4}",
            t, inputs[i][0], inputs[i][1], output[0]
        );
    }

    // Verify outputs are bounded
    let all_finite = outputs.iter().all(|o| o.iter().all(|&v| v.is_finite()));
    println!("\n  All outputs finite: {}", all_finite);

    if !all_finite {
        return Err("LTC network produced non-finite values".into());
    }

    println!("\n✓ LTC network test passed!");
    Ok(())
}

/// Tests the CfC (Closed-form Continuous) network.
fn test_cfc_network(
    units: usize,
    length: usize,
    no_gate: bool,
) -> std::result::Result<(), Box<dyn Error>> {
    use cfc::CfcBuilder;

    println!("\n========================================");
    println!("   CfC Network Test");
    println!("========================================\n");

    println!("Configuration:");
    println!("  Hidden units: {}", units);
    println!("  Sequence length: {}", length);
    println!("  No-gate mode: {}", no_gate);

    // Create network
    let network = CfcBuilder::new(2, 1)
        .hidden_units(units)
        .no_gate(no_gate)
        .build();

    // Generate sine wave input
    let inputs: Vec<Vec<f64>> = (0..length)
        .map(|i| {
            let t = i as f64 * 0.1;
            vec![t.sin(), t.cos()]
        })
        .collect();

    // Process sequence
    let start = std::time::Instant::now();
    let outputs = network.forward_sequence(&inputs);
    let elapsed = start.elapsed();

    println!("\nResults:");
    println!("  Processing time: {:?}", elapsed);
    println!("  Output samples (first 5):");
    for (i, output) in outputs.iter().take(5).enumerate() {
        let t = i as f64 * 0.1;
        println!(
            "    t={:.1}: input=({:.3}, {:.3}) -> output={:.4}",
            t, inputs[i][0], inputs[i][1], output[0]
        );
    }

    // Verify outputs are bounded
    let all_finite = outputs.iter().all(|o| o.iter().all(|&v| v.is_finite()));
    println!("\n  All outputs finite: {}", all_finite);

    if !all_finite {
        return Err("CfC network produced non-finite values".into());
    }

    println!("\n✓ CfC network test passed!");
    Ok(())
}

/// Generates a waveform next-step prediction task.
///
/// Returns `(inputs, targets)` where `targets[i]` is the waveform value at step `i` and
/// `inputs[i] = [target[i - 1], cos(t_i)]` (previous value plus a phase reference).
fn generate_wave_task(task: &str, length: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let targets: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 * 0.05;
            match task {
                "square" => {
                    if (t * 2.0).sin() > 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                "sawtooth" => (t % (2.0 * std::f64::consts::PI)) / std::f64::consts::PI - 1.0,
                _ => t.sin(), // sine wave by default
            }
        })
        .collect();

    let inputs: Vec<Vec<f64>> = (0..length)
        .map(|i| {
            let prev = if i > 0 { targets[i - 1] } else { 0.0 };
            vec![prev, (i as f64 * 0.05).cos()]
        })
        .collect();

    (inputs, targets)
}

/// Prints a sample-prediction table (every 20th step) and returns the MSE.
fn report_predictions(outputs: &[Vec<f64>], targets: &[f64]) -> f64 {
    let length = targets.len();
    let mse: f64 = outputs
        .iter()
        .zip(targets.iter())
        .map(|(o, &t)| (o[0] - t).powi(2))
        .sum::<f64>()
        / length as f64;

    println!("\nResults:");
    println!("  Mean Squared Error: {:.6}", mse);
    println!("\n  Sample predictions (showing every 20th step):");
    println!(
        "    {:>6} {:>10} {:>10} {:>10}",
        "Step", "Target", "Predicted", "Error"
    );
    for i in (0..length).step_by(20) {
        let target = targets[i];
        let predicted = outputs[i][0];
        let error = (target - predicted).abs();
        println!(
            "    {:>6} {:>10.4} {:>10.4} {:>10.4}",
            i, target, predicted, error
        );
    }

    mse
}

/// Trains a seeded [`TrainableCfc`] on a waveform task, logging the loss every
/// `max(1, epochs / 10)` epochs. Returns the trained network and the final training loss.
fn train_cfc_on_task(
    inputs: &[Vec<f64>],
    targets: &[f64],
    units: usize,
    epochs: usize,
    lr: f64,
    seed: u64,
) -> (TrainableCfc, f64) {
    let target_rows: Vec<Vec<f64>> = targets.iter().map(|&t| vec![t]).collect();

    let mut net = TrainableCfcBuilder::new(2, units, 1)
        .learning_rate(lr)
        .truncation_length(inputs.len())
        .seed(seed)
        .build();

    let log_every = std::cmp::max(1, epochs / 10);
    let mut last_loss = f64::NAN;
    for epoch in 1..=epochs {
        last_loss = net.train_step(inputs, &target_rows);
        if epoch % log_every == 0 || epoch == epochs {
            println!("  epoch {:>5}: loss = {:.6}", epoch, last_loss);
        }
    }

    (net, last_loss)
}

/// Trains a model on a waveform prediction task (the `train` subcommand).
fn run_training(
    model: &str,
    task: &str,
    epochs: usize,
    lr: f64,
    seed: u64,
    units: usize,
    save: Option<&str>,
) -> std::result::Result<(), Box<dyn Error>> {
    if model.to_lowercase() != "cfc" {
        return Err(format!(
            "unsupported model '{}': only 'cfc' is currently trainable",
            model
        )
        .into());
    }

    println!("\n========================================");
    println!("   CfC Training (BPTT + Adam)");
    println!("========================================\n");

    println!("Configuration:");
    println!("  Model: {}", model);
    println!("  Task: {} wave prediction", task);
    println!("  Hidden units: {}", units);
    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", lr);
    println!("  Seed: {}", seed);

    let length = 200;
    let (inputs, targets) = generate_wave_task(task, length);
    let target_rows: Vec<Vec<f64>> = targets.iter().map(|&t| vec![t]).collect();

    // Untrained baseline (same seed / architecture, before any updates).
    let baseline = TrainableCfcBuilder::new(2, units, 1).seed(seed).build();
    let untrained_mse = baseline.loss(&inputs, &target_rows);
    println!("\nUntrained MSE: {:.6}", untrained_mse);

    println!("\nTraining:");
    let start = std::time::Instant::now();
    let (net, _) = train_cfc_on_task(&inputs, &targets, units, epochs, lr, seed);
    let elapsed = start.elapsed();

    let outputs = net.predict(&inputs);
    let final_mse = report_predictions(&outputs, &targets);

    println!("\n  Training time: {:?}", elapsed);
    println!("  Final MSE: {:.6}", final_mse);
    println!(
        "  Improvement over untrained: {:.1}x",
        untrained_mse / final_mse
    );

    if let Some(path) = save {
        let json = serde_json::to_string_pretty(&net)?;
        std::fs::write(path, json)?;
        println!("  Saved trained model to '{}'", path);
    }

    Ok(())
}

/// Runs a time-series prediction example.
///
/// The CfC model is trained by default (BPTT + Adam, 200 epochs, lr 0.01, seed 42);
/// `--no-train` restores the untrained forward pass. LTC and classic LNN models have no
/// training path and always run untrained.
fn run_time_series_example(
    model: &str,
    units: usize,
    task: &str,
    no_train: bool,
) -> std::result::Result<(), Box<dyn Error>> {
    use cfc::CfcBuilder;
    use ltc::LtcNetwork;

    println!("\n========================================");
    println!("   Time Series Prediction Example");
    println!("========================================\n");

    let model_kind = model.to_lowercase();
    let trained = model_kind == "cfc" && !no_train;

    println!("Configuration:");
    println!("  Model: {}", model);
    println!("  Hidden units: {}", units);
    println!("  Task: {} wave prediction", task);
    println!(
        "  Training: {}",
        if trained { "enabled" } else { "disabled" }
    );

    let length = 200;
    let (inputs, targets) = generate_wave_task(task, length);

    // Run with selected model
    let outputs = match model_kind.as_str() {
        "cfc" if trained => {
            let (epochs, lr, seed) = (200, 0.01, 42);
            println!(
                "\nTraining (epochs: {}, lr: {}, seed: {}):",
                epochs, lr, seed
            );
            let (net, _) = train_cfc_on_task(&inputs, &targets, units, epochs, lr, seed);
            net.predict(&inputs)
        }
        "cfc" => {
            let network = CfcBuilder::new(2, 1).hidden_units(units).build();
            network.forward_sequence(&inputs)
        }
        "ltc" => {
            let network = LtcNetwork::new(2, units, 1);
            network.forward_sequence(&inputs)
        }
        _ => {
            // Use classic LNN
            let mut lnn = LiquidNeuralNetwork::new(units);
            lnn.set_input_neurons(vec![0, 1]);
            lnn.set_output_neurons(vec![units - 1]);
            let converted_inputs: Vec<Vec<f64>> = inputs.clone();
            lnn.run_simulation(1, converted_inputs)?
        }
    };

    report_predictions(&outputs, &targets);

    if !trained {
        println!("\nNote: This is a forward pass demonstration. The network is not trained,");
        println!("so predictions are based on random initial weights.");
        if model_kind == "cfc" {
            println!("Drop --no-train to train the CfC before predicting.");
        } else {
            println!("Training is currently only supported for the cfc model.");
        }
    }

    Ok(())
}

#[cfg(test)]
mod integration_tests {
    use liquid_neural_network::prelude::*;

    #[test]
    fn test_full_pipeline() {
        // Create a small network
        let mut lnn = LnnBuilder::new(10)
            .weight_range(-0.5, 0.5)
            .threshold_range(0.3, 0.7)
            .build();

        // Configure input/output
        lnn.set_input_neurons(vec![0, 1]);
        lnn.set_output_neurons(vec![8, 9]);

        // Run simulation
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let outputs = lnn.run_simulation(5, inputs).unwrap();

        // Verify outputs
        assert_eq!(outputs.len(), 3);
        for output in &outputs {
            assert_eq!(output.len(), 2); // 2 output neurons
        }
    }

    #[test]
    fn test_network_persistence() {
        let lnn = LnnBuilder::new(20).spectral_radius(0.9).build();

        // Serialize
        let json = serde_json::to_string(&lnn).unwrap();

        // Deserialize
        let restored: LiquidNeuralNetwork = serde_json::from_str(&json).unwrap();

        assert_eq!(lnn.neuron_count(), restored.neuron_count());
    }

    #[test]
    fn test_logic_gate_and() {
        let mut lnn = LiquidNeuralNetwork::new(10);
        lnn.configure_for_logic_gate("AND").unwrap();

        // Test AND gate behavior
        let test_cases = vec![
            (vec![0.0, 0.0], false),
            (vec![0.0, 1.0], false),
            (vec![1.0, 0.0], false),
            (vec![1.0, 1.0], true),
        ];

        for (input, _expected) in test_cases {
            let output = lnn.run_simulation(10, vec![input]).unwrap();
            assert!(!output.is_empty());
            lnn.reset_state();
        }
    }

    #[test]
    fn test_logic_gate_or() {
        let mut lnn = LiquidNeuralNetwork::new(10);
        lnn.configure_for_logic_gate("OR").unwrap();

        let test_cases = vec![
            (vec![0.0, 0.0], false),
            (vec![0.0, 1.0], true),
            (vec![1.0, 0.0], true),
            (vec![1.0, 1.0], true),
        ];

        for (input, _expected) in test_cases {
            let output = lnn.run_simulation(10, vec![input]).unwrap();
            assert!(!output.is_empty());
            lnn.reset_state();
        }
    }

    #[test]
    fn test_spectral_radius_scaling() {
        let lnn = LnnBuilder::new(20).spectral_radius(0.95).build();

        let stats = lnn.statistics();
        assert!(stats.synapse_count > 0);
    }

    #[test]
    fn test_builder_fluent_api() {
        let lnn = LiquidNeuralNetwork::builder(30)
            .weight_range(-1.0, 1.0)
            .delay_range(1, 3)
            .threshold_range(0.4, 0.8)
            .connection_probability(0.5)
            .build();

        assert_eq!(lnn.neuron_count(), 30);
    }

    #[test]
    fn test_input_output_configuration() {
        let mut lnn = LiquidNeuralNetwork::new(20);

        lnn.set_input_neurons(vec![0, 1, 2, 3, 4]);
        lnn.set_output_neurons(vec![15, 16, 17, 18, 19]);

        assert_eq!(lnn.input_neurons().len(), 5);
        assert_eq!(lnn.output_neurons().len(), 5);
    }

    #[test]
    fn test_network_reset() {
        let mut lnn = LiquidNeuralNetwork::new(10);
        lnn.set_input_neurons(vec![0, 1]);

        // Encode some input
        lnn.encode_input(&[1.0, 2.0]);

        // Reset
        lnn.reset_state();

        // Verify all neurons are reset
        for neuron in &lnn.neurons {
            assert_eq!(neuron.membrane_potential, 0.0);
        }
    }
}
