//main.rs

use clap::Parser;
use env_logger::Builder;
use log::{info, LevelFilter};
use ndarray::Array2;
use ndarray_rand::RandomExt; // Add this line to import RandomExt
use rand_distr::Uniform;
use std;
use std::error::Error;

mod activation;
mod input;
mod lnn;
mod network_components;
mod output;

// Define CLI arguments struct using Clap.
#[derive(Parser)]
#[clap(version = "1.0", author = "postrv")]
struct Opts {
    // TODO Define command line arguments here.
}

fn determine_neuron_count(inputs: &[Vec<f64>]) -> usize {
    // A simple example heuristic: set the neuron count to be
    // a multiple of the number of features from the input.
    // Adjust this heuristic based on experimental results.

    if let Some(first_input) = inputs.first() {
        let feature_count = first_input.len();
        let multiplier = 10; // Adjust this multiplier as needed
        feature_count * multiplier
    } else {
        // Default to some minimum size if no input is found
        100
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the logger
    Builder::new().filter(None, LevelFilter::Info).init();

    // Parse the command line arguments
    let _opts: Opts = Opts::parse();

    // The main logic of application will go here.
    // For now, we'll just log an information message.
    info!("Liquid Neural Network application started");

    let inputs = input::read_input_from_csv("input_data.csv")?;

    let neuron_count = determine_neuron_count(&inputs); // implement this function
    let lnn = lnn::LiquidNeuralNetwork::new(neuron_count);

    // Here I will set up the neural network, start the simulation, etc.

    // Placeholder example of using ndarray, rand and rayon
    let _rng = rand::thread_rng();
    let range = Uniform::new(-1.0, 1.0);
    let mut random_matrix: Array2<f64> = Array2::random((10, 10), range);
    let processed_matrix: () = random_matrix.par_mapv_inplace(|elem| (elem.sin()).powi(2)); // using rayon's parallel iterator here; // applying a non-linear transformation

    // This would be replaced by LNN logic
    println!("{:?}", processed_matrix);

    // For demonstration purposes
    info!("Processing completed");

    let output_array2 = lnn.decode_output();
    // Convert the output to a Vec<Vec<f64>>
    let outputs: Vec<Vec<f64>> = output_array2.into_iter().map(|elem| vec![elem]).collect();
    output::write_output_to_csv("output_data.csv", &outputs)?;

    // Write output data to CSV
    output::write_output_to_csv("output_data.csv", &outputs)?;

    // Finalize application logic
    info!("Processing completed");

    Ok(())
}
