//! Configures the reservoir as classic logic gates and prints their truth tables.
//!
//! Run with:
//! ```bash
//! cargo run --example logic_gates
//! ```
//!
//! Each gate is realised analytically inside a 10-neuron reservoir by
//! [`LiquidNeuralNetwork::configure_for_logic_gate`]. The gate's truth value is read from
//! the output neuron: `sigmoid(potential) > 0.5` (i.e. potential > 0) counts as logical 1.

use liquid_neural_network::prelude::*;

/// Number of timesteps to run the reservoir for each input pattern. The synchronous,
/// delayed dynamics settle well within this window for every gate.
const TIMESTEPS: usize = 10;

fn main() -> Result<()> {
    // (gate name, input patterns). NOT is unary; the rest are binary.
    let two_input = [
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let not_input = [vec![0.0], vec![1.0]];

    let gates: [(&str, &[Vec<f64>]); 6] = [
        ("AND", &two_input),
        ("OR", &two_input),
        ("NOT", &not_input),
        ("NAND", &two_input),
        ("NOR", &two_input),
        ("XOR", &two_input),
    ];

    for (name, patterns) in gates {
        // A fresh 10-neuron reservoir configured for this gate.
        let mut lnn = LiquidNeuralNetwork::new(10);
        lnn.configure_for_logic_gate(name)?;

        println!("{name} gate");
        println!("{}", "-".repeat(24));
        for pattern in patterns {
            let outputs = lnn.run_simulation(TIMESTEPS, vec![pattern.clone()])?;
            let raw = outputs
                .last()
                .and_then(|row| row.first())
                .copied()
                .unwrap_or(0.0);
            let bit = u8::from(raw > 0.5);
            let inputs: Vec<u8> = pattern.iter().map(|&x| x as u8).collect();
            println!("  {inputs:?} -> {bit}  (raw {raw:.4})");

            // Reset membrane potentials before the next pattern.
            lnn.reset_state();
        }
        println!();
    }

    Ok(())
}
