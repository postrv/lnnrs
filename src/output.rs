// src/output.rs

use csv;
use std::error::Error;
use std::fs::File;

/// Writes output data to a CSV file.
pub fn write_output_to_csv(file_path: &str, outputs: &[Vec<f64>]) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_writer(File::create(file_path)?);

    for output in outputs {
        wtr.write_record(output.iter().map(|&val| val.to_string()))?;
    }
    wtr.flush()?;

    Ok(())
}