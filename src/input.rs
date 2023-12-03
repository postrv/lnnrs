// src/input.rs

use csv;
use std::error::Error;
use std::fs::File;

/// Reads input data from a CSV file into a Vec<Vec<f64>>.
pub fn read_input_from_csv(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_reader(File::open(file_path)?);
    let mut inputs = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let input_values: Vec<f64> = record
            .iter()
            .map(|field| field.parse::<f64>().unwrap_or_default())
            .collect();
        inputs.push(input_values);
    }

    Ok(inputs)
}
