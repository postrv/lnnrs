// src/input.rs
//! Input handling for the Liquid Neural Network.
//!
//! Provides functionality for reading input data from CSV files.

use crate::error::{LnnError, Result};
use std::fs::File;
use std::path::Path;

/// Reads input data from a CSV file into a `Vec<Vec<f64>>`.
///
/// Each row in the CSV file becomes a `Vec<f64>` containing the parsed values.
/// Non-numeric values are parsed as 0.0.
///
/// # Arguments
/// * `file_path` - Path to the CSV file to read
///
/// # Returns
/// * `Ok(Vec<Vec<f64>>)` - The parsed input data
/// * `Err(LnnError)` - If the file cannot be opened or parsed
///
/// # Example
/// ```ignore
/// let inputs = read_input_from_csv("data/inputs.csv")?;
/// for row in inputs {
///     println!("Row: {:?}", row);
/// }
/// ```
pub fn read_input_from_csv<P: AsRef<Path>>(file_path: P) -> Result<Vec<Vec<f64>>> {
    let path = file_path.as_ref();
    let file = File::open(path).map_err(|e| {
        LnnError::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to open file '{}': {}", path.display(), e),
        ))
    })?;

    let mut rdr = csv::Reader::from_reader(file);
    let mut inputs = Vec::new();

    for (row_idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| {
            LnnError::Csv(csv::Error::from(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Error reading row {}: {}", row_idx + 1, e),
            )))
        })?;

        let input_values: Vec<f64> = record
            .iter()
            .map(|field| field.trim().parse::<f64>().unwrap_or(0.0))
            .collect();

        inputs.push(input_values);
    }

    log::debug!("Read {} input rows from '{}'", inputs.len(), path.display());
    Ok(inputs)
}

/// Reads input data from a CSV string.
///
/// This is useful for testing or when the CSV data is already in memory.
///
/// # Arguments
/// * `csv_data` - A string containing CSV-formatted data
///
/// # Returns
/// * `Ok(Vec<Vec<f64>>)` - The parsed input data
/// * `Err(LnnError)` - If the data cannot be parsed
pub fn read_input_from_string(csv_data: &str) -> Result<Vec<Vec<f64>>> {
    let mut rdr = csv::Reader::from_reader(csv_data.as_bytes());
    let mut inputs = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let input_values: Vec<f64> = record
            .iter()
            .map(|field| field.trim().parse::<f64>().unwrap_or(0.0))
            .collect();
        inputs.push(input_values);
    }

    Ok(inputs)
}

/// Validates that input data has consistent dimensions.
///
/// # Arguments
/// * `inputs` - The input data to validate
///
/// # Returns
/// * `Ok(())` if all rows have the same length
/// * `Err(LnnError)` if rows have inconsistent lengths
pub fn validate_inputs(inputs: &[Vec<f64>]) -> Result<()> {
    if inputs.is_empty() {
        return Ok(());
    }

    let expected_len = inputs[0].len();
    for (idx, row) in inputs.iter().enumerate() {
        if row.len() != expected_len {
            return Err(LnnError::Configuration(format!(
                "Inconsistent input dimensions: row {} has {} elements, expected {}",
                idx,
                row.len(),
                expected_len
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_from_string() {
        let csv_data = "a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0";
        let inputs = read_input_from_string(csv_data).unwrap();

        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(inputs[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_read_from_string_with_invalid_values() {
        let csv_data = "a,b,c\n1.0,invalid,3.0\n4.0,5.0,6.0";
        let inputs = read_input_from_string(csv_data).unwrap();

        assert_eq!(inputs[0], vec![1.0, 0.0, 3.0]); // "invalid" becomes 0.0
    }

    #[test]
    fn test_read_from_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "a,b,c").unwrap();
        writeln!(temp_file, "1.0,2.0,3.0").unwrap();
        writeln!(temp_file, "4.0,5.0,6.0").unwrap();

        let inputs = read_input_from_csv(temp_file.path()).unwrap();

        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_read_nonexistent_file() {
        let result = read_input_from_csv("/nonexistent/path/file.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_consistent_inputs() {
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        assert!(validate_inputs(&inputs).is_ok());
    }

    #[test]
    fn test_validate_inconsistent_inputs() {
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]]; // Different lengths
        assert!(validate_inputs(&inputs).is_err());
    }

    #[test]
    fn test_validate_empty_inputs() {
        let inputs: Vec<Vec<f64>> = vec![];
        assert!(validate_inputs(&inputs).is_ok());
    }
}
