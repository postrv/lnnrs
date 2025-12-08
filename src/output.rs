// src/output.rs
//! Output handling for the Liquid Neural Network.
//!
//! Provides functionality for writing output data to CSV files.

use crate::error::Result;
use std::fs::File;
use std::path::Path;

/// Writes output data to a CSV file.
///
/// Each Vec<f64> in the outputs becomes a row in the CSV file.
///
/// # Arguments
/// * `file_path` - Path to the output CSV file
/// * `outputs` - The output data to write
///
/// # Returns
/// * `Ok(())` if the write succeeds
/// * `Err(LnnError)` if the file cannot be created or written
///
/// # Example
/// ```ignore
/// let outputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// write_output_to_csv("results.csv", &outputs)?;
/// ```
pub fn write_output_to_csv<P: AsRef<Path>>(file_path: P, outputs: &[Vec<f64>]) -> Result<()> {
    let path = file_path.as_ref();
    let mut wtr = csv::Writer::from_writer(File::create(path)?);

    for output in outputs {
        wtr.write_record(output.iter().map(|&val| val.to_string()))?;
    }

    wtr.flush()?;
    log::debug!("Wrote {} output rows to '{}'", outputs.len(), path.display());
    Ok(())
}

/// Writes output data to a string in CSV format.
///
/// This is useful for testing or when the output needs to be captured in memory.
///
/// # Arguments
/// * `outputs` - The output data to write
///
/// # Returns
/// * `Ok(String)` containing the CSV-formatted data
/// * `Err(LnnError)` if serialization fails
pub fn write_output_to_string(outputs: &[Vec<f64>]) -> Result<String> {
    let mut wtr = csv::Writer::from_writer(Vec::new());

    for output in outputs {
        wtr.write_record(output.iter().map(|&val| val.to_string()))?;
    }

    let data = wtr.into_inner().map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    })?;

    Ok(String::from_utf8_lossy(&data).to_string())
}

/// Writes output data with headers to a CSV file.
///
/// # Arguments
/// * `file_path` - Path to the output CSV file
/// * `headers` - Column headers
/// * `outputs` - The output data to write
///
/// # Returns
/// * `Ok(())` if the write succeeds
/// * `Err(LnnError)` if the file cannot be created or written
pub fn write_output_with_headers<P: AsRef<Path>>(
    file_path: P,
    headers: &[&str],
    outputs: &[Vec<f64>],
) -> Result<()> {
    let path = file_path.as_ref();
    let mut wtr = csv::Writer::from_writer(File::create(path)?);

    // Write headers
    wtr.write_record(headers)?;

    // Write data
    for output in outputs {
        wtr.write_record(output.iter().map(|&val| val.to_string()))?;
    }

    wtr.flush()?;
    log::debug!(
        "Wrote {} output rows with headers to '{}'",
        outputs.len(),
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::fs;

    #[test]
    fn test_write_to_string() {
        let outputs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let csv_string = write_output_to_string(&outputs).unwrap();

        assert!(csv_string.contains("1"));
        assert!(csv_string.contains("2"));
        assert!(csv_string.contains("3"));
    }

    #[test]
    fn test_write_to_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let outputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        write_output_to_csv(temp_file.path(), &outputs).unwrap();

        let contents = fs::read_to_string(temp_file.path()).unwrap();
        assert!(contents.contains("1"));
        assert!(contents.contains("2"));
    }

    #[test]
    fn test_write_empty_outputs() {
        let temp_file = NamedTempFile::new().unwrap();
        let outputs: Vec<Vec<f64>> = vec![];

        let result = write_output_to_csv(temp_file.path(), &outputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_write_with_headers() {
        let temp_file = NamedTempFile::new().unwrap();
        let headers = vec!["x", "y", "z"];
        let outputs = vec![vec![1.0, 2.0, 3.0]];

        write_output_with_headers(temp_file.path(), &headers, &outputs).unwrap();

        let contents = fs::read_to_string(temp_file.path()).unwrap();
        assert!(contents.contains("x,y,z"));
        assert!(contents.contains("1"));
    }
}
