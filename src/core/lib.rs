use std::path::PathBuf;

use pyo3::prelude::*;

pub mod compile_data;
pub mod glob_expansion;

#[pyfunction]
fn build_file_list(search_dir: &str, pattern: &str) -> PyResult<Vec<PathBuf>> {
    let file_list =
        glob_expansion::find_files(search_dir, pattern).expect("No files could be found.");

    Ok(file_list)
}

#[pyfunction]
fn collate_results(file_list: Vec<PathBuf>) -> PyResult<()> {
    compile_data::variant_compilation(file_list).expect("Variant data could not be compiled.");

    Ok(())
}

/// A Python data compiler module implemented in Rust.
#[pymodule]
fn amplityper_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(collate_results, m)?)?;
    m.add_function(wrap_pyfunction!(build_file_list, m)?)?;
    Ok(())
}
