use globmatch;
use std::path::PathBuf;

pub fn find_files(search_dir: &str, pattern: &str) -> Result<Vec<PathBuf>, String> {
    let builder = globmatch::Builder::new(pattern)
        .case_sensitive(true)
        .build(search_dir)?;

    let paths: Vec<_> = builder
        .into_iter()
        .filter_entry(|p| !globmatch::is_hidden_entry(p))
        .flatten()
        .collect();

    Ok(paths)
}
