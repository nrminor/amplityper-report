# Amplityper Core Rust Functions

When finding and reading very large numbers of files, Python may not be the nest choice. To speed up these tasks, along with potentially very large glob wildcard expansions, we have rewritten the relevant functions in Rust and wrapped them into Python functions with [PyO3](https://pyo3.rs/v0.20.0/). To build these core functions, simply run `pip install maturin` globally or in your virtual environment, and then run `maturin develop --release`. This will compile the Rust library, create Python bindings, and expose them to the Amplityper Report python module.
