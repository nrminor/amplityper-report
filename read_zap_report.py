#!/usr/bin/env python3

"""
This module compiles and evaluates the results from the READ-ZAP amplicon-based
haplotype phasing pipeline. In particular, it reports the haplotypes at each amplicon,
sorted by the frequencies, alongside the synonymous and nonsynonymous mutations in
those haplotypes.

Example usage:
```
usage: read_zap_report.py [-h] --results_dir RESULTS_DIR [--config CONFIG]

options:
    -h, --help          show this help message and exit
    --results_dir RESULTS_DIR, -d RESULTS_DIR
                        Results 'root' directory to traverse in search if iVar tables and
                        other files.
    --config CONFIG, -c CONFIG
                        YAML file used to configure module such that it avoids harcoding.
```
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import cast, List
from result import Ok, Err, Result
from strictyaml import load, Map, YAMLError, Str  # , Str, Int  # type: ignore
import polars as pl


@dataclass
class ConfigParams:
    """
    The available config parameters provided in YAML format to this module are
    converted into this dataclass for easier access downstream and more strict
    typing of each field.
    """

    ivar_pattern: Path
    fasta_pattern: Path


def parse_command_line_args() -> Result[argparse.Namespace, str]:
    """
    Parse command line arguments while passing errors onto main.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        "-d",
        type=Path,
        required=True,
        help="Results 'root' directory to traverse in search if iVar tables and other files..",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        default="config.yaml",
        help="YAML file used to configure module such that it avoids harcoding.",
    )
    args = parser.parse_args()
    if len(vars(args)) < 1:
        return Err("Arguments could not properly be parsed.")

    return Ok(args)


def parse_configurations(config_path: Path) -> Result[ConfigParams, str]:
    """
        This module uses the library `strictYAML` to perform very strict, statically
        typed configuration. This helps catch errors downstream while also doing away with
        most hardcoding, which is likely to cause runtime errors for other users. Like
        the other functions in this module, it uses Rust-like result types to handle
        potential errors and pass them "up" to the main function.

    Args:
        `config_path: str`: A simple file path to a YAML-formatted config file in string
        format.

    Returns:
        `Result[ConfigParams, str]`: A result-type containing either an instance of the
        dataclass `ConfigParams` or an error message in string format.
    """

    # define the statically typed schema for the config parameters
    schema = Map(
        {
            "ivar_pattern": Str(),
            "fasta_pattern": Str(),
        }
    )

    # use the schema and the provided path to parse out a dictionary of parameters
    try:
        config_dict = cast(
            dict, load((Path(os.getcwd()) / config_path).read_text(), schema).data
        )
    except TypeError as message:
        return Err(f"The provided YAML file could not be parsed:\n{message}")
    except YAMLError as message:
        return Err(f"The provided YAML file could not be parsed:\n{message}")

    params = ConfigParams(
        ivar_pattern=cast(Path, config_dict.get("ivar_pattern")),
        fasta_pattern=cast(Path, config_dict.get("fasta_pattern")),
    )

    return Ok(params)


def construct_file_list(
    results_dir: Path, glob_pattern: Path
) -> Result[List[Path], str]:
    """
        Function `construct_file_list()` uses the provided path of wildcards
        to expand out all available files to be compiled downstream.

    Args:
        - `results_dir: Path`: A Pathlib path instance recording the results
        "root" directory, which is the top of the READ-ZAP results hierarchy.
        - `ivar_pattern: Path`: A Pathlib path instance containing wildcards
        that can be expanded to the desired files.

    Returns:
        - `Result[List[Path], str]`: A Result type instance containing either
        a list of paths to the desired files, or an error message string.
    """

    # collect a list of all the files to search
    files_to_query = list(results_dir.glob(str(glob_pattern)))

    # make sure that there aren't duplicates
    try:
        set(files_to_query)
    except ValueError as message:
        return Err(f"Redudant files present that may corrupt results:\n{message}")

    if len(files_to_query) == 0:
        return Err(
            f"No files found that match the wildcard path:\n{glob_pattern}\nwithin {results_dir}"
        )

    return Ok(files_to_query)


def compile_data_with_io(file_list: List[Path]) -> Result[pl.LazyFrame, str]:
    """
        Function `compile_data_with_io()` takes the list of paths and
        reads each file, parsing it with Polars, and writing it into one
        large temporary TSV file. This method for compiling all files into
        one involves a great deal of read-write, but it also avoids potential
        type mismatch issues between a type schema inferred for one dataframe
        and the type schema inferred for the next. Downstream, the new
        TSV can be parsed into a single dataframe where it's possible to
        infer a type scheme from many rows.

    Args:
        - `file_list: List[Path]`: A list of paths, where each path points to
        a TSV file generated by iVar.

    Returns:
        - `pl.LazyFrame`: A Polars LazyFrame to be queries and transformed
        downstream.
    """

    # Double check that a tempfile from a previous run isn't present
    if os.path.isfile("tmp.tsv"):
        os.remove("tmp.tsv")

    if len(file_list) == 0:
        return Err("No files found to compile data from.")

    # compile all tables into one large temporary table
    for i, file in enumerate(file_list):
        with open("tmp.tsv", "a", encoding="utf-8") as temp:
            # Parse out information in the file path to add into the dataframe
            # NOTE: this hardcoding will eventually be replaced with config params
            amplicon = str(file.parent).split("results/amplicon_")[1].split("/")[0]
            simplename = os.path.basename(file).replace(".tsv", "")
            sample_id = simplename.split("_")[0]
            contig = simplename.split("_")[-1]
            if len(sample_id) == 1:
                continue

            # quick test of whether the header should be written. It is only written
            # the first time
            write_header = bool(i == 0)

            # Read the csv, modify it, and write it onto the growing temporary tsv
            pl.read_csv(
                file, separator="\t", raise_if_empty=True, null_values=["NA", ""]
            ).with_columns(
                pl.lit(sample_id).alias("Sample ID"),
                pl.lit(amplicon).alias("Amplicon"),
                pl.lit(f"{amplicon}-{sample_id}-{contig}").alias(
                    "Amplicon-Sample-Contig"
                ),
            ).write_csv(temp, separator="\t", include_header=write_header)

    # lazily scan the new tmp tsv for usage downstream
    all_contigs = pl.scan_csv("tmp.tsv", separator="\t", infer_schema_length=1500).sort(
        "POS"
    )

    return Ok(all_contigs)


def main() -> None:
    """
    Main coordinates the flow of data through the functions defined in `read_zap_report.py`.
    It also handles errors and error messages so that the script can be debugged without
    long tracebacks.
    """

    # parse command line arguments while handling any errors
    args_result = parse_command_line_args()
    if isinstance(args_result, Err):
        sys.exit(
            f"Command line argument parsing encountered an error.\n{args_result.unwrap_err()}"
        )
    results_dir = args_result.unwrap().results_dir
    config = args_result.unwrap().config

    # parse configurations from the config YAML file
    config_result = parse_configurations(config)
    if isinstance(config_result, Err):
        sys.exit(
            f"Config file parsing parsing encountered an error.\n{config_result.unwrap_err()}"
        )
    ivar_pattern = config_result.unwrap().ivar_pattern

    # make a list of the iVar files to query based on the provided wildcard path
    ivar_list_result = construct_file_list(results_dir, ivar_pattern)
    if isinstance(config_result, Err):
        sys.exit(
            f"No files found at the provided wildcard path:\n{ivar_list_result.unwrap_err()}"
        )
    ivar_list = ivar_list_result.unwrap_or([])

    # compile all files into one Polars LazyFrame to be queried downstream
    all_contigs_result = compile_data_with_io(ivar_list)
    if isinstance(all_contigs_result, Err):
        sys.exit(
            f"A dataframe could not be compiled.\n{all_contigs_result.unwrap_err()}"
        )
    all_contigs = all_contigs_result.unwrap()

    # temporarily write file to TSV for debugging
    all_contigs.sink_ipc("debug.arrow", compression="zstd")
    os.remove("tmp.tsv")

    # aggregate the lazyframe so that a comma-delimited list of
    # nucleotide and amino-acid substitutions are in their own columns,
    # with synonymous and nonsynonymous mutations parsed out into two
    # more columns. Then, select the handful of columns of interest and
    # deduplicate rows.

    # Make a list of FASTA files to pull information from

    # add FASTA information about depth of coverage per-contig consensus
    # onto the lazyframe with a join

    # Sort the lazyframe first by contig frequency and then by position/amplicon


if __name__ == "__main__":
    main()
