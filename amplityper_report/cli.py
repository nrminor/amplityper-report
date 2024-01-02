#!/usr/bin/env python3

"""
`cli` handles command line argument parsing as well as configuration YAML parsing.
"""

import argparse
import os
from pathlib import Path
from typing import cast

from icecream import ic  # type: ignore # pylint: disable=import-error
from pydantic.dataclasses import dataclass
from result import Err, Ok, Result
from strictyaml import Int, Map, Str, YAMLError, load  # type: ignore


@dataclass(frozen=True, kw_only=True)
class ConfigParams:
    """
    The available config parameters provided in YAML format to this module are
    converted into this dataclass for easier access downstream and more strict
    typing of each field.
    """

    ivar_pattern: Path
    fasta_pattern: Path
    tidyvcf_pattern: Path
    fasta_split_char: str
    fasta_split_index: int
    ivar_split_char: str
    id_split_index: int
    hap_split_index: int


def parse_command_line_args() -> Result[argparse.Namespace, str]:
    """
        Parse command line arguments while passing errors onto main.

    Args:
        `None`

    Returns:
        `Result[argparse.Namespace, str]`: A Result type containing an `argparse` namespace
        or an error message string.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        "-d",
        type=Path,
        required=True,
        help="Results 'root' directory to traverse in search if iVar tables and other files",
    )
    parser.add_argument(
        "--gene_bed",
        "-b",
        type=Path,
        required=True,
        help="BED of amplicons where the final column contains genes associated with each amplicon",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        default="config.yaml",
        help="YAML file used to configure module such that it avoids harcoding",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=bool,
        required=False,
        default=False,
        help="Whether to resume if intermediate files from previous runs are detected.",
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
            "tidyvcf_pattern": Str(),
            "fasta_split_char": Str(),
            "fasta_split_index": Int(),
            "ivar_split_char": Str(),
            "id_split_index": Int(),
            "hap_split_index": Int(),
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
        tidyvcf_pattern=cast(Path, config_dict.get("tidyvcf_pattern")),
        fasta_split_char=str(config_dict.get("fasta_split_char")),
        fasta_split_index=cast(int, config_dict.get("fasta_split_index")),
        ivar_split_char=str(config_dict.get("ivar_split_char")),
        id_split_index=cast(int, config_dict.get("id_split_index")),
        hap_split_index=cast(int, config_dict.get("hap_split_index")),
    )

    ic("Configurations parsed.")
    ic(params)

    return Ok(params)
