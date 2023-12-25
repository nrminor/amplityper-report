#!/usr/bin/env python3

"""
This module compiles and evaluates the results from the Amplityper amplicon-based
haplotype phasing pipeline. In particular, it reports the haplotypes at each amplicon,
sorted by the frequencies, alongside the synonymous and nonsynonymous mutations in
those haplotypes.


```
usage: amplityper_report.py [-h] --results_dir RESULTS_DIR --gene_bed GENE_BED [--config CONFIG]

options:
    -h, --help            show this help message and exit
    --results_dir RESULTS_DIR, -d RESULTS_DIR
                        Results 'root' directory to traverse in search if iVar tables and other files
    --gene_bed GENE_BED, -b GENE_BED
                        BED of amplicons where the final column contains genes associated with each amplicon
    --config CONFIG, -c CONFIG
                        YAML file used to configure module such that it avoids harcoding
```
"""

import argparse
import os
import sys
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import polars as pl
from icecream import ic  # type: ignore # pylint: disable=import-error
from pydantic.dataclasses import dataclass
from result import Err, Ok, Result
from strictyaml import Int, Map, Str, YAMLError, load  # type: ignore
from tqdm import tqdm  # type: ignore # pylint: disable=import-error


@dataclass
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


def construct_file_list(
    results_dir: Path, glob_pattern: Path
) -> Result[List[Path], str]:
    """
        Function `construct_file_list()` uses the provided path of wildcards
        to expand out all available files to be compiled downstream.

    Args:
        `results_dir: Path`: A Pathlib path instance recording the results
        "root" directory, which is the top of the Amplityper results hierarchy.
        `ivar_pattern: Path`: A Pathlib path instance containing wildcards
        that can be expanded to the desired files.

    Returns:
        `Result[List[Path], str]`: A Result type instance containing either
        a list of paths to the desired files, or an error message string.
    """

    ic("Constructing file list:")
    ic(glob_pattern)

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


def _check_cleanliness(path_parts: tuple[str, ...]) -> bool:
    """
    Helper function to make sure that the file matched with a wildcard
    pattern is not a macOS '._' file generated during Spotlight indexing
    or Time Machine backups.
    """

    for part in path_parts:
        if "._" in part:
            return False
    return True


def compile_data_with_io(
    file_list: List[Path], config: ConfigParams
) -> Result[pl.LazyFrame, str]:
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
        `file_list: List[Path]`: A list of paths, where each path points to
        a TSV file generated by iVar.

    Returns:
        `pl.LazyFrame`: A Polars LazyFrame to be queries and transformed
        downstream.
    """

    ic("Compiling variant data for each contig.")

    # Double check that foles from a previous run aren't present
    if os.path.isfile("tmp.tsv"):
        os.remove("tmp.tsv")
    if os.path.isfile("tmp.arrow"):
        os.remove("tmp.arrow")

    if len(file_list) == 0:
        return Err("No files found to compile data from.")

    # compile all tables into one large temporary table
    progress_bar = tqdm(total=len(file_list), ncols=100)
    with open("tmp.tsv", "a", encoding="utf-8") as temp:
        for i, file in enumerate(file_list):
            progress_bar.update(1)
            # Parse out information in the file path to add into the dataframe
            simplename = os.path.basename(file).replace("_ivar.tsv", "")
            sample_id = simplename.split(config.ivar_split_char)[config.id_split_index]
            haplotype = simplename.split(config.ivar_split_char)[config.hap_split_index]
            amplicon = simplename.replace(sample_id, "").replace(haplotype, "")

            amplicon = amplicon[1:] if amplicon[0] == "_" else amplicon
            amplicon = amplicon[:-1] if amplicon[-1] == "_" else amplicon

            # quick test of whether the header should be written. It is only written
            # the first time
            write_header = bool(i == 0)

            # Read the csv, modify it, and write it onto the growing temporary tsv
            ivar_df = pl.read_csv(
                file, separator="\t", raise_if_empty=True, null_values=["NA", ""]
            )

            # make empty row for haplotype info if iVar called no variants
            if ivar_df.shape[0] == 0:
                ivar_df.vstack(ivar_df.clear(n=1), in_place=True)

            # Add sample, amplicon, and haplotype information
            ivar_df.with_columns(
                pl.lit(sample_id).alias("Sample ID"),
                pl.lit(amplicon).alias("Amplicon"),
                pl.lit(haplotype).alias("Contig"),
                pl.lit(f"{amplicon}-{sample_id}-{haplotype}").alias(
                    "Amplicon-Sample-Contig"
                ),
            ).write_csv(temp, separator="\t", include_header=write_header)
    progress_bar.close()

    ic("Converting variant data to compressed arrow format.")

    # lazily scan the new tmp tsv for usage downstream
    pl.scan_csv("tmp.tsv", separator="\t", infer_schema_length=1500).sort(
        "POS"
    ).sink_ipc("tmp.arrow", compression="zstd")

    all_contigs = pl.scan_ipc("tmp.arrow", memory_map=False)
    os.remove("tmp.tsv")

    return Ok(all_contigs)


def collect_file_lists(
    results_dir: Path, pattern1: Path, pattern2: Path, pattern3: Path
) -> Result[Tuple[List[Path], List[Path], List[Path]], str]:
    """
        Function `collect_file_lists()` quarterbacks sequential executions of
        the `construct_file_list()` function, each with a different wildcard
        pattern. Doing so keeps the size of `main()` more reasonable and will
        also make potential refactoring easier in the future.

    Args:
        `results_dir: Path`: A Pathlib Path type pointing to the results "root"
        directory to be searched with the following glob wildcard patterns.
        `pattern1: Path`: A Pathlib Path type, which can contain wildcards to
        be expanded with the glob library.
        `pattern2: Path`: A Pathlib Path type, which can contain wildcards to
        be expanded with the glob library.
        `pattern3: Path`: A Pathlib Path type, which can contain wildcards to
        be expanded with the glob library.

    Returns:
        `Result[List[List[Path]], str]`: A Result type containing eaither a list of
        lists or an error message string.
    """

    # make a list of the iVar files to query based on the provided wildcard path
    ivar_list_result = construct_file_list(results_dir, pattern1)
    if isinstance(ivar_list_result, Err):
        return Err(
            f"No files found at the provided wildcard path:\n{ivar_list_result.unwrap_err()}"
        )
    clean_ivar_list = [
        file
        for file in ivar_list_result.unwrap_or([])
        if _check_cleanliness(file.parts)
    ]

    # Make a list of FASTA files to pull information from
    fasta_list_result = construct_file_list(results_dir, pattern2)
    if isinstance(fasta_list_result, Err):
        return Err(
            f"No FASTAs found at the provided wildcard path:\n{fasta_list_result.unwrap_err()}"
        )
    clean_fasta_list = [
        file for file in fasta_list_result.unwrap() if _check_cleanliness(file.parts)
    ]

    # Make a list of "tidy" vcf files to pull codon information from
    tvcf_result = construct_file_list(results_dir, pattern3)
    if isinstance(tvcf_result, Err):
        return Err(
            f"No 'tidy' VCF tables found at the provided wildcard path:\n{tvcf_result.unwrap_err()}"
        )
    clean_tvcf_list = [
        file for file in tvcf_result.unwrap() if _check_cleanliness(file.parts)
    ]

    return Ok((clean_ivar_list, clean_fasta_list, clean_tvcf_list))


def _try_parse_int(value: str) -> Optional[int]:
    """
    Helper function that handles the possibility that a read support
    cannot be parsed as an integer from the FASTA defline and returns
    `None` instead of raising an unrecoverable error.
    """
    try:
        return int(value)
    except ValueError:
        return None


def _try_parse_identifier(defline: str) -> Optional[str]:
    """
    Helper function that splits a contig's FASTA defline by the "_" symbol,
    pulls a sample id assuming it is the second item in the underscore-split
    defline, parses out the contig name assuming that string starts with
    "contig", and constructs an amplicon name-sample id-contig identifier
    that will serve as a unique key for each contig across all samples
    and amplicons.
    """

    cleaned_defline = defline.replace(">", "").split(" ")[0]
    items = cleaned_defline.split("_")
    sample_id = items[0]
    (haplotype,) = [item for item in items if "haplotype" in item]
    amplicon = cleaned_defline.replace(sample_id, "").replace(haplotype, "")

    amplicon = amplicon[1:] if amplicon[0] == "_" else amplicon
    amplicon = amplicon[:-1] if amplicon[-1] == "_" else amplicon

    identifier = f"{amplicon}-{sample_id}-{haplotype}"

    return identifier


def _is_valid_utf8(fasta_line: str) -> bool:
    """
    Helper function that double checks that each FASTA defline is valid
    UTF-8 text.
    """
    try:
        fasta_line.encode("utf-8").decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def generate_seq_dict(
    input_fasta: List[str], config: ConfigParams
) -> Optional[Dict[Optional[str], Optional[int]]]:
    """
        The function `generate_seq_dict()` uses a series of list comprehensions
        to 1) Test that a FASTA file can be decoded into proper UTF-8 text; 2)
        Parse out the name of the current amplicon; And 3) Parse out a FASTA's
        defline and extract a sample ID and a depth of coverage. It then saves
        these pieces of information into a dictionary, where each key is a
        contig's unique identifier, and each value is the integer depth-of-
        coverage.

    Args:
        `input_fasta: List[str]`: Parsed FASTA lines store as strings in a list.
        `split_char: str`: The character, e.g. "-", to use for splitting the
        FASTA defline to parse out information.

    Returns:
        `Optional[Dict[Optional[str], Optional[int]]]`: An option type, with
        option types being a union of type T or None, that wraps the contig
        identifier-depth dictionary. Each item in the dictionary can also be
        None.
    """

    # make sure the lines can be decoded
    decodable = [_is_valid_utf8(line) for line in input_fasta]
    if False in decodable:
        return None

    deflines = [line for line in input_fasta if line.startswith(">")]
    supports = [
        _try_parse_int(line.split(config.fasta_split_char)[config.fasta_split_index])
        for line in input_fasta
        if line.startswith(">")
    ]
    identifiers = [_try_parse_identifier(defline) for defline in deflines]

    assert len(deflines) == len(
        identifiers
    ), "Mismatch between the number of deflines and number of sequences"

    seq_dict = dict(zip(identifiers, supports))

    return seq_dict


def compile_mutation_codons(tvcf_list: List[Path]) -> pl.LazyFrame:
    """
        Function `compile_mutation_codons()` loops through all "tidy" VCF files
        in a provided list of Paths and creates a Polars LazyFrame containing
        1) A column of nucleotide mutation strings, and 2) a column of codon
        numbers for use with amino acid substitutions.

    Args:
        `tvcf_list: List[Path]`: A list containing Pathlib Path types pointing to
        any number of tidy VCF files to be assessed.

    Returns:
        `pl.LazyFrame`: A Polars LazyFrame query that will be evaluated downstream.
    """

    ic("Compiling codon numbers for all coding mutations.")

    if os.path.isfile("tmp.tvcf"):
        os.remove("tmp.tvcf")

    header_written = False

    progress_bar = tqdm(total=len(tvcf_list), ncols=100)
    with open("tmp.tvcf", "a", encoding="utf-8") as tmp_file:
        for tidy_vcf in tvcf_list:
            progress_bar.update(1)
            variants = pl.read_csv(tidy_vcf, separator="\t")

            # do a few checks to make sure the loop doesn't hang on unexpected
            # file writes
            if variants.shape[0] == 0:
                continue
            if len(set(variants.columns).intersection({"ref", "pos", "alt"})) != 3:
                continue
            if "info_ANN" not in variants.columns and "NUC_SUB" not in variants.columns:
                continue

            variants = variants.with_columns(
                pl.concat_str(
                    [pl.col("ref"), pl.col("pos"), pl.col("alt")], separator="-"
                ).alias("NUC_SUB")
            ).select(["NUC_SUB", "info_ANN"])

            # separate out nucleotide substitutions and annotations
            nuc_subs = variants.select("NUC_SUB").to_series().to_list()
            anns = variants.select("info_ANN").to_series().to_list()

            # Separate out the tenth annotation, which is the amino acid substitution
            aa_vars = [ann.split("|")[10] for ann in anns]

            # Separate out the codon numbers and make sure they are a numeric type
            codons = ["".join((x for x in codon if x.isdigit())) for codon in aa_vars]
            num_codons = [int(codon) if codon != "" else None for codon in codons]

            # decide whether to write header or just append rows
            if header_written is False:
                write_header = True
                header_written = True
            else:
                write_header = False

            # write out
            pl.DataFrame({"NUC_SUB": nuc_subs, "CODON": num_codons}).unique().write_csv(
                tmp_file, separator="\t", include_header=write_header
            )
    progress_bar.close()

    ic("All codons compiled in temporary file.")

    # When the full dataset is ammassed read and "lazify" it
    amassed_codons = pl.read_csv("tmp.tvcf", separator="\t", ignore_errors=True).lazy()
    os.remove("tmp.tvcf")
    return amassed_codons


def compile_contig_depths(fasta_list: List[Path], config: ConfigParams) -> pl.LazyFrame:
    """
        Function `compile_contig_depths()` loops through all FASTA files
        in a provided list of FASTA Paths and creates a Polars LazyFrame containing
        1) A column of unique identifiers for each contig, and 2) a column of read
        supports/depths of coverage for each contig.

    Args:
        `fasta_list: List[Path]`: A list containing Pathlib Path types pointing to
        any number of FASTA files to be assessed.

    Returns:
        `pl.LazyFrame`: A Polars LazyFrame query that will be evaluated downstream.
    """

    ic("Compiling contig depths for each contig FASTA.")

    seq_dicts = []

    progress_bar = tqdm(total=len(fasta_list), ncols=100)
    for fasta in fasta_list:
        progress_bar.update(1)
        with open(fasta, "r", encoding="utf-8") as fasta_contents:
            try:
                fasta_lines = fasta_contents.readlines()
            except UnicodeDecodeError:
                print(
                    f"The FASTA at the following path could not be decoded to utf-8:\n{fasta}"
                )
                continue
            seq_dict = generate_seq_dict(fasta_lines, config)
            if seq_dict is None:
                print(
                    f"The FASTA at the following path could not be decoded to utf-8:\n{fasta}"
                )
                continue
            seq_dicts.append(seq_dict)
    progress_bar.close()

    identifiers = list(
        itertools.chain.from_iterable(seq_dict.keys() for seq_dict in seq_dicts)
    )
    supports = list(
        itertools.chain.from_iterable(seq_dict.values() for seq_dict in seq_dicts)
    )

    depth_df = pl.LazyFrame(
        {"Amplicon-Sample-Contig": identifiers, "Depth of Coverage": supports}
    )

    return depth_df


def generate_gene_df(gene_bed: Path) -> pl.LazyFrame:
    """
        Function `generate_gene_df()` uses series of joins on a BED file of amplicons
        and genes to extract an "amplicon basename" alongside the start/stop positions
        and genes associated with each amplicon.

    Args:
        `gene_bed: Path`: A Pathlib Path type pointing to the location of a BED file
        with of all primers, which must contain start and stop positions for each primer,
        amplicon names, and genes

    Returns:
        `pl.LazyFrame`: A Polars LazyFrame query that will be evaluated downstream.
    """

    ic("Generating gene dataframe.")

    gene_df = (
        pl.scan_csv(
            gene_bed,
            separator="\t",
            has_header=False,
            new_columns=[
                "Ref",
                "Start Position",
                "Stop Position",
                "NAME",
                "RESPLICE_NAME",
                "INDEX",
                "Gene",
            ],
        )
        .select("RESPLICE_NAME", "Gene")
        .with_columns(
            pl.col("RESPLICE_NAME")
            .str.replace("_RIGHT", "")
            .str.replace("_LEFT", "")
            .alias("Amplicon")
        )
        .drop("RESPLICE_NAME")
        .unique()
        .join(
            (
                pl.scan_csv(
                    gene_bed,
                    separator="\t",
                    has_header=False,
                    new_columns=[
                        "Ref",
                        "Start Position",
                        "Stop Position",
                        "NAME",
                        "RESPLICE_NAME",
                        "INDEX",
                        "Gene",
                    ],
                )
                .drop("Ref", "INDEX")
                .filter(pl.col("RESPLICE_NAME").str.contains("_LEFT"))
                .drop("Stop Position")
                .with_columns(
                    pl.col("RESPLICE_NAME")
                    .str.replace("_RIGHT", "")
                    .str.replace("_LEFT", "")
                    .alias("Amplicon")
                )
                .drop("NAME", "RESPLICE_NAME", "Gene")
                .unique()
            ),
            how="left",
            on="Amplicon",
        )
        .join(
            (
                pl.scan_csv(
                    gene_bed,
                    separator="\t",
                    has_header=False,
                    new_columns=[
                        "Ref",
                        "Start Position",
                        "Stop Position",
                        "NAME",
                        "RESPLICE_NAME",
                        "INDEX",
                        "Gene",
                    ],
                )
                .drop("Ref", "INDEX")
                .filter(pl.col("RESPLICE_NAME").str.contains("_RIGHT"))
                .drop("Start Position")
                .with_columns(
                    pl.col("RESPLICE_NAME")
                    .str.replace("_RIGHT", "")
                    .str.replace("_LEFT", "")
                    .alias("Amplicon")
                )
                .drop("NAME", "RESPLICE_NAME", "Gene")
                .unique()
            ),
            how="left",
            on="Amplicon",
        )
        .sort("Start Position", "Stop Position")
    )

    return gene_df


def construct_long_df(
    all_contigs: pl.LazyFrame, gene_df: pl.LazyFrame, codon_df: pl.LazyFrame
) -> pl.LazyFrame:
    """
        Function `construct_long_df()` constructs a "long" dataframe (or more specifically,
        a Polars LazyFrame query that, when evaluated downstream, will produce a dataframe)
        that lists each mutation in each contig alongside whether it is noncoding, synonymous,
        and nonsynonymous.

    Args:
        `all_contigs: pl.LazyFrame`: A Polars lazyframe that, when queried, will produce
        a dataframe out of all iVar tables merged together.
        `gene_df: pl.LazyFrame`: A Polars lazyframe that, when queried, will produce a
        dataframe where each row represents an amplicon, its start and stop positions, and
        which gene it is in.
        `codon_df: pl.LazyFrame`: A Polars lazyframe that, when queried, will be a table of
        nucleotide mutations and associated codons within the protein versions of the gene

    Returns:
        `pl.LazyFrame`: A Polars LazyFrame query that will be evaluated downstream.
    """

    ic("Using a series of joins to construct a long dataframe of all mutations.")

    long_df = (
        all_contigs.select(
            [
                "REGION",
                "POS",
                "REF",
                "ALT",
                "REF_AA",
                "ALT_AA",
                "Amplicon",
                "Sample ID",
                "Contig",
                "Amplicon-Sample-Contig",
            ]
        )
        .unique()
        .with_columns(
            pl.concat_str(
                [pl.col("REF"), pl.col("POS"), pl.col("ALT")], separator="-"
            ).alias("NUC_SUB")
        )
        .join(gene_df, how="left", on="Amplicon")
        .join(codon_df, how="left", on="NUC_SUB")
        .unique()
        .with_columns(
            pl.when(pl.col("CODON").is_null())
            .then(
                pl.concat_str(
                    [
                        pl.col("REF_AA"),
                        pl.lit("->"),
                        pl.col("ALT_AA"),
                        pl.lit("at nuc. position"),
                        pl.col("POS"),
                    ],
                    separator=" ",
                )
            )
            .otherwise(
                pl.concat_str(
                    [pl.col("REF_AA"), pl.col("CODON"), pl.col("ALT_AA")], separator="-"
                )
            )
            .alias("AA_SUB")
        )
        .drop(["REF", "POS", "ALT"])
        .with_columns(
            pl.concat_str([pl.col("Gene"), pl.col("AA_SUB")], separator=": ").alias(
                "AA_SUB"
            )
        )
        .with_columns(
            (pl.col("ALT_AA").is_null() & pl.col("REF_AA").is_null()).alias("Noncoding")
        )
        .with_columns(
            (
                # pylint: disable-next=singleton-comparison
                (pl.col("ALT_AA") == pl.col("REF_AA")) & (pl.col("Noncoding") == False)  # noqa: E712
            ).alias("Synonymous")
        )
        .with_columns(
            (
                # pylint: disable-next=singleton-comparison
                (pl.col("ALT_AA") != pl.col("REF_AA")) & (pl.col("Noncoding") == False)  # noqa: E712
            ).alias("Nonsynonymous")
        )
        .drop(["REF_AA", "ALT_AA", "CODON"])
    )

    long_df.collect().write_ipc("haplotype_long_table.arrow", compression="zstd")

    return long_df


def construct_short_df(long_df: pl.LazyFrame, gene_df: pl.LazyFrame) -> pl.LazyFrame:
    """
        Function `construct_short_df()` essentially "pivots" the polars lazyframe produced
        in `construct_long_df()` such that mutations are stored in comma-delimited list and
        counts of nucleotide, synonymous, and non-synonymous mutations are added.

    Args:
        `long_df: pl.LazyFrame`: A Polars lazyframe that, when queried, will produce
        a long dataframe of all individual mutations in all contigs.
        `gene_df: pl.LazyFrame`: A Polars lazyframe that, when queried, will produce a
        dataframe where each row represents an amplicon, its start and stop positions, and
        which gene it is in.

    Returns:
        `pl.LazyFrame`: A Polars LazyFrame query that will be evaluated downstream.
    """

    ic("Constructing a pivoted dataframe of all unique contigs from each amplicon.")

    short_df = (
        long_df.unique(subset="Amplicon-Sample-Contig")
        .select(["Amplicon-Sample-Contig"])
        .join(
            long_df.select(
                ["Amplicon", "Sample ID", "Amplicon-Sample-Contig"]
            ).with_columns(
                pl.concat_str(
                    [pl.col("Amplicon"), pl.col("Sample ID")], separator="-"
                ).alias("Amplicon-Sample")
            ),
            on="Amplicon-Sample-Contig",
            how="left",
        )
        .unique()
        .join(gene_df, how="left", on="Amplicon")
        .join(
            long_df.select(["Amplicon-Sample-Contig", "NUC_SUB"])
            .group_by("Amplicon-Sample-Contig")
            .agg(pl.col("NUC_SUB"))
            .with_columns(
                pl.col("NUC_SUB").list.join(", ").alias("Nucleotide Substitutions")
            )
            .drop("NUC_SUB"),
            on="Amplicon-Sample-Contig",
            how="left",
        )
        .join(
            long_df.select(["Amplicon-Sample-Contig", "NUC_SUB"])
            .group_by("Amplicon-Sample-Contig")
            .agg(pl.col("NUC_SUB").count())
            .with_columns(pl.col("NUC_SUB").alias("Nuc Mut Count"))
            .drop("NUC_SUB"),
            on="Amplicon-Sample-Contig",
            how="left",
        )
        .join(
            long_df.select(["Amplicon-Sample-Contig", "AA_SUB", "Synonymous"])
            .filter(pl.col("Synonymous"))
            .group_by("Amplicon-Sample-Contig")
            .agg(pl.col("AA_SUB"))
            .with_columns(
                pl.col("AA_SUB").list.join(", ").alias("Synonymous Mutations")
            )
            .drop("AA_SUB"),
            on="Amplicon-Sample-Contig",
            how="left",
        )
        .join(
            long_df.select(["Amplicon-Sample-Contig", "AA_SUB", "Synonymous"])
            .filter(pl.col("Synonymous"))
            .group_by("Amplicon-Sample-Contig")
            .agg(pl.col("AA_SUB").count())
            .with_columns(pl.col("AA_SUB").alias("Syn count"))
            .drop("AA_SUB"),
            on="Amplicon-Sample-Contig",
            how="left",
        )
        .join(
            long_df.select(["Amplicon-Sample-Contig", "AA_SUB", "Nonsynonymous"])
            .filter(pl.col("Nonsynonymous"))
            .group_by("Amplicon-Sample-Contig")
            .agg(pl.col("AA_SUB"))
            .with_columns(
                pl.col("AA_SUB").list.join(", ").alias("Nonsynonymous Mutations")
            )
            .drop("AA_SUB"),
            on="Amplicon-Sample-Contig",
            how="left",
        )
        .join(
            long_df.select(["Amplicon-Sample-Contig", "AA_SUB", "Nonsynonymous"])
            .filter(pl.col("Nonsynonymous"))
            .group_by("Amplicon-Sample-Contig")
            .agg(pl.col("AA_SUB").count())
            .with_columns(pl.col("AA_SUB").alias("Nonsyn count"))
            .drop("AA_SUB"),
            on="Amplicon-Sample-Contig",
            how="left",
        )
        .drop("Amplicon-Sample")
        .unique()
    )

    return short_df


def compute_crude_dnds_ratio(short_df: pl.LazyFrame) -> pl.LazyFrame:
    """
        Function `compute_crude_dnds_ratio()` constructs a very crude approximation of
        πN/πS, the ratio of nonsynonymous to synonymous mutations and joins it onto
        the "short" lazyframe from `construct_short_df()`.

    Args:
        `short_df: pl.LazyFrame`: A Polars lazyframe that, when queried, will produce
        a short dataframe of all contigs, where mutations are comma-separated in a single
        column.

    Returns:
        `pl.LazyFrame`: A Polars LazyFrame query that will be evaluated downstream.
    """

    ic("Computing a crude dN/dS ratio for each contig in each sample.")

    new_df = (
        short_df.with_columns(
            (pl.col("Stop Position") - pl.col("Start Position")).alias(
                "Amplicon Length"
            )
        )
        .with_columns(
            (
                pl.col("Nonsyn count")
                / ((pl.col("Amplicon Length") * 2) - pl.col("Syn count"))
            ).alias("pn")
        )
        .with_columns(
            (
                pl.col("Syn count")
                / ((pl.col("Amplicon Length") * 2) - pl.col("Nonsyn count"))
            ).alias("ps")
        )
        .with_columns((-(3 / 4) * (1 - (4 * pl.col("pn") / 3)).log()).alias("dn"))
        .with_columns((-(3 / 4) * (1 - (4 * pl.col("ps") / 3)).log()).alias("ds"))
        .with_columns((pl.col("dn") / pl.col("ds")).alias("Crude dN/dS Ratio"))
        .drop("pn", "ps", "dn", "ds")
        .unique()
    )

    return new_df


def assign_haplotype_names(unnamed_df: pl.LazyFrame) -> pl.DataFrame:
    """
        Function `assign_haplotype_names()` sorts haplotypes/contigs within each sample
        in descending order by how well-supported by reads they are. It then gives those
        sorted haplotypes a name based on their frequences.

    Args:
        `unnamed_df: pl.LazyFrame`: A Polars lazyframe that, when queried, will produce
        a short dataframe of all contigs, where mutations are comma-separated in a single
        column, along with crude N/S ratios.

    Returns:
        `pl.DataFrame`: A collected Polars DataFrame with complete information for
        reporting.
    """

    ic("Assigning names to each putative haplotype based on their frequencies.")

    sample_amp_dfs = unnamed_df.collect().partition_by(
        "Amplicon", "Sample ID", maintain_order=False
    )

    ic("Naming each haplotype.")

    progress_bar = tqdm(total=len(sample_amp_dfs), ncols=100)
    for i, df in enumerate(sample_amp_dfs):
        progress_bar.update(1)
        cols = df.columns
        new_cols = ["Haplotype"] + cols
        new_df = (
            df.unique()
            .sort("Depth of Coverage", descending=True)
            .with_row_count(offset=1)
            .cast({"row_nr": pl.Utf8})
            .with_columns(
                pl.concat_str(
                    [pl.col("Amplicon"), pl.lit("Haplotype"), pl.col("row_nr")],
                    separator=" ",
                ).alias("Haplotype")
            )
            .drop("row_nr")
            .select(new_cols)
        )
        sample_amp_dfs[i] = new_df
    progress_bar.close()

    final_df = pl.concat(sample_amp_dfs, rechunk=True)

    return final_df


def aggregate_haplotype_df(
    long_contigs: pl.LazyFrame,
    tvcf_list: List[Path],
    clean_fasta_list: List[Path],
    gene_bed: Path,
    config: ConfigParams,
) -> pl.DataFrame:
    """
        Function `aggregate_haplotype_df()` oversees the construction of a final
        dataframe, which will be reported out as an intricately sorted Excel file.

    Args:
        `long_contigs: pl.LazyFrame`: A Polars LazyFrame query that was amassed
        from many iVar tables.
        `clean_fasta_list: List[Path]`: A list of Pathlib Paths pointing toward
        any number of FASTA files.
        `gene_bed: Path`: A Pathlib Path type pointing to the location of a BED file
        with of all primers, which must contain start and stop positions for each primer,
        amplicon names, and genes

    Returns:
        `pl.DataFrame`: A fully evaluated Polars dataframe that can be written
        out as an excel file.
    """

    ic(
        "Running function that manages the many steps of aggregating the haplotype table."
    )

    # construct data frame mapping genes to amplicons
    gene_df = generate_gene_df(gene_bed)

    # construct a codon table for all mutations
    codon_df = compile_mutation_codons(tvcf_list)

    # construct long dataframe
    long_df = construct_long_df(long_contigs, gene_df, codon_df)

    # aggregate into short dataframe
    short_df = construct_short_df(long_df, gene_df)

    # Compile depths from contig FASTA files
    depth_df = compile_contig_depths(clean_fasta_list, config)

    # add FASTA information about depth of coverage per-contig consensus
    # onto the lazyframe with a join
    ic("Joining per-contig depth information.")
    short_df_with_depth = short_df.join(
        depth_df, on="Amplicon-Sample-Contig", how="left"
    ).filter(~pl.col("Depth of Coverage").is_null())

    # generate crude dn/ds ratios
    ratio_df = compute_crude_dnds_ratio(short_df_with_depth)

    # Dynamically assign haplotype names per amplicon-sample combination
    final_df = assign_haplotype_names(ratio_df)

    return final_df.sort(
        "Sample ID",
        "Start Position",
        "Depth of Coverage",
        "Nonsyn count",
        descending=[False, False, True, True],
    ).drop("Amplicon-Sample-Contig")


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
    gene_bed = args_result.unwrap().gene_bed

    # parse configurations from the config YAML file
    config_result = parse_configurations(config)
    if isinstance(config_result, Err):
        sys.exit(
            f"Config file parsing parsing encountered an error.\n{config_result.unwrap_err()}"
        )

    # collect all necessary file lists
    collect_results = collect_file_lists(
        results_dir,
        config_result.unwrap().ivar_pattern,
        config_result.unwrap().fasta_pattern,
        config_result.unwrap().tidyvcf_pattern,
    )
    if isinstance(collect_results, Err):
        sys.exit(
            f"Glob-based file listing encountered an error.\n{collect_results.unwrap_err()}"
        )
    # pylint: disable-next=assignment-from-no-return
    clean_ivar_list, clean_fasta_list, clean_tvcf_list = collect_results.unwrap()

    # compile all files into one Polars LazyFrame to be queried downstream
    all_contigs_result = compile_data_with_io(clean_ivar_list, config_result.unwrap())
    if isinstance(all_contigs_result, Err):
        sys.exit(
            f"A dataframe could not be compiled.\n{all_contigs_result.unwrap_err()}"
        )

    # aggregate a dataframe containing information to be reported out for review
    final_df = aggregate_haplotype_df(
        all_contigs_result.unwrap(),
        clean_tvcf_list,
        clean_fasta_list,
        gene_bed,
        config_result.unwrap(),
    )

    # Sort the lazyframe first by contig frequency and then by position/amplicon
    final_df.write_excel(
        "final_report.xlsx",
        autofit=False,
        freeze_panes=(1, 0),
        header_format={"bold": True},
        conditional_formats={
            "Crude dN/dS Ratio": {
                "type": "3_color_scale",
                "mid_value": 1,
                "min_value": 0,
                "mid_color": "#FFFFFF",
            }
        },
    )

    # clear temporary arrow structure
    if os.path.isfile("tmp.arrow"):
        os.remove("tmp.arrow")


if __name__ == "__main__":
    main()
