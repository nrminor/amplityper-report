#!/usr/bin/env python3

"""
`transform_data` contains functions that perform a series of transformations on Polars
Lazy/DataFrames, which will ultimately serve as the final report.
"""

from pathlib import Path
from typing import List

import polars as pl
from icecream import ic  # type: ignore # pylint: disable=import-error
from tqdm import tqdm  # type: ignore # pylint: disable=import-error

from .cli import ConfigParams
from .compile_data import compile_contig_depths, compile_mutation_codons


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
            pl.when(pl.col("ALT").is_null())
            .then(None)
            .otherwise(
                pl.concat_str(
                    [pl.col("REF"), pl.col("POS"), pl.col("ALT")], separator="-"
                )
            )
            .alias("NUC_SUB")
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

    progress_bar = tqdm(total=len(sample_amp_dfs))
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
    whether_resume: bool,
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
    codon_df = compile_mutation_codons(tvcf_list, whether_resume)

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
