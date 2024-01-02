#!/usr/bin/env python3

"""
This module compiles and evaluates the results from the Amplityper amplicon-based
haplotype phasing pipeline. In particular, it reports the haplotypes at each amplicon,
sorted by the frequencies, alongside the synonymous and nonsynonymous mutations in
those haplotypes.


```
usage: at_report [-h] --results_dir RESULTS_DIR --gene_bed GENE_BED [--config CONFIG] [--resume RESUME]

options:
  -h, --help            show this help message and exit
  --results_dir RESULTS_DIR, -d RESULTS_DIR
                        Results 'root' directory to traverse in search of iVar tables and other files
  --gene_bed GENE_BED, -b GENE_BED
                        BED of amplicons where the final column contains genes associated with each amplicon
  --config CONFIG, -c CONFIG
                        YAML file used to configure module such that it avoids harcoding
  --resume RESUME, -r RESUME
                        Whether to resume if intermediate files from previous runs are detected.
```
"""

import os
import sys

from result import Err

from .cli import parse_command_line_args, parse_configurations
from .compile_data import (
    compile_data_with_io,
    collect_file_lists
)
from .transform_data import aggregate_haplotype_df


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
    whether_resume = args_result.unwrap().resume

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
    all_contigs_result = compile_data_with_io(
        clean_ivar_list, config_result.unwrap(), whether_resume
    )
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
        whether_resume,
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
    if os.path.isfile("tmp.arrow") and whether_resume is False:
        os.remove("tmp.arrow")


if __name__ == "__main__":
    main()
