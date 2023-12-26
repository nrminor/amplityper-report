// Inpired by Python prototype:
/*
    simplename = os.path.basename(file).replace("_ivar.tsv", "")
    sample_id = simplename.split(config.ivar_split_char)[config.id_split_index]
    haplotype = simplename.split(config.ivar_split_char)[config.hap_split_index]
    amplicon = simplename.replace(sample_id, "").replace(haplotype, "")
    amplicon = amplicon[1:] if amplicon[0] == "_" else amplicon
    amplicon = amplicon[:-1] if amplicon[-1] == "_" else amplicon
*/
fn _parse_ivar_name() {}

// Inspired by Python prototype:
/*
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
*/
fn _compile_with_io() {}

// Inspired by Python prototype (which is still Rust under the hood):
/*
    # lazily scan the new tmp tsv for usage downstream
    pl.scan_csv("tmp.tsv", separator="\t", infer_schema_length=1500).sort(
        "POS"
    ).sink_ipc("tmp.arrow", compression="zstd")
*/
fn _convert_to_arrow() {}

// Meant to replace the entire loop in `compile_data_with_io()` in the py module
pub fn variant_compilation() {}
