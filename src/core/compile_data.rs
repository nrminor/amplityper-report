use polars::prelude::IpcCompression::ZSTD;
use polars::prelude::*;
use polars_io::ipc::IpcWriter;
use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::{
    fs::{self},
    path::{Path, PathBuf},
};

struct IVarMetadata {
    amplicon: String,
    sample_id: String,
    haplotype: String,
}

fn parse_ivar_name(ivar_path: &Path) -> Result<IVarMetadata, String> {
    let simplename = ivar_path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .replace("_ivar.tsv", "");

    let sample_id = simplename.split('_').collect::<Vec<&str>>()[0].to_owned();
    let haplotype = simplename
        .split('_')
        .collect::<Vec<&str>>()
        .last()
        .unwrap()
        .to_string();
    let mut amplicon = simplename
        .replace(&sample_id, "")
        .replace(&haplotype, "")
        .to_string();
    if let Some('_') = amplicon.chars().next() {
        amplicon = amplicon[1..amplicon.len() - 1].to_string();
    }

    let meta = IVarMetadata {
        amplicon,
        sample_id,
        haplotype,
    };

    Ok(meta)
}

fn compile_with_io(
    file: &PathBuf,
    writer: &mut BufWriter<fs::File>,
    write_header: bool,
    file_meta: &IVarMetadata,
) -> Result<(), String> {
    // read the ivar table
    let mut ivar_lf = LazyCsvReader::new(file)
        .with_separator(b'\t')
        .finish()
        .unwrap();

    // add empty row if iVar found no variants
    if ivar_lf.clone().collect().unwrap().height() == 0 {
        let ivar_df = ivar_lf.collect().unwrap();
        let colnames = ivar_df.get_column_names();

        // Create a vector to store the new Series
        let mut new_series_vec: Vec<Series> = Vec::with_capacity(colnames.len());

        for col in colnames {
            let series = Series::new_null(col, 1);
            new_series_vec.push(series);
        }

        ivar_lf = DataFrame::new(new_series_vec).unwrap().lazy();
    }

    // prepare a new column of haplotype IDs
    ivar_lf = ivar_lf
        .with_column(lit(file_meta.amplicon.clone()).alias("Amplicon"))
        .with_column(lit(file_meta.sample_id.clone()).alias("Sample ID"))
        .with_column(lit(file_meta.haplotype.clone()).alias("Contig"))
        .with_column(
            lit(format!(
                "{}-{}-{}",
                file_meta.amplicon, file_meta.sample_id, file_meta.haplotype
            ))
            .alias("Amplicon-Sample-Contig"),
        );

    // append the dataframe to the temp tsv
    CsvWriter::new(writer)
        .include_header(write_header)
        .with_separator(b'\t')
        .finish(&mut ivar_lf.collect().unwrap())
        .unwrap();

    Ok(())
}

fn _convert_to_arrow() -> Result<DataFrame, PolarsError> {
    let mut tmp_arrow = File::create("tmp.arrow").expect("could not create arrow file.");

    {
        let mut tmp = CsvReader::from_path("tmp.tsv").unwrap().finish().unwrap();

        IpcWriter::new(&mut tmp_arrow)
            .with_compression(Some(ZSTD))
            .finish(&mut tmp)
            .expect("Could not write to arrow file.");
    }

    IpcReader::new(tmp_arrow).finish()
}

// Meant to replace the entire loop in `compile_data_with_io()` in the py module
pub fn variant_compilation(file_list: Vec<PathBuf>) -> Result<(), String> {
    // Do away with files from previous runs (this will be replaced with tempfiles)
    if Path::new("tmp.tsv").exists() {
        let _ = fs::remove_file(Path::new("tmp.tsv"));
    }
    if Path::new("tmp.arrow").exists() {
        let _ = fs::remove_file(Path::new("tmp.arrow"));
    }

    // Make sure there are actually iVar files to go through
    if file_list.is_empty() {
        return Err(String::from("No files found to compile data from."));
    }

    // Create a temporary TSV file with append set to true
    let temp_tsv = OpenOptions::new().create(true).append(true).open("tmp.tsv").expect(
        "Failed to create new temporary TSV file. Please double check file writing permissions.",
    );

    // buffer the new TSV
    let mut writer = BufWriter::new(temp_tsv);

    // For each iVar file in the glob expansion, read its contents and append them
    // to a temporary TSV
    for (i, file) in file_list.iter().enumerate() {
        let file_meta = parse_ivar_name(file).expect("File metadata could not be parsed.");

        // quick test of whether the header should be written. It is only written
        // the first time
        let write_header = i == 0;

        // compile the data
        compile_with_io(file, &mut writer, write_header, &file_meta).unwrap();
    }

    Ok(())
}
