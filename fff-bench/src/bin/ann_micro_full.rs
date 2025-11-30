//! End-to-end micro-index benchmark: generate vectors, write via FFF writer,
//! then issue micro-indexed reads with the official reader.

use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::time::Instant;
use std::{fs, sync::Arc};

use anyhow::Result;
use arrow_array::{
    builder::{FixedSizeListBuilder, Float32Builder},
    RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use fff_poc::{
    options::FileWriterOptionsBuilder,
    reader::FileReaderV2Builder,
    writer::FileWriter,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Parser, Debug)]
struct Args {
    /// Number of vectors to generate.
    #[arg(long, default_value_t = 16_384)]
    rows: usize,
    /// Dimensionality of each vector.
    #[arg(long, default_value_t = 16)]
    dim: usize,
    /// Number of queries to run.
    #[arg(long, default_value_t = 8)]
    queries: usize,
    /// Reuse existing file if present.
    #[arg(long)]
    reuse: bool,
    /// Output path for the generated FFF file.
    #[arg(long, default_value = "tmp_test/ann_micro_full.fff")]
    output: PathBuf,
}

fn build_batch(rows: usize, dim: usize, rng: &mut StdRng) -> RecordBatch {
    let schema = Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        ),
        false,
    )]);
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for _ in 0..rows {
        for _ in 0..dim {
            builder.values().append_value(rng.gen::<f32>());
        }
        builder.append(true);
    }
    let array = Arc::new(builder.finish());
    RecordBatch::try_new(Arc::new(schema), vec![array]).unwrap()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(42);
    let batch = if args.reuse && args.output.exists() {
        None
    } else {
        Some(build_batch(args.rows, args.dim, &mut rng))
    };

    if let Some(batch) = batch.as_ref() {
        if let Some(parent) = args.output.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&args.output)?;
        let options = FileWriterOptionsBuilder::with_defaults()
            .enable_micro_index(true)
            .build();
        let start = Instant::now();
        let mut writer = FileWriter::try_new(batch.schema(), file, options)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        writer
            .write_batch(batch)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let counters = writer.finish().map_err(|e| anyhow::anyhow!(e.to_string()))?;
        println!(
            "write done: rows={}, dim={}, chunks_written={:?}, elapsed={:?}",
            args.rows,
            args.dim,
            counters.iter().map(|c| c.index_size).collect::<Vec<_>>(),
            start.elapsed()
        );
    } else {
        println!("reusing existing file at {}", args.output.display());
    }

    let mut queries: Vec<Vec<f32>> = (0..args.queries)
        .map(|_| (0..args.dim).map(|_| rng.gen::<f32>()).collect())
        .collect();

    let file = File::open(&args.output)?;
    let mut reader = FileReaderV2Builder::new(Arc::new(file))
        .with_enable_micro_index(true)
        .build()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let start = Instant::now();
    let mut total_slices = 0usize;
    for (i, q) in queries.drain(..).enumerate() {
        let slices = reader
            .search_vector_column(0, &q, 1)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        total_slices += slices.iter().map(|a| a.len()).sum::<usize>();
        println!(
            "query {} -> {} slices (first len={})",
            i,
            slices.len(),
            slices.first().map(|a| a.len()).unwrap_or(0)
        );
    }
    println!(
        "micro search: queries={}, total_vectors={}, elapsed={:?}",
        args.queries,
        total_slices,
        start.elapsed()
    );

    Ok(())
}
