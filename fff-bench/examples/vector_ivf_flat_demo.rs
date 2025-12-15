use anyhow::{anyhow, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::FixedSizeListArray;
use clap::Parser;
use fff_poc::reader::{FileReaderV2Builder, Projection, Selection};
use fff_vindex::artifact::ivf_flat::{
    build_ivf_flat_artifact_with_options, load_ivf_flat_artifact, search_ivf_flat_artifact_native,
    IvfFlatArtifactBuildOptions, PostingCodec,
};
use fff_vindex::artifact::wasm_ivf_flat::WasmIvfFlatKernel;
use fff_vindex::ivf_flat::{
    build_ivf_flat_sidecar, load_ivf_flat_index, search_ivf_flat, IvfFlatBuildOptions,
};
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    base_f3: PathBuf,

    /// Leaf column index of the vector column in the base file schema.
    #[arg(long)]
    vector_leaf_index: usize,

    /// Vector dimension.
    #[arg(long)]
    dim: usize,

    #[arg(long, default_value = "emb_ivf_flat")]
    index_name: String,

    #[arg(long, default_value_t = 1024)]
    nlist: usize,

    #[arg(long, default_value_t = 200_000)]
    train_sample: usize,

    #[arg(long, default_value_t = 1)]
    seed: u64,

    #[arg(long, default_value_t = 20)]
    max_kmeans_iters: usize,

    #[arg(long, default_value_t = 10)]
    k: usize,

    #[arg(long, default_value_t = 8)]
    nprobe: usize,

    /// Number of queries; queries are read from rows `[0..nq)`.
    #[arg(long, default_value_t = 1)]
    nq: usize,

    /// Build and query the IVF artifact container instead of the legacy ivf_flat file.
    #[arg(long, default_value_t = false)]
    artifact: bool,

    /// Run IVF search via a Wasm kernel (path to the `.wasm` module). Implies `--artifact`.
    #[arg(long)]
    artifact_wasm_kernel: Option<PathBuf>,

    /// Posting codec for artifact: `raw` or `row_id_delta_varint_v1`.
    #[arg(long, default_value = "raw")]
    artifact_posting_codec: String,

    /// Disable host-side chunk cache for wasm kernel (forces reads each time).
    #[arg(long, default_value_t = false)]
    artifact_wasm_no_cache: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let queries = read_first_vectors(&args.base_f3, args.vector_leaf_index, args.dim, args.nq)?;

    let build_opts = IvfFlatBuildOptions {
        nlist: args.nlist,
        train_sample: args.train_sample,
        seed: args.seed,
        max_kmeans_iters: args.max_kmeans_iters,
    };

    let (index_path, results) = if let Some(wasm_path) = &args.artifact_wasm_kernel {
        let posting_codec = match args.artifact_posting_codec.as_str() {
            "raw" => PostingCodec::Raw,
            "row_id_delta_varint_v1" => PostingCodec::RowIdDeltaVarintV1,
            other => return Err(anyhow!("unsupported --artifact-posting-codec: {other}")),
        };
        let index_path = build_ivf_flat_artifact_with_options(
            &args.base_f3,
            args.vector_leaf_index,
            args.dim,
            &args.index_name,
            build_opts,
            IvfFlatArtifactBuildOptions { posting_codec },
        )
        .with_context(|| "build ivf-flat artifact")?;
        let artifact = load_ivf_flat_artifact(&index_path)?;
        let artifact = Arc::new(artifact);
        let mut kernel = WasmIvfFlatKernel::load(wasm_path, Arc::clone(&artifact))
            .with_context(|| "load wasm ivf-flat kernel")?;
        kernel.set_cache_enabled(!args.artifact_wasm_no_cache);
        kernel.reset_stats();
        let results_batch = if args.nq == 1 {
            vec![kernel.search(&artifact, &queries, args.k, args.nprobe)?]
        } else {
            kernel.search_batch(&artifact, &queries, args.nq, args.k, args.nprobe)?
        };
        let stats = kernel.stats();
        println!(
            "wasm stats: cache_hits={} chunks_fetched={} compressed_bytes_in={} raw_bytes_decoded={} fetch_time_ms={:.3} total_time_ms={:.3}",
            stats.cache_hits,
            stats.chunks_fetched,
            stats.compressed_bytes_in,
            stats.raw_bytes_decoded,
            (stats.fetch_time_ns as f64) / 1e6,
            (stats.total_time_ns as f64) / 1e6,
        );
        println!(
            "wasm chunks(sample): {:?}",
            stats.fetched_chunk_ids_sample
        );
        println!(
            "wasm chunks(miss sample): {:?}",
            stats.cache_miss_chunk_ids_sample
        );
        (index_path, results_batch.into_iter().next().unwrap_or_default())
    } else if args.artifact {
        let posting_codec = match args.artifact_posting_codec.as_str() {
            "raw" => PostingCodec::Raw,
            "row_id_delta_varint_v1" => PostingCodec::RowIdDeltaVarintV1,
            other => return Err(anyhow!("unsupported --artifact-posting-codec: {other}")),
        };
        let index_path = build_ivf_flat_artifact_with_options(
            &args.base_f3,
            args.vector_leaf_index,
            args.dim,
            &args.index_name,
            build_opts,
            IvfFlatArtifactBuildOptions { posting_codec },
        )
        .with_context(|| "build ivf-flat artifact")?;
        let artifact = load_ivf_flat_artifact(&index_path)?;
        let mut all = Vec::new();
        for q in 0..args.nq {
            let query = &queries[q * args.dim..(q + 1) * args.dim];
            all.push(search_ivf_flat_artifact_native(&artifact, query, args.k, args.nprobe)?);
        }
        (index_path, all.into_iter().next().unwrap_or_default())
    } else {
        let index_path = build_ivf_flat_sidecar(
            &args.base_f3,
            args.vector_leaf_index,
            args.dim,
            &args.index_name,
            build_opts,
        )
        .with_context(|| "build ivf-flat index")?;
        let index = load_ivf_flat_index(&index_path)?;
        let mut all = Vec::new();
        for q in 0..args.nq {
            let query = &queries[q * args.dim..(q + 1) * args.dim];
            all.push(search_ivf_flat(&index, query, args.k, args.nprobe)?);
        }
        (index_path, all.into_iter().next().unwrap_or_default())
    };

    println!("index: {}", index_path.display());
    println!("top{} (nprobe={}):", results.len(), args.nprobe);
    for r in results {
        println!("row_id={} dist={}", r.row_id, r.distance);
    }
    Ok(())
}

fn read_first_vectors(base_f3: &PathBuf, leaf: usize, dim: usize, nq: usize) -> Result<Vec<f32>> {
    let file = File::open(base_f3).with_context(|| format!("open {}", base_f3.display()))?;
    let row_indexes: Vec<u64> = (0..nq as u64).collect();
    let mut reader = FileReaderV2Builder::new(Arc::new(file))
        .with_projections(Projection::All)
        .with_selection(Selection::RowIndexes(row_indexes))
        .build()
        .map_err(|e| anyhow!(e.to_string()))?;
    let batches = reader
        .read_file()
        .map_err(|e| anyhow!(e.to_string()))
        .with_context(|| "read first vector")?;
    let schema = batches
        .first()
        .ok_or_else(|| anyhow!("no batches"))?
        .schema();
    let batch = if batches.len() == 1 {
        batches[0].clone()
    } else {
        arrow::compute::concat_batches(&schema, &batches)?
    };
    if leaf >= batch.num_columns() {
        return Err(anyhow!(
            "vector_leaf_index out of range: {} >= {}",
            leaf,
            batch.num_columns()
        ));
    }
    let col = batch.column(leaf);
    let fsl = col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| anyhow!("projected column is not FixedSizeListArray"))?;
    if fsl.value_length() as usize != dim {
        return Err(anyhow!(
            "dim mismatch: expected {}, got {}",
            dim,
            fsl.value_length()
        ));
    }
    let values = fsl.values().as_primitive::<arrow::datatypes::Float32Type>();
    let total = nq * dim;
    if values.values().len() < total {
        return Err(anyhow!(
            "not enough values for nq={}, dim={}: have {}",
            nq,
            dim,
            values.values().len()
        ));
    }
    Ok(values.values()[..total].to_vec())
}
