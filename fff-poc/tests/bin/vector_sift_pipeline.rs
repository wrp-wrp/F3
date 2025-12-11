use std::{
    collections::HashSet,
    fs::{self, File},
    io::{self, BufReader},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context, Result};
use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use fff_core::errors as f3_errors;
use fff_poc::{
    options::FileWriterOptions,
    reader::FileReaderV2Builder,
    vector_index::{
        encode_bruteforce_index, encode_hnsw_index, QuantizationSpec, VectorDistanceMetric,
        VectorIndexAlgorithm, VectorIndexConfig, VectorSearchResult,
    },
    writer::FileWriter,
};
use tempfile::tempfile;

#[derive(Parser, Debug)]
#[command(
    name = "vector-sift-pipeline",
    about = "Build an F3 vector index from the SIFT1M dataset and run recall measurements."
)]
struct Args {
    /// Directory that contains the unpacked sift_base.fvecs / sift_query.fvecs / sift_groundtruth.ivecs files.
    #[arg(long, default_value = "data/sift1m")]
    dataset_dir: PathBuf,

    /// Number of base vectors to use when building the index (0 = entire file).
    #[arg(long, default_value_t = 100_000)]
    train_size: usize,

    /// Number of queries to evaluate (0 = entire query file).
    #[arg(long, default_value_t = 1_000)]
    query_size: usize,

    /// Number of nearest neighbors per query.
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Algorithm to evaluate: "brute", "hnsw", or "wasm".
    #[arg(long, default_value = "wasm")]
    algorithm: String,

    /// Maximum HNSW degree (ignored for brute-force).
    #[arg(long, default_value_t = 32)]
    hnsw_m: usize,

    /// ef_search parameter for HNSW queries (ignored for brute-force).
    #[arg(long, default_value_t = 64)]
    hnsw_ef_search: usize,

    /// Path to the compiled Wasm module when --algorithm=wasm is selected.
    #[arg(long)]
    wasm_module: Option<PathBuf>,

    /// Optional path to dump the raw index blob for later reuse.
    #[arg(long)]
    output_blob: Option<PathBuf>,

    /// Optional path to persist the generated F3 file.
    #[arg(long)]
    output_file: Option<PathBuf>,

    /// Compare against the native HNSW runtime (same index blob, no Wasm).
    #[arg(long)]
    compare_native: bool,

    /// Number of queries to bundle per Wasm call (1 = disabled).
    #[arg(long, default_value_t = 1)]
    wasm_batch_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineAlgorithm {
    Brute,
    Hnsw,
    Wasm,
}

#[derive(Clone, Debug)]
struct VariantReport {
    label: String,
    algorithm: PipelineAlgorithm,
    stats: EvalStats,
    build_time: Duration,
    index_blob_bytes: usize,
    output_file: Option<PathBuf>,
}

struct RunVariantArgs<'a> {
    label: &'a str,
    algorithm: PipelineAlgorithm,
    base_vectors: &'a [Vec<f32>],
    queries: &'a [Vec<f32>],
    truth: &'a [Vec<u32>],
    k: usize,
    hnsw_m: usize,
    hnsw_ef_search: usize,
    wasm_module: Option<Vec<u8>>,
    output_blob: Option<PathBuf>,
    output_file: Option<PathBuf>,
    available_rows: usize,
    wasm_batch_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let base_path = args.dataset_dir.join("sift_base.fvecs");
    let query_path = args.dataset_dir.join("sift_query.fvecs");
    let truth_path = args.dataset_dir.join("sift_groundtruth.ivecs");

    ensure!(
        base_path.exists() && query_path.exists() && truth_path.exists(),
        "Expected SIFT files (sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs) under {:?}",
        args.dataset_dir
    );

    let base_vectors =
        read_fvecs(&base_path, args.train_size).context("failed to read sift_base.fvecs")?;
    let queries =
        read_fvecs(&query_path, args.query_size).context("failed to read sift_query.fvecs")?;
    let truth =
        read_ivecs(&truth_path, queries.len()).context("failed to read sift_groundtruth.ivecs")?;

    ensure!(
        !base_vectors.is_empty(),
        "No base vectors loaded from {:?}",
        base_path
    );
    ensure!(
        !queries.is_empty(),
        "No queries loaded from {:?}",
        query_path
    );
    ensure!(
        base_vectors[0].len() == queries[0].len(),
        "Base dimension ({}) != query dimension ({})",
        base_vectors[0].len(),
        queries[0].len()
    );

    ensure!(
        args.wasm_batch_size >= 1,
        "--wasm-batch-size must be at least 1"
    );

    let pipeline_algo = parse_algorithm(&args.algorithm)?;
    if args.compare_native {
        ensure!(
            matches!(pipeline_algo, PipelineAlgorithm::Wasm),
            "--compare-native requires --algorithm=wasm"
        );
    }

    let wasm_module_bytes = match pipeline_algo {
        PipelineAlgorithm::Wasm => {
            let wasm_path = args.wasm_module.as_ref().ok_or_else(|| {
                anyhow::anyhow!("--wasm-module must be provided when --algorithm=wasm")
            })?;
            Some(read_wasm_module(wasm_path)?)
        }
        _ => None,
    };

    println!(
        "Dataset size: {} vectors, {} queries (dim = {}, train_size = {}, query_size = {})",
        base_vectors.len(),
        queries.len(),
        base_vectors[0].len(),
        args.train_size,
        args.query_size
    );

    let primary_label = format!("primary-{}", algorithm_label(pipeline_algo));
    let primary_report = run_variant(RunVariantArgs {
        label: &primary_label,
        algorithm: pipeline_algo,
        base_vectors: &base_vectors,
        queries: &queries,
        truth: &truth,
        k: args.k,
        hnsw_m: args.hnsw_m,
        hnsw_ef_search: args.hnsw_ef_search,
        wasm_module: wasm_module_bytes.clone(),
        output_blob: args.output_blob.clone(),
        output_file: args.output_file.clone(),
        available_rows: base_vectors.len(),
        wasm_batch_size: args.wasm_batch_size,
    })?;
    print_variant_report(&primary_report);

    if args.compare_native {
        let baseline_report = run_variant(RunVariantArgs {
            label: "native-hnsw",
            algorithm: PipelineAlgorithm::Hnsw,
            base_vectors: &base_vectors,
            queries: &queries,
            truth: &truth,
            k: args.k,
            hnsw_m: args.hnsw_m,
            hnsw_ef_search: args.hnsw_ef_search,
            wasm_module: None,
            output_blob: None,
            output_file: None,
            available_rows: base_vectors.len(),
            wasm_batch_size: 1,
        })?;
        print_variant_report(&baseline_report);
        print_comparison(&primary_report, &baseline_report);
    }

    Ok(())
}

fn algorithm_label(algorithm: PipelineAlgorithm) -> &'static str {
    match algorithm {
        PipelineAlgorithm::Brute => "brute-force",
        PipelineAlgorithm::Hnsw => "native-hnsw",
        PipelineAlgorithm::Wasm => "wasm-hnsw",
    }
}

fn read_wasm_module(path: &PathBuf) -> Result<Vec<u8>> {
    if !path.exists() {
        bail!(
            "Wasm module path {:?} does not exist. Build the module first.",
            path
        );
    }
    fs::read(path).with_context(|| format!("Failed to read Wasm module from {:?}", path))
}

fn run_variant(args: RunVariantArgs<'_>) -> Result<VariantReport> {
    const VECTOR_INDEX_ID: u32 = 1;
    let RunVariantArgs {
        label,
        algorithm,
        base_vectors,
        queries,
        truth,
        k,
        hnsw_m,
        hnsw_ef_search,
        wasm_module,
        output_blob,
        output_file,
        available_rows,
        wasm_batch_size,
    } = args;

    if matches!(algorithm, PipelineAlgorithm::Wasm) && wasm_module.is_none() {
        bail!("Wasm module bytes missing for Wasm pipeline run");
    }
    let use_batch = matches!(algorithm, PipelineAlgorithm::Wasm) && wasm_batch_size > 1;

    let build_start = Instant::now();
    let index_blob = match algorithm {
        PipelineAlgorithm::Brute => map_vector_result(encode_bruteforce_index(base_vectors))?,
        PipelineAlgorithm::Hnsw | PipelineAlgorithm::Wasm => {
            map_vector_result(encode_hnsw_index(base_vectors, hnsw_m, hnsw_ef_search))?
        }
    };
    let build_time = build_start.elapsed();
    let index_blob_bytes = index_blob.len();

    if let Some(path) = output_blob {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, &index_blob)
            .with_context(|| format!("unable to write blob to {:?}", path))?;
    }

    let output_path = output_file;
    let file_handle: Arc<File> = if let Some(path) = output_path.as_ref() {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        Arc::new(
            File::create(path)
                .with_context(|| format!("Failed to create output file at {:?}", path))?,
        )
    } else {
        Arc::new(tempfile().context("failed to create temporary output file")?)
    };

    let schema = Arc::new(Schema::new(vec![Field::new(
        "values",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )?;

    let vector_config = VectorIndexConfig {
        index_id: VECTOR_INDEX_ID,
        column: "values".to_string(),
        algorithm: match algorithm {
            PipelineAlgorithm::Brute => VectorIndexAlgorithm::BruteForce,
            PipelineAlgorithm::Hnsw => VectorIndexAlgorithm::Hnsw,
            PipelineAlgorithm::Wasm => VectorIndexAlgorithm::CustomWasm,
        },
        distance_metric: VectorDistanceMetric::L2,
        priority: 0,
        usage_hint: Some(label.to_string()),
        quantization: QuantizationSpec {
            dimension: base_vectors[0].len() as u32,
            segments: vec![],
        },
        custom_params: vec![],
        data: index_blob,
        wasm_module,
    };
    let options = FileWriterOptions::builder()
        .add_vector_index(vector_config)
        .build();
    let mut writer = map_vector_result(FileWriter::try_new(schema, file_handle.clone(), options))?;
    map_vector_result(writer.write_batch(&batch))?;
    map_vector_result(writer.finish())?;

    let mut reader = map_vector_result(FileReaderV2Builder::new(file_handle.clone()).build())?;
    let stats = if use_batch {
        evaluate_recall_batch(
            queries,
            truth,
            k,
            available_rows,
            wasm_batch_size,
            |batch, k| map_vector_result(reader.vector_knn_l2_batch(VECTOR_INDEX_ID, batch, k)),
        )?
    } else {
        evaluate_recall(queries, truth, k, available_rows, |query, k| {
            map_vector_result(reader.vector_knn_l2(VECTOR_INDEX_ID, query, k))
        })?
    };

    Ok(VariantReport {
        label: label.to_string(),
        algorithm,
        stats,
        build_time,
        index_blob_bytes,
        output_file: output_path,
    })
}

fn print_variant_report(report: &VariantReport) {
    println!(
        "[{}] algorithm = {} | recall = {:.4} | avg latency = {:.3} ms | QPS = {:.2} | total query time = {:.2}s",
        report.label,
        algorithm_label(report.algorithm),
        report.stats.recall,
        report.stats.avg_ms,
        report.stats.qps,
        report.stats.total_time.as_secs_f64()
    );
    println!(
        "[{}] index build = {:.2}s | blob size = {:.2} MB",
        report.label,
        report.build_time.as_secs_f64(),
        report.index_blob_bytes as f64 / (1024.0 * 1024.0)
    );
    if let Some(path) = &report.output_file {
        println!("[{}] F3 file written to {:?}", report.label, path);
    }
}

fn print_comparison(primary: &VariantReport, baseline: &VariantReport) {
    let qps_ratio = if baseline.stats.qps > 0.0 {
        primary.stats.qps / baseline.stats.qps
    } else {
        0.0
    };
    println!(
        "[compare] {} vs {} | recall delta = {:+.4} | avg latency delta = {:+.3} ms | QPS ratio = {:.2}",
        primary.label,
        baseline.label,
        primary.stats.recall - baseline.stats.recall,
        primary.stats.avg_ms - baseline.stats.avg_ms,
        qps_ratio
    );
}

fn parse_algorithm(value: &str) -> Result<PipelineAlgorithm> {
    match value.to_ascii_lowercase().as_str() {
        "brute" | "bruteforce" | "brute-force" => Ok(PipelineAlgorithm::Brute),
        "hnsw" | "hnsw-native" => Ok(PipelineAlgorithm::Hnsw),
        "wasm" | "custom" => Ok(PipelineAlgorithm::Wasm),
        other => bail!(
            "Unsupported algorithm '{}'. Use 'brute', 'hnsw', or 'wasm'.",
            other
        ),
    }
}

fn read_fvecs(path: &Path, limit: usize) -> Result<Vec<Vec<f32>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut vectors = Vec::new();
    loop {
        if limit > 0 && vectors.len() >= limit {
            break;
        }
        let dim = match reader.read_i32::<LittleEndian>() {
            Ok(value) => value as usize,
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err.into()),
        };
        ensure!(dim > 0, "vector dimension must be positive");
        let mut vec = vec![0f32; dim];
        for value in vec.iter_mut() {
            *value = reader.read_f32::<LittleEndian>()?;
        }
        vectors.push(vec);
    }
    Ok(vectors)
}

fn read_ivecs(path: &Path, limit: usize) -> Result<Vec<Vec<u32>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    loop {
        if limit > 0 && rows.len() >= limit {
            break;
        }
        let dim = match reader.read_i32::<LittleEndian>() {
            Ok(value) => value as usize,
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err.into()),
        };
        ensure!(dim > 0, "ground-truth record dimension must be positive");
        let mut row = Vec::with_capacity(dim);
        for _ in 0..dim {
            row.push(reader.read_i32::<LittleEndian>()? as u32);
        }
        rows.push(row);
    }
    Ok(rows)
}

#[derive(Clone, Debug)]
struct EvalStats {
    recall: f64,
    avg_ms: f64,
    qps: f64,
    total_time: Duration,
}

fn evaluate_recall<F>(
    queries: &[Vec<f32>],
    truth: &[Vec<u32>],
    k: usize,
    available_rows: usize,
    mut knn: F,
) -> Result<EvalStats>
where
    F: FnMut(&[f32], usize) -> Result<Vec<VectorSearchResult>>,
{
    ensure!(!queries.is_empty(), "no queries available for evaluation");
    ensure!(
        truth.len() >= queries.len(),
        "ground truth rows ({}) < query count ({})",
        truth.len(),
        queries.len()
    );
    let mut total_hits = 0usize;
    let mut total_targets = 0usize;
    let mut total_time = Duration::ZERO;

    for (idx, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = knn(query, k)?;
        total_time += start.elapsed();
        let truth_row = &truth[idx];
        let expected = build_expected_set(truth_row, k, available_rows);
        total_targets += expected.len();
        for result in results {
            if expected.contains(&(result.row_id as u32)) {
                total_hits += 1;
            }
        }
    }

    let recall = if total_targets == 0 {
        0.0
    } else {
        total_hits as f64 / total_targets as f64
    };
    let avg_ms = total_time.as_secs_f64() * 1_000.0 / queries.len() as f64;
    let qps = if total_time.is_zero() {
        0.0
    } else {
        queries.len() as f64 / total_time.as_secs_f64()
    };

    Ok(EvalStats {
        recall,
        avg_ms,
        qps,
        total_time,
    })
}

fn evaluate_recall_batch<F>(
    queries: &[Vec<f32>],
    truth: &[Vec<u32>],
    k: usize,
    available_rows: usize,
    batch_size: usize,
    mut knn_batch: F,
) -> Result<EvalStats>
where
    F: FnMut(&[Vec<f32>], usize) -> Result<Vec<Vec<VectorSearchResult>>>,
{
    ensure!(!queries.is_empty(), "no queries available for evaluation");
    ensure!(
        truth.len() >= queries.len(),
        "ground truth rows ({}) < query count ({})",
        truth.len(),
        queries.len()
    );
    ensure!(batch_size > 0, "batch_size must be positive");
    let mut total_hits = 0usize;
    let mut total_targets = 0usize;
    let mut total_time = Duration::ZERO;

    let mut idx = 0;
    while idx < queries.len() {
        let end = (idx + batch_size).min(queries.len());
        let chunk = &queries[idx..end];
        let start = Instant::now();
        let batch_results = knn_batch(chunk, k)?;
        total_time += start.elapsed();
        ensure!(
            batch_results.len() == chunk.len(),
            "batch executor returned {} results but {} queries were provided",
            batch_results.len(),
            chunk.len()
        );
        for (offset, results) in batch_results.into_iter().enumerate() {
            let truth_row = &truth[idx + offset];
            let expected = build_expected_set(truth_row, k, available_rows);
            total_targets += expected.len();
            for result in results {
                if expected.contains(&(result.row_id as u32)) {
                    total_hits += 1;
                }
            }
        }
        idx = end;
    }

    let recall = if total_targets == 0 {
        0.0
    } else {
        total_hits as f64 / total_targets as f64
    };
    let avg_ms = total_time.as_secs_f64() * 1_000.0 / queries.len() as f64;
    let qps = if total_time.is_zero() {
        0.0
    } else {
        queries.len() as f64 / total_time.as_secs_f64()
    };

    Ok(EvalStats {
        recall,
        avg_ms,
        qps,
        total_time,
    })
}

fn build_expected_set(truth_row: &[u32], k: usize, available_rows: usize) -> HashSet<u32> {
    let mut expected = Vec::with_capacity(k);
    for &neighbor in truth_row {
        if neighbor < available_rows as u32 {
            expected.push(neighbor);
        }
        if expected.len() == k {
            break;
        }
    }
    expected.into_iter().collect()
}

fn map_vector_result<T>(result: f3_errors::Result<T>) -> Result<T> {
    result.map_err(|err| anyhow::anyhow!(err.to_string()))
}
