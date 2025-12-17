use anyhow::Result;
use fff_vindex::ivf_flat::IvfFlatBuildOptions;
use fff_vindex::artifact::ivf_pq::{build_ivf_pq_artifact, IvfPqArtifact, IvfPqArtifactBuildOptions};
use fff_vindex::artifact::wasm_ivf_pq::WasmIvfPqKernel;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tempfile::tempdir;

use arrow_schema::{Schema, Field, DataType};
use arrow_array::{RecordBatch, FixedSizeListArray, Float32Array};
use arrow_array::types::Float32Type;
use fff_poc::writer::FileWriter;
use fff_poc::options::FileWriterOptions;

fn generate_random_f3_file(path: &Path, n: usize, dim: usize, rng: &mut StdRng) -> Result<()> {
    let f = File::create(path)?;
    
    let field = Field::new("vector", DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32
    ), false);
    let schema = Arc::new(Schema::new(vec![field]));
    
    let options = FileWriterOptions::default();
    let mut writer = FileWriter::try_new(schema.clone(), f, options)
        .map_err(|e| anyhow::anyhow!("FileWriter new error: {:?}", e))?;
    
    // Generate data
    let total_values = n * dim;
    let mut values = Vec::with_capacity(total_values);
    for _ in 0..total_values {
        values.push(rng.gen::<f32>());
    }
    
    let values_array = Float32Array::from(values);
    let fsl = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values_array),
        None,
    );
    
    let batch = RecordBatch::try_new(schema, vec![Arc::new(fsl)])?;
    writer.write_batch(&batch)
        .map_err(|e| anyhow::anyhow!("FileWriter write_batch error: {:?}", e))?;
    writer.finish()
        .map_err(|e| anyhow::anyhow!("FileWriter finish error: {:?}", e))?;
    
    Ok(())
}

fn main() -> Result<()> {
    let dir = tempdir()?;
    let base_path = dir.path().join("base.f3");
    let dim = 128;
    let n_base = 10_000;
    let n_train = 2_000; // Subset for training
    let mut rng = StdRng::seed_from_u64(42);

    println!("Generating random base data...");
    generate_random_f3_file(&base_path, n_base, dim, &mut rng)?;

    let ivf_options = IvfFlatBuildOptions {
        nlist: 100, // 100 clusters
        train_sample: n_train,
        max_kmeans_iters: 10,
        seed: 42,
    };
    
    // PQ options: m=16 -> d_sub=8
    // m * d_sub = dim => 16 * 8 = 128. Correct.
    let pq_options = IvfPqArtifactBuildOptions {
        num_subspaces: 16,
        subspace_dim: 8,
        max_kmeans_iters: 10,
        seed: 42,
        train_sample: n_train,
    };

    println!("Building IvfPq artifact...");
    let start_build = Instant::now();
    let artifact_path = build_ivf_pq_artifact(
        &base_path,
        0, // vector_leaf_index (assuming simple f3)
        dim,
        "test_ivf_pq",
        ivf_options,
        pq_options,
    )?;
    println!("Artifact built in {:?} at {}", start_build.elapsed(), artifact_path.display());

    // Load artifact
    println!("Loading artifact...");
    let artifact = IvfPqArtifact::open(&artifact_path)?;
    let artifact = Arc::new(artifact);

    // Load Wasm Kernel
    // Path to wasm file. We assume running from fff-vindex crate root.
    // The target path relative to workspace root is: target/wasm32-wasip1/debug/ivf_kernel_basic.wasm
    // From fff-vindex, it is ../target/...
    let wasm_path = PathBuf::from("../target/wasm32-wasip1/debug/ivf_kernel_basic.wasm");
    if !wasm_path.exists() {
        eprintln!("Wasm file not found at: {}", wasm_path.display());
        return Ok(());
    }

    println!("Loading Wasm kernel...");
    let mut kernel = WasmIvfPqKernel::load(&wasm_path, artifact.clone())?;
    
    // Generate query
    let mut query = vec![0.0f32; dim];
    for i in 0..dim {
        query[i] = rng.gen();
    }

    println!("Running search...");
    let k = 10;
    let nprobe = 10;
    
    let start_search = Instant::now();
    let results = kernel.search(&artifact, &query, k, nprobe)?;
    let elapsed = start_search.elapsed();
    
    println!("Search completed in {:?}", elapsed);
    println!("Found {} results", results.len());
    for (i, res) in results.iter().enumerate() {
        println!("#{}: row_id={}, dist={}", i, res.row_id, res.distance);
    }
    
    // Verify results are sorted
    for w in results.windows(2) {
        if w[1].distance < w[0].distance {
            println!("WARNING: Results not sorted!");
        }
    }

    Ok(())
}
