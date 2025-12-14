use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use fff_poc::options::FileWriterOptions;
use fff_poc::writer::FileWriter;
use fff_vindex::ivf_flat::{build_ivf_flat_sidecar, load_ivf_flat_index, search_ivf_flat, IvfFlatBuildOptions};
use rand::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::sync::Arc;
 
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}
 
fn exact_topk(vectors: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<u32> {
    let n = vectors.len() / dim;
    let mut dists: Vec<(f32, u32)> = (0..n as u32)
        .map(|i| {
            let start = i as usize * dim;
            let v = &vectors[start..start + dim];
            (l2_sq(query, v), i)
        })
        .collect();
    dists.sort_by(|a, b| a.0.total_cmp(&b.0));
    dists.truncate(k);
    dists.into_iter().map(|(_, i)| i).collect()
}
 
/// Recall test:
/// - build clustered vectors so IVF (nlist=clusters) is well-behaved and deterministic
/// - compare IVF results to exact brute-force results
#[test]
fn ivf_flat_recall_at_k_is_high() {
    let tmp = tempfile::tempdir().unwrap();
    let base_path = tmp.path().join("base.f3");
 
    let dim = 8usize;
    let clusters = 10usize;
    let per_cluster = 20usize;
    let n = clusters * per_cluster;
    let k = 5usize;
    // Use multiple probes to tolerate that kmeans may split a tight cluster across multiple lists.
    let nprobe = 3usize;
 
    let mut rng = StdRng::seed_from_u64(123);
 
    // Data schema: vector + payload id
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "v",
            DataType::FixedSizeList(Arc::clone(&item_field), dim as i32),
            false,
        ),
        Field::new("id", DataType::Int32, false),
    ]));
 
    // Generate clustered vectors: cluster centers are far apart, with small noise.
    let mut vectors = vec![0.0f32; n * dim];
    for c in 0..clusters {
        let base = (c as f32) * 100.0;
        for j in 0..per_cluster {
            let row = c * per_cluster + j;
            for d in 0..dim {
                // Keep clusters tight so IVF training/assignment is stable in tests.
                let noise = rng.gen_range(-0.05f32..=0.05f32);
                vectors[row * dim + d] = base + (d as f32) * 0.01 + noise;
            }
        }
    }
 
    let v_values = Float32Array::from(vectors.clone());
    let v = FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(v_values), None).unwrap();
    let ids = Int32Array::from((0..n as i32).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(v), Arc::new(ids)]).unwrap();
 
    {
        let file = File::create(&base_path).unwrap();
        let mut writer = FileWriter::try_new(schema, file, FileWriterOptions::default()).unwrap();
        writer.write_batch(&batch).unwrap();
        writer.finish().unwrap();
    }
 
    let index_path = build_ivf_flat_sidecar(
        &base_path,
        /* vector_leaf_index */ 0,
        dim,
        "recall",
        IvfFlatBuildOptions {
            nlist: clusters,
            train_sample: n,
            seed: 7,
            max_kmeans_iters: 30,
        },
    )
    .unwrap();
    let index = load_ivf_flat_index(&index_path).unwrap();
 
    // Evaluate recall@k over a handful of queries. Use exact vectors as queries for determinism.
    let queries = 30usize;
    let mut total_recall = 0.0f32;
    for _ in 0..queries {
        let qi = rng.gen_range(0..n);
        let query = &vectors[qi * dim..(qi + 1) * dim];
        let approx = search_ivf_flat(&index, query, k, nprobe).unwrap();
        let approx_set: HashSet<u32> = approx.into_iter().map(|r| r.row_id).collect();
 
        let exact = exact_topk(&vectors, dim, query, k);
        let exact_set: HashSet<u32> = exact.into_iter().collect();
 
        let hit = approx_set.intersection(&exact_set).count();
        total_recall += (hit as f32) / (k as f32);
    }
 
    let avg_recall = total_recall / (queries as f32);
    // With well-separated clusters and queries taken from the dataset, IVF with a few probes should be near-perfect.
    assert!(
        avg_recall >= 0.99,
        "avg_recall@{} too low: {}",
        k,
        avg_recall
    );
}
