use anyhow::{bail, Context};
use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use byteorder::{LittleEndian, ReadBytesExt};
use fff_poc::options::FileWriterOptions;
use fff_poc::writer::FileWriter;
use fff_vindex::ivf_flat::{build_ivf_flat_sidecar, load_ivf_flat_index, search_ivf_flat, IvfFlatBuildOptions};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
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

fn read_fvecs(path: &Path, max_vectors: usize) -> anyhow::Result<(usize, Vec<f32>)> {
    let f = File::open(path).with_context(|| format!("open fvecs {}", path.display()))?;
    let mut r = BufReader::new(f);

    // Peek dim.
    let dim = r
        .read_i32::<LittleEndian>()
        .with_context(|| format!("read dim from {}", path.display()))? as usize;
    if dim == 0 || dim > 10_000 {
        bail!("invalid dim={dim} in {}", path.display());
    }

    // Read first vector (we already consumed dim).
    let mut out = Vec::<f32>::new();
    out.reserve(max_vectors.saturating_mul(dim));
    for _ in 0..dim {
        out.push(r.read_f32::<LittleEndian>()?);
    }

    // Read remaining vectors.
    while out.len() / dim < max_vectors {
        let d = match r.read_i32::<LittleEndian>() {
            Ok(d) => d as usize,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        if d != dim {
            bail!("dim mismatch in {}: expected {dim}, got {d}", path.display());
        }
        for _ in 0..dim {
            out.push(r.read_f32::<LittleEndian>()?);
        }
    }
    Ok((dim, out))
}

fn exact_topk(vectors: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<u32> {
    let n = vectors.len() / dim;
    let mut dists: Vec<(f32, u32)> = (0..n as u32)
        .map(|i| {
            let start = i as usize * dim;
            (l2_sq(query, &vectors[start..start + dim]), i)
        })
        .collect();
    if k == 0 {
        return vec![];
    }
    let kth = k.saturating_sub(1).min(dists.len().saturating_sub(1));
    dists.select_nth_unstable_by(kth, |a, b| a.0.total_cmp(&b.0));
    dists.truncate(k);
    dists.sort_by(|a, b| a.0.total_cmp(&b.0));
    dists.into_iter().map(|(_, i)| i).collect()
}

fn default_sift_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/sift")
}

/// End-to-end SIFT100K recall test (local-only).
///
/// Prereq:
/// - run `scripts/download_sift_100k.sh` (or set `SIFT_DIR`)
///
/// Run:
/// - `cargo test -p fff-vindex --release --test sift_100k -- --ignored --nocapture`
#[test]
#[ignore]
fn sift_100k_ivf_flat_recall_at_10() {
    if cfg!(debug_assertions) && std::env::var_os("SIFT_RUN_DEBUG").is_none() {
        eprintln!("skipping in debug build (set SIFT_RUN_DEBUG=1 to force)");
        return;
    }

    let sift_dir = std::env::var_os("SIFT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_sift_dir);
    let base_path = sift_dir.join("learn.fvecs");
    let query_path = sift_dir.join("query.fvecs");

    assert!(
        base_path.exists() && query_path.exists(),
        "missing SIFT files; run `scripts/download_sift_100k.sh` or set SIFT_DIR (got {})",
        sift_dir.display()
    );

    // Base: 100k vectors (learn set).
    let (dim, base_vectors) = read_fvecs(&base_path, 100_000).unwrap();
    assert_eq!(dim, 128);
    let n = base_vectors.len() / dim;
    assert!(n >= 100_000);

    // Queries: brute-force is O(N*dim) per query; keep moderate for `--release`.
    let queries = 50usize;
    let (_, query_vectors) = read_fvecs(&query_path, queries).unwrap();
    assert_eq!(query_vectors.len(), queries * dim);

    // Build base.f3 in tempdir: vector + payload id (row_id).
    let tmp = tempfile::tempdir().unwrap();
    let base_f3 = tmp.path().join("sift100k.f3");
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "v",
            DataType::FixedSizeList(Arc::clone(&item_field), dim as i32),
            false,
        ),
        Field::new("id", DataType::Int32, false),
    ]));
    let v_values = Float32Array::from(base_vectors.clone());
    let v = FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(v_values), None).unwrap();
    let ids = Int32Array::from((0..n as i32).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(v), Arc::new(ids)]).unwrap();
    {
        let f = File::create(&base_f3).unwrap();
        let mut w = FileWriter::try_new(schema, f, FileWriterOptions::default()).unwrap();
        w.write_batch(&batch).unwrap();
        w.finish().unwrap();
    }

    // Build IVFFlat index.
    // NOTE: our current IVF implementation does O(train_sample*nlist*dim*iters) kmeans training and
    // O(N*nlist*dim) assignment. These defaults aim to finish in a reasonable time in `--release`.
    let nlist = 64usize;
    let nprobe = 16usize;
    let k = 10usize;
    let index_path = build_ivf_flat_sidecar(
        &base_f3,
        /* vector_leaf_index */ 0,
        dim,
        "sift_ivf_flat",
        IvfFlatBuildOptions {
            nlist,
            train_sample: 5_000,
            seed: 1,
            max_kmeans_iters: 5,
        },
    )
    .unwrap();
    let index = load_ivf_flat_index(&index_path).unwrap();

    let mut total_recall = 0.0f32;
    for qi in 0..queries {
        let q = &query_vectors[qi * dim..(qi + 1) * dim];
        let approx = search_ivf_flat(&index, q, k, nprobe).unwrap();
        let approx_set: HashSet<u32> = approx.into_iter().map(|r| r.row_id).collect();

        let exact = exact_topk(&base_vectors, dim, q, k);
        let exact_set: HashSet<u32> = exact.into_iter().collect();

        let hit = approx_set.intersection(&exact_set).count();
        total_recall += (hit as f32) / (k as f32);
    }

    let avg_recall = total_recall / (queries as f32);
    println!(
        "SIFT100K IVF_FLAT avg_recall@{} over {} queries: nlist={}, nprobe={} => {}",
        k, queries, nlist, nprobe, avg_recall
    );

    // Tune by adjusting nlist/nprobe (and/or improving IVF assignment) if you want higher recall.
    assert!(avg_recall >= 0.60, "avg_recall@{k} too low: {avg_recall}");
}
