use crate::manifest::{BaseFileBinding, IndexEntry, IndexKind, IndexManifest};
use anyhow::{anyhow, bail, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::{Array, FixedSizeListArray, Float32Array, ListArray, UInt32Array};
use byteorder::{ByteOrder, LittleEndian};
use fff_poc::reader::{FileReaderV2Builder, Projection, Selection};
use ordered_float::NotNan;
use rand::prelude::*;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const IVF_FLAT_MAGIC: &[u8; 8] = b"F3IVFF1\0";
const IVF_FLAT_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct IvfFlatBuildOptions {
    pub nlist: usize,
    pub train_sample: usize,
    pub seed: u64,
    pub max_kmeans_iters: usize,
}

impl Default for IvfFlatBuildOptions {
    fn default() -> Self {
        Self {
            nlist: 1024,
            train_sample: 200_000,
            seed: 1,
            max_kmeans_iters: 20,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IvfFlatIndex {
    pub dim: usize,
    pub nlist: usize,
    pub centroids: Vec<f32>,    // nlist * dim
    pub list_offsets: Vec<u64>, // nlist + 1
    pub row_ids: Vec<u32>,      // N
    pub vectors: Vec<f32>,      // N * dim, aligned with row_ids
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub row_id: u32,
    pub distance: f32,
}

pub fn build_ivf_flat_sidecar(
    base_f3_path: impl AsRef<Path>,
    vector_leaf_index: usize,
    dim: usize,
    index_name: &str,
    build_options: IvfFlatBuildOptions,
) -> Result<PathBuf> {
    let base_f3_path = base_f3_path.as_ref();
    let index_dir = PathBuf::from(format!("{}.vindex", base_f3_path.display()));
    fs::create_dir_all(&index_dir)
        .with_context(|| format!("create index dir {}", index_dir.display()))?;

    let (base_schema_checksum, base_data_checksum) = read_base_checksums(base_f3_path)
        .with_context(|| "read base file checksums (schema/data)")?;
    let base_size = fs::metadata(base_f3_path)
        .with_context(|| format!("stat base file {}", base_f3_path.display()))?
        .len();

    let base_binding = BaseFileBinding {
        path: base_f3_path.to_path_buf(),
        size: base_size,
        schema_checksum: base_schema_checksum,
        data_checksum: base_data_checksum,
    };

    let mut rng = StdRng::seed_from_u64(build_options.seed);

    let mut sample_vectors = Vec::<f32>::new();
    sample_vectors.reserve(build_options.train_sample.saturating_mul(dim));

    let mut all_vectors = Vec::<f32>::new();
    let mut all_row_ids = Vec::<u32>::new();

    let mut next_row_id: u32 = 0;
    scan_vectors_from_f3(
        base_f3_path,
        vector_leaf_index,
        dim,
        |batch_vectors| {
            for v in batch_vectors.chunks_exact(dim) {
                if sample_vectors.len() / dim < build_options.train_sample {
                    sample_vectors.extend_from_slice(v);
                } else {
                    let seen = next_row_id as u64 + 1;
                    let j = rng.gen_range(0..seen);
                    if (j as usize) < build_options.train_sample {
                        let offset = (j as usize) * dim;
                        sample_vectors[offset..offset + dim].copy_from_slice(v);
                    }
                }
                all_vectors.extend_from_slice(v);
                all_row_ids.push(next_row_id);
                next_row_id = next_row_id.wrapping_add(1);
            }
            Ok(())
        },
    )?;

    if all_row_ids.is_empty() {
        bail!("no vectors found in base file");
    }
    if sample_vectors.is_empty() {
        bail!("no training vectors sampled");
    }
    if sample_vectors.len() % dim != 0 {
        bail!("internal: sample vector buffer not aligned to dim");
    }

    let nlist = build_options.nlist;
    if nlist == 0 {
        bail!("nlist must be > 0");
    }
    if (sample_vectors.len() / dim) < nlist {
        bail!(
            "train_sample too small: have {} samples but nlist={}",
            sample_vectors.len() / dim,
            nlist
        );
    }

    let centroids = kmeans_l2(
        &sample_vectors,
        dim,
        nlist,
        build_options.max_kmeans_iters,
        &mut rng,
    )?;

    let index = assign_ivf_flat(&centroids, dim, nlist, &all_row_ids, &all_vectors)?;

    let index_path = index_dir.join(format!("{index_name}.ivf_flat"));
    write_ivf_flat_index_file(&index_path, &base_binding, vector_leaf_index, &index)?;

    let manifest_path = index_dir.join("manifest.json");
    let mut manifest = if manifest_path.exists() {
        IndexManifest::load(&manifest_path)?
    } else {
        IndexManifest {
            base: base_binding,
            indexes: vec![],
        }
    };
    manifest.base = BaseFileBinding {
        path: base_f3_path.to_path_buf(),
        size: base_size,
        schema_checksum: base_schema_checksum,
        data_checksum: base_data_checksum,
    };
    manifest.add_or_replace_index(IndexEntry {
        name: index_name.to_string(),
        path: index_path.clone(),
        kind: IndexKind::IvfFlat,
        vector_leaf_index: vector_leaf_index as u32,
        dim: dim as u32,
        metric: "l2".to_string(),
        build_params: serde_json::json!({
            "nlist": build_options.nlist,
            "train_sample": build_options.train_sample,
            "seed": build_options.seed,
            "max_kmeans_iters": build_options.max_kmeans_iters,
        }),
        quantization: None,
    });
    manifest.save_pretty(&manifest_path)?;

    Ok(index_path)
}

pub fn load_ivf_flat_index(index_path: impl AsRef<Path>) -> Result<IvfFlatIndex> {
    let index_path = index_path.as_ref();
    // Prefer the custom binary format; fall back to legacy `.f3` index files for compatibility.
    let mut f = File::open(index_path)
        .with_context(|| format!("open ivf-flat index {}", index_path.display()))?;
    let mut magic = [0u8; 8];
    match f.read_exact(&mut magic) {
        Ok(()) if &magic == IVF_FLAT_MAGIC => load_ivf_flat_index_binary(index_path),
        Ok(()) => load_ivf_flat_index_f3(index_path),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            bail!("index file too small: {}", index_path.display())
        }
        Err(e) => Err(e.into()),
    }
}

pub fn search_ivf_flat(
    index: &IvfFlatIndex,
    query: &[f32],
    k: usize,
    nprobe: usize,
) -> Result<Vec<SearchResult>> {
    if query.len() != index.dim {
        bail!("query dim mismatch: expected {}, got {}", index.dim, query.len());
    }
    if k == 0 {
        return Ok(vec![]);
    }
    let nprobe = nprobe.min(index.nlist).max(1);

    let mut centroid_dists: Vec<(usize, f32)> = (0..index.nlist)
        .map(|cid| {
            let c = &index.centroids[cid * index.dim..(cid + 1) * index.dim];
            (cid, l2_sq(query, c))
        })
        .collect();
    centroid_dists.select_nth_unstable_by(nprobe - 1, |a, b| a.1.total_cmp(&b.1));
    centroid_dists.truncate(nprobe);

    let mut heap: BinaryHeap<(NotNan<f32>, u32)> = BinaryHeap::new();
    for (cid, _) in centroid_dists {
        let start = index.list_offsets[cid] as usize;
        let end = index.list_offsets[cid + 1] as usize;
        for pos in start..end {
            let row_id = index.row_ids[pos];
            let v = &index.vectors[pos * index.dim..(pos + 1) * index.dim];
            let dist = l2_sq(query, v);
            let dist_nn = NotNan::new(dist).map_err(|_| anyhow!("distance is NaN"))?;
            if heap.len() < k {
                heap.push((dist_nn, row_id));
            } else if let Some(&(worst, _)) = heap.peek() {
                if dist_nn < worst {
                    heap.pop();
                    heap.push((dist_nn, row_id));
                }
            }
        }
    }

    let mut out = heap
        .into_sorted_vec()
        .into_iter()
        .map(|(distance, row_id)| SearchResult {
            row_id,
            distance: distance.into_inner(),
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    Ok(out)
}

pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn load_ivf_flat_index_binary(index_path: &Path) -> Result<IvfFlatIndex> {
    let file = File::open(index_path)
        .with_context(|| format!("open ivf-flat index {}", index_path.display()))?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != IVF_FLAT_MAGIC {
        bail!("ivf-flat index magic mismatch");
    }

    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    r.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != IVF_FLAT_VERSION {
        bail!("unsupported ivf-flat index version: {version}");
    }

    r.read_exact(&mut buf4)?;
    let dim = u32::from_le_bytes(buf4) as usize;
    r.read_exact(&mut buf4)?;
    let nlist = u32::from_le_bytes(buf4) as usize;
    r.read_exact(&mut buf4)?;
    let _vector_leaf_index = u32::from_le_bytes(buf4);

    // Base checksums for quick self-check (also validated via manifest).
    r.read_exact(&mut buf8)?;
    let _base_schema_checksum = u64::from_le_bytes(buf8);
    r.read_exact(&mut buf8)?;
    let _base_data_checksum = u64::from_le_bytes(buf8);

    r.read_exact(&mut buf8)?;
    let centroids_len = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8)?;
    let list_offsets_len = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8)?;
    let row_ids_len = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8)?;
    let vectors_len = u64::from_le_bytes(buf8) as usize;

    if dim == 0 {
        bail!("invalid dim=0 in ivf-flat index header");
    }
    if nlist == 0 {
        bail!("invalid nlist=0 in ivf-flat index header");
    }
    if centroids_len != nlist * dim {
        bail!(
            "invalid centroids_len: expected nlist*dim={}, got {}",
            nlist * dim,
            centroids_len
        );
    }
    if list_offsets_len != nlist + 1 {
        bail!(
            "invalid list_offsets_len: expected nlist+1={}, got {}",
            nlist + 1,
            list_offsets_len
        );
    }
    if vectors_len != row_ids_len * dim {
        bail!(
            "invalid vectors_len: expected row_ids_len*dim={}, got {}",
            row_ids_len * dim,
            vectors_len
        );
    }

    let mut centroids = Vec::<f32>::with_capacity(centroids_len);
    for _ in 0..centroids_len {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        centroids.push(f32::from_le_bytes(b));
    }

    let mut list_offsets = Vec::<u64>::with_capacity(list_offsets_len);
    for _ in 0..list_offsets_len {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        list_offsets.push(u64::from_le_bytes(b));
    }

    let mut row_ids = Vec::<u32>::with_capacity(row_ids_len);
    for _ in 0..row_ids_len {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        row_ids.push(u32::from_le_bytes(b));
    }

    let mut vectors = Vec::<f32>::with_capacity(vectors_len);
    for _ in 0..vectors_len {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        vectors.push(f32::from_le_bytes(b));
    }

    Ok(IvfFlatIndex {
        dim,
        nlist,
        centroids,
        list_offsets,
        row_ids,
        vectors,
    })
}

fn load_ivf_flat_index_f3(index_path: &Path) -> Result<IvfFlatIndex> {
    let file = File::open(index_path)
        .with_context(|| format!("open ivf-flat index {}", index_path.display()))?;
    let mut reader = FileReaderV2Builder::new(Arc::new(file))
        .with_projections(Projection::All)
        .with_selection(Selection::All)
        .build()
        .map_err(|e| anyhow!(e.to_string()))?;
    let batches = reader
        .read_file()
        .map_err(|e| anyhow!(e.to_string()))
        .with_context(|| "read ivf-flat index file")?;
    if batches.is_empty() {
        bail!("index file has no record batches");
    }

    let schema = batches[0].schema();
    let batch = if batches.len() == 1 {
        batches[0].clone()
    } else {
        arrow::compute::concat_batches(&schema, &batches)
            .map_err(|e| anyhow!(e.to_string()))
            .with_context(|| "concat ivf-flat index batches")?
    };

    let nlist = batch.num_rows();
    if nlist == 0 {
        bail!("ivf-flat index has 0 rows (nlist=0)");
    }

    let centroids_col = batch
        .column_by_name("centroids")
        .ok_or_else(|| anyhow!("missing column centroids"))?;
    let centroids_fsl = centroids_col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| anyhow!("centroids is not FixedSizeListArray"))?;
    if centroids_fsl.null_count() != 0 {
        bail!("centroids contains nulls; not supported");
    }
    let dim = centroids_fsl.value_length() as usize;
    if dim == 0 {
        bail!("invalid dim=0 in ivf-flat index");
    }
    let centroids_values = centroids_fsl
        .values()
        .as_primitive::<arrow::datatypes::Float32Type>();
    let centroids = centroids_values.values().to_vec();

    let ids_col = batch
        .column_by_name("postings_row_ids")
        .ok_or_else(|| anyhow!("missing column postings_row_ids"))?;
    let ids_list = ids_col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("postings_row_ids is not a ListArray"))?;
    if ids_list.null_count() != 0 {
        bail!("postings_row_ids contains nulls; not supported");
    }
    let mut list_offsets = Vec::<u64>::with_capacity(nlist + 1);
    for off in ids_list.offsets().iter() {
        if *off < 0 {
            bail!("postings_row_ids offsets contains negative value");
        }
        list_offsets.push(*off as u64);
    }
    let row_ids_arr: &UInt32Array = ids_list
        .values()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| anyhow!("postings_row_ids values is not UInt32Array"))?;
    let row_ids = row_ids_arr.values().to_vec();

    let vecs_col = batch
        .column_by_name("postings_vectors")
        .ok_or_else(|| anyhow!("missing column postings_vectors"))?;
    let vecs_list = vecs_col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("postings_vectors is not a ListArray"))?;
    if vecs_list.null_count() != 0 {
        bail!("postings_vectors contains nulls; not supported");
    }
    let vecs_arr: &Float32Array = vecs_list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| anyhow!("postings_vectors values is not Float32Array"))?;
    let vectors = vecs_arr.values().to_vec();

    if vectors.len() % dim != 0 {
        bail!(
            "vectors length mismatch: expected multiple of dim={}, got {}",
            dim,
            vectors.len()
        );
    }
    if vectors.len() / dim != row_ids.len() {
        bail!(
            "vectors/row_ids length mismatch: vectors rows={}, row_ids={}",
            vectors.len() / dim,
            row_ids.len()
        );
    }

    Ok(IvfFlatIndex {
        dim,
        nlist,
        centroids,
        list_offsets,
        row_ids,
        vectors,
    })
}

pub(crate) fn kmeans_l2(
    samples: &[f32],
    dim: usize,
    k: usize,
    max_iters: usize,
    rng: &mut StdRng,
) -> Result<Vec<f32>> {
    let n = samples.len() / dim;
    if n < k {
        bail!("kmeans: n < k");
    }
    let mut centroids = vec![0f32; k * dim];
    let mut chosen = std::collections::HashSet::<usize>::new();
    for cid in 0..k {
        loop {
            let idx = rng.gen_range(0..n);
            if chosen.insert(idx) {
                centroids[cid * dim..(cid + 1) * dim]
                    .copy_from_slice(&samples[idx * dim..(idx + 1) * dim]);
                break;
            }
        }
    }

    let mut assignments = vec![0usize; n];
    let mut counts = vec![0usize; k];
    let mut sums = vec![0f32; k * dim];

    for _ in 0..max_iters {
        counts.fill(0);
        sums.fill(0.0);

        for i in 0..n {
            let v = &samples[i * dim..(i + 1) * dim];
            let mut best = (0usize, f32::INFINITY);
            for cid in 0..k {
                let c = &centroids[cid * dim..(cid + 1) * dim];
                let d = l2_sq(v, c);
                if d < best.1 {
                    best = (cid, d);
                }
            }
            assignments[i] = best.0;
            counts[best.0] += 1;
            let sum = &mut sums[best.0 * dim..(best.0 + 1) * dim];
            for j in 0..dim {
                sum[j] += v[j];
            }
        }

        for cid in 0..k {
            if counts[cid] == 0 {
                let idx = rng.gen_range(0..n);
                centroids[cid * dim..(cid + 1) * dim]
                    .copy_from_slice(&samples[idx * dim..(idx + 1) * dim]);
            } else {
                let inv = 1.0 / (counts[cid] as f32);
                let c = &mut centroids[cid * dim..(cid + 1) * dim];
                let sum = &sums[cid * dim..(cid + 1) * dim];
                for j in 0..dim {
                    c[j] = sum[j] * inv;
                }
            }
        }
    }

    Ok(centroids)
}

pub(crate) fn assign_ivf_flat(
    centroids: &[f32],
    dim: usize,
    nlist: usize,
    row_ids: &[u32],
    vectors: &[f32],
) -> Result<IvfFlatIndex> {
    if vectors.len() != row_ids.len() * dim {
        bail!("vectors length mismatch: expected row_ids.len()*dim");
    }
    let n = row_ids.len();
    let mut counts = vec![0usize; nlist];
    let mut lists = vec![0usize; n];

    for i in 0..n {
        let v = &vectors[i * dim..(i + 1) * dim];
        let mut best = (0usize, f32::INFINITY);
        for cid in 0..nlist {
            let c = &centroids[cid * dim..(cid + 1) * dim];
            let d = l2_sq(v, c);
            if d < best.1 {
                best = (cid, d);
            }
        }
        lists[i] = best.0;
        counts[best.0] += 1;
    }

    let mut offsets = vec![0u64; nlist + 1];
    for i in 0..nlist {
        offsets[i + 1] = offsets[i] + counts[i] as u64;
    }

    let mut cursor = offsets[..nlist].iter().map(|&x| x as usize).collect::<Vec<_>>();
    let mut out_row_ids = vec![0u32; n];
    let mut out_vectors = vec![0f32; n * dim];
    for i in 0..n {
        let list_id = lists[i];
        let pos = cursor[list_id];
        cursor[list_id] += 1;
        out_row_ids[pos] = row_ids[i];
        out_vectors[pos * dim..(pos + 1) * dim]
            .copy_from_slice(&vectors[i * dim..(i + 1) * dim]);
    }

    Ok(IvfFlatIndex {
        dim,
        nlist,
        centroids: centroids.to_vec(),
        list_offsets: offsets,
        row_ids: out_row_ids,
        vectors: out_vectors,
    })
}

fn write_ivf_flat_index_file(
    index_path: &Path,
    base: &BaseFileBinding,
    vector_leaf_index: usize,
    index: &IvfFlatIndex,
) -> Result<()> {
    if index.list_offsets.len() != index.nlist + 1 {
        bail!(
            "invalid list_offsets: expected nlist+1={}, got {}",
            index.nlist + 1,
            index.list_offsets.len()
        );
    }
    if index.centroids.len() != index.nlist * index.dim {
        bail!(
            "invalid centroids: expected nlist*dim={}, got {}",
            index.nlist * index.dim,
            index.centroids.len()
        );
    }
    if index.vectors.len() % index.dim != 0 {
        bail!(
            "invalid vectors: expected multiple of dim={}, got {}",
            index.dim,
            index.vectors.len()
        );
    }
    if index.vectors.len() / index.dim != index.row_ids.len() {
        bail!(
            "invalid vectors/row_ids: vectors rows={}, row_ids={}",
            index.vectors.len() / index.dim,
            index.row_ids.len()
        );
    }

    let file = File::create(index_path)
        .with_context(|| format!("create ivf-flat index file {}", index_path.display()))?;
    let mut w = BufWriter::new(file);

    w.write_all(IVF_FLAT_MAGIC)?;
    w.write_all(&IVF_FLAT_VERSION.to_le_bytes())?;

    let dim_u32: u32 = index
        .dim
        .try_into()
        .map_err(|_| anyhow!("dim too large: {}", index.dim))?;
    let nlist_u32: u32 = index
        .nlist
        .try_into()
        .map_err(|_| anyhow!("nlist too large: {}", index.nlist))?;
    let vector_leaf_index_u32: u32 = vector_leaf_index
        .try_into()
        .map_err(|_| anyhow!("vector_leaf_index too large: {}", vector_leaf_index))?;

    w.write_all(&dim_u32.to_le_bytes())?;
    w.write_all(&nlist_u32.to_le_bytes())?;
    w.write_all(&vector_leaf_index_u32.to_le_bytes())?;

    // Base checksums (self-check; manifest also stores these).
    w.write_all(&base.schema_checksum.to_le_bytes())?;
    w.write_all(&base.data_checksum.to_le_bytes())?;

    w.write_all(&(index.centroids.len() as u64).to_le_bytes())?;
    w.write_all(&(index.list_offsets.len() as u64).to_le_bytes())?;
    w.write_all(&(index.row_ids.len() as u64).to_le_bytes())?;
    w.write_all(&(index.vectors.len() as u64).to_le_bytes())?;

    for &v in &index.centroids {
        w.write_all(&v.to_le_bytes())?;
    }
    for &v in &index.list_offsets {
        w.write_all(&v.to_le_bytes())?;
    }
    for &v in &index.row_ids {
        w.write_all(&v.to_le_bytes())?;
    }
    for &v in &index.vectors {
        w.write_all(&v.to_le_bytes())?;
    }

    w.flush()?;
    Ok(())
}

pub(crate) fn read_base_checksums(base_f3_path: &Path) -> Result<(u64, u64)> {
    use fff_poc::io::reader::Reader as _;
    use fff_format::{MAGIC, POSTSCRIPT_SIZE};

    let f = File::open(base_f3_path)
        .with_context(|| format!("open base f3 {}", base_f3_path.display()))?;
    let size = f.size().map_err(|e| anyhow!(e.to_string()))?;

    let mut postscript_buffer: [u8; POSTSCRIPT_SIZE as usize] = [0; POSTSCRIPT_SIZE as usize];
    f.read_exact_at(&mut postscript_buffer, size - POSTSCRIPT_SIZE)
        .map_err(|e| anyhow!(e.to_string()))?;
    if postscript_buffer[postscript_buffer.len() - 2..] != *MAGIC {
        bail!("base file magic mismatch");
    }
    let data_checksum = LittleEndian::read_u64(&postscript_buffer[10..18]);
    let schema_checksum = LittleEndian::read_u64(&postscript_buffer[18..26]);
    Ok((schema_checksum, data_checksum))
}

pub(crate) fn scan_vectors_from_f3(
    base_f3_path: &Path,
    vector_leaf_index: usize,
    dim: usize,
    mut on_vectors: impl FnMut(&[f32]) -> Result<()>,
) -> Result<()> {
    let file = File::open(base_f3_path)
        .with_context(|| format!("open base f3 {}", base_f3_path.display()))?;
    let mut reader = FileReaderV2Builder::new(Arc::new(file))
        // NOTE: `Projection::LeafColumnIndexes` currently selects *physical* columns in the file
        // footer, and does not map logical schema field indexes to their physical column spans.
        // For `FixedSizeList`, a single logical field maps to multiple physical columns.
        // To keep vindex robust, we read all and then pick the logical column.
        .with_projections(Projection::All)
        .with_selection(Selection::All)
        .build()
        .map_err(|e| anyhow!(e.to_string()))?;
    let batches = reader
        .read_file()
        .map_err(|e| anyhow!(e.to_string()))
        .with_context(|| "read base f3 for vectors")?;
    for batch in batches {
        if vector_leaf_index >= batch.num_columns() {
            bail!(
                "vector_leaf_index out of range: {} >= {}",
                vector_leaf_index,
                batch.num_columns()
            );
        }
        let col = batch.column(vector_leaf_index);
        let fsl = col
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| anyhow!("vector column is not FixedSizeListArray"))?;
        if fsl.value_length() as usize != dim {
            bail!(
                "vector dim mismatch: expected {dim}, got {}",
                fsl.value_length()
            );
        }
        if fsl.null_count() != 0 {
            bail!("vector column contains nulls; not supported in vindex build yet");
        }
        let values = fsl.values().as_primitive::<arrow::datatypes::Float32Type>();
        on_vectors(values.values())?;
    }
    Ok(())
}
