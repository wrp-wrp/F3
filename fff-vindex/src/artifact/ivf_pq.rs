use crate::ivf_flat::{
    read_base_checksums, scan_vectors_from_f3, IvfFlatBuildOptions, SearchResult,
    kmeans_l2, assign_ivf_flat, IvfFlatIndex,
};
use crate::manifest::{BaseFileBinding, IndexEntry, IndexKind, IndexManifest};
use anyhow::{bail, Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::sync::OnceLock;
use std::cell::RefCell;
use std::collections::HashMap;

const ARTIFACT_MAGIC: &[u8; 8] = b"F3VIDX2\0";
const ARTIFACT_FOOTER_MAGIC: &[u8; 8] = b"F3VIDXF\0";
const ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkType {
    Codebooks,
    Centroids,
    ListOffsets,
    PostingList,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PostingCodec {
    IvfPq,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDesc {
    pub chunk_type: ChunkType,
    #[serde(default)]
    pub list_id: Option<u32>,
    pub offset: u64,
    pub len: u64,
    pub raw_len: u64,
    pub codec: PostingCodec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPqArtifactFooter {
    pub base: BaseFileBinding,
    pub index_name: String,
    pub kind: IndexKind,
    pub vector_leaf_index: u32,
    pub dim: u32,
    pub nlist: u32,
    pub num_subspaces: u32,
    pub subspace_dim: u32,
    pub metric: String,
    #[serde(default)]
    pub build_params: serde_json::Value,
    pub chunks: Vec<ChunkDesc>,
}

#[derive(Debug)]
pub struct IvfPqArtifact {
    path: PathBuf,
    footer: IvfPqArtifactFooter,
}

#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, mut dst: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    let mut off = offset;
    while !dst.is_empty() {
        let n = file.read_at(dst, off)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "read_at returned 0",
            ));
        }
        off += n as u64;
        dst = &mut dst[n..];
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<()> {
    let mut f = file.try_clone()?;
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(dst)
}


impl IvfPqArtifact {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut f = File::open(&path).with_context(|| format!("open artifact {}", path.display()))?;

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != ARTIFACT_MAGIC {
            bail!("artifact magic mismatch: {}", path.display());
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != ARTIFACT_VERSION {
            bail!("unsupported artifact version: {version}");
        }
        let mut buf8 = [0u8; 8];
        f.read_exact(&mut buf8)?;
        let footer_offset = u64::from_le_bytes(buf8);
        if footer_offset == 0 {
            bail!("artifact footer_offset=0: {}", path.display());
        }

        f.seek(SeekFrom::Start(footer_offset))?;
        let mut footer_magic = [0u8; 8];
        f.read_exact(&mut footer_magic)?;
        if &footer_magic != ARTIFACT_FOOTER_MAGIC {
            bail!("artifact footer magic mismatch: {}", path.display());
        }
        f.read_exact(&mut buf4)?;
        let footer_version = u32::from_le_bytes(buf4);
        if footer_version != ARTIFACT_VERSION {
            bail!("unsupported artifact footer version: {footer_version}");
        }
        f.read_exact(&mut buf4)?;
        let json_len = u32::from_le_bytes(buf4) as usize;
        let mut json = vec![0u8; json_len];
        f.read_exact(&mut json)?;
        let footer: IvfPqArtifactFooter =
            serde_json::from_slice(&json).with_context(|| "parse artifact footer json")?;

        Ok(Self { path, footer })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn footer(&self) -> &IvfPqArtifactFooter {
        &self.footer
    }

    pub fn validate_base(&self) -> Result<()> {
        let meta = fs::metadata(&self.footer.base.path)
            .with_context(|| format!("stat base file {}", self.footer.base.path.display()))?;
        if meta.len() != self.footer.base.size {
            bail!(
                "base file size mismatch: expected {}, got {}",
                self.footer.base.size,
                meta.len()
            );
        }
        Ok(())
    }

    pub fn validate_base_checksums(&self) -> Result<()> {
        let (schema_checksum, data_checksum) = read_base_checksums(&self.footer.base.path)
            .with_context(|| "read base file checksums (schema/data)")?;
        if schema_checksum != self.footer.base.schema_checksum {
            bail!(
                "base schema checksum mismatch: expected {}, got {}",
                self.footer.base.schema_checksum,
                schema_checksum
            );
        }
        if data_checksum != self.footer.base.data_checksum {
            bail!(
                "base data checksum mismatch: expected {}, got {}",
                self.footer.base.data_checksum,
                data_checksum
            );
        }
        Ok(())
    }

    pub fn open_data_file(&self) -> Result<File> {
        File::open(&self.path).with_context(|| format!("open artifact {}", self.path.display()))
    }

    fn find_chunk(&self, chunk_type: ChunkType, list_id: Option<u32>) -> Result<&ChunkDesc> {
        self.footer
            .chunks
            .iter()
            .find(|c| c.chunk_type == chunk_type && c.list_id == list_id)
            .ok_or_else(|| anyhow::anyhow!("chunk not found: {:?} {:?}", chunk_type, list_id))
    }

    fn read_chunk_bytes(&self, desc: &ChunkDesc) -> Result<Vec<u8>> {
        let mut f = BufReader::new(
            File::open(&self.path).with_context(|| format!("open artifact {}", self.path.display()))?,
        );
        f.seek(SeekFrom::Start(desc.offset))?;
        let mut buf = vec![0u8; desc.len as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn read_chunk_bytes_from_file(&self, file: &File, desc: &ChunkDesc) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; desc.len as usize];
        read_exact_at(file, desc.offset, &mut buf)
            .with_context(|| format!("read artifact bytes at off={} len={}", desc.offset, desc.len))?;
        Ok(buf)
    }

    pub(crate) fn read_chunk_bytes_by_index(&self, chunk_id: u32) -> Result<Vec<u8>> {
        let idx = chunk_id as usize;
        let desc = self
            .footer
            .chunks
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("chunk_id out of range: {chunk_id}"))?;
        self.read_chunk_bytes(desc)
    }

    pub(crate) fn chunk_len_by_index(&self, chunk_id: u32) -> Result<u32> {
        let idx = chunk_id as usize;
        let desc = self
            .footer
            .chunks
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("chunk_id out of range: {chunk_id}"))?;
        Ok(u32::try_from(desc.len).unwrap_or(0))
    }

    pub(crate) fn read_chunk_bytes_by_index_from_file(
        &self,
        file: &File,
        chunk_id: u32,
    ) -> Result<Vec<u8>> {
        let idx = chunk_id as usize;
        let desc = self
            .footer
            .chunks
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("chunk_id out of range: {chunk_id}"))?;
        self.read_chunk_bytes_from_file(file, desc)
    }

    pub(crate) fn find_chunk_id(&self, chunk_type: ChunkType, list_id: Option<u32>) -> Result<u32> {
        let idx = self
            .footer
            .chunks
            .iter()
            .position(|c| c.chunk_type == chunk_type && c.list_id == list_id)
            .ok_or_else(|| anyhow::anyhow!("chunk not found: {:?} {:?}", chunk_type, list_id))?;
        Ok(idx as u32)
    }

    pub fn read_codebooks_f32(&self) -> Result<Vec<f32>> {
        let desc = self.find_chunk(ChunkType::Codebooks, None)?;
        let bytes = self.read_chunk_bytes(desc)?;
        if bytes.len() % 4 != 0 {
            bail!("codebooks chunk not aligned to f32");
        }
        let mut out = vec![0f32; bytes.len() / 4];
        LittleEndian::read_f32_into(&bytes, &mut out);
        Ok(out)
    }

    pub fn read_codebooks_f32_from_file(&self, file: &File) -> Result<Vec<f32>> {
        let desc = self.find_chunk(ChunkType::Codebooks, None)?;
        let bytes = self.read_chunk_bytes_from_file(file, desc)?;
        if bytes.len() % 4 != 0 {
            bail!("codebooks chunk not aligned to f32");
        }
        let mut out = vec![0f32; bytes.len() / 4];
        LittleEndian::read_f32_into(&bytes, &mut out);
        Ok(out)
    }

    pub fn read_list_offsets_u64(&self) -> Result<Vec<u64>> {
        let desc = self.find_chunk(ChunkType::ListOffsets, None)?;
        let bytes = self.read_chunk_bytes(desc)?;
        if bytes.len() % 8 != 0 {
            bail!("list_offsets chunk not aligned to u64");
        }
        let mut out = vec![0u64; bytes.len() / 8];
        LittleEndian::read_u64_into(&bytes, &mut out);
        Ok(out)
    }

    pub fn read_posting_list(&self, list_id: u32) -> Result<(Vec<u32>, Vec<u8>)> {
        let desc = self.find_chunk(ChunkType::PostingList, Some(list_id))?;
        let bytes = self.read_chunk_bytes(desc)?;
        match desc.codec {
            PostingCodec::IvfPq => {
                 if bytes.len() < 4 {
                    bail!("posting_list(ivfpq) chunk too small");
                }
                let count = LittleEndian::read_u32(&bytes[0..4]) as usize;
                
                // Decode RowIDs (delta-varint)
                // Start after count (4 bytes)
                // Need to assume row_ids start at offset 4? No, format?
                // Standard Format: [count: u32] [first_row_id: u32] [varints...] [codes...]
                if bytes.len() < 8 && count > 0 {
                     bail!("posting_list(ivfpq) chunk too small for row_ids");
                }
                
                let mut row_ids = Vec::<u32>::with_capacity(count);
                let codes_start_offset;
                
                if count > 0 {
                    let mut cur = LittleEndian::read_u32(&bytes[4..8]);
                    let mut offset = 8usize;
                    row_ids.push(cur);
                    for _ in 1..count {
                        let (delta, used) = decode_uleb128_u32(&bytes[offset..])?;
                        offset += used;
                        cur = cur.wrapping_add(delta);
                        row_ids.push(cur);
                    }
                    codes_start_offset = offset;
                } else {
                    codes_start_offset = 4;
                }

                let m = self.footer.num_subspaces as usize;
                let codes_len = count * m;
                let expected_total = codes_start_offset + codes_len;
                
                if bytes.len() != expected_total {
                    bail!("posting_list(ivfpq) length mismatch: expected {}, got {}", expected_total, bytes.len());
                }

                let codes = bytes[codes_start_offset..].to_vec();
                Ok((row_ids, codes))
            }
        }
    }

    pub fn read_posting_list_from_file(
        &self,
        file: &File,
        list_id: u32,
    ) -> Result<(Vec<u32>, Vec<u8>)> {
        let desc = self.find_chunk(ChunkType::PostingList, Some(list_id))?;
        let bytes = self.read_chunk_bytes_from_file(file, desc)?;
        match desc.codec {
            PostingCodec::IvfPq => {
                 if bytes.len() < 4 {
                    bail!("posting_list(ivfpq) chunk too small");
                }
                let count = LittleEndian::read_u32(&bytes[0..4]) as usize;
                
                let mut row_ids = Vec::<u32>::with_capacity(count);
                let codes_start_offset;
                
                if count > 0 {
                    if bytes.len() < 8 { bail!("truncated"); }
                    let mut cur = LittleEndian::read_u32(&bytes[4..8]);
                    let mut offset = 8usize;
                    row_ids.push(cur);
                    for _ in 1..count {
                        let (delta, used) = decode_uleb128_u32(&bytes[offset..])?;
                        offset += used;
                        cur = cur.wrapping_add(delta);
                        row_ids.push(cur);
                    }
                    codes_start_offset = offset;
                } else {
                    codes_start_offset = 4;
                }

                let m = self.footer.num_subspaces as usize;
                let codes_len = count * m;
                let codes = bytes[codes_start_offset..codes_start_offset + codes_len].to_vec();
                Ok((row_ids, codes))
            }
        }
    }
}

fn decode_uleb128_u32(mut input: &[u8]) -> Result<(u32, usize)> {
    let mut value: u32 = 0;
    let mut shift: u32 = 0;
    let mut used = 0usize;
    loop {
        let Some(&b) = input.first() else {
            bail!("uleb128 truncated");
        };
        input = &input[1..];
        used += 1;
        let low = (b & 0x7f) as u32;
        if shift >= 32 && low != 0 {
            bail!("uleb128 overflow");
        }
        value |= low.wrapping_shl(shift);
        if (b & 0x80) == 0 {
            return Ok((value, used));
        }
        shift = shift.saturating_add(7);
        if used > 5 {
            bail!("uleb128 too long for u32");
        }
    }
}

fn encode_uleb128_u32(mut v: u32, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7f) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}







#[derive(Debug, Clone)]
pub struct IvfPqArtifactBuildOptions {
    pub seed: u64,
    pub train_sample: usize,
    pub max_kmeans_iters: usize,
    pub num_subspaces: u32,
    pub subspace_dim: u32,
}

impl Default for IvfPqArtifactBuildOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            train_sample: 1000,
            max_kmeans_iters: 10,
            num_subspaces: 16, // Default M=16
            subspace_dim: 8,   // Default 128 / 16 = 8
        }
    }
}

pub fn load_ivf_pq_artifact(path: impl AsRef<Path>) -> Result<IvfPqArtifact> {
    IvfPqArtifact::open(path)
}

pub fn build_ivf_pq_artifact(
    base_f3_path: impl AsRef<Path>,
    vector_leaf_index: usize,
    dim: usize,
    index_name: &str,
    ivf_options: IvfFlatBuildOptions, // Reuse build options for scan part
    pq_options: IvfPqArtifactBuildOptions,
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

    let mut rng = StdRng::seed_from_u64(pq_options.seed);

    // 1. Train Codebooks
    // We need to sample vectors for PQ training.
    // For simplicity, we sample from the base file same as IVF training, or separate?
    // Let's reuse IvfFlatBuildOptions.train_sample for centroid training,
    // and pq_options.train_sample for PQ training.
    
    // Sample vectors
    // To train PQ, we need a lot of residual vectors (vectors - assigned_centroid).
    // This implies we must first train IVF centroids?
    // User signature: `ivf_options`.
    // We can assume user WANTS to build IVF-PQ.
    // So we build IVF centroids first.
    
    // TODO: The `IvfFlatIndex` logic is in `ivf_flat.rs`.
    // We probably want to reuse `assign_ivf_flat` and centroid training.
    // Actually, `IvfFlatIndex` is in memory.
    // Let's scan and build IVF Index first (in memory or just centroids).
    
    // Step 0: Train IVF Centroids (if not provided? Usually we build from scratch)
    // Reusing logic from `ivf_flat.rs` is tricky if not modular.
    // Copy-paste simplified training loop.

    let mut sample_vectors = Vec::<f32>::new();
    sample_vectors.reserve(pq_options.train_sample.saturating_mul(dim));

    let mut all_vectors = Vec::<f32>::new();
    let mut all_row_ids = Vec::<u32>::new();

    let mut next_row_id: u32 = 0;
    scan_vectors_from_f3(
        base_f3_path,
        vector_leaf_index,
        dim,
        |batch_vectors| {
            for v in batch_vectors.chunks_exact(dim) {
                if sample_vectors.len() / dim < pq_options.train_sample {
                    sample_vectors.extend_from_slice(v);
                } else {
                    let seen = next_row_id as u64 + 1;
                    let j = rng.gen_range(0..seen);
                    if (j as usize) < pq_options.train_sample {
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

    let nlist = ivf_options.nlist;
    if nlist == 0 {
        bail!("nlist must be > 0");
    }
    
    // 2. Train IVF Centroids
    let ivf_centroids = kmeans_l2(
        &sample_vectors,
        dim,
        nlist,
        ivf_options.max_kmeans_iters,
        &mut rng,
    )?;

    // 3. Assign to IVF Lists (In-Memory)
    // We reuse IvfFlatIndex to organize vectors by list
    let ivf_index = assign_ivf_flat(&ivf_centroids, dim, nlist, &all_row_ids, &all_vectors)?;

    // 4. Train PQ Codebooks
    // We need residuals from the SAMPLE vectors to train PQ.
    // Re-assign sample vectors to find their nearest centroid and get residual.
    let m = pq_options.num_subspaces as usize;
    let d_sub = pq_options.subspace_dim as usize;
    
    if m * d_sub != dim {
         bail!("PQ params mismatch: m({}) * d_sub({}) != dim({})", m, d_sub, dim);
    }

    let mut residual_samples = Vec::with_capacity(sample_vectors.len());
    for v in sample_vectors.chunks_exact(dim) {
        let (list_id, _) = find_nearest_centroid(v, &ivf_centroids);
        let centroid = &ivf_centroids[list_id * dim..(list_id + 1) * dim];
        for i in 0..dim {
            residual_samples.push(v[i] - centroid[i]);
        }
    }

    let codebooks = train_pq_codebooks(
        &residual_samples,
        m,
        d_sub,
        256, // K_sub is always 256 for u8 codes
        pq_options.max_kmeans_iters,
        &mut rng,
    )?;

    // 5. Write Artifact
    let index_path = index_dir.join(format!("{index_name}.ivf_pq"));
    let mut f = BufWriter::new(File::create(&index_path)?);

    f.write_all(ARTIFACT_MAGIC)?;
    f.write_all(&u32::to_le_bytes(ARTIFACT_VERSION))?;
    // Placeholder for footer offset
    f.write_all(&0u64.to_le_bytes())?; // 8 bytes for u64

    let chunks = write_ivf_pq_content(
        &mut f,
        &ivf_index,
        &codebooks,
        m,
        d_sub,
    )?;

    let footer_offset = f.stream_position()?;
    
    let footer = IvfPqArtifactFooter {
        base: base_binding.clone(),
        index_name: index_name.to_string(),
        kind: IndexKind::IvfPq,
        vector_leaf_index: vector_leaf_index as u32,
        dim: dim as u32,
        nlist: nlist as u32,
        num_subspaces: m as u32,
        subspace_dim: d_sub as u32,
        metric: "l2".to_string(),
        build_params: serde_json::json!({
             "ivf": {
                 "nlist": ivf_options.nlist,
                 "train_sample": ivf_options.train_sample,
             },
             "pq": {
                 "m": m,
                 "d_sub": d_sub,
                 "train_sample": pq_options.train_sample,
             }
        }),
        chunks,
    };

    f.write_all(ARTIFACT_FOOTER_MAGIC)?;
    f.write_all(&u32::to_le_bytes(ARTIFACT_VERSION))?;
    let json = serde_json::to_vec(&footer)?;
    f.write_all(&u32::to_le_bytes(json.len() as u32))?;
    f.write_all(&json)?;

    // Patch footer offset
    f.seek(SeekFrom::Start(12))?; // 8 (magic) + 4 (version) = 12
    f.write_all(&u64::to_le_bytes(footer_offset))?;
    
    f.flush()?;

    // Update Manifest
    let manifest_path = index_dir.join("manifest.json");
    let mut manifest = if manifest_path.exists() {
        IndexManifest::load(&manifest_path)?
    } else {
        IndexManifest {
            base: base_binding,
            indexes: vec![],
        }
    };
    // Ensure base checksums are up to date if reusing manifest
    manifest.base.schema_checksum = base_schema_checksum;
    manifest.base.data_checksum = base_data_checksum;
    manifest.base.size = base_size;

    manifest.add_or_replace_index(IndexEntry {
        name: index_name.to_string(),
        path: index_path.clone(),
        kind: IndexKind::IvfPq,
        vector_leaf_index: vector_leaf_index as u32,
        dim: dim as u32,
        metric: "l2".to_string(),
        build_params: footer.build_params.clone(),
        quantization: Some(serde_json::json!({ "type": "pq", "m": m, "d_sub": d_sub })),
    });
    manifest.save_pretty(&manifest_path)?;

    Ok(index_path)
}

fn find_nearest_centroid(v: &[f32], centroids: &[f32]) -> (usize, f32) {
    let dim = v.len();
    let num_centroids = centroids.len() / dim;
    let mut min_dist = f32::MAX;
    let mut best_id = 0;
    
    for i in 0..num_centroids {
        let c = &centroids[i * dim..(i + 1) * dim];
        let d = crate::ivf_flat::l2_sq(v, c);
        if d < min_dist {
            min_dist = d;
            best_id = i;
        }
    }
    (best_id, min_dist)
}

fn train_pq_codebooks(
    residual_samples: &[f32],
    m: usize,
    d_sub: usize,
    k_sub: usize,
    max_iters: usize,
    rng: &mut StdRng,
) -> Result<Vec<Vec<f32>>> {
    // 4. Train PQ Codebooks
    // codebooks: M x (K_sub * D_sub)
    let mut codebooks = Vec::with_capacity(m);
    let num_samples = residual_samples.len() / (m * d_sub);

    for sub in 0..m {
        // Gather valid samples for this subspace
        let mut sub_samples = Vec::with_capacity(num_samples * d_sub);
        for i in 0..num_samples {
             let offset = i * (m * d_sub) + sub * d_sub;
             sub_samples.extend_from_slice(&residual_samples[offset..offset+d_sub]);
        }

        let centroids = kmeans_l2(
            &sub_samples,
            d_sub,
            k_sub,
            max_iters,
            rng,
        )?;
        codebooks.push(centroids);
    }
    Ok(codebooks)
}

fn write_ivf_pq_content(
    f: &mut BufWriter<File>,
    ivf_index: &IvfFlatIndex,
    codebooks: &[Vec<f32>],
    m: usize,
    d_sub: usize,
) -> Result<Vec<ChunkDesc>> {
    let mut chunks = Vec::new();
    
    // 1. Write Codebooks Chunk
    // Format: Flat f32 array of all codebooks concatenated?
    // or M separate chunks?
    // Let's do single "Codebooks" chunk: [M * 256 * D_sub] floats.
    let codebooks_start = f.stream_position()?;
    let mut flattened_cb = Vec::new();
    for cb in codebooks {
        flattened_cb.extend(cb);
    }
    let cb_bytes_len = flattened_cb.len() * 4;
    let mut cb_bytes = vec![0u8; cb_bytes_len];
    LittleEndian::write_f32_into(&flattened_cb, &mut cb_bytes);
    f.write_all(&cb_bytes)?;
    
    chunks.push(ChunkDesc {
        chunk_type: ChunkType::Codebooks,
        list_id: None,
        offset: codebooks_start,
        len: cb_bytes_len as u64,
        raw_len: cb_bytes_len as u64,
        codec: PostingCodec::IvfPq, // misuse? Codebooks are raw floats
    });
    
    // 2. Write Centroids Chunk
    let centroids_start = f.stream_position()?;
    let mut centroids_bytes = vec![0u8; ivf_index.centroids.len() * 4];
    LittleEndian::write_f32_into(&ivf_index.centroids, &mut centroids_bytes);
    f.write_all(&centroids_bytes)?;
    chunks.push(ChunkDesc {
        chunk_type: ChunkType::Centroids,
        list_id: None,
        offset: centroids_start,
        len: centroids_bytes.len() as u64,
        raw_len: centroids_bytes.len() as u64,
        codec: PostingCodec::IvfPq,
    });

    // 3. Write Posting Lists
    // ivf_index has `list_offsets` which index into big `row_ids` and `vectors`.
    // We iterate lists.
    let num_lists = ivf_index.nlist;
    for i in 0..num_lists {
        let start = ivf_index.list_offsets[i] as usize;
        let end = ivf_index.list_offsets[i+1] as usize;
        let count = end - start;
        
        if count == 0 {
             // Empty list, still write chunk? Or skip?
             // Usually skip or write empty.
             // Artifact expects dense chunks?
             // `find_chunk(..., Some(list_id))` needs to find it.
             // So we MUST write it.
             let offset = f.stream_position()?;
             f.write_all(&u32::to_le_bytes(0))?; // count = 0
             chunks.push(ChunkDesc {
                chunk_type: ChunkType::PostingList,
                list_id: Some(i as u32),
                offset,
                len: 4,
                raw_len: 0,
                codec: PostingCodec::IvfPq,
            });
            continue;
        }

        let chunk_start = f.stream_position()?;
        f.write_all(&u32::to_le_bytes(count as u32))?;

        // Write Row IDs (Delta Varint)
        let list_row_ids = &ivf_index.row_ids[start..end];
        let mut cur = list_row_ids[0];
        f.write_all(&u32::to_le_bytes(cur))?;
        
        let mut row_id_bytes_len = 4;
        let mut varint_buf = Vec::with_capacity(16);
        for &rid in &list_row_ids[1..] {
             let delta = rid.wrapping_sub(cur);
             varint_buf.clear();
             encode_uleb128_u32(delta, &mut varint_buf);
             f.write_all(&varint_buf)?;
             row_id_bytes_len += varint_buf.len();
             cur = rid;
        }

        // Write Codes
        // Need to encode vectors[start..end]
        let list_vectors = &ivf_index.vectors[start * (m * d_sub) .. end * (m * d_sub)];
        let centroid = &ivf_index.centroids[i * (m * d_sub) .. (i+1) * (m * d_sub)];
        
        let mut codes = Vec::with_capacity(count * m);
        for j in 0..count {
            let v = &list_vectors[j * (m * d_sub) .. (j+1) * (m * d_sub)];
            // Compute residual `v - centroid`
            // Then encode
            for sub in 0..m {
                let sub_offset = sub * d_sub;
                let sub_v = &v[sub_offset..sub_offset + d_sub];
                let sub_c = &centroid[sub_offset..sub_offset + d_sub]; // centroid subspace
                // residual
                let mut res = Vec::with_capacity(d_sub);
                for k in 0..d_sub { res.push(sub_v[k] - sub_c[k]); }
                
                // Find nearest code
                let codebook = &codebooks[sub]; // K * D_sub
                let (code_idx, _) = find_nearest_centroid(&res, codebook);
                codes.push(code_idx as u8);
            }
        }
        f.write_all(&codes)?;

        let chunk_len = f.stream_position()? - chunk_start;
        chunks.push(ChunkDesc {
            chunk_type: ChunkType::PostingList,
            list_id: Some(i as u32),
            offset: chunk_start,
            len: chunk_len,
            raw_len: (count * (m * d_sub) * 4) as u64, // Estimate original f32 size
            codec: PostingCodec::IvfPq,
        });
    }

    Ok(chunks)
}

