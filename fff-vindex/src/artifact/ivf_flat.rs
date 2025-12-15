use crate::ivf_flat::{
    assign_ivf_flat, kmeans_l2, read_base_checksums, scan_vectors_from_f3, IvfFlatBuildOptions,
    IvfFlatIndex, SearchResult,
};
use crate::manifest::{BaseFileBinding, IndexEntry, IndexKind, IndexManifest};
use anyhow::{bail, Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use half::f16;
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

const ARTIFACT_MAGIC: &[u8; 8] = b"F3VIDX1\0";
const ARTIFACT_FOOTER_MAGIC: &[u8; 8] = b"F3VIDXF\0";
const ARTIFACT_VERSION: u32 = 1;

fn f16_lut() -> &'static [f32] {
    static LUT: OnceLock<Vec<f32>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut v = Vec::with_capacity(65536);
        for bits in 0u32..=0xffff {
            v.push(f16::from_bits(bits as u16).to_f32());
        }
        v
    })
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkType {
    Centroids,
    ListOffsets,
    PostingList,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PostingCodec {
    Raw,
    RowIdDeltaVarintV1,
    RawF16,
    RowIdDeltaVarintV1F16,
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
pub struct IvfFlatArtifactFooter {
    pub base: BaseFileBinding,
    pub index_name: String,
    pub kind: IndexKind,
    pub vector_leaf_index: u32,
    pub dim: u32,
    pub nlist: u32,
    pub metric: String,
    #[serde(default)]
    pub build_params: serde_json::Value,
    pub chunks: Vec<ChunkDesc>,
}

#[derive(Debug)]
pub struct IvfFlatArtifact {
    path: PathBuf,
    footer: IvfFlatArtifactFooter,
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

impl IvfFlatArtifact {
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
        let footer: IvfFlatArtifactFooter =
            serde_json::from_slice(&json).with_context(|| "parse artifact footer json")?;

        Ok(Self { path, footer })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn footer(&self) -> &IvfFlatArtifactFooter {
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

    pub fn read_centroids_f32(&self) -> Result<Vec<f32>> {
        let desc = self.find_chunk(ChunkType::Centroids, None)?;
        let bytes = self.read_chunk_bytes(desc)?;
        if bytes.len() % 4 != 0 {
            bail!("centroids chunk not aligned to f32");
        }
        let mut out = vec![0f32; bytes.len() / 4];
        LittleEndian::read_f32_into(&bytes, &mut out);
        Ok(out)
    }

    pub fn read_centroids_f32_from_file(&self, file: &File) -> Result<Vec<f32>> {
        let desc = self.find_chunk(ChunkType::Centroids, None)?;
        let bytes = self.read_chunk_bytes_from_file(file, desc)?;
        if bytes.len() % 4 != 0 {
            bail!("centroids chunk not aligned to f32");
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

    pub fn read_posting_list(&self, list_id: u32) -> Result<(Vec<u32>, Vec<f32>)> {
        let desc = self.find_chunk(ChunkType::PostingList, Some(list_id))?;
        let bytes = self.read_chunk_bytes(desc)?;
        let dim = self.footer.dim as usize;
        match desc.codec {
            PostingCodec::Raw => {
                if bytes.len() < 4 {
                    bail!("posting_list chunk too small");
                }
                let count = LittleEndian::read_u32(&bytes[0..4]) as usize;
                let row_ids_bytes = 4 + count * 4;
                let vectors_bytes = count * dim * 4;
                let expected = row_ids_bytes + vectors_bytes;
                if bytes.len() != expected {
                    bail!(
                        "posting_list size mismatch: expected {expected} bytes, got {}",
                        bytes.len()
                    );
                }
                let mut row_ids = vec![0u32; count];
                LittleEndian::read_u32_into(&bytes[4..row_ids_bytes], &mut row_ids);
                let mut vectors = vec![0f32; count * dim];
                LittleEndian::read_f32_into(&bytes[row_ids_bytes..], &mut vectors);
                Ok((row_ids, vectors))
            }
            PostingCodec::RowIdDeltaVarintV1 => {
                if bytes.len() < 8 {
                    bail!("posting_list(delta-varint) chunk too small");
                }
                let count = LittleEndian::read_u32(&bytes[0..4]) as usize;
                let dim = self.footer.dim as usize;
                let mut cur = LittleEndian::read_u32(&bytes[4..8]);
                let mut offset = 8usize;
                let mut row_ids = Vec::<u32>::with_capacity(count);
                if count > 0 {
                    row_ids.push(cur);
                    for _ in 1..count {
                        let (delta, used) = decode_uleb128_u32(&bytes[offset..])?;
                        offset += used;
                        cur = cur.wrapping_add(delta);
                        row_ids.push(cur);
                    }
                }
                let vectors_bytes = count * dim * 4;
                if bytes.len() != offset + vectors_bytes {
                    bail!(
                        "posting_list(delta-varint) size mismatch: expected {} bytes, got {}",
                        offset + vectors_bytes,
                        bytes.len()
                    );
                }
                let mut vectors = vec![0f32; count * dim];
                LittleEndian::read_f32_into(&bytes[offset..], &mut vectors);
                Ok((row_ids, vectors))
            }
            PostingCodec::RawF16 => {
                if bytes.len() < 4 {
                    bail!("posting_list(f16) chunk too small");
                }
                let count = LittleEndian::read_u32(&bytes[0..4]) as usize;
                let row_ids_bytes = 4 + count * 4;
                let vectors_bytes = count * dim * 2;
                let expected = row_ids_bytes + vectors_bytes;
                if bytes.len() != expected {
                    bail!(
                        "posting_list(f16) size mismatch: expected {expected} bytes, got {}",
                        bytes.len()
                    );
                }
                let mut row_ids = vec![0u32; count];
                LittleEndian::read_u32_into(&bytes[4..row_ids_bytes], &mut row_ids);
                let mut vectors = vec![0f32; count * dim];
                let mut off = row_ids_bytes;
                let lut = f16_lut();
                for i in 0..(count * dim) {
                    let bits = LittleEndian::read_u16(&bytes[off..off + 2]);
                    vectors[i] = lut[bits as usize];
                    off += 2;
                }
                Ok((row_ids, vectors))
            }
            PostingCodec::RowIdDeltaVarintV1F16 => {
                if bytes.len() < 8 {
                    bail!("posting_list(delta-varint+f16) chunk too small");
                }
                let count = LittleEndian::read_u32(&bytes[0..4]) as usize;
                let mut cur = LittleEndian::read_u32(&bytes[4..8]);
                let mut offset = 8usize;
                let mut row_ids = Vec::<u32>::with_capacity(count);
                if count > 0 {
                    row_ids.push(cur);
                    for _ in 1..count {
                        let (delta, used) = decode_uleb128_u32(&bytes[offset..])?;
                        offset += used;
                        cur = cur.wrapping_add(delta);
                        row_ids.push(cur);
                    }
                }
                let vectors_bytes = count * dim * 2;
                if bytes.len() != offset + vectors_bytes {
                    bail!(
                        "posting_list(delta-varint+f16) size mismatch: expected {} bytes, got {}",
                        offset + vectors_bytes,
                        bytes.len()
                    );
                }
                let mut vectors = vec![0f32; count * dim];
                let mut off = offset;
                let lut = f16_lut();
                for i in 0..(count * dim) {
                    let bits = LittleEndian::read_u16(&bytes[off..off + 2]);
                    vectors[i] = lut[bits as usize];
                    off += 2;
                }
                Ok((row_ids, vectors))
            }
        }
    }

    pub fn read_posting_list_from_file(
        &self,
        file: &File,
        list_id: u32,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let desc = self.find_chunk(ChunkType::PostingList, Some(list_id))?;
        let bytes = self.read_chunk_bytes_from_file(file, desc)?;
        let dim = self.footer.dim as usize;
        match desc.codec {
            PostingCodec::Raw => {
                if bytes.len() < 4 {
                    bail!("posting_list chunk too small");
                }
                let count = LittleEndian::read_u32(&bytes[0..4]) as usize;
                let row_ids_bytes = 4 + count * 4;
                let vectors_bytes = count * dim * 4;
                let expected = row_ids_bytes + vectors_bytes;
                if bytes.len() != expected {
                    bail!(
                        "posting_list size mismatch: expected {expected} bytes, got {}",
                        bytes.len()
                    );
                }
                let mut row_ids = vec![0u32; count];
                LittleEndian::read_u32_into(&bytes[4..row_ids_bytes], &mut row_ids);
                let mut vectors = vec![0f32; count * dim];
                LittleEndian::read_f32_into(&bytes[row_ids_bytes..], &mut vectors);
                Ok((row_ids, vectors))
            }
            PostingCodec::RowIdDeltaVarintV1 => {
                if bytes.len() < 8 {
                    bail!("posting_list(delta-varint) chunk too small");
                }
                let count = LittleEndian::read_u32(&bytes[0..4]) as usize;
                let dim = self.footer.dim as usize;
                let mut cur = LittleEndian::read_u32(&bytes[4..8]);
                let mut offset = 8usize;
                let mut row_ids = Vec::<u32>::with_capacity(count);
                if count > 0 {
                    row_ids.push(cur);
                    for _ in 1..count {
                        let (delta, used) = decode_uleb128_u32(&bytes[offset..])?;
                        offset += used;
                        cur = cur.wrapping_add(delta);
                        row_ids.push(cur);
                    }
                }
                let vectors_bytes = count * dim * 4;
                if bytes.len() != offset + vectors_bytes {
                    bail!(
                        "posting_list(delta-varint) size mismatch: expected {} bytes, got {}",
                        offset + vectors_bytes,
                        bytes.len()
                    );
                }
                let mut vectors = vec![0f32; count * dim];
                LittleEndian::read_f32_into(&bytes[offset..], &mut vectors);
                Ok((row_ids, vectors))
            }
            PostingCodec::RawF16 => {
                if bytes.len() < 4 {
                    bail!("posting_list(f16) chunk too small");
                }
                let count = LittleEndian::read_u32(&bytes[0..4]) as usize;
                let row_ids_bytes = 4 + count * 4;
                let vectors_bytes = count * dim * 2;
                let expected = row_ids_bytes + vectors_bytes;
                if bytes.len() != expected {
                    bail!(
                        "posting_list(f16) size mismatch: expected {expected} bytes, got {}",
                        bytes.len()
                    );
                }
                let mut row_ids = vec![0u32; count];
                LittleEndian::read_u32_into(&bytes[4..row_ids_bytes], &mut row_ids);
                let mut vectors = vec![0f32; count * dim];
                let mut off = row_ids_bytes;
                let lut = f16_lut();
                for i in 0..(count * dim) {
                    let bits = LittleEndian::read_u16(&bytes[off..off + 2]);
                    vectors[i] = lut[bits as usize];
                    off += 2;
                }
                Ok((row_ids, vectors))
            }
            PostingCodec::RowIdDeltaVarintV1F16 => {
                if bytes.len() < 8 {
                    bail!("posting_list(delta-varint+f16) chunk too small");
                }
                let count = LittleEndian::read_u32(&bytes[0..4]) as usize;
                let mut cur = LittleEndian::read_u32(&bytes[4..8]);
                let mut offset = 8usize;
                let mut row_ids = Vec::<u32>::with_capacity(count);
                if count > 0 {
                    row_ids.push(cur);
                    for _ in 1..count {
                        let (delta, used) = decode_uleb128_u32(&bytes[offset..])?;
                        offset += used;
                        cur = cur.wrapping_add(delta);
                        row_ids.push(cur);
                    }
                }
                let vectors_bytes = count * dim * 2;
                if bytes.len() != offset + vectors_bytes {
                    bail!(
                        "posting_list(delta-varint+f16) size mismatch: expected {} bytes, got {}",
                        offset + vectors_bytes,
                        bytes.len()
                    );
                }
                let mut vectors = vec![0f32; count * dim];
                let mut off = offset;
                let lut = f16_lut();
                for i in 0..(count * dim) {
                    let bits = LittleEndian::read_u16(&bytes[off..off + 2]);
                    vectors[i] = lut[bits as usize];
                    off += 2;
                }
                Ok((row_ids, vectors))
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
pub struct IvfFlatArtifactBuildOptions {
    pub posting_codec: PostingCodec,
}

impl Default for IvfFlatArtifactBuildOptions {
    fn default() -> Self {
        Self {
            posting_codec: PostingCodec::Raw,
        }
    }
}

pub fn load_ivf_flat_artifact(path: impl AsRef<Path>) -> Result<IvfFlatArtifact> {
    IvfFlatArtifact::open(path)
}

pub fn build_ivf_flat_artifact(
    base_f3_path: impl AsRef<Path>,
    vector_leaf_index: usize,
    dim: usize,
    index_name: &str,
    build_options: IvfFlatBuildOptions,
) -> Result<PathBuf> {
    build_ivf_flat_artifact_with_options(
        base_f3_path,
        vector_leaf_index,
        dim,
        index_name,
        build_options,
        IvfFlatArtifactBuildOptions::default(),
    )
}

pub fn build_ivf_flat_artifact_with_options(
    base_f3_path: impl AsRef<Path>,
    vector_leaf_index: usize,
    dim: usize,
    index_name: &str,
    build_options: IvfFlatBuildOptions,
    artifact_options: IvfFlatArtifactBuildOptions,
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

    let index_path = index_dir.join(format!("{index_name}.ivf_flat.artifact"));
    write_ivf_flat_artifact_file(
        &index_path,
        &base_binding,
        vector_leaf_index,
        dim,
        index_name,
        &build_options,
        &artifact_options,
        &index,
    )?;

    let manifest_path = index_dir.join("manifest.json");
    let mut manifest = if manifest_path.exists() {
        IndexManifest::load(&manifest_path)?
    } else {
        IndexManifest {
            base: base_binding.clone(),
            indexes: vec![],
        }
    };
    manifest.base = base_binding.clone();
    manifest.add_or_replace_index(IndexEntry {
        name: index_name.to_string(),
        path: index_path.clone(),
        kind: IndexKind::IvfFlat,
        vector_leaf_index: vector_leaf_index as u32,
        dim: dim as u32,
        metric: "l2".to_string(),
        build_params: serde_json::json!({
            "format": "artifact_v1",
            "posting_codec": serde_json::to_value(artifact_options.posting_codec).unwrap_or(serde_json::Value::Null),
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

fn write_ivf_flat_artifact_file(
    index_path: &Path,
    base: &BaseFileBinding,
    vector_leaf_index: usize,
    dim: usize,
    index_name: &str,
    build_options: &IvfFlatBuildOptions,
    artifact_options: &IvfFlatArtifactBuildOptions,
    index: &IvfFlatIndex,
) -> Result<()> {
    let f = File::create(index_path)
        .with_context(|| format!("create artifact {}", index_path.display()))?;
    let mut w = BufWriter::new(f);

    w.write_all(ARTIFACT_MAGIC)?;
    w.write_all(&ARTIFACT_VERSION.to_le_bytes())?;
    let footer_offset_pos = w.stream_position()?;
    w.write_all(&0u64.to_le_bytes())?;

    let mut chunks = Vec::<ChunkDesc>::new();

    // centroids chunk (raw f32[])
    let offset = w.stream_position()?;
    let mut buf = vec![0u8; index.centroids.len() * 4];
    LittleEndian::write_f32_into(&index.centroids, &mut buf);
    w.write_all(&buf)?;
    chunks.push(ChunkDesc {
        chunk_type: ChunkType::Centroids,
        list_id: None,
        offset,
        len: buf.len() as u64,
        raw_len: buf.len() as u64,
        codec: PostingCodec::Raw,
    });

    // list_offsets chunk (raw u64[])
    let offset = w.stream_position()?;
    let mut buf = vec![0u8; index.list_offsets.len() * 8];
    LittleEndian::write_u64_into(&index.list_offsets, &mut buf);
    w.write_all(&buf)?;
    chunks.push(ChunkDesc {
        chunk_type: ChunkType::ListOffsets,
        list_id: None,
        offset,
        len: buf.len() as u64,
        raw_len: buf.len() as u64,
        codec: PostingCodec::Raw,
    });

    // posting_list chunks: codec-specific layout, see `PostingCodec`.
    for list_id in 0..index.nlist {
        let start = index.list_offsets[list_id] as usize;
        let end = index.list_offsets[list_id + 1] as usize;
        let count = end.saturating_sub(start);
        let offset = w.stream_position()?;

        let mut encoded = Vec::<u8>::new();
        let raw_len: u64;
        match artifact_options.posting_codec {
            PostingCodec::Raw => {
                let vectors_slice = &index.vectors[start * dim..end * dim];
                let mut vectors_buf = vec![0u8; count * dim * 4];
                LittleEndian::write_f32_into(vectors_slice, &mut vectors_buf);
                encoded.reserve(4 + count * 4 + vectors_buf.len());
                let mut header = [0u8; 4];
                LittleEndian::write_u32(&mut header, count as u32);
                encoded.extend_from_slice(&header);
                let row_ids_slice = &index.row_ids[start..end];
                let mut row_ids_buf = vec![0u8; count * 4];
                LittleEndian::write_u32_into(row_ids_slice, &mut row_ids_buf);
                encoded.extend_from_slice(&row_ids_buf);
                encoded.extend_from_slice(&vectors_buf);
                raw_len = encoded.len() as u64;
            }
            PostingCodec::RowIdDeltaVarintV1 => {
                let vectors_slice = &index.vectors[start * dim..end * dim];
                let mut vectors_buf = vec![0u8; count * dim * 4];
                LittleEndian::write_f32_into(vectors_slice, &mut vectors_buf);
                encoded.reserve(8 + count * 2 + vectors_buf.len());
                let mut header = [0u8; 4];
                LittleEndian::write_u32(&mut header, count as u32);
                encoded.extend_from_slice(&header);
                let row_ids_slice = &index.row_ids[start..end];
                let first = row_ids_slice.first().copied().unwrap_or(0);
                let mut first_buf = [0u8; 4];
                LittleEndian::write_u32(&mut first_buf, first);
                encoded.extend_from_slice(&first_buf);
                for wpair in row_ids_slice.windows(2) {
                    let delta = wpair[1].wrapping_sub(wpair[0]);
                    encode_uleb128_u32(delta, &mut encoded);
                }
                let row_ids_raw_len = 4 + count * 4;
                raw_len = (row_ids_raw_len + vectors_buf.len()) as u64;
                encoded.extend_from_slice(&vectors_buf);
            }
            PostingCodec::RawF16 => {
                let vectors_slice = &index.vectors[start * dim..end * dim];
                let mut vectors_buf = Vec::<u8>::with_capacity(count * dim * 2);
                for &x in vectors_slice {
                    vectors_buf.extend_from_slice(&f16::from_f32(x).to_bits().to_le_bytes());
                }
                encoded.reserve(4 + count * 4 + vectors_buf.len());
                let mut header = [0u8; 4];
                LittleEndian::write_u32(&mut header, count as u32);
                encoded.extend_from_slice(&header);
                let row_ids_slice = &index.row_ids[start..end];
                let mut row_ids_buf = vec![0u8; count * 4];
                LittleEndian::write_u32_into(row_ids_slice, &mut row_ids_buf);
                encoded.extend_from_slice(&row_ids_buf);
                encoded.extend_from_slice(&vectors_buf);
                raw_len = (4 + count * 4 + count * dim * 4) as u64;
            }
            PostingCodec::RowIdDeltaVarintV1F16 => {
                let vectors_slice = &index.vectors[start * dim..end * dim];
                let mut vectors_buf = Vec::<u8>::with_capacity(count * dim * 2);
                for &x in vectors_slice {
                    vectors_buf.extend_from_slice(&f16::from_f32(x).to_bits().to_le_bytes());
                }
                encoded.reserve(8 + count * 2 + vectors_buf.len());
                let mut header = [0u8; 4];
                LittleEndian::write_u32(&mut header, count as u32);
                encoded.extend_from_slice(&header);
                let row_ids_slice = &index.row_ids[start..end];
                let first = row_ids_slice.first().copied().unwrap_or(0);
                let mut first_buf = [0u8; 4];
                LittleEndian::write_u32(&mut first_buf, first);
                encoded.extend_from_slice(&first_buf);
                for wpair in row_ids_slice.windows(2) {
                    let delta = wpair[1].wrapping_sub(wpair[0]);
                    encode_uleb128_u32(delta, &mut encoded);
                }
                let row_ids_raw_len = 4 + count * 4;
                raw_len = (row_ids_raw_len + count * dim * 4) as u64;
                encoded.extend_from_slice(&vectors_buf);
            }
        }
        w.write_all(&encoded)?;

        let len = encoded.len() as u64;
        chunks.push(ChunkDesc {
            chunk_type: ChunkType::PostingList,
            list_id: Some(list_id as u32),
            offset,
            len,
            raw_len,
            codec: artifact_options.posting_codec,
        });
    }

    let footer_offset = w.stream_position()?;
    let footer = IvfFlatArtifactFooter {
        base: base.clone(),
        index_name: index_name.to_string(),
        kind: IndexKind::IvfFlat,
        vector_leaf_index: vector_leaf_index as u32,
        dim: dim as u32,
        nlist: index.nlist as u32,
        metric: "l2".to_string(),
        build_params: serde_json::json!({
            "format": "artifact_v1",
            "posting_codec": serde_json::to_value(artifact_options.posting_codec).unwrap_or(serde_json::Value::Null),
            "nlist": build_options.nlist,
            "train_sample": build_options.train_sample,
            "seed": build_options.seed,
            "max_kmeans_iters": build_options.max_kmeans_iters,
        }),
        chunks,
    };
    let footer_json = serde_json::to_vec(&footer).with_context(|| "serialize footer json")?;
    if footer_json.len() > (u32::MAX as usize) {
        bail!("footer json too large");
    }
    w.write_all(ARTIFACT_FOOTER_MAGIC)?;
    w.write_all(&ARTIFACT_VERSION.to_le_bytes())?;
    w.write_all(&(footer_json.len() as u32).to_le_bytes())?;
    w.write_all(&footer_json)?;
    w.flush()?;

    // Patch header.footer_offset
    let mut f = w.into_inner().with_context(|| "flush artifact writer")?;
    f.seek(SeekFrom::Start(footer_offset_pos))?;
    f.write_all(&footer_offset.to_le_bytes())?;
    f.flush()?;

    Ok(())
}

pub fn search_ivf_flat_artifact_native(
    artifact: &IvfFlatArtifact,
    query: &[f32],
    k: usize,
    nprobe: usize,
) -> Result<Vec<SearchResult>> {
    let footer = artifact.footer();
    let dim = footer.dim as usize;
    let nlist = footer.nlist as usize;
    if query.len() != dim {
        bail!("query dim mismatch: expected {dim}, got {}", query.len());
    }
    if k == 0 {
        return Ok(vec![]);
    }
    let nprobe = nprobe.min(nlist).max(1);

    // Load small, hot parts into memory; postings are loaded per list.
    let centroids = artifact.read_centroids_f32()?;
    if centroids.len() != nlist * dim {
        bail!("centroids len mismatch");
    }

    let mut centroid_dists: Vec<(usize, f32)> = (0..nlist)
        .map(|cid| {
            let c = &centroids[cid * dim..(cid + 1) * dim];
            (cid, crate::ivf_flat::l2_sq(query, c))
        })
        .collect();
    centroid_dists.select_nth_unstable_by(nprobe - 1, |a, b| a.1.total_cmp(&b.1));
    centroid_dists.truncate(nprobe);

    let mut heap: std::collections::BinaryHeap<(ordered_float::NotNan<f32>, u32)> =
        std::collections::BinaryHeap::new();
    for (cid, _) in centroid_dists {
        let (row_ids, vectors) = artifact.read_posting_list(cid as u32)?;
        for (pos, &row_id) in row_ids.iter().enumerate() {
            let start = pos * dim;
            let v = &vectors[start..start + dim];
            let dist = crate::ivf_flat::l2_sq(query, v);
            let dist_nn = ordered_float::NotNan::new(dist)
                .map_err(|_| anyhow::anyhow!("distance is NaN"))?;
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

pub struct IvfFlatArtifactSearcher {
    artifact: IvfFlatArtifact,
    file: File,
    centroids: Vec<f32>,
    posting_cache: RefCell<HashMap<u32, (Vec<u32>, Vec<f32>)>>,
    posting_cache_enabled: bool,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct IvfFlatSearchStageStats {
    pub centroid_ns: u64,
    pub posting_decode_ns: u64,
    pub dist_ns: u64,
    pub heap_ns: u64,
    pub posting_cache_hits: u64,
    pub posting_cache_misses: u64,
}

impl IvfFlatArtifactSearcher {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(path, /*posting_cache_enabled=*/true)
    }

    pub fn open_with_options(path: impl AsRef<Path>, posting_cache_enabled: bool) -> Result<Self> {
        let artifact = IvfFlatArtifact::open(path)?;
        let file = artifact.open_data_file()?;
        let centroids = artifact.read_centroids_f32_from_file(&file)?;
        Ok(Self {
            artifact,
            file,
            centroids,
            posting_cache: RefCell::new(HashMap::new()),
            posting_cache_enabled,
        })
    }

    pub fn artifact(&self) -> &IvfFlatArtifact {
        &self.artifact
    }

    pub fn clear_posting_cache(&self) {
        self.posting_cache.borrow_mut().clear();
    }

    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Result<Vec<SearchResult>> {
        let footer = self.artifact.footer();
        let dim = footer.dim as usize;
        let nlist = footer.nlist as usize;
        if query.len() != dim {
            bail!("query dim mismatch: expected {dim}, got {}", query.len());
        }
        if k == 0 {
            return Ok(vec![]);
        }
        let nprobe = nprobe.min(nlist).max(1);

        if self.centroids.len() != nlist * dim {
            bail!("centroids len mismatch");
        }

        let mut centroid_dists: Vec<(usize, f32)> = (0..nlist)
            .map(|cid| {
                let c = &self.centroids[cid * dim..(cid + 1) * dim];
                (cid, crate::ivf_flat::l2_sq(query, c))
            })
            .collect();
        centroid_dists.select_nth_unstable_by(nprobe - 1, |a, b| a.1.total_cmp(&b.1));
        centroid_dists.truncate(nprobe);

        let mut heap: std::collections::BinaryHeap<(ordered_float::NotNan<f32>, u32)> =
            std::collections::BinaryHeap::new();
        for (cid, _) in centroid_dists {
            let cid_u32 = cid as u32;
            if self.posting_cache_enabled {
                if !self.posting_cache.borrow().contains_key(&cid_u32) {
                    let decoded = self
                        .artifact
                        .read_posting_list_from_file(&self.file, cid_u32)?;
                    self.posting_cache.borrow_mut().insert(cid_u32, decoded);
                }
                let cache = self.posting_cache.borrow();
                let Some((row_ids, vectors)) = cache.get(&cid_u32) else {
                    continue;
                };
                for (pos, &row_id) in row_ids.iter().enumerate() {
                    let start = pos * dim;
                    let v = &vectors[start..start + dim];
                    let dist = crate::ivf_flat::l2_sq(query, v);
                    let dist_nn = ordered_float::NotNan::new(dist)
                        .map_err(|_| anyhow::anyhow!("distance is NaN"))?;
                    if heap.len() < k {
                        heap.push((dist_nn, row_id));
                    } else if let Some(&(worst, _)) = heap.peek() {
                        if dist_nn < worst {
                            heap.pop();
                            heap.push((dist_nn, row_id));
                        }
                    }
                }
                continue;
            } else {
                let decoded = self
                    .artifact
                    .read_posting_list_from_file(&self.file, cid_u32)?;
                // Use locals for this list only.
                // Keep owned buffers alive until we're done scanning.
                let (r, v) = decoded;

                // Scan using the owned buffers, then continue.
                for (pos, &row_id) in r.iter().enumerate() {
                    let start = pos * dim;
                    let vec_slice = &v[start..start + dim];
                    let dist = crate::ivf_flat::l2_sq(query, vec_slice);
                    let dist_nn = ordered_float::NotNan::new(dist)
                        .map_err(|_| anyhow::anyhow!("distance is NaN"))?;
                    if heap.len() < k {
                        heap.push((dist_nn, row_id));
                    } else if let Some(&(worst, _)) = heap.peek() {
                        if dist_nn < worst {
                            heap.pop();
                            heap.push((dist_nn, row_id));
                        }
                    }
                }
                continue;
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

    pub fn search_profiled(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<(Vec<SearchResult>, IvfFlatSearchStageStats)> {
        let mut stats = IvfFlatSearchStageStats::default();

        let footer = self.artifact.footer();
        let dim = footer.dim as usize;
        let nlist = footer.nlist as usize;
        if query.len() != dim {
            bail!("query dim mismatch: expected {dim}, got {}", query.len());
        }
        if k == 0 {
            return Ok((vec![], stats));
        }
        let nprobe = nprobe.min(nlist).max(1);

        if self.centroids.len() != nlist * dim {
            bail!("centroids len mismatch");
        }

        let start_centroids = Instant::now();
        let mut centroid_dists: Vec<(usize, f32)> = (0..nlist)
            .map(|cid| {
                let c = &self.centroids[cid * dim..(cid + 1) * dim];
                (cid, crate::ivf_flat::l2_sq(query, c))
            })
            .collect();
        centroid_dists.select_nth_unstable_by(nprobe - 1, |a, b| a.1.total_cmp(&b.1));
        centroid_dists.truncate(nprobe);
        stats.centroid_ns += start_centroids.elapsed().as_nanos() as u64;

        let mut heap: std::collections::BinaryHeap<(ordered_float::NotNan<f32>, u32)> =
            std::collections::BinaryHeap::new();
        let mut scratch_dists: Vec<f32> = Vec::new();

        for (cid, _) in centroid_dists {
            let cid_u32 = cid as u32;

            if self.posting_cache_enabled {
                if self.posting_cache.borrow().contains_key(&cid_u32) {
                    stats.posting_cache_hits += 1;
                } else {
                    stats.posting_cache_misses += 1;
                    let start_decode = Instant::now();
                    let decoded = self
                        .artifact
                        .read_posting_list_from_file(&self.file, cid_u32)?;
                    stats.posting_decode_ns += start_decode.elapsed().as_nanos() as u64;
                    self.posting_cache.borrow_mut().insert(cid_u32, decoded);
                }

                let cache = self.posting_cache.borrow();
                let Some((row_ids, vectors)) = cache.get(&cid_u32) else {
                    continue;
                };

                scratch_dists.clear();
                scratch_dists.reserve(row_ids.len());
                let start_dist = Instant::now();
                for pos in 0..row_ids.len() {
                    let start = pos * dim;
                    let v = &vectors[start..start + dim];
                    scratch_dists.push(crate::ivf_flat::l2_sq(query, v));
                }
                stats.dist_ns += start_dist.elapsed().as_nanos() as u64;

                let start_heap = Instant::now();
                for (pos, &row_id) in row_ids.iter().enumerate() {
                    let dist_nn = ordered_float::NotNan::new(scratch_dists[pos])
                        .map_err(|_| anyhow::anyhow!("distance is NaN"))?;
                    if heap.len() < k {
                        heap.push((dist_nn, row_id));
                    } else if let Some(&(worst, _)) = heap.peek() {
                        if dist_nn < worst {
                            heap.pop();
                            heap.push((dist_nn, row_id));
                        }
                    }
                }
                stats.heap_ns += start_heap.elapsed().as_nanos() as u64;
                continue;
            }

            stats.posting_cache_misses += 1;
            let start_decode = Instant::now();
            let decoded = self
                .artifact
                .read_posting_list_from_file(&self.file, cid_u32)?;
            stats.posting_decode_ns += start_decode.elapsed().as_nanos() as u64;

            let (row_ids, vectors) = decoded;

            scratch_dists.clear();
            scratch_dists.reserve(row_ids.len());
            let start_dist = Instant::now();
            for pos in 0..row_ids.len() {
                let start = pos * dim;
                let v = &vectors[start..start + dim];
                scratch_dists.push(crate::ivf_flat::l2_sq(query, v));
            }
            stats.dist_ns += start_dist.elapsed().as_nanos() as u64;

            let start_heap = Instant::now();
            for (pos, &row_id) in row_ids.iter().enumerate() {
                let dist_nn = ordered_float::NotNan::new(scratch_dists[pos])
                    .map_err(|_| anyhow::anyhow!("distance is NaN"))?;
                if heap.len() < k {
                    heap.push((dist_nn, row_id));
                } else if let Some(&(worst, _)) = heap.peek() {
                    if dist_nn < worst {
                        heap.pop();
                        heap.push((dist_nn, row_id));
                    }
                }
            }
            stats.heap_ns += start_heap.elapsed().as_nanos() as u64;
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
        Ok((out, stats))
    }
}
