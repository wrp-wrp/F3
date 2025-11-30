use std::mem::size_of;
use std::sync::Arc;

use arrow_array::{
    builder::{BinaryBuilder, FixedSizeListBuilder, Float32Builder},
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Float32Array,
};
use arrow_schema::DataType;
use fff_core::errors::{Error, Result};

/// Parsed auxiliary metadata for a vector block micro-index.
#[derive(Debug, Clone)]
pub struct MicroIndexBlob {
    pub dim: u32,
    pub k: u32,
    pub centroids: Vec<f32>,
    /// Per-bucket spans of (start, len) in row-order.
    pub bucket_spans: Vec<Vec<(u32, u32)>>,
}

const MAGIC: &[u8; 4] = b"MIVF";
const VERSION: u32 = 1;

pub fn serialize_micro_blob(blob: &MicroIndexBlob) -> Vec<u8> {
    let spans_slots: usize = blob
        .bucket_spans
        .iter()
        .map(|v| 1 + v.len() * 2)
        .sum();
    let mut out = Vec::with_capacity(
        16 + blob.centroids.len() * size_of::<f32>() + spans_slots * size_of::<u32>(),
    );
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&blob.dim.to_le_bytes());
    out.extend_from_slice(&blob.k.to_le_bytes());
    for &c in &blob.centroids {
        out.extend_from_slice(&c.to_le_bytes());
    }
    for spans in &blob.bucket_spans {
        out.extend_from_slice(&(spans.len() as u32).to_le_bytes());
        for &(start, len) in spans {
            out.extend_from_slice(&start.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
        }
    }
    out
}

pub fn parse_micro_blob(buf: &[u8]) -> Result<MicroIndexBlob> {
    if buf.len() < 16 {
        return Err(Error::General("micro index aux too small".into()));
    }
    if &buf[0..4] != MAGIC {
        return Err(Error::General("micro index aux magic mismatch".into()));
    }
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(Error::General("micro index aux version mismatch".into()));
    }
    let dim = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let k = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    let mut offset = 16;
    let centroid_len = (dim as usize) * (k as usize);
    if buf.len() < offset + centroid_len * size_of::<f32>() {
        return Err(Error::General("micro index aux truncated centroids".into()));
    }
    let mut centroids = Vec::with_capacity(centroid_len);
    for chunk in buf[offset..offset + centroid_len * size_of::<f32>()]
        .chunks_exact(size_of::<f32>())
    {
        centroids.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    offset += centroid_len * size_of::<f32>();
    let mut bucket_spans = Vec::with_capacity(k as usize);
    for _ in 0..k {
        if buf.len() < offset + size_of::<u32>() {
            return Err(Error::General("micro index aux truncated spans".into()));
        }
        let span_cnt = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut spans = Vec::with_capacity(span_cnt);
        for _ in 0..span_cnt {
            if buf.len() < offset + 8 {
                return Err(Error::General("micro index aux truncated span entries".into()));
            }
            let start = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
            let len = u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().unwrap());
            offset += 8;
            spans.push((start, len));
        }
        bucket_spans.push(spans);
    }
    Ok(MicroIndexBlob {
        dim,
        k,
        centroids,
        bucket_spans,
    })
}

/// Convert FixedSizeList<Float32> to a BinaryArray so it can be encoded with existing binary path.
pub fn vector_to_binary(array: &FixedSizeListArray) -> Result<BinaryArray> {
    let dim = array.value_length() as usize;
    let values = array
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| Error::General("vector child type must be Float32".into()))?;
    let mut builder = BinaryBuilder::with_capacity(array.len(), array.len() * dim * 4);
    for i in 0..array.len() {
        if array.is_null(i) {
            builder.append_null();
        } else {
            let start = array.value_offset(i) as usize;
            let slice = values.values()[start..start + dim]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            builder.append_value(slice);
        }
    }
    Ok(builder.finish())
}

/// Convert a BinaryArray back into FixedSizeList<Float32>.
pub fn binary_to_vector(
    array: &BinaryArray,
    dim: i32,
    nullable: bool,
) -> Result<ArrayRef> {
    let dim_usize = dim as usize;
    let value_builder = Float32Builder::with_capacity(array.len() * dim_usize);
    let mut list_builder = FixedSizeListBuilder::new(value_builder, dim);
    for i in 0..array.len() {
        if array.is_null(i) {
            list_builder.append(false);
            if nullable {
                list_builder.values().append_nulls(dim_usize);
            }
        } else {
            let val = array.value(i);
            if val.len() != dim_usize * 4 {
                return Err(Error::General("invalid vector byte length".into()));
            }
            for chunk in val.chunks_exact(4) {
                list_builder
                    .values()
                    .append_value(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            list_builder.append(true);
        }
    }
    let finished = list_builder.finish();
    Ok(Arc::new(finished) as ArrayRef)
}

pub fn is_supported_vector(dt: &DataType) -> bool {
    matches!(dt, DataType::FixedSizeList(child, _) if matches!(child.data_type(), DataType::Float32))
}

/// Build a tiny IVF-like micro index using evenly spaced seeds.
pub fn build_mini_ivf(dim: usize, vectors: &[f32], row_count: usize) -> MicroIndexBlob {
    let safe_dim = dim.max(1);
    let safe_rows = row_count.max(1);
    let k = safe_rows.clamp(1, 8);
    let stride = (safe_rows + k - 1) / k;
    let mut centroids = Vec::with_capacity(k * safe_dim);
    for i in 0..k {
        let idx = std::cmp::min(i * stride, safe_rows - 1);
        let start = idx * safe_dim;
        let end = start + safe_dim;
        let slice = &vectors[start..end];
        centroids.extend_from_slice(slice);
    }
    // assign
    let mut bucket_spans: Vec<Vec<(u32, u32)>> = vec![Vec::new(); k];
    for row in 0..safe_rows {
        let base = row * safe_dim;
        let vec_slice = &vectors[base..base + safe_dim];
        let mut best = (0usize, f32::INFINITY);
        for (cid, centroid) in centroids.chunks(safe_dim).enumerate() {
            let mut dist = 0f32;
            for j in 0..safe_dim {
                let d = vec_slice[j] - centroid[j];
                dist += d * d;
            }
            if dist < best.1 {
                best = (cid, dist);
            }
        }
        let spans = &mut bucket_spans[best.0];
        if let Some(last) = spans.last_mut() {
            if last.0 + last.1 == row as u32 {
                last.1 += 1;
                continue;
            }
        }
        spans.push((row as u32, 1));
    }
    MicroIndexBlob {
        dim: safe_dim as u32,
        k: k as u32,
        centroids,
        bucket_spans,
    }
}
