use crate::file::footer::MetadataSection;
use byteorder::{LittleEndian, ReadBytesExt};
use fff_core::errors::{Error, Result};
use fff_format::File::fff::flatbuf as fb;
use fff_format::ToFlatBuffer;
use fff_ude_wasm::Runtime as WasmRuntime;
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::io::Cursor;
use wide::f32x4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorIndexAlgorithm {
    BruteForce,
    Hnsw,
    Ivf,
    CustomWasm,
}

impl From<VectorIndexAlgorithm> for fb::VectorIndexAlgorithm {
    fn from(value: VectorIndexAlgorithm) -> Self {
        match value {
            VectorIndexAlgorithm::BruteForce => fb::VectorIndexAlgorithm::BRUTE_FORCE,
            VectorIndexAlgorithm::Hnsw => fb::VectorIndexAlgorithm::HNSW,
            VectorIndexAlgorithm::Ivf => fb::VectorIndexAlgorithm::IVF,
            VectorIndexAlgorithm::CustomWasm => fb::VectorIndexAlgorithm::CUSTOM_WASM,
        }
    }
}

impl From<fb::VectorIndexAlgorithm> for VectorIndexAlgorithm {
    fn from(value: fb::VectorIndexAlgorithm) -> Self {
        match value {
            fb::VectorIndexAlgorithm::HNSW => VectorIndexAlgorithm::Hnsw,
            fb::VectorIndexAlgorithm::IVF => VectorIndexAlgorithm::Ivf,
            fb::VectorIndexAlgorithm::CUSTOM_WASM => VectorIndexAlgorithm::CustomWasm,
            _ => VectorIndexAlgorithm::BruteForce,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorDistanceMetric {
    L2,
    Cosine,
    Dot,
}

impl From<VectorDistanceMetric> for fb::VectorDistanceMetric {
    fn from(value: VectorDistanceMetric) -> Self {
        match value {
            VectorDistanceMetric::L2 => fb::VectorDistanceMetric::L2,
            VectorDistanceMetric::Cosine => fb::VectorDistanceMetric::COSINE,
            VectorDistanceMetric::Dot => fb::VectorDistanceMetric::DOT,
        }
    }
}

impl From<fb::VectorDistanceMetric> for VectorDistanceMetric {
    fn from(value: fb::VectorDistanceMetric) -> Self {
        match value {
            fb::VectorDistanceMetric::COSINE => VectorDistanceMetric::Cosine,
            fb::VectorDistanceMetric::DOT => VectorDistanceMetric::Dot,
            _ => VectorDistanceMetric::L2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantizationMethod {
    None,
    Scalar,
    Product,
    Custom,
}

impl Default for QuantizationMethod {
    fn default() -> Self {
        QuantizationMethod::None
    }
}

impl From<QuantizationMethod> for fb::VectorQuantizationMethod {
    fn from(value: QuantizationMethod) -> Self {
        match value {
            QuantizationMethod::None => fb::VectorQuantizationMethod::NONE,
            QuantizationMethod::Scalar => fb::VectorQuantizationMethod::SCALAR,
            QuantizationMethod::Product => fb::VectorQuantizationMethod::PRODUCT,
            QuantizationMethod::Custom => fb::VectorQuantizationMethod::CUSTOM,
        }
    }
}

impl From<fb::VectorQuantizationMethod> for QuantizationMethod {
    fn from(value: fb::VectorQuantizationMethod) -> Self {
        match value {
            fb::VectorQuantizationMethod::SCALAR => QuantizationMethod::Scalar,
            fb::VectorQuantizationMethod::PRODUCT => QuantizationMethod::Product,
            fb::VectorQuantizationMethod::CUSTOM => QuantizationMethod::Custom,
            _ => QuantizationMethod::None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QuantizationSegment {
    pub start_dim: u32,
    pub end_dim: u32,
    pub method: QuantizationMethod,
    pub bits: u8,
    pub params: Vec<u8>,
}

impl QuantizationSegment {
    fn to_fb<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<fb::QuantizationSegment<'fb>> {
        let params = if self.params.is_empty() {
            None
        } else {
            Some(fbb.create_vector(&self.params))
        };
        fb::QuantizationSegment::create(
            fbb,
            &fb::QuantizationSegmentArgs {
                start_dim: self.start_dim,
                end_dim: self.end_dim,
                method: self.method.clone().into(),
                bits: self.bits,
                params,
            },
        )
    }
}

impl From<&fb::QuantizationSegment<'_>> for QuantizationSegment {
    fn from(segment: &fb::QuantizationSegment<'_>) -> Self {
        Self {
            start_dim: segment.start_dim(),
            end_dim: segment.end_dim(),
            method: segment.method().into(),
            bits: segment.bits(),
            params: segment
                .params()
                .map(|p| p.bytes().to_vec())
                .unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QuantizationSpec {
    pub dimension: u32,
    pub segments: Vec<QuantizationSegment>,
}

impl QuantizationSpec {
    fn to_fb<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> Option<WIPOffset<fb::QuantizationSpec<'fb>>> {
        if self.dimension == 0 && self.segments.is_empty() {
            return None;
        }
        let segments = self
            .segments
            .iter()
            .map(|segment| segment.to_fb(fbb))
            .collect::<Vec<_>>();
        let segments_vec = if segments.is_empty() {
            None
        } else {
            Some(fbb.create_vector(&segments))
        };
        Some(fb::QuantizationSpec::create(
            fbb,
            &fb::QuantizationSpecArgs {
                dimension: self.dimension,
                segments: segments_vec,
            },
        ))
    }
}

impl From<&fb::QuantizationSpec<'_>> for QuantizationSpec {
    fn from(spec: &fb::QuantizationSpec<'_>) -> Self {
        Self {
            dimension: spec.dimension(),
            segments: spec
                .segments()
                .map(|segments| {
                    segments
                        .iter()
                        .map(|s| QuantizationSegment::from(&s))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VectorIndexDescriptor {
    pub index_id: u32,
    pub column: String,
    pub algorithm: VectorIndexAlgorithm,
    pub distance_metric: VectorDistanceMetric,
    pub priority: u16,
    pub usage_hint: Option<String>,
    pub quantization: QuantizationSpec,
    pub custom_params: Vec<u8>,
    pub data_section: Option<MetadataSection>,
    pub wasm_section: Option<MetadataSection>,
}

impl VectorIndexDescriptor {
    pub fn data_section(&self) -> Option<&MetadataSection> {
        self.data_section.as_ref()
    }

    pub fn wasm_section(&self) -> Option<&MetadataSection> {
        self.wasm_section.as_ref()
    }

    pub fn to_fb<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<fb::VectorIndexDescriptor<'fb>> {
        let column = fbb.create_string(&self.column);
        let usage_hint = self.usage_hint.as_ref().map(|hint| fbb.create_string(hint));
        let custom_params = if self.custom_params.is_empty() {
            None
        } else {
            Some(fbb.create_vector(&self.custom_params))
        };
        let quantization = self.quantization.to_fb(fbb);
        let data_section = self.data_section.as_ref().map(|section| section.to_fb(fbb));
        let wasm_section = self.wasm_section.as_ref().map(|section| section.to_fb(fbb));
        let args = fb::VectorIndexDescriptorArgs {
            index_id: self.index_id,
            column: Some(column),
            algorithm: self.algorithm.clone().into(),
            metric: self.distance_metric.clone().into(),
            priority: self.priority,
            usage_hint,
            data_section,
            wasm_section,
            quantization,
            custom_params,
        };
        fb::VectorIndexDescriptor::create(fbb, &args)
    }
}

impl<'a> From<&fb::VectorIndexDescriptor<'a>> for VectorIndexDescriptor {
    fn from(desc: &fb::VectorIndexDescriptor<'a>) -> Self {
        Self {
            index_id: desc.index_id(),
            column: desc.column().unwrap_or_default().to_string(),
            algorithm: desc.algorithm().into(),
            distance_metric: desc.metric().into(),
            priority: desc.priority(),
            usage_hint: desc.usage_hint().map(|s| s.to_string()),
            quantization: desc
                .quantization()
                .map(|q| QuantizationSpec::from(&q))
                .unwrap_or_default(),
            custom_params: desc
                .custom_params()
                .map(|p| p.bytes().to_vec())
                .unwrap_or_default(),
            data_section: desc.data_section().map(|sec| MetadataSection::from(&sec)),
            wasm_section: desc.wasm_section().map(|sec| MetadataSection::from(&sec)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VectorIndexConfig {
    pub index_id: u32,
    pub column: String,
    pub algorithm: VectorIndexAlgorithm,
    pub distance_metric: VectorDistanceMetric,
    pub priority: u16,
    pub usage_hint: Option<String>,
    pub quantization: QuantizationSpec,
    pub custom_params: Vec<u8>,
    pub data: Vec<u8>,
    pub wasm_module: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub enum VectorIndexBuildAlgorithm {
    BruteForce,
    Hnsw {
        max_neighbors: usize,
        ef_search: usize,
    },
}

#[derive(Clone, Debug)]
pub struct VectorIndexBuildConfig {
    pub index_id: u32,
    pub column: String,
    pub algorithm: VectorIndexBuildAlgorithm,
    pub distance_metric: VectorDistanceMetric,
    pub priority: u16,
    pub usage_hint: Option<String>,
    pub quantization: QuantizationSpec,
    pub custom_params: Vec<u8>,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            index_id: 0,
            column: String::new(),
            algorithm: VectorIndexAlgorithm::BruteForce,
            distance_metric: VectorDistanceMetric::L2,
            priority: 0,
            usage_hint: None,
            quantization: QuantizationSpec::default(),
            custom_params: Vec::new(),
            data: Vec::new(),
            wasm_module: None,
        }
    }
}

impl VectorIndexConfig {
    pub fn into_descriptor(
        self,
        data_section: Option<MetadataSection>,
        wasm_section: Option<MetadataSection>,
    ) -> VectorIndexDescriptor {
        VectorIndexDescriptor {
            index_id: self.index_id,
            column: self.column,
            algorithm: self.algorithm,
            distance_metric: self.distance_metric,
            priority: self.priority,
            usage_hint: self.usage_hint,
            quantization: self.quantization,
            custom_params: self.custom_params,
            data_section,
            wasm_section,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorSearchResult {
    pub row_id: u64,
    pub distance: f32,
}

pub enum VectorIndexRuntime {
    BruteForce(BruteForceIndex),
    Hnsw(HnswIndex),
    CustomWasm(WasmVectorIndex),
}

impl VectorIndexRuntime {
    pub fn from_descriptor(
        desc: &VectorIndexDescriptor,
        bytes: &[u8],
        wasm_blob: Option<&[u8]>,
    ) -> Result<Self> {
        match desc.algorithm {
            VectorIndexAlgorithm::BruteForce => {
                if desc.distance_metric != VectorDistanceMetric::L2 {
                    return Err(Error::General(
                        "Brute-force index currently supports only L2 metric".to_string(),
                    ));
                }
                Ok(Self::BruteForce(BruteForceIndex::from_bytes(bytes)?))
            }
            VectorIndexAlgorithm::Hnsw => {
                if desc.distance_metric != VectorDistanceMetric::L2 {
                    return Err(Error::General(
                        "HNSW index currently supports only L2 metric".to_string(),
                    ));
                }
                Ok(Self::Hnsw(HnswIndex::from_bytes(bytes)?))
            }
            VectorIndexAlgorithm::CustomWasm => {
                let wasm_blob = wasm_blob.ok_or_else(|| {
                    Error::General(
                        "Custom Wasm vector index is missing its wasm_module bytes".to_string(),
                    )
                })?;
                Ok(Self::CustomWasm(WasmVectorIndex::new(
                    bytes,
                    wasm_blob,
                    desc.distance_metric,
                )?))
            }
            other => Err(Error::General(format!(
                "Vector algorithm {:?} is not supported",
                other
            ))),
        }
    }

    pub fn knn_l2(&self, query: &[f32], k: usize) -> Result<Vec<VectorSearchResult>> {
        match self {
            Self::BruteForce(index) => index.knn_l2(query, k),
            Self::Hnsw(index) => index.knn_l2(query, k),
            Self::CustomWasm(index) => index.knn_l2(query, k),
        }
    }

    pub fn knn_l2_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<VectorSearchResult>>> {
        match self {
            Self::BruteForce(index) => {
                let mut out = Vec::with_capacity(queries.len());
                for query in queries {
                    out.push(index.knn_l2(query, k)?);
                }
                Ok(out)
            }
            Self::Hnsw(index) => {
                let mut out = Vec::with_capacity(queries.len());
                for query in queries {
                    out.push(index.knn_l2(query, k)?);
                }
                Ok(out)
            }
            Self::CustomWasm(index) => index.knn_l2_batch(queries, k),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BruteForceIndex {
    dimension: usize,
    num_vectors: usize,
    values: Vec<f32>,
}

impl BruteForceIndex {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::General(
                "Vector index blob too small for header".to_string(),
            ));
        }
        let num_vectors = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let dimension = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let expected = 8 + num_vectors
            .checked_mul(dimension)
            .ok_or_else(|| Error::General("Vector index size overflow".to_string()))?
            .checked_mul(4)
            .ok_or_else(|| Error::General("Vector index size overflow".to_string()))?;
        if bytes.len() != expected {
            return Err(Error::General(format!(
                "Vector index blob size mismatch. expected {}, got {}",
                expected,
                bytes.len()
            )));
        }
        let mut values = Vec::with_capacity(num_vectors * dimension);
        let mut cursor = Cursor::new(&bytes[8..]);
        for _ in 0..num_vectors * dimension {
            values.push(
                cursor
                    .read_f32::<LittleEndian>()
                    .map_err(|e| Error::General(format!("Unable to read vector value: {e}")))?,
            );
        }
        Ok(Self {
            dimension,
            num_vectors,
            values,
        })
    }

    pub fn knn_l2(&self, query: &[f32], k: usize) -> Result<Vec<VectorSearchResult>> {
        if query.len() != self.dimension {
            return Err(Error::General(format!(
                "Query dimension {} does not match index dimension {}",
                query.len(),
                self.dimension
            )));
        }
        let mut results = Vec::with_capacity(self.num_vectors);
        for row in 0..self.num_vectors {
            let base = row * self.dimension;
            let mut dist = 0.0f32;
            for d in 0..self.dimension {
                let diff = self.values[base + d] - query[d];
                dist += diff * diff;
            }
            results.push(VectorSearchResult {
                row_id: row as u64,
                distance: dist,
            });
        }
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        let limit = k.min(results.len());
        results.truncate(limit);
        Ok(results)
    }
}

pub fn encode_bruteforce_index(vectors: &[Vec<f32>]) -> Result<Vec<u8>> {
    if vectors.is_empty() {
        return Err(Error::General(
            "Cannot build vector index with zero vectors".to_string(),
        ));
    }
    let dimension = vectors[0].len();
    if dimension == 0 {
        return Err(Error::General(
            "Vector dimension must be greater than 0".to_string(),
        ));
    }
    if vectors.iter().any(|v| v.len() != dimension) {
        return Err(Error::General(
            "All vectors must share the same dimension".to_string(),
        ));
    }
    let mut buffer = Vec::with_capacity(8 + vectors.len() * dimension * 4);
    buffer.extend_from_slice(&(vectors.len() as u32).to_le_bytes());
    buffer.extend_from_slice(&(dimension as u32).to_le_bytes());
    for vec in vectors {
        for value in vec {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(buffer)
}

#[derive(Clone, Debug)]
pub struct HnswIndex {
    dimension: usize,
    num_vectors: usize,
    values: Vec<f32>,
    levels: Vec<Vec<Vec<usize>>>,
    entry_point: usize,
    ef_search: usize,
    max_level: usize,
    _m: usize,
}

impl HnswIndex {
    const HEADER_SIZE: usize = 24;

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::HEADER_SIZE {
            return Err(Error::General(
                "HNSW index blob too small for header".to_string(),
            ));
        }
        let num_vectors = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let dimension = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let m = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let ef_search = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let entry_point = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let level_count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        if num_vectors == 0 {
            return Err(Error::General(
                "HNSW index must contain at least one vector".to_string(),
            ));
        }
        if dimension == 0 {
            return Err(Error::General(
                "HNSW vector dimension must be greater than 0".to_string(),
            ));
        }
        if ef_search == 0 {
            return Err(Error::General(
                "HNSW ef_search must be greater than 0".to_string(),
            ));
        }
        if entry_point >= num_vectors {
            return Err(Error::General(
                "HNSW entry point must reference an existing vector".to_string(),
            ));
        }
        if level_count == 0 {
            return Err(Error::General(
                "HNSW level count must be greater than 0".to_string(),
            ));
        }
        let values_len = num_vectors
            .checked_mul(dimension)
            .ok_or_else(|| Error::General("HNSW vector length overflow".to_string()))?
            .checked_mul(4)
            .ok_or_else(|| Error::General("HNSW vector byte length overflow".to_string()))?;
        if bytes.len() < Self::HEADER_SIZE + values_len {
            return Err(Error::General(
                "HNSW blob truncated before vector payload".to_string(),
            ));
        }
        let mut values = Vec::with_capacity(num_vectors * dimension);
        let mut cursor = Cursor::new(&bytes[Self::HEADER_SIZE..Self::HEADER_SIZE + values_len]);
        for _ in 0..num_vectors * dimension {
            values.push(
                cursor
                    .read_f32::<LittleEndian>()
                    .map_err(|e| Error::General(format!("Unable to read vector value: {e}")))?,
            );
        }

        let mut offset = Self::HEADER_SIZE + values_len;
        let mut levels = vec![vec![Vec::new(); num_vectors]; level_count];
        for node in 0..num_vectors {
            if offset + 4 > bytes.len() {
                return Err(Error::General(
                    "HNSW adjacency section truncated".to_string(),
                ));
            }
            let node_levels =
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let actual_levels = node_levels.min(level_count);
            for lvl in 0..actual_levels {
                if offset + 4 > bytes.len() {
                    return Err(Error::General("HNSW neighbor list truncated".to_string()));
                }
                let neighbor_count =
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;
                let needed = neighbor_count
                    .checked_mul(4)
                    .ok_or_else(|| Error::General("HNSW neighbor list overflow".to_string()))?;
                if offset + needed > bytes.len() {
                    return Err(Error::General("HNSW neighbor list truncated".to_string()));
                }
                let mut neighbors = Vec::with_capacity(neighbor_count);
                for _ in 0..neighbor_count {
                    let idx =
                        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                    if idx >= num_vectors {
                        return Err(Error::General(
                            "HNSW neighbor index out of bounds".to_string(),
                        ));
                    }
                    neighbors.push(idx);
                    offset += 4;
                }
                levels[lvl][node] = neighbors;
            }
            for _ in actual_levels..level_count {
                if offset + 4 > bytes.len() {
                    return Err(Error::General("HNSW padding truncated".to_string()));
                }
                let padding = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                if padding != 0 {
                    return Err(Error::General(
                        "HNSW padding value must be zero".to_string(),
                    ));
                }
                offset += 4;
            }
        }
        if offset != bytes.len() {
            return Err(Error::General(
                "HNSW blob has trailing bytes beyond adjacency lists".to_string(),
            ));
        }
        Ok(Self {
            dimension,
            num_vectors,
            values,
            levels,
            entry_point,
            ef_search,
            max_level: level_count.saturating_sub(1),
            _m: m,
        })
    }

    fn l2_distance(&self, row_id: usize, query: &[f32]) -> f32 {
        let base = row_id * self.dimension;
        let mut acc = f32x4::splat(0.0);
        let chunks = query.chunks_exact(4);
        let remainder = chunks.remainder();
        for (idx, chunk) in chunks.enumerate() {
            let start = base + idx * 4;
            let a = load_f32x4(&self.values[start..start + 4]);
            let b = load_f32x4(chunk);
            let diff = a - b;
            acc += diff * diff;
        }
        let mut sum = acc.reduce_add();
        let tail_start = query.len() - remainder.len();
        for i in 0..remainder.len() {
            let diff = self.values[base + tail_start + i] - remainder[i];
            sum += diff * diff;
        }
        sum
    }

    pub fn knn_l2(&self, query: &[f32], k: usize) -> Result<Vec<VectorSearchResult>> {
        if query.len() != self.dimension {
            return Err(Error::General(format!(
                "Query dimension {} does not match HNSW index dimension {}",
                query.len(),
                self.dimension
            )));
        }
        let mut current = self.entry_point;
        let mut current_dist = self.l2_distance(current, query);
        for level in (1..=self.max_level).rev() {
            current = self.greedy_search(current, query, level, current_dist);
            current_dist = self.l2_distance(current, query);
        }
        let ef = self.ef_search.max(k.max(1));
        let candidates = self.search_layer(current, query, 0, ef);
        let mut results: Vec<VectorSearchResult> = candidates
            .into_iter()
            .map(|entry| VectorSearchResult {
                row_id: entry.index as u64,
                distance: entry.distance.0,
            })
            .collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        if results.len() > k {
            results.truncate(k);
        }
        Ok(results)
    }

    fn greedy_search(
        &self,
        mut current: usize,
        query: &[f32],
        level: usize,
        mut current_dist: f32,
    ) -> usize {
        loop {
            let mut improved = false;
            for &neighbor in &self.levels[level][current] {
                let dist = self.l2_distance(neighbor, query);
                if dist < current_dist {
                    current_dist = dist;
                    current = neighbor;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        current
    }

    fn search_layer(
        &self,
        entry: usize,
        query: &[f32],
        level: usize,
        ef: usize,
    ) -> Vec<DistanceEntry> {
        let mut visited = vec![false; self.num_vectors];
        let mut candidates = BinaryHeap::new();
        let mut best = BinaryHeap::new();
        let entry_dist = self.l2_distance(entry, query);
        let entry_entry = DistanceEntry::new(entry_dist, entry);
        candidates.push(Reverse(entry_entry));
        best.push(entry_entry);
        visited[entry] = true;

        while let Some(Reverse(candidate)) = candidates.pop() {
            let worst = best
                .peek()
                .map(|w| w.distance.0)
                .unwrap_or(candidate.distance.0);
            if candidate.distance.0 > worst && best.len() >= ef {
                break;
            }
            for &neighbor in &self.levels[level][candidate.index] {
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;
                let dist = self.l2_distance(neighbor, query);
                let entry = DistanceEntry::new(dist, neighbor);
                if best.len() < ef {
                    best.push(entry);
                    candidates.push(Reverse(entry));
                } else if dist < best.peek().unwrap().distance.0 {
                    best.pop();
                    best.push(entry);
                    candidates.push(Reverse(entry));
                }
            }
        }
        best.into_iter().collect()
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct OrderedF32(pub f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct DistanceEntry {
    distance: OrderedF32,
    index: usize,
}

impl DistanceEntry {
    fn new(distance: f32, index: usize) -> Self {
        Self {
            distance: OrderedF32(distance),
            index,
        }
    }
}

impl Ord for DistanceEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.distance.cmp(&other.distance) {
            Ordering::Equal => self.index.cmp(&other.index),
            ord => ord,
        }
    }
}

impl PartialOrd for DistanceEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn encode_hnsw_index(
    vectors: &[Vec<f32>],
    max_neighbors: usize,
    ef_search: usize,
) -> Result<Vec<u8>> {
    if vectors.is_empty() {
        return Err(Error::General(
            "Cannot build HNSW index with zero vectors".to_string(),
        ));
    }
    if max_neighbors == 0 {
        return Err(Error::General(
            "HNSW max_neighbors must be greater than 0".to_string(),
        ));
    }
    if ef_search == 0 {
        return Err(Error::General(
            "HNSW ef_search must be greater than 0".to_string(),
        ));
    }
    let dimension = vectors[0].len();
    if dimension == 0 {
        return Err(Error::General(
            "Vector dimension must be greater than 0".to_string(),
        ));
    }
    if vectors.iter().any(|v| v.len() != dimension) {
        return Err(Error::General(
            "All vectors must share the same dimension".to_string(),
        ));
    }
    let mut builder = HnswBuilder::new(dimension, max_neighbors, ef_search)?;
    for vec in vectors {
        builder.add_vector(vec.clone())?;
    }
    builder.serialize()
}

struct HnswBuilder {
    dimension: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    rng: StdRng,
    vectors: Vec<Vec<f32>>,
    adjacency: Vec<Vec<Vec<usize>>>,
    node_levels: Vec<usize>,
    entry_point: usize,
    max_level: isize,
}

impl HnswBuilder {
    fn new(dimension: usize, max_neighbors: usize, ef_search: usize) -> Result<Self> {
        let m = max_neighbors.min(64).max(2);
        let m0 = (m * 2).min(128);
        let ef_construction = ef_search.max(m0);
        Ok(Self {
            dimension,
            m,
            m0,
            ef_construction,
            ef_search: ef_search.max(1),
            rng: StdRng::seed_from_u64(0xF3F3_F3F3_F3F3_F3F3),
            vectors: Vec::new(),
            adjacency: Vec::new(),
            node_levels: Vec::new(),
            entry_point: 0,
            max_level: -1,
        })
    }

    fn add_vector(&mut self, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::General(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }
        let node_id = self.vectors.len();
        let level = self.sample_level();
        self.ensure_level_storage(level, node_id);
        self.vectors.push(vector);
        self.node_levels.push(level);

        if node_id == 0 {
            self.entry_point = 0;
            self.max_level = level as isize;
            return Ok(());
        }

        let mut current = self.entry_point;
        let mut current_dist = simd_l2_distance(&self.vectors[current], &self.vectors[node_id]);
        for lvl in ((level as isize + 1)..=self.max_level).rev() {
            current = self.greedy_search(current, node_id, lvl as usize, current_dist);
            current_dist = simd_l2_distance(&self.vectors[current], &self.vectors[node_id]);
        }

        let top_level = std::cmp::min(level as isize, self.max_level) as usize;
        for lvl in (0..=top_level).rev() {
            let candidates = self.search_layer(current, node_id, lvl, self.ef_construction);
            let max_degree = if lvl == 0 { self.m0 } else { self.m };
            let selected = self.select_neighbors(node_id, candidates, max_degree);
            self.connect_new_node(node_id, &selected, lvl, max_degree);
            if !selected.is_empty() {
                current = selected[0].0;
            }
        }

        if (level as isize) > self.max_level {
            self.max_level = level as isize;
            self.entry_point = node_id;
        }

        Ok(())
    }

    fn serialize(mut self) -> Result<Vec<u8>> {
        let num_vectors = self.vectors.len();
        let mut buffer =
            Vec::with_capacity(HnswIndex::HEADER_SIZE + num_vectors * self.dimension * 4);
        buffer.extend_from_slice(&(num_vectors as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.dimension as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.m as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.ef_search as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.entry_point as u32).to_le_bytes());
        buffer.extend_from_slice(&((self.max_level + 1) as u32).to_le_bytes());

        for vec in &self.vectors {
            for value in vec {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
        }

        let total_levels = (self.max_level + 1).max(0) as usize;
        for level in &mut self.adjacency {
            while level.len() < num_vectors {
                level.push(Vec::new());
            }
        }

        for node in 0..num_vectors {
            let level_count = self.node_levels[node] + 1;
            buffer.extend_from_slice(&(level_count as u32).to_le_bytes());
            for lvl in 0..level_count {
                let neighbors = &self.adjacency[lvl][node];
                buffer.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
                for &neighbor in neighbors {
                    buffer.extend_from_slice(&(neighbor as u32).to_le_bytes());
                }
            }
            for _ in level_count..total_levels {
                buffer.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        Ok(buffer)
    }

    fn ensure_level_storage(&mut self, level: usize, node_id: usize) {
        while self.adjacency.len() <= level {
            self.adjacency.push(vec![Vec::new(); node_id]);
        }
        for lvl in 0..self.adjacency.len() {
            if self.adjacency[lvl].len() < node_id + 1 {
                self.adjacency[lvl].push(Vec::new());
            }
        }
    }

    fn sample_level(&mut self) -> usize {
        let mut level = 0;
        let lambda = 1.0 / (self.m as f64).ln();
        while self.rng.gen::<f64>() < (-lambda).exp() {
            level += 1;
        }
        level
    }

    fn greedy_search(
        &self,
        mut current: usize,
        target_node: usize,
        level: usize,
        mut current_dist: f32,
    ) -> usize {
        loop {
            let mut improved = false;
            for &neighbor in &self.adjacency[level][current] {
                let dist = simd_l2_distance(&self.vectors[neighbor], &self.vectors[target_node]);
                if dist < current_dist {
                    current_dist = dist;
                    current = neighbor;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        current
    }

    fn search_layer(
        &self,
        entry: usize,
        target_node: usize,
        level: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        let query = &self.vectors[target_node];
        let mut visited = vec![false; self.vectors.len()];
        let mut candidates = BinaryHeap::new();
        let mut best = BinaryHeap::new();
        let entry_dist = simd_l2_distance(&self.vectors[entry], query);
        let entry_entry = DistanceEntry::new(entry_dist, entry);
        candidates.push(Reverse(entry_entry));
        best.push(entry_entry);
        visited[entry] = true;

        while let Some(Reverse(candidate)) = candidates.pop() {
            let worst = best
                .peek()
                .map(|w| w.distance.0)
                .unwrap_or(candidate.distance.0);
            if candidate.distance.0 > worst && best.len() >= ef {
                break;
            }
            for &neighbor in &self.adjacency[level][candidate.index] {
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;
                let dist = simd_l2_distance(&self.vectors[neighbor], query);
                let entry = DistanceEntry::new(dist, neighbor);
                if best.len() < ef {
                    best.push(entry);
                    candidates.push(Reverse(entry));
                } else if dist < best.peek().unwrap().distance.0 {
                    best.pop();
                    best.push(entry);
                    candidates.push(Reverse(entry));
                }
            }
        }

        let mut results: Vec<_> = best
            .into_iter()
            .map(|entry| (entry.index, entry.distance.0))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    fn select_neighbors(
        &self,
        node_id: usize,
        mut candidates: Vec<(usize, f32)>,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let query = &self.vectors[node_id];
        for candidate in &mut candidates {
            if candidate.1.is_nan() {
                candidate.1 = simd_l2_distance(query, &self.vectors[candidate.0]);
            }
        }
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(limit);
        candidates
    }

    fn connect_new_node(
        &mut self,
        node_id: usize,
        neighbors: &[(usize, f32)],
        level: usize,
        max_degree: usize,
    ) {
        for &(neighbor, _) in neighbors {
            if !self.adjacency[level][node_id].contains(&neighbor) {
                self.adjacency[level][node_id].push(neighbor);
            }
            if !self.adjacency[level][neighbor].contains(&node_id) {
                self.adjacency[level][neighbor].push(node_id);
            }
            self.prune_connections(neighbor, level, max_degree);
        }
        self.prune_connections(node_id, level, max_degree);
    }

    fn prune_connections(&mut self, node_id: usize, level: usize, max_degree: usize) {
        let adj = &mut self.adjacency[level][node_id];
        if adj.len() <= max_degree {
            return;
        }
        let mut scored: Vec<_> = adj
            .iter()
            .map(|&neighbor| {
                let dist = simd_l2_distance(&self.vectors[node_id], &self.vectors[neighbor]);
                (neighbor, dist)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(max_degree);
        *adj = scored.into_iter().map(|(idx, _)| idx).collect();
    }
}

fn simd_l2_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut acc = f32x4::splat(0.0);
    let chunks = lhs.chunks_exact(4);
    let remainder = chunks.remainder();
    for (chunk_l, chunk_r) in chunks.zip(rhs.chunks_exact(4)) {
        let a = load_f32x4(chunk_l);
        let b = load_f32x4(chunk_r);
        let diff = a - b;
        acc += diff * diff;
    }
    let mut sum = acc.reduce_add();
    let tail_start = lhs.len() - remainder.len();
    for i in 0..remainder.len() {
        let diff = lhs[tail_start + i] - rhs[tail_start + i];
        sum += diff * diff;
    }
    sum
}

#[inline]
fn load_f32x4(slice: &[f32]) -> f32x4 {
    f32x4::from([slice[0], slice[1], slice[2], slice[3]])
}

pub struct WasmVectorIndex {
    runtime: WasmRuntime,
    metric: VectorDistanceMetric,
    payload_buffer: RefCell<Vec<u8>>,
}

impl WasmVectorIndex {
    fn new(index_blob: &[u8], wasm_module: &[u8], metric: VectorDistanceMetric) -> Result<Self> {
        if metric != VectorDistanceMetric::L2 {
            return Err(Error::General(
                "Custom Wasm vector index currently supports only L2 metric".to_string(),
            ));
        }
        let runtime = WasmRuntime::try_new(wasm_module).map_err(|e| {
            Error::General(format!(
                "Failed to initialize WASM runtime for vector index: {e}"
            ))
        })?;
        runtime
            .call_scalar_function_owned("vector_init_ffi", index_blob)
            .map_err(|e| Error::General(format!("Vector index WASM init failed: {e}")))?;
        Ok(Self {
            runtime,
            metric,
            payload_buffer: RefCell::new(Vec::new()),
        })
    }

    fn knn_l2(&self, query: &[f32], k: usize) -> Result<Vec<VectorSearchResult>> {
        if self.metric != VectorDistanceMetric::L2 {
            return Err(Error::General(
                "Custom Wasm vector index currently supports only L2 metric".to_string(),
            ));
        }
        let response = {
            let mut payload = self.payload_buffer.borrow_mut();
            payload.clear();
            payload.reserve(8 + query.len() * 4);
            payload.extend_from_slice(&(query.len() as u32).to_le_bytes());
            payload.extend_from_slice(&(k as u32).to_le_bytes());
            for value in query {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            self.runtime
                .call_scalar_function_owned("vector_knn_ffi", &payload)
                .map_err(|e| Error::General(format!("Vector index WASM query failed: {e}")))?
        };
        decode_wasm_results(&response)
    }

    fn knn_l2_batch(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<VectorSearchResult>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        if self.metric != VectorDistanceMetric::L2 {
            return Err(Error::General(
                "Custom Wasm vector index currently supports only L2 metric".to_string(),
            ));
        }
        let dimension = queries[0].len();
        for (idx, query) in queries.iter().enumerate() {
            if query.len() != dimension {
                return Err(Error::General(format!(
                    "Query {} dimension {} does not match HNSW index dimension {}",
                    idx,
                    query.len(),
                    dimension
                )));
            }
        }
        let response = {
            let mut payload = self.payload_buffer.borrow_mut();
            payload.clear();
            payload.reserve(12 + queries.len() * dimension * 4);
            payload.extend_from_slice(&(dimension as u32).to_le_bytes());
            payload.extend_from_slice(&(k as u32).to_le_bytes());
            payload.extend_from_slice(&(queries.len() as u32).to_le_bytes());
            for query in queries {
                for value in query {
                    payload.extend_from_slice(&value.to_le_bytes());
                }
            }
            self.runtime
                .call_scalar_function_owned("vector_knn_batch_ffi", &payload)
                .map_err(|e| Error::General(format!("Vector index WASM batch query failed: {e}")))?
        };
        decode_wasm_batch_results(&response, queries.len())
    }
}

fn decode_wasm_results(bytes: &[u8]) -> Result<Vec<VectorSearchResult>> {
    if bytes.len() < 4 {
        return Err(Error::General(
            "Vector WASM response too small to contain result count".to_string(),
        ));
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let expected = 4 + count * (8 + 4);
    if bytes.len() != expected {
        return Err(Error::General(format!(
            "Vector WASM response size mismatch. expected {} bytes, got {}",
            expected,
            bytes.len()
        )));
    }
    let mut results = Vec::with_capacity(count);
    let mut offset = 4;
    for _ in 0..count {
        let row_id = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let distance = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        results.push(VectorSearchResult { row_id, distance });
    }
    Ok(results)
}

fn decode_wasm_batch_results(
    bytes: &[u8],
    expected_batches: usize,
) -> Result<Vec<Vec<VectorSearchResult>>> {
    if bytes.len() < 4 {
        return Err(Error::General(
            "Vector WASM batch response too small to contain batch count".to_string(),
        ));
    }
    let batch_count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    if batch_count != expected_batches {
        return Err(Error::General(format!(
            "Vector WASM batch response mismatch. expected {} batches, got {}",
            expected_batches, batch_count
        )));
    }
    let mut offset = 4;
    let mut all_results = Vec::with_capacity(batch_count);
    for _ in 0..batch_count {
        if offset + 4 > bytes.len() {
            return Err(Error::General(
                "Vector WASM batch response truncated before result count".to_string(),
            ));
        }
        let count = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut results = Vec::with_capacity(count);
        for _ in 0..count {
            if offset + 12 > bytes.len() {
                return Err(Error::General(
                    "Vector WASM batch response truncated in result payload".to_string(),
                ));
            }
            let row_id = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
            let distance = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;
            results.push(VectorSearchResult { row_id, distance });
        }
        all_results.push(results);
    }
    if offset != bytes.len() {
        return Err(Error::General(
            "Vector WASM batch response has trailing bytes".to_string(),
        ));
    }
    Ok(all_results)
}
