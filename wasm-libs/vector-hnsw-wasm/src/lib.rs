use byteorder::{LittleEndian, ReadBytesExt};
use fff_core::errors::{Error, Result};
use fff_ude::ffi::scalar_wrapper;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::Cursor;
use std::sync::OnceLock;
use wide::f32x4;

static INDEX: OnceLock<HnswIndex> = OnceLock::new();

fn vector_init(input: &[u8]) -> Result<Box<[u8]>> {
    let index = HnswIndex::from_bytes(input)?;
    INDEX
        .set(index)
        .map_err(|_| Error::General("HNSW index already initialized".to_string()))?;
    Ok(Vec::new().into_boxed_slice())
}

fn vector_knn(input: &[u8]) -> Result<Box<[u8]>> {
    if input.len() < 8 {
        return Err(Error::General(
            "vector knn payload must contain dimension and k".to_string(),
        ));
    }
    let dimension = u32::from_le_bytes(input[0..4].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
    let expected = 8 + dimension * 4;
    if input.len() != expected {
        return Err(Error::General(format!(
            "vector knn payload dimension mismatch. expected {} bytes, got {}",
            expected,
            input.len()
        )));
    }
    let mut query = vec![0f32; dimension];
    let mut cursor = Cursor::new(&input[8..]);
    for value in query.iter_mut() {
        *value = cursor
            .read_f32::<LittleEndian>()
            .map_err(|e| Error::General(format!("Unable to read query value: {e}")))?;
    }
    let index = INDEX
        .get()
        .ok_or_else(|| Error::General("HNSW index not initialized".to_string()))?;
    let results = index.knn_l2(&query, k)?;
    let mut buffer = Vec::with_capacity(4 + results.len() * (8 + 4));
    buffer.extend_from_slice(&(results.len() as u32).to_le_bytes());
    for res in results {
        buffer.extend_from_slice(&(res.row_id as u64).to_le_bytes());
        buffer.extend_from_slice(&res.distance.to_le_bytes());
    }
    Ok(buffer.into_boxed_slice())
}

#[no_mangle]
pub unsafe extern "C" fn vector_init_ffi(
    ptr: *const u8,
    len: usize,
    out: *mut fff_ude::ffi::CSlice,
) -> i32 {
    scalar_wrapper(vector_init, ptr, len, out)
}

#[no_mangle]
pub unsafe extern "C" fn vector_knn_ffi(
    ptr: *const u8,
    len: usize,
    out: *mut fff_ude::ffi::CSlice,
) -> i32 {
    scalar_wrapper(vector_knn, ptr, len, out)
}

#[derive(Clone, Debug)]
struct HnswIndex {
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

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
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
                let mut list = Vec::with_capacity(neighbor_count);
                for _ in 0..neighbor_count {
                    let idx =
                        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                    if idx >= num_vectors {
                        return Err(Error::General(
                            "HNSW neighbor index out of bounds".to_string(),
                        ));
                    }
                    list.push(idx);
                    offset += 4;
                }
                levels[lvl][node] = list;
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

    fn knn_l2(&self, query: &[f32], k: usize) -> Result<Vec<VectorSearchResult>> {
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
        let mut results: Vec<VectorSearchResult> = self
            .search_layer(current, query, 0, ef)
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
        candidates.push(std::cmp::Reverse(entry_entry));
        best.push(entry_entry);
        visited[entry] = true;

        while let Some(std::cmp::Reverse(candidate)) = candidates.pop() {
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
                    candidates.push(std::cmp::Reverse(entry));
                } else if dist < best.peek().unwrap().distance.0 {
                    best.pop();
                    best.push(entry);
                    candidates.push(std::cmp::Reverse(entry));
                }
            }
        }
        best.into_iter().collect()
    }
}

#[derive(Clone, Debug)]
struct VectorSearchResult {
    row_id: u64,
    distance: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct OrderedF32(f32);

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

#[inline]
fn load_f32x4(slice: &[f32]) -> f32x4 {
    f32x4::from([slice[0], slice[1], slice[2], slice[3]])
}
