use ordered_float::NotNan;
use std::alloc::{alloc, dealloc, Layout};
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

#[link(wasm_import_module = "env")]
extern "C" {
    fn host_chunk_len(chunk_id: u32) -> u32;
    fn host_read_chunk(chunk_id: u32, dst_ptr: u32, dst_len: u32) -> u32;
}

#[no_mangle]
pub unsafe extern "C" fn alloc_ffi(len: u32, align: u32) -> u32 {
    if len == 0 {
        return 0;
    }
    let Ok(layout) = Layout::from_size_align(len as usize, align as usize) else {
        return 0;
    };
    let ptr = alloc(layout);
    ptr as u32
}

#[no_mangle]
pub unsafe extern "C" fn dealloc_ffi(ptr: u32, len: u32, align: u32) {
    if ptr == 0 || len == 0 {
        return;
    }
    let Ok(layout) = Layout::from_size_align(len as usize, align as usize) else {
        return;
    };
    dealloc(ptr as *mut u8, layout);
}

#[derive(Clone)]
struct DecodedPosting {
    row_ids: Vec<u32>,
    vectors: Vec<f32>, // count * dim
    bytes: usize,
}

#[derive(Default)]
struct DecodedCache {
    map: HashMap<u32, DecodedPosting>,
    fifo: VecDeque<u32>,
    bytes: usize,
    budget: usize,
}

impl DecodedCache {
    fn with_default_budget() -> Self {
        Self {
            budget: 192 * 1024 * 1024,
            ..Default::default()
        }
    }

    fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
        self.evict_if_needed(0);
    }

    fn evict_if_needed(&mut self, incoming: usize) {
        if self.budget == 0 {
            self.map.clear();
            self.fifo.clear();
            self.bytes = 0;
            return;
        }
        while self.bytes + incoming > self.budget {
            let Some(old) = self.fifo.pop_front() else {
                break;
            };
            if let Some(v) = self.map.remove(&old) {
                self.bytes = self.bytes.saturating_sub(v.bytes);
            }
        }
    }

    fn get(&self, chunk_id: u32) -> Option<&DecodedPosting> {
        self.map.get(&chunk_id)
    }

    fn insert(&mut self, chunk_id: u32, posting: DecodedPosting) {
        if self.budget == 0 {
            return;
        }
        let incoming = posting.bytes;
        self.evict_if_needed(incoming);
        if incoming > self.budget {
            return;
        }
        if let Some(old) = self.map.remove(&chunk_id) {
            self.bytes = self.bytes.saturating_sub(old.bytes);
        }
        self.bytes += posting.bytes;
        self.map.insert(chunk_id, posting);
        self.fifo.push_back(chunk_id);
    }
}

fn decoded_cache() -> &'static Mutex<DecodedCache> {
    static CACHE: OnceLock<Mutex<DecodedCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(DecodedCache::with_default_budget()))
}

#[derive(Default, Clone, Copy)]
struct LastStats {
    decode_ns: u64,
    compute_ns: u64,
    decoded_cache_hits: u64,
    decoded_cache_misses: u64,
    decoded_cache_bytes: u64,
}

fn last_stats_cell() -> &'static Mutex<LastStats> {
    static STATS: OnceLock<Mutex<LastStats>> = OnceLock::new();
    STATS.get_or_init(|| Mutex::new(LastStats::default()))
}

#[no_mangle]
pub extern "C" fn ivf_set_decoded_cache_budget_ffi(bytes: u32) {
    decoded_cache().lock().unwrap().set_budget(bytes as usize);
}

/// Writes 5x u64 to `out_ptr`:
/// - decode_ns, compute_ns, decoded_cache_hits, decoded_cache_misses, decoded_cache_bytes
#[no_mangle]
pub unsafe extern "C" fn ivf_last_stats_ffi(out_ptr: u32) -> u32 {
    if out_ptr == 0 {
        return 0;
    }
    let s = *last_stats_cell().lock().unwrap();
    let out = std::slice::from_raw_parts_mut(out_ptr as *mut u64, 5);
    out[0] = s.decode_ns;
    out[1] = s.compute_ns;
    out[2] = s.decoded_cache_hits;
    out[3] = s.decoded_cache_misses;
    out[4] = s.decoded_cache_bytes;
    1
}

#[repr(C)]
struct Pair {
    row_id: u32,
    dist: NotNan<f32>,
}

impl Eq for Pair {}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.row_id == other.row_id
    }
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by distance (worst first), tie-break by row_id for stability.
        self.dist
            .cmp(&other.dist)
            .then_with(|| self.row_id.cmp(&other.row_id))
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn read_u32_le(buf: &[u8], offset: usize) -> Option<u32> {
    buf.get(offset..offset + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
}

fn decode_uleb128_u32(mut input: &[u8]) -> Option<(u32, usize)> {
    let mut value: u32 = 0;
    let mut shift: u32 = 0;
    let mut used = 0usize;
    loop {
        let b = *input.first()?;
        input = &input[1..];
        used += 1;
        let low = (b & 0x7f) as u32;
        value |= low.wrapping_shl(shift);
        if (b & 0x80) == 0 {
            return Some((value, used));
        }
        shift = shift.saturating_add(7);
        if used > 5 {
            return None;
        }
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    // IEEE-754 half -> float conversion.
    // Based on common reference implementations; handles subnormals/inf/nan.
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    let out_sign = sign << 31;
    let out: u32 = if exp == 0 {
        if frac == 0 {
            out_sign
        } else {
            // subnormal: normalize
            let mut e = -14i32;
            let mut f = frac;
            while (f & 0x0400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x03ff;
            let exp32 = (e + 127) as u32;
            out_sign | (exp32 << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        // inf/nan
        out_sign | 0x7f800000 | (frac << 13)
    } else {
        let exp32 = (exp as i32 - 15 + 127) as u32;
        out_sign | (exp32 << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

fn f16_lut() -> &'static [f32] {
    static LUT: OnceLock<Vec<f32>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut v = Vec::with_capacity(65536);
        for bits in 0u32..=0xffff {
            v.push(f16_bits_to_f32(bits as u16));
        }
        v
    })
}

fn l2_sq_f16(query: &[f32], vec_f16_bytes: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let mut off = 0usize;
    let lut = f16_lut();
    for &q in query {
        let bits = u16::from_le_bytes([vec_f16_bytes[off], vec_f16_bytes[off + 1]]);
        let v = lut[bits as usize];
        let d = q - v;
        sum += d * d;
        off += 2;
    }
    sum
}

fn decode_row_ids_raw(bytes: &[u8], count: usize) -> Option<Vec<u32>> {
    if bytes.len() < count * 4 {
        return None;
    }
    let mut out = Vec::<u32>::with_capacity(count);
    let mut off = 0usize;
    for _ in 0..count {
        let v = u32::from_le_bytes(bytes.get(off..off + 4)?.try_into().ok()?);
        out.push(v);
        off += 4;
    }
    Some(out)
}

fn decode_vectors_f16_to_f32(bytes: &[u8], count: usize, dim: usize) -> Option<Vec<f32>> {
    if bytes.len() < count * dim * 2 {
        return None;
    }
    let lut = f16_lut();
    let mut out = Vec::<f32>::with_capacity(count * dim);
    let mut off = 0usize;
    for _ in 0..(count * dim) {
        let b0 = *bytes.get(off)?;
        let b1 = *bytes.get(off + 1)?;
        let bits = u16::from_le_bytes([b0, b1]);
        out.push(lut[bits as usize]);
        off += 2;
    }
    Some(out)
}

fn decode_posting_raw_f16(bytes: &[u8], dim: usize) -> Option<DecodedPosting> {
    if bytes.len() < 4 {
        return None;
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let row_ids_bytes = 4 + count * 4;
    let vectors_bytes = count * dim * 2;
    if bytes.len() != row_ids_bytes + vectors_bytes {
        return None;
    }
    let row_ids = decode_row_ids_raw(&bytes[4..row_ids_bytes], count)?;
    let vectors = decode_vectors_f16_to_f32(&bytes[row_ids_bytes..], count, dim)?;
    let mem_bytes = row_ids.len() * 4 + vectors.len() * 4;
    Some(DecodedPosting {
        row_ids,
        vectors,
        bytes: mem_bytes,
    })
}

fn decode_posting_delta_f16(bytes: &[u8], dim: usize) -> Option<DecodedPosting> {
    if bytes.len() < 8 {
        return None;
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let mut cur_row = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    let mut offset = 8usize;
    let mut row_ids = Vec::<u32>::with_capacity(count);
    if count > 0 {
        row_ids.push(cur_row);
        for _ in 1..count {
            let (delta, used) = decode_uleb128_u32(&bytes[offset..])?;
            offset += used;
            cur_row = cur_row.wrapping_add(delta);
            row_ids.push(cur_row);
        }
    }
    let vectors_bytes = count * dim * 2;
    if bytes.len() != offset + vectors_bytes {
        return None;
    }
    let vectors = decode_vectors_f16_to_f32(&bytes[offset..], count, dim)?;
    let mem_bytes = row_ids.len() * 4 + vectors.len() * 4;
    Some(DecodedPosting {
        row_ids,
        vectors,
        bytes: mem_bytes,
    })
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

unsafe fn fetch_chunk(chunk_id: u32) -> Option<Vec<u8>> {
    let len = host_chunk_len(chunk_id);
    if len == 0 {
        return None;
    }
    let mut buf = Vec::<u8>::with_capacity(len as usize);
    buf.set_len(len as usize);
    let got = host_read_chunk(chunk_id, buf.as_mut_ptr() as u32, len);
    if got != len {
        return None;
    }
    Some(buf)
}

fn heap_push_topk(heap: &mut std::collections::BinaryHeap<Pair>, k: usize, row_id: u32, dist: f32) {
    let Ok(dist_nn) = NotNan::new(dist) else {
        return;
    };
    if heap.len() < k {
        heap.push(Pair {
            row_id,
            dist: dist_nn,
        });
    } else if let Some(worst) = heap.peek() {
        if dist_nn < worst.dist {
            let _ = heap.pop();
            heap.push(Pair {
                row_id,
                dist: dist_nn,
            });
        }
    }
}

fn write_out_pairs(
    out_words: &mut [u32],
    out_cap: usize,
    out: Vec<Pair>,
) -> u32 {
    let mut out = out;
    out.sort_by(|a, b| a.dist.cmp(&b.dist).then_with(|| a.row_id.cmp(&b.row_id)));
    let out_n = out.len().min(out_cap) as u32;
    for (i, pair) in out.into_iter().take(out_n as usize).enumerate() {
        out_words[i * 2] = pair.row_id;
        out_words[i * 2 + 1] = pair.dist.into_inner().to_bits();
    }
    out_n
}

/// Directory layout (little-endian):
/// - dim: u32
/// - nlist: u32
/// - centroids_chunk_id: u32
/// - posting_codec: u32 (0=raw, 1=row_id_delta_varint_v1)
/// - posting_chunk_ids: u32[nlist] (index by list_id)
///
/// Query layout:
/// - query: f32[dim]
///
/// Output layout:
/// - repeated (row_id:u32, dist_bits:u32) for up to `k` results.
#[no_mangle]
pub unsafe extern "C" fn ivf_flat_search_ffi(
    dir_ptr: u32,
    dir_len: u32,
    query_ptr: u32,
    dim: u32,
    k: u32,
    nprobe: u32,
    out_ptr: u32,
    out_cap: u32,
) -> u32 {
    if k == 0 || out_cap == 0 {
        return 0;
    }
    if dir_ptr == 0 || query_ptr == 0 || out_ptr == 0 {
        return 0;
    }

    let dir = std::slice::from_raw_parts(dir_ptr as *const u8, dir_len as usize);
    if dir.len() < 16 {
        return 0;
    }
    let dir_dim = read_u32_le(dir, 0).unwrap();
    let nlist = read_u32_le(dir, 4).unwrap();
    let centroids_chunk_id = read_u32_le(dir, 8).unwrap();
    let posting_codec = read_u32_le(dir, 12).unwrap();
    if dir_dim != dim || nlist == 0 {
        return 0;
    }
    let needed = 16usize + (nlist as usize) * 4;
    if dir.len() < needed {
        return 0;
    }
    let posting_ids_bytes = &dir[16..16 + (nlist as usize) * 4];
    let posting_chunk_ids: &[u32] = std::slice::from_raw_parts(
        posting_ids_bytes.as_ptr() as *const u32,
        nlist as usize,
    );

    let query = std::slice::from_raw_parts(query_ptr as *const f32, dim as usize);

    let centroids_bytes = match fetch_chunk(centroids_chunk_id) {
        Some(b) => b,
        None => return 0,
    };
    if centroids_bytes.len() != (nlist as usize) * (dim as usize) * 4 {
        return 0;
    }
    let centroids: &[f32] = std::slice::from_raw_parts(
        centroids_bytes.as_ptr() as *const f32,
        (nlist as usize) * (dim as usize),
    );

    // Pick nprobe lists.
    let mut centroid_dists = Vec::<(u32, f32)>::with_capacity(nlist as usize);
    for cid in 0..nlist as usize {
        let c = &centroids[cid * dim as usize..(cid + 1) * dim as usize];
        centroid_dists.push((cid as u32, l2_sq(query, c)));
    }
    let nprobe = nprobe.clamp(1, nlist);
    centroid_dists.select_nth_unstable_by((nprobe - 1) as usize, |a, b| a.1.total_cmp(&b.1));
    centroid_dists.truncate(nprobe as usize);

    let mut heap = std::collections::BinaryHeap::<Pair>::new();
    let k = k.min(out_cap) as usize;
    for (cid, _) in centroid_dists {
        let posting_chunk_id = posting_chunk_ids.get(cid as usize).copied().unwrap_or(0);
        if posting_chunk_id == 0 {
            continue;
        }
        let bytes = match fetch_chunk(posting_chunk_id) {
            Some(b) => b,
            None => continue,
        };
        if bytes.len() < 4 {
            continue;
        }
        let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let vectors_bytes_f32 = count * dim as usize * 4;
        let vectors_bytes_f16 = count * dim as usize * 2;

        match posting_codec {
            0 => {
                // raw: [count:u32][row_ids:u32*count][vectors]
                let row_ids_bytes = 4 + count * 4;
                if bytes.len() != row_ids_bytes + vectors_bytes_f32 {
                    continue;
                }
                let row_ids: &[u32] = std::slice::from_raw_parts(
                    bytes[4..row_ids_bytes].as_ptr() as *const u32,
                    count,
                );
                let vectors: &[f32] = std::slice::from_raw_parts(
                    bytes[row_ids_bytes..].as_ptr() as *const f32,
                    count * dim as usize,
                );
                for (pos, &row_id) in row_ids.iter().enumerate() {
                    let v = &vectors[pos * dim as usize..(pos + 1) * dim as usize];
                    heap_push_topk(&mut heap, k, row_id, l2_sq(query, v));
                }
            }
            1 => {
                // row_id_delta_varint_v1: [count:u32][first:u32][deltas:uleb128*(count-1)][vectors]
                if bytes.len() < 8 {
                    continue;
                }
                let mut offset = 8usize;
                if count > 1 {
                    for _ in 1..count {
                        let Some((_, used)) = decode_uleb128_u32(&bytes[offset..]) else {
                            offset = usize::MAX;
                            break;
                        };
                        offset += used;
                    }
                }
                if offset == usize::MAX {
                    continue;
                }
                if bytes.len() != offset + vectors_bytes_f32 {
                    continue;
                }
                let vectors: &[f32] = std::slice::from_raw_parts(
                    bytes[offset..].as_ptr() as *const f32,
                    count * dim as usize,
                );

                let mut cur_row = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                let mut off2 = 8usize;
                for pos in 0..count {
                    if pos > 0 {
                        let Some((delta, used)) = decode_uleb128_u32(&bytes[off2..]) else {
                            break;
                        };
                        off2 += used;
                        cur_row = cur_row.wrapping_add(delta);
                    }
                    let v = &vectors[pos * dim as usize..(pos + 1) * dim as usize];
                    heap_push_topk(&mut heap, k, cur_row, l2_sq(query, v));
                }
            }
            2 => {
                // raw_f16: [count:u32][row_ids:u32*count][vectors:f16]
                let row_ids_bytes = 4 + count * 4;
                if bytes.len() != row_ids_bytes + vectors_bytes_f16 {
                    continue;
                }
                let row_ids: &[u32] = std::slice::from_raw_parts(
                    bytes[4..row_ids_bytes].as_ptr() as *const u32,
                    count,
                );
                let vectors_bytes = &bytes[row_ids_bytes..];
                for (pos, &row_id) in row_ids.iter().enumerate() {
                    let off = pos * dim as usize * 2;
                    let v = &vectors_bytes[off..off + dim as usize * 2];
                    heap_push_topk(&mut heap, k, row_id, l2_sq_f16(query, v));
                }
            }
            3 => {
                // row_id_delta_varint_v1_f16: [count:u32][first:u32][deltas:uleb128*(count-1)][vectors:f16]
                if bytes.len() < 8 {
                    continue;
                }
                let mut offset = 8usize;
                if count > 1 {
                    for _ in 1..count {
                        let Some((_, used)) = decode_uleb128_u32(&bytes[offset..]) else {
                            offset = usize::MAX;
                            break;
                        };
                        offset += used;
                    }
                }
                if offset == usize::MAX {
                    continue;
                }
                if bytes.len() != offset + vectors_bytes_f16 {
                    continue;
                }
                let vectors_bytes = &bytes[offset..];

                let mut cur_row = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                let mut off2 = 8usize;
                for pos in 0..count {
                    if pos > 0 {
                        let Some((delta, used)) = decode_uleb128_u32(&bytes[off2..]) else {
                            break;
                        };
                        off2 += used;
                        cur_row = cur_row.wrapping_add(delta);
                    }
                    let off = pos * dim as usize * 2;
                    let v = &vectors_bytes[off..off + dim as usize * 2];
                    heap_push_topk(&mut heap, k, cur_row, l2_sq_f16(query, v));
                }
            }
            _ => continue,
        }
    }

    let out_words = std::slice::from_raw_parts_mut(out_ptr as *mut u32, (out_cap as usize) * 2);
    write_out_pairs(out_words, k, heap.into_vec())
}

/// Batch variant.
/// Queries layout: f32[nq * dim]
/// Out layout: (row_id:u32, dist_bits:u32) repeated for each query, with fixed `out_cap`.
/// Counts layout: u32[nq], each is number of valid results (<= out_cap).
#[no_mangle]
pub unsafe extern "C" fn ivf_flat_search_batch_ffi(
    dir_ptr: u32,
    dir_len: u32,
    queries_ptr: u32,
    nq: u32,
    dim: u32,
    k: u32,
    nprobe: u32,
    out_ptr: u32,
    out_cap: u32,
    counts_ptr: u32,
) -> u32 {
    if nq == 0 || k == 0 || out_cap == 0 {
        return 0;
    }
    if dir_ptr == 0 || queries_ptr == 0 || out_ptr == 0 || counts_ptr == 0 {
        return 0;
    }

    let dir = std::slice::from_raw_parts(dir_ptr as *const u8, dir_len as usize);
    if dir.len() < 16 {
        return 0;
    }
    let dir_dim = read_u32_le(dir, 0).unwrap();
    let nlist = read_u32_le(dir, 4).unwrap();
    let centroids_chunk_id = read_u32_le(dir, 8).unwrap();
    let posting_codec = read_u32_le(dir, 12).unwrap();
    if dir_dim != dim || nlist == 0 {
        return 0;
    }
    let needed = 16usize + (nlist as usize) * 4;
    if dir.len() < needed {
        return 0;
    }
    let posting_ids_bytes = &dir[16..16 + (nlist as usize) * 4];
    let posting_chunk_ids: &[u32] = std::slice::from_raw_parts(
        posting_ids_bytes.as_ptr() as *const u32,
        nlist as usize,
    );

    let queries =
        std::slice::from_raw_parts(queries_ptr as *const f32, (nq as usize) * (dim as usize));
    let counts = std::slice::from_raw_parts_mut(counts_ptr as *mut u32, nq as usize);

    let centroids_bytes = match fetch_chunk(centroids_chunk_id) {
        Some(b) => b,
        None => return 0,
    };
    if centroids_bytes.len() != (nlist as usize) * (dim as usize) * 4 {
        return 0;
    }
    let centroids: &[f32] = std::slice::from_raw_parts(
        centroids_bytes.as_ptr() as *const f32,
        (nlist as usize) * (dim as usize),
    );

    let out_words = std::slice::from_raw_parts_mut(
        out_ptr as *mut u32,
        (nq as usize) * (out_cap as usize) * 2,
    );
    let out_cap = out_cap as usize;
    let k = (k.min(out_cap as u32)) as usize;

    let mut decode_ns: u64 = 0;
    let mut compute_ns: u64 = 0;
    let mut decoded_cache_hits: u64 = 0;
    let mut decoded_cache_misses: u64 = 0;

    for q in 0..(nq as usize) {
        let query = &queries[q * dim as usize..(q + 1) * dim as usize];

        let mut centroid_dists = Vec::<(u32, f32)>::with_capacity(nlist as usize);
        for cid in 0..nlist as usize {
            let c = &centroids[cid * dim as usize..(cid + 1) * dim as usize];
            centroid_dists.push((cid as u32, l2_sq(query, c)));
        }
        let nprobe = nprobe.clamp(1, nlist);
        centroid_dists.select_nth_unstable_by((nprobe - 1) as usize, |a, b| a.1.total_cmp(&b.1));
        centroid_dists.truncate(nprobe as usize);

        let mut heap = std::collections::BinaryHeap::<Pair>::new();

        for (cid, _) in &centroid_dists {
            let posting_chunk_id = posting_chunk_ids.get(*cid as usize).copied().unwrap_or(0);
            if posting_chunk_id == 0 {
                continue;
            }

            // For f16 codecs, cache decoded postings to avoid repeated dynamic decoding.
            if posting_codec == 2 || posting_codec == 3 {
                if let Some(posting) = decoded_cache().lock().unwrap().get(posting_chunk_id) {
                    decoded_cache_hits += 1;
                    let start_compute = Instant::now();
                    for (pos, &row_id) in posting.row_ids.iter().enumerate() {
                        let v = &posting.vectors[pos * dim as usize..(pos + 1) * dim as usize];
                        heap_push_topk(&mut heap, k, row_id, l2_sq(query, v));
                    }
                    compute_ns += start_compute.elapsed().as_nanos() as u64;
                    continue;
                }
                decoded_cache_misses += 1;
                let bytes = match fetch_chunk(posting_chunk_id) {
                    Some(b) => b,
                    None => continue,
                };
                let start_decode = Instant::now();
                let decoded = match posting_codec {
                    2 => decode_posting_raw_f16(&bytes, dim as usize),
                    3 => decode_posting_delta_f16(&bytes, dim as usize),
                    _ => None,
                };
                decode_ns += start_decode.elapsed().as_nanos() as u64;
                let Some(decoded) = decoded else { continue; };
                let start_compute = Instant::now();
                for (pos, &row_id) in decoded.row_ids.iter().enumerate() {
                    let v = &decoded.vectors[pos * dim as usize..(pos + 1) * dim as usize];
                    heap_push_topk(&mut heap, k, row_id, l2_sq(query, v));
                }
                compute_ns += start_compute.elapsed().as_nanos() as u64;
                decoded_cache().lock().unwrap().insert(posting_chunk_id, decoded);
                continue;
            }

            let bytes = match fetch_chunk(posting_chunk_id) {
                Some(b) => b,
                None => continue,
            };
            if bytes.len() < 4 {
                continue;
            }
            let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
            let vectors_bytes_f32 = count * dim as usize * 4;
            let vectors_bytes_f16 = count * dim as usize * 2;

            match posting_codec {
                0 => {
                    let row_ids_bytes = 4 + count * 4;
                    if bytes.len() != row_ids_bytes + vectors_bytes_f32 {
                        continue;
                    }
                    let row_ids: &[u32] = std::slice::from_raw_parts(
                        bytes[4..row_ids_bytes].as_ptr() as *const u32,
                        count,
                    );
                    let vectors: &[f32] = std::slice::from_raw_parts(
                        bytes[row_ids_bytes..].as_ptr() as *const f32,
                        count * dim as usize,
                    );
                    let start_compute = Instant::now();
                    for (pos, &row_id) in row_ids.iter().enumerate() {
                        let v = &vectors[pos * dim as usize..(pos + 1) * dim as usize];
                        heap_push_topk(&mut heap, k, row_id, l2_sq(query, v));
                    }
                    compute_ns += start_compute.elapsed().as_nanos() as u64;
                }
                1 => {
                    if bytes.len() < 8 {
                        continue;
                    }
                    let mut offset = 8usize;
                    if count > 1 {
                        for _ in 1..count {
                            let Some((_, used)) = decode_uleb128_u32(&bytes[offset..]) else {
                                offset = usize::MAX;
                                break;
                            };
                            offset += used;
                        }
                    }
                    if offset == usize::MAX {
                        continue;
                    }
                    if bytes.len() != offset + vectors_bytes_f32 {
                        continue;
                    }
                    let vectors: &[f32] = std::slice::from_raw_parts(
                        bytes[offset..].as_ptr() as *const f32,
                        count * dim as usize,
                    );

                    let mut cur_row = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                    let mut off2 = 8usize;
                    let start_compute = Instant::now();
                    for pos in 0..count {
                        if pos > 0 {
                            let Some((delta, used)) = decode_uleb128_u32(&bytes[off2..]) else {
                                break;
                            };
                            off2 += used;
                            cur_row = cur_row.wrapping_add(delta);
                        }
                        let v = &vectors[pos * dim as usize..(pos + 1) * dim as usize];
                        heap_push_topk(&mut heap, k, cur_row, l2_sq(query, v));
                    }
                    compute_ns += start_compute.elapsed().as_nanos() as u64;
                }
                2 => {
                    let row_ids_bytes = 4 + count * 4;
                    if bytes.len() != row_ids_bytes + vectors_bytes_f16 {
                        continue;
                    }
                    let row_ids: &[u32] = std::slice::from_raw_parts(
                        bytes[4..row_ids_bytes].as_ptr() as *const u32,
                        count,
                    );
                    let vectors_bytes = &bytes[row_ids_bytes..];
                    let start_compute = Instant::now();
                    for (pos, &row_id) in row_ids.iter().enumerate() {
                        let off = pos * dim as usize * 2;
                        let v = &vectors_bytes[off..off + dim as usize * 2];
                        heap_push_topk(&mut heap, k, row_id, l2_sq_f16(query, v));
                    }
                    compute_ns += start_compute.elapsed().as_nanos() as u64;
                }
                3 => {
                    if bytes.len() < 8 {
                        continue;
                    }
                    let mut offset = 8usize;
                    if count > 1 {
                        for _ in 1..count {
                            let Some((_, used)) = decode_uleb128_u32(&bytes[offset..]) else {
                                offset = usize::MAX;
                                break;
                            };
                            offset += used;
                        }
                    }
                    if offset == usize::MAX {
                        continue;
                    }
                    if bytes.len() != offset + vectors_bytes_f16 {
                        continue;
                    }
                    let vectors_bytes = &bytes[offset..];

                    let mut cur_row = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                    let mut off2 = 8usize;
                    let start_compute = Instant::now();
                    for pos in 0..count {
                        if pos > 0 {
                            let Some((delta, used)) = decode_uleb128_u32(&bytes[off2..]) else {
                                break;
                            };
                            off2 += used;
                            cur_row = cur_row.wrapping_add(delta);
                        }
                        let off = pos * dim as usize * 2;
                        let v = &vectors_bytes[off..off + dim as usize * 2];
                        heap_push_topk(&mut heap, k, cur_row, l2_sq_f16(query, v));
                    }
                    compute_ns += start_compute.elapsed().as_nanos() as u64;
                }
                _ => continue,
            }
        }

        let base = q * out_cap * 2;
        counts[q] = write_out_pairs(&mut out_words[base..base + out_cap * 2], k, heap.into_vec());
    }

    let cache_bytes = decoded_cache().lock().unwrap().bytes as u64;
    *last_stats_cell().lock().unwrap() = LastStats {
        decode_ns,
        compute_ns,
        decoded_cache_hits,
        decoded_cache_misses,
        decoded_cache_bytes: cache_bytes,
    };

    1
}
