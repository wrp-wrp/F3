use ordered_float::NotNan;
use std::alloc::{alloc, dealloc, Layout};
use core::arch::wasm32::*;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

#[link(wasm_import_module = "env")]
extern "C" {
    fn host_chunk_len(chunk_id: u32) -> u32;
    fn host_read_chunk(chunk_id: u32, dst_ptr: u32, dst_len: u32) -> u32;
    fn host_l2_sq_batch_f32(query_ptr: u32, vectors_ptr: u32, count: u32, dim: u32, out_ptr: u32) -> u32;
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
    centroid_ns: u64,
    decode_ns: u64,
    compute_ns: u64,
    dist_ns: u64,
    heap_ns: u64,
    decoded_cache_hits: u64,
    decoded_cache_misses: u64,
    decoded_cache_bytes: u64,
}

fn last_stats_cell() -> &'static Mutex<LastStats> {
    static STATS: OnceLock<Mutex<LastStats>> = OnceLock::new();
    STATS.get_or_init(|| Mutex::new(LastStats::default()))
}

fn profile_stages_cell() -> &'static Mutex<bool> {
    static PROFILE: OnceLock<Mutex<bool>> = OnceLock::new();
    PROFILE.get_or_init(|| Mutex::new(false))
}

fn profile_stages_enabled() -> bool {
    *profile_stages_cell().lock().unwrap()
}

fn use_host_dist_cell() -> &'static Mutex<bool> {
    static USE: OnceLock<Mutex<bool>> = OnceLock::new();
    USE.get_or_init(|| Mutex::new(false))
}

fn use_host_dist_enabled() -> bool {
    *use_host_dist_cell().lock().unwrap()
}

#[no_mangle]
pub extern "C" fn ivf_set_decoded_cache_budget_ffi(bytes: u32) {
    decoded_cache().lock().unwrap().set_budget(bytes as usize);
}

#[no_mangle]
pub extern "C" fn ivf_set_profile_stages_ffi(enabled: u32) {
    *profile_stages_cell().lock().unwrap() = enabled != 0;
}

#[no_mangle]
pub extern "C" fn ivf_set_use_host_dist_ffi(enabled: u32) {
    *use_host_dist_cell().lock().unwrap() = enabled != 0;
}

#[inline]
fn l2_sq_batch_f32(query: &[f32], vectors: &[f32], dim: usize, out: &mut Vec<f32>) {
    let dim = dim.max(1);
    let count = vectors.len() / dim;
    out.clear();
    out.resize(count, 0.0);
    if use_host_dist_enabled() {
        let ok = unsafe {
            host_l2_sq_batch_f32(
                query.as_ptr() as u32,
                vectors.as_ptr() as u32,
                count as u32,
                dim as u32,
                out.as_mut_ptr() as u32,
            )
        };
        if ok != 0 {
            return;
        }
    }
    for i in 0..count {
        let base = i * dim;
        out[i] = l2_sq(query, &vectors[base..base + dim]);
    }
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

/// Writes 8x u64 to `out_ptr`:
/// - centroid_ns, decode_ns, dist_ns, heap_ns, compute_ns, decoded_cache_hits, decoded_cache_misses, decoded_cache_bytes
#[no_mangle]
pub unsafe extern "C" fn ivf_last_stats_v2_ffi(out_ptr: u32) -> u32 {
    if out_ptr == 0 {
        return 0;
    }
    let s = *last_stats_cell().lock().unwrap();
    let out = std::slice::from_raw_parts_mut(out_ptr as *mut u64, 8);
    out[0] = s.centroid_ns;
    out[1] = s.decode_ns;
    out[2] = s.dist_ns;
    out[3] = s.heap_ns;
    out[4] = s.compute_ns;
    out[5] = s.decoded_cache_hits;
    out[6] = s.decoded_cache_misses;
    out[7] = s.decoded_cache_bytes;
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

fn decode_vectors_f32(bytes: &[u8], count: usize, dim: usize) -> Option<Vec<f32>> {
    let bytes_len = count.checked_mul(dim)?.checked_mul(4)?;
    if bytes.len() < bytes_len {
        return None;
    }
    let mut out = Vec::<f32>::with_capacity(count * dim);
    let mut off = 0usize;
    for _ in 0..(count * dim) {
        let b = bytes.get(off..off + 4)?;
        out.push(f32::from_le_bytes(b.try_into().ok()?));
        off += 4;
    }
    Some(out)
}

fn decode_posting_raw_f32(bytes: &[u8], dim: usize) -> Option<DecodedPosting> {
    if bytes.len() < 4 {
        return None;
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let row_ids_bytes = 4 + count * 4;
    let vectors_bytes = count * dim * 4;
    if bytes.len() != row_ids_bytes + vectors_bytes {
        return None;
    }
    let row_ids = decode_row_ids_raw(&bytes[4..row_ids_bytes], count)?;
    let vectors = decode_vectors_f32(&bytes[row_ids_bytes..], count, dim)?;
    let mem_bytes = row_ids.len() * 4 + vectors.len() * 4;
    Some(DecodedPosting {
        row_ids,
        vectors,
        bytes: mem_bytes,
    })
}

fn decode_posting_delta_f32(bytes: &[u8], dim: usize) -> Option<DecodedPosting> {
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
    let vectors_bytes = count * dim * 4;
    if bytes.len() != offset + vectors_bytes {
        return None;
    }
    let vectors = decode_vectors_f32(&bytes[offset..], count, dim)?;
    let mem_bytes = row_ids.len() * 4 + vectors.len() * 4;
    Some(DecodedPosting {
        row_ids,
        vectors,
        bytes: mem_bytes,
    })
}

#[inline]
fn posting_codec_uses_decoded_cache(posting_codec: u32) -> bool {
    matches!(posting_codec, 0 | 1 | 2 | 3)
}

fn decode_posting_to_f32(bytes: &[u8], posting_codec: u32, dim: usize) -> Option<DecodedPosting> {
    match posting_codec {
        0 => decode_posting_raw_f32(bytes, dim),
        1 => decode_posting_delta_f32(bytes, dim),
        2 => decode_posting_raw_f16(bytes, dim),
        3 => decode_posting_delta_f16(bytes, dim),
        _ => None,
    }
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
    l2_sq_impl(a, b)
}

/// Microbenchmark helper for the distance kernel.
///
/// Computes `iters * count` distances between:
/// - `query_ptr`: a single `f32[dim]`
/// - `vectors_ptr`: `f32[count * dim]` (row-major, contiguous)
///
/// Writes the accumulated sum (to prevent dead-code elimination) to `out_ptr` as `f32`.
#[no_mangle]
pub unsafe extern "C" fn l2_microbench_query_vs_vectors_f32_ffi(
    query_ptr: u32,
    vectors_ptr: u32,
    count: u32,
    dim: u32,
    iters: u32,
    out_ptr: u32,
) -> u32 {
    if query_ptr == 0 || vectors_ptr == 0 {
        return 0;
    }
    let dim = dim as usize;
    let count = count as usize;
    if dim == 0 || count == 0 {
        return 0;
    }
    let Some(vectors_len) = count.checked_mul(dim) else {
        return 0;
    };

    let query = std::slice::from_raw_parts(query_ptr as *const f32, dim);
    let vectors = std::slice::from_raw_parts(vectors_ptr as *const f32, vectors_len);

    let mut acc = 0.0f32;
    for _ in 0..(iters as usize) {
        for i in 0..count {
            let base = i * dim;
            let v = &vectors[base..base + dim];
            acc += l2_sq(query, v);
        }
    }
    std::hint::black_box(acc);

    if out_ptr != 0 {
        *(out_ptr as *mut f32) = acc;
    }
    1
}

#[cfg(not(target_feature = "simd128"))]
#[inline]
fn l2_sq_impl(a: &[f32], b: &[f32]) -> f32 {
    l2_sq_scalar(a, b)
}

#[cfg(target_feature = "simd128")]
#[inline]
fn l2_sq_impl(a: &[f32], b: &[f32]) -> f32 {
    unsafe { l2_sq_simd128(a, b) }
}

#[cfg(not(target_feature = "simd128"))]
#[inline]
fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[cfg(target_feature = "simd128")]
#[inline]
unsafe fn l2_sq_simd128(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;
    let n = a.len().min(b.len());
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    // Unroll to reduce loop overhead and expose ILP (similar to native NEON kernel).
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut acc2 = f32x4_splat(0.0);
    let mut acc3 = f32x4_splat(0.0);

    let mut i = 0usize;
    while i + 16 <= n {
        let a0 = v128_load(ap.add(i) as *const v128);
        let b0 = v128_load(bp.add(i) as *const v128);
        let d0 = f32x4_sub(a0, b0);
        acc0 = f32x4_add(acc0, f32x4_mul(d0, d0));

        let a1v = v128_load(ap.add(i + 4) as *const v128);
        let b1v = v128_load(bp.add(i + 4) as *const v128);
        let d1 = f32x4_sub(a1v, b1v);
        acc1 = f32x4_add(acc1, f32x4_mul(d1, d1));

        let a2v = v128_load(ap.add(i + 8) as *const v128);
        let b2v = v128_load(bp.add(i + 8) as *const v128);
        let d2 = f32x4_sub(a2v, b2v);
        acc2 = f32x4_add(acc2, f32x4_mul(d2, d2));

        let a3v = v128_load(ap.add(i + 12) as *const v128);
        let b3v = v128_load(bp.add(i + 12) as *const v128);
        let d3 = f32x4_sub(a3v, b3v);
        acc3 = f32x4_add(acc3, f32x4_mul(d3, d3));

        i += 16;
    }

    let mut acc = f32x4_add(f32x4_add(acc0, acc1), f32x4_add(acc2, acc3));
    while i + 4 <= n {
        let va = v128_load(ap.add(i) as *const v128);
        let vb = v128_load(bp.add(i) as *const v128);
        let d = f32x4_sub(va, vb);
        acc = f32x4_add(acc, f32x4_mul(d, d));
        i += 4;
    }

    // Horizontal add 4 lanes.
    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    while i < n {
        let d = *ap.add(i) - *bp.add(i);
        sum += d * d;
        i += 1;
    }
    sum
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
                let row_ids_bytes = 4 + count * 4;
                if bytes.len() != row_ids_bytes + vectors_bytes_f32 {
                    continue;
                }
                let vectors: &[f32] = std::slice::from_raw_parts(
                    bytes[row_ids_bytes..].as_ptr() as *const f32,
                    count * dim as usize,
                );
                for (pos, _) in (0..count).enumerate() {
                    let mut row_id_buf = [0u8; 4];
                    row_id_buf.copy_from_slice(&bytes[4 + pos * 4..4 + pos * 4 + 4]);
                    let row_id = u32::from_le_bytes(row_id_buf);

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
                let row_ids_bytes = 4 + count * 4;
                if bytes.len() != row_ids_bytes + vectors_bytes_f16 {
                    continue;
                }
                let vectors_bytes = &bytes[row_ids_bytes..];
                for (pos, _) in (0..count).enumerate() {
                    let mut row_id_buf = [0u8; 4];
                    row_id_buf.copy_from_slice(&bytes[4 + pos * 4..4 + pos * 4 + 4]);
                    let row_id = u32::from_le_bytes(row_id_buf);

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
            4 => {
                // RawU8: [count:u32][scale:f32][zero_point:f32][row_ids:u32*count][vectors:u8]
                let row_ids_bytes = 12 + count * 4;
                let vectors_bytes_u8 = count * dim as usize;
                
                if bytes.len() != row_ids_bytes + vectors_bytes_u8 {
                    continue;
                }
                let scale = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
                let zero_point = f32::from_le_bytes(bytes[8..12].try_into().unwrap());

                let q_vectors = &bytes[row_ids_bytes..];

                // Dequantize on the fly
                for (pos, _) in (0..count).enumerate() {
                    let mut row_id_buf = [0u8; 4];
                    row_id_buf.copy_from_slice(&bytes[12 + pos * 4..12 + pos * 4 + 4]);
                    let row_id = u32::from_le_bytes(row_id_buf);

                    let base = pos * dim as usize;
                    let q_vec = &q_vectors[base..base + dim as usize];
                    // Simple scalar dequant loop.
                    let mut sum = 0.0f32;
                    for j in 0..dim as usize {
                        let val = (q_vec[j] as f32 - zero_point) / scale;
                        let d = query[j] - val;
                        sum += d * d;
                    }
                    heap_push_topk(&mut heap, k, row_id, sum);
                }
            }
            5 => {
                // RowIdDeltaVarintV1U8: [count:u32][scale:f32][zero_point:f32][first:u32][deltas...][vectors:u8]
                if bytes.len() < 16 {
                    continue;
                }
                let scale = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
                let zero_point = f32::from_le_bytes(bytes[8..12].try_into().unwrap());

                let mut offset = 16usize;
                if count > 1 {
                    for _ in 1..count {
                        let Some((_, used)) = decode_uleb128_u32(&bytes[offset..]) else {
                            offset = usize::MAX;
                            break;
                        };
                        offset += used;
                    }
                }
                let vectors_bytes_u8 = count * dim as usize;
                if offset == usize::MAX || bytes.len() != offset + vectors_bytes_u8 {
                    continue;
                }
                let q_vectors = &bytes[offset..];

                let mut cur_row = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
                let mut off2 = 16usize;

                for pos in 0..count {
                    if pos > 0 {
                        let Some((delta, used)) = decode_uleb128_u32(&bytes[off2..]) else {
                            break;
                        };
                        off2 += used;
                        cur_row = cur_row.wrapping_add(delta);
                    }
                    let base = pos * dim as usize;
                    let q_vec = &q_vectors[base..base + dim as usize];
                    // Simple scalar dequant loop.
                    let mut sum = 0.0f32;
                    for j in 0..dim as usize {
                        let val = (q_vec[j] as f32 - zero_point) / scale;
                        let d = query[j] - val;
                        sum += d * d;
                    }
                    heap_push_topk(&mut heap, k, cur_row, sum);
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
    
    let mut codebooks_chunk_id = 0;
    let posting_ids_bytes;
    
    if posting_codec == 6 {
        // IvfPq: Extra u32 for codebooks_chunk_id
        if dir.len() < 20 { return 0; }
        codebooks_chunk_id = read_u32_le(dir, 16).unwrap();
        let needed = 20usize + (nlist as usize) * 4;
        if dir.len() < needed { return 0; }
        posting_ids_bytes = &dir[20..20 + (nlist as usize) * 4];
    } else {
        let needed = 16usize + (nlist as usize) * 4;
        if dir.len() < needed {
            return 0;
        }
        posting_ids_bytes = &dir[16..16 + (nlist as usize) * 4];
    }

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

    let profile = profile_stages_enabled();
    let mut decode_ns: u64 = 0;
    let mut compute_ns: u64 = 0;
    let mut centroid_ns: u64 = 0;
    let mut dist_ns: u64 = 0;
    let mut heap_ns: u64 = 0;
    let mut decoded_cache_hits: u64 = 0;
    let mut decoded_cache_misses: u64 = 0;

    for q in 0..(nq as usize) {
        let query = &queries[q * dim as usize..(q + 1) * dim as usize];

        let mut scratch_dists = Vec::<f32>::new();
        let mut scratch_row_ids = Vec::<u32>::new();

        let start_centroids = if profile { Some(Instant::now()) } else { None };
        let mut centroid_dists = Vec::<(u32, f32)>::with_capacity(nlist as usize);
        for cid in 0..nlist as usize {
            let c = &centroids[cid * dim as usize..(cid + 1) * dim as usize];
            centroid_dists.push((cid as u32, l2_sq(query, c)));
        }
        let nprobe = nprobe.clamp(1, nlist);
        centroid_dists.select_nth_unstable_by((nprobe - 1) as usize, |a, b| a.1.total_cmp(&b.1));
        centroid_dists.truncate(nprobe as usize);
        if let Some(start) = start_centroids {
            centroid_ns += start.elapsed().as_nanos() as u64;
        }

        let mut heap = std::collections::BinaryHeap::<Pair>::new();

	        for (cid, _) in &centroid_dists {
	            let posting_chunk_id = posting_chunk_ids.get(*cid as usize).copied().unwrap_or(0);
	            if posting_chunk_id == 0 {
	                continue;
	            }

	            // Logic dispatch based on codec
	            if posting_codec == 6 {
	                let posting_bytes = match fetch_chunk(posting_chunk_id) {
	                    Some(b) => b,
	                    None => continue,
	                };
	                // IvfPq Search
	                // 1. Load Codebooks (lazy load if not present?)
	                // We should load codebooks once outside loop.
	                // But loop is over lists.
                // Codebooks are global.
                // We can cache them in a static? Or just load every time (host cache makes it fast).
                // Let's load inside loop but cache it? No, load once per batch call is better.
                // BUT `codebooks_chunk_id` was read outside.
                // Let's defer loading to first use or load before q loop.
                // Re-fetch from host is cheap (Arc copy).
                // Parsing codebooks is cheap (cast to slice).
                // But parsing needs to happen.
                // Codebooks layout: raw flat f32.
                // We need to know `m` and `d_sub`.
                // `m` = codes_len / count? No.
                // `dim` is given.
                // `d_sub` = dim / m.
                // We don't know `m` explicitly from dir?
                // We must infer `m` from Codebooks size?
                // Codebooks size = m * 256 * d_sub * 4.
                // m * 256 * (dim / m) * 4 = 256 * dim * 4.
                // So size is constant regardless of `m`!
                // Wait. `256 * dim * 4` bytes.
                // So we cannot infer `m`.
                // We need `m`.
                // `ChunkDesc` or `Footer` has `num_subspaces`.
                // Kernel doesn't see Footer.
                // Kernel assumes `m`. Typically `m` is part of configuration or encoded in posting list?
                // Posting list has `count`, then `row_ids`.
                // Then `codes`. `codes.len() = count * m`.
                // So `m = codes.len() / count`.
                // We can infer `m` from the posting list!
                // Iterate over `centroid_dists`:
                
                let codebooks_bytes = match fetch_chunk(codebooks_chunk_id) {
                    Some(b) => b,
                    None => continue,
                };
                let codebooks: &[f32] = bytemuck::cast_slice(&codebooks_bytes);
                // Validation? 
                
	                // Parse Posting List
	                let bytes = &posting_bytes;
	                if bytes.len() < 4 { continue; }
	                let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
	                if count == 0 { continue; }
                
                // Decode Row IDs
                let mut row_ids = Vec::with_capacity(count);
                let first_row_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                let mut codes_offset = 8;
                row_ids.push(first_row_id);
                
                let mut cur = first_row_id;
                for _ in 1..count {
                     if let Some((delta, used)) = decode_uleb128_u32(&bytes[codes_offset..]) {
                         codes_offset += used;
                         cur = cur.wrapping_add(delta);
                         row_ids.push(cur);
                     } else {
                         break;
                     }
                }
                
                if row_ids.len() != count { continue; }
                
                let codes = &bytes[codes_offset..];
                if codes.len() % count != 0 { continue; }
                let m = codes.len() / count;
                let d_sub = dim as usize / m;
                
                if codebooks.len() != 256 * dim as usize {
                     // Error or continue
                     continue; 
                }
                
                // Precompute LUT for this (Query - Centroid)
                // Centroid C is corresponding to `cid`.
                // We already have `cid`.
                let centroid_vec = &centroids[(*cid as usize) * (dim as usize)..((*cid as usize)+1) * (dim as usize)];
                
                // q_res = query - centroid
                // We can compute LUT directly without explicit q_res allocation.
                // LUT[sub][code] = || (q[sub] - C[sub]) - Codebook[sub][code] ||^2
                //                = || q[sub] - (C[sub] + Codebook[sub][code]) ||^2
                
                let mut lut = vec![0.0f32; m * 256];
                for sub in 0..m {
                     let q_sub_start = sub * d_sub;
                     let q_sub = &query[q_sub_start..q_sub_start + d_sub];
                     let c_sub = &centroid_vec[q_sub_start..q_sub_start + d_sub];
                     
                     // Subspace codebook start
                     // Codebooks stored as: [sub0_c0, sub0_c1 ... sub0_c255, sub1_c0 ...]
                     // i.e. Blocked by subspace.
                     // (Verified by `train_pq_codebooks` pushing centroids sequentially per subspace)
                     let cb_sub_start = sub * 256 * d_sub;
                     
                     for code in 0..256 {
                         let cb_vec = &codebooks[cb_sub_start + code * d_sub .. cb_sub_start + (code+1) * d_sub];
                         let mut d2 = 0.0f32;
                         for k in 0..d_sub {
                             let diff = q_sub[k] - (c_sub[k] + cb_vec[k]);
                             d2 += diff * diff;
                         }
                         lut[sub * 256 + code] = d2;
                     }
                }
                
                // Scan codes
                for i in 0..count {
                    let mut dist = 0.0f32;
                    let vec_codes = &codes[i * m .. (i+1) * m];
                    for sub in 0..m {
                        let code = vec_codes[sub] as usize;
                        dist += lut[sub * 256 + code];
                    }
                    heap_push_topk(&mut heap, k, row_ids[i], dist);
                }
                
                continue;
            }

            // Cache decoded postings inside Wasm to avoid repeated host->wasm copies for hot chunks.
            // This is critical for RQ1: boundary/copy should be amortizable in steady state.
            if posting_codec_uses_decoded_cache(posting_codec) {
                if let Some(posting) = decoded_cache().lock().unwrap().get(posting_chunk_id) {
                    decoded_cache_hits += 1;
                    if profile {
                        scratch_dists.clear();
                        let start_dist = Instant::now();
                        l2_sq_batch_f32(query, &posting.vectors, dim as usize, &mut scratch_dists);
                        dist_ns += start_dist.elapsed().as_nanos() as u64;

                        let start_heap = Instant::now();
                        for (pos, &row_id) in posting.row_ids.iter().enumerate() {
                            heap_push_topk(&mut heap, k, row_id, scratch_dists[pos]);
                        }
                        heap_ns += start_heap.elapsed().as_nanos() as u64;
                    } else {
                        let start_compute = Instant::now();
                        scratch_dists.clear();
                        l2_sq_batch_f32(query, &posting.vectors, dim as usize, &mut scratch_dists);
                        for (pos, &row_id) in posting.row_ids.iter().enumerate() {
                            heap_push_topk(&mut heap, k, row_id, scratch_dists[pos]);
                        }
                        compute_ns += start_compute.elapsed().as_nanos() as u64;
                    }
                    continue;
                }
                
                decoded_cache_misses += 1;
                let bytes = match fetch_chunk(posting_chunk_id) {
                    Some(b) => b,
                    None => continue,
                };
                let start_decode = Instant::now();
                let decoded = decode_posting_to_f32(&bytes, posting_codec, dim as usize);
                decode_ns += start_decode.elapsed().as_nanos() as u64;
                let Some(decoded) = decoded else { continue; };
                if profile {
                    scratch_dists.clear();
                    let start_dist = Instant::now();
                    l2_sq_batch_f32(query, &decoded.vectors, dim as usize, &mut scratch_dists);
                    dist_ns += start_dist.elapsed().as_nanos() as u64;

                    let start_heap = Instant::now();
                    for (pos, &row_id) in decoded.row_ids.iter().enumerate() {
                        heap_push_topk(&mut heap, k, row_id, scratch_dists[pos]);
                    }
                    heap_ns += start_heap.elapsed().as_nanos() as u64;
                } else {
                    let start_compute = Instant::now();
                    scratch_dists.clear();
                    l2_sq_batch_f32(query, &decoded.vectors, dim as usize, &mut scratch_dists);
                    for (pos, &row_id) in decoded.row_ids.iter().enumerate() {
                        heap_push_topk(&mut heap, k, row_id, scratch_dists[pos]);
                    }
                    compute_ns += start_compute.elapsed().as_nanos() as u64;
                }
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
                    let vectors: &[f32] = std::slice::from_raw_parts(
                        bytes[row_ids_bytes..].as_ptr() as *const f32,
                        count * dim as usize,
                    );
                    if profile {
                        scratch_dists.clear();
                        scratch_dists.reserve(count);
                        let start_dist = Instant::now();
                        for pos in 0..count {
                            let v = &vectors[pos * dim as usize..(pos + 1) * dim as usize];
                            scratch_dists.push(l2_sq(query, v));
                        }
                        dist_ns += start_dist.elapsed().as_nanos() as u64;

                        let start_heap = Instant::now();
                        for (pos, _) in (0..count).enumerate() {
                            let mut row_id_buf = [0u8; 4];
                            row_id_buf.copy_from_slice(&bytes[4 + pos * 4..4 + pos * 4 + 4]);
                            let row_id = u32::from_le_bytes(row_id_buf);
                            heap_push_topk(&mut heap, k, row_id, scratch_dists[pos]);
                        }
                        heap_ns += start_heap.elapsed().as_nanos() as u64;
                    } else {
                        for (pos, _) in (0..count).enumerate() {
                            let mut row_id_buf = [0u8; 4];
                            row_id_buf.copy_from_slice(&bytes[4 + pos * 4..4 + pos * 4 + 4]);
                            let row_id = u32::from_le_bytes(row_id_buf);

                            let v = &vectors[pos * dim as usize..(pos + 1) * dim as usize];
                            heap_push_topk(&mut heap, k, row_id, l2_sq(query, v));
                        }
                        // compute_ns += start_compute.elapsed().as_nanos() as u64; -- removed Instant::now
                    }
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

                    if profile {
                        scratch_row_ids.clear();
                        scratch_row_ids.reserve(count);
                        let start_decode = Instant::now();
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
                            scratch_row_ids.push(cur_row);
                        }
                        decode_ns += start_decode.elapsed().as_nanos() as u64;

                        scratch_dists.clear();
                        scratch_dists.reserve(scratch_row_ids.len());
                        let start_dist = Instant::now();
                        for pos in 0..scratch_row_ids.len() {
                            let v = &vectors[pos * dim as usize..(pos + 1) * dim as usize];
                            scratch_dists.push(l2_sq(query, v));
                        }
                        dist_ns += start_dist.elapsed().as_nanos() as u64;

                        let start_heap = Instant::now();
                        for (pos, &row_id) in scratch_row_ids.iter().enumerate() {
                            heap_push_topk(&mut heap, k, row_id, scratch_dists[pos]);
                        }
                        heap_ns += start_heap.elapsed().as_nanos() as u64;
                    } else {
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
                    }
                4 => {
                    // RawU8: [count:u32][scale:f32][zero_point:f32][row_ids:u32*count][vectors:u8]
                    process_raw_u8(
                        &bytes,
                        count,
                        dim as usize,
                        &query,
                        k as usize,
                        &mut heap,
                        profile,
                        &mut scratch_dists,
                        &mut dist_ns,
                        &mut heap_ns,
                        &mut compute_ns,
                    );
                }
                2 => {
                    let row_ids_bytes = 4 + count * 4;
                    if bytes.len() != row_ids_bytes + vectors_bytes_f16 {
                        continue;
                    }
                    let vectors_bytes = &bytes[row_ids_bytes..];
                    if profile {
                        scratch_dists.clear();
                        scratch_dists.reserve(count);
                        let start_dist = Instant::now();
                        for pos in 0..count {
                            let off = pos * dim as usize * 2;
                            let v = &vectors_bytes[off..off + dim as usize * 2];
                            scratch_dists.push(l2_sq_f16(query, v));
                        }
                        dist_ns += start_dist.elapsed().as_nanos() as u64;

                        let start_heap = Instant::now();
                        for (pos, _) in (0..count).enumerate() {
                            let mut row_id_buf = [0u8; 4];
                            row_id_buf.copy_from_slice(&bytes[4 + pos * 4..4 + pos * 4 + 4]);
                            let row_id = u32::from_le_bytes(row_id_buf);
                            heap_push_topk(&mut heap, k, row_id, scratch_dists[pos]);
                        }
                        heap_ns += start_heap.elapsed().as_nanos() as u64;
                    } else {
                        for (pos, _) in (0..count).enumerate() {
                            let mut row_id_buf = [0u8; 4];
                            row_id_buf.copy_from_slice(&bytes[4 + pos * 4..4 + pos * 4 + 4]);
                            let row_id = u32::from_le_bytes(row_id_buf);

                            let off = pos * dim as usize * 2;
                            let v = &vectors_bytes[off..off + dim as usize * 2];
                            heap_push_topk(&mut heap, k, row_id, l2_sq_f16(query, v));
                        }
                        // compute_ns += start_compute.elapsed().as_nanos() as u64;
                    }
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
                    if profile {
                        scratch_row_ids.clear();
                        scratch_row_ids.reserve(count);
                        scratch_dists.clear();
                        scratch_dists.reserve(count);
                        let start_decode = Instant::now();
                        for pos in 0..count {
                            if pos > 0 {
                                let Some((delta, used)) = decode_uleb128_u32(&bytes[off2..]) else {
                                    break;
                                };
                                off2 += used;
                                cur_row = cur_row.wrapping_add(delta);
                            }
                            scratch_row_ids.push(cur_row);
                        }
                        decode_ns += start_decode.elapsed().as_nanos() as u64;

                        let start_dist = Instant::now();
                        for pos in 0..scratch_row_ids.len() {
                            let off = pos * dim as usize * 2;
                            let v = &vectors_bytes[off..off + dim as usize * 2];
                            scratch_dists.push(l2_sq_f16(query, v));
                        }
                        dist_ns += start_dist.elapsed().as_nanos() as u64;

                        let start_heap = Instant::now();
                        for (pos, &row_id) in scratch_row_ids.iter().enumerate() {
                            heap_push_topk(&mut heap, k, row_id, scratch_dists[pos]);
                        }
                        heap_ns += start_heap.elapsed().as_nanos() as u64;
                    } else {
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
                }
                _ => continue,
            }
        }

        let base = q * out_cap * 2;
        counts[q] = write_out_pairs(&mut out_words[base..base + out_cap * 2], k, heap.into_vec());
    }

    let cache_bytes = decoded_cache().lock().unwrap().bytes as u64;
    if profile {
        compute_ns = dist_ns.saturating_add(heap_ns);
    }
    *last_stats_cell().lock().unwrap() = LastStats {
        centroid_ns,
        decode_ns,
        compute_ns,
        dist_ns,
        heap_ns,
        decoded_cache_hits,
        decoded_cache_misses,
        decoded_cache_bytes: cache_bytes,
    };

    1
}

#[inline(always)]
fn process_raw_u8(
    bytes: &[u8],
    count: usize,
    dim: usize,
    query: &[f32],
    k: usize,
    mut heap: &mut std::collections::BinaryHeap<Pair>,
    profile: bool,
    scratch_dists: &mut Vec<f32>,
    dist_ns: &mut u64,
    heap_ns: &mut u64,
    compute_ns: &mut u64,
) {
    let row_ids_bytes = 12 + count * 4;
    let vectors_bytes_u8 = count * dim as usize;
    if bytes.len() != row_ids_bytes + vectors_bytes_u8 {
        return;
    }
    let scale = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let zero_point = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let vectors_bytes = &bytes[row_ids_bytes..];

    // Precompute Transform Phase 0: T = S*q + Z, C = Sum(T^2)
    let mut transformed_query = vec![0.0f32; dim];
    let mut sum_t_sq = 0.0f32;
    for i in 0..dim {
        let t = scale * query[i] + zero_point;
        transformed_query[i] = t;
        sum_t_sq += t * t;
    }
    let inv_scale_sq = 1.0 / (scale * scale);

    if profile {
        scratch_dists.clear();
        scratch_dists.reserve(count);
        // let start_dist = Instant::now();
        for pos in 0..count {
            let off = pos * dim as usize;
            let q_vec = &vectors_bytes[off..off + dim as usize];
            
            // SIMD Optimization
            let dist = unsafe { l2_sq_u8_simd_fast(&transformed_query, q_vec, dim, inv_scale_sq, sum_t_sq) };
            scratch_dists.push(dist);
        }
        // *dist_ns += start_dist.elapsed().as_nanos() as u64;

        // let start_heap = Instant::now();
        for (pos, _) in (0..count).enumerate() {
            let mut row_id_buf = [0u8; 4];
            row_id_buf.copy_from_slice(&bytes[12 + pos * 4..12 + pos * 4 + 4]);
            let row_id = u32::from_le_bytes(row_id_buf);
            heap_push_topk(&mut heap, k, row_id, scratch_dists[pos]);
        }
        // *heap_ns += start_heap.elapsed().as_nanos() as u64;
    } else {
        // let start_compute = Instant::now();
        for (pos, _) in (0..count).enumerate() {
            let mut row_id_buf = [0u8; 4];
            row_id_buf.copy_from_slice(&bytes[12 + pos * 4..12 + pos * 4 + 4]);
            let row_id = u32::from_le_bytes(row_id_buf);

            let off = pos * dim as usize;
            let q_vec = &vectors_bytes[off..off + dim as usize];
            
            // SIMD Optimization
            let dist = unsafe { l2_sq_u8_simd_fast(&transformed_query, q_vec, dim, inv_scale_sq, sum_t_sq) };
            heap_push_topk(&mut heap, k, row_id, dist);
        }
        // *compute_ns += start_compute.elapsed().as_nanos() as u64;
    }
}

#[target_feature(enable = "simd128")]
unsafe fn l2_sq_u8_simd_fast(
    transformed_query: &[f32], // T = S*q + Z
    vector: &[u8],
    dim: usize,
    inv_scale_sq: f32, // 1/S^2
    sum_t_sq: f32,     // Sum(T^2)
) -> f32 {
    let mut i = 0;
    
    let mut sum_r_sq_i32_0 = i32x4_splat(0);
    let mut sum_r_sq_i32_1 = i32x4_splat(0);
    let mut sum_r_sq_i32_2 = i32x4_splat(0);
    let mut sum_r_sq_i32_3 = i32x4_splat(0);

    let mut sum_tr_0 = f32x4_splat(0.0);
    let mut sum_tr_1 = f32x4_splat(0.0);
    let mut sum_tr_2 = f32x4_splat(0.0);
    let mut sum_tr_3 = f32x4_splat(0.0);

    let v_ptr = vector.as_ptr();
    let t_ptr = transformed_query.as_ptr();

    while i + 64 <= dim {
        // Block 0
        {
            let v_u8 = v128_load(v_ptr.add(i) as *const v128);
            let v_lo_16 = i16x8_extend_low_u8x16(v_u8);
            let v_hi_16 = i16x8_extend_high_u8x16(v_u8);
            let sq_lo = i32x4_dot_i16x8(v_lo_16, v_lo_16);
            let sq_hi = i32x4_dot_i16x8(v_hi_16, v_hi_16);
            sum_r_sq_i32_0 = i32x4_add(sum_r_sq_i32_0, i32x4_add(sq_lo, sq_hi));

            let v_f0 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(v_lo_16));
            let v_f1 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(v_lo_16));
            let v_f2 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(v_hi_16));
            let v_f3 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(v_hi_16));

            let t0 = v128_load(t_ptr.add(i) as *const v128);
            let t1 = v128_load(t_ptr.add(i+4) as *const v128);
            let t2 = v128_load(t_ptr.add(i+8) as *const v128);
            let t3 = v128_load(t_ptr.add(i+12) as *const v128);

            sum_tr_0 = f32x4_add(sum_tr_0, f32x4_mul(t0, v_f0));
            sum_tr_0 = f32x4_add(sum_tr_0, f32x4_mul(t1, v_f1));
            sum_tr_0 = f32x4_add(sum_tr_0, f32x4_mul(t2, v_f2));
            sum_tr_0 = f32x4_add(sum_tr_0, f32x4_mul(t3, v_f3));
        }

        // Block 1
        {
            let v_u8 = v128_load(v_ptr.add(i+16) as *const v128);
            let v_lo_16 = i16x8_extend_low_u8x16(v_u8);
            let v_hi_16 = i16x8_extend_high_u8x16(v_u8);
            let sq_lo = i32x4_dot_i16x8(v_lo_16, v_lo_16);
            let sq_hi = i32x4_dot_i16x8(v_hi_16, v_hi_16);
            sum_r_sq_i32_1 = i32x4_add(sum_r_sq_i32_1, i32x4_add(sq_lo, sq_hi));

            let v_f0 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(v_lo_16));
            let v_f1 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(v_lo_16));
            let v_f2 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(v_hi_16));
            let v_f3 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(v_hi_16));

            let t0 = v128_load(t_ptr.add(i+16) as *const v128);
            let t1 = v128_load(t_ptr.add(i+20) as *const v128);
            let t2 = v128_load(t_ptr.add(i+24) as *const v128);
            let t3 = v128_load(t_ptr.add(i+28) as *const v128);

            sum_tr_1 = f32x4_add(sum_tr_1, f32x4_mul(t0, v_f0));
            sum_tr_1 = f32x4_add(sum_tr_1, f32x4_mul(t1, v_f1));
            sum_tr_1 = f32x4_add(sum_tr_1, f32x4_mul(t2, v_f2));
            sum_tr_1 = f32x4_add(sum_tr_1, f32x4_mul(t3, v_f3));
        }

        // Block 2
        {
            let v_u8 = v128_load(v_ptr.add(i+32) as *const v128);
            let v_lo_16 = i16x8_extend_low_u8x16(v_u8);
            let v_hi_16 = i16x8_extend_high_u8x16(v_u8);
            let sq_lo = i32x4_dot_i16x8(v_lo_16, v_lo_16);
            let sq_hi = i32x4_dot_i16x8(v_hi_16, v_hi_16);
            sum_r_sq_i32_2 = i32x4_add(sum_r_sq_i32_2, i32x4_add(sq_lo, sq_hi));

            let v_f0 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(v_lo_16));
            let v_f1 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(v_lo_16));
            let v_f2 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(v_hi_16));
            let v_f3 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(v_hi_16));

            let t0 = v128_load(t_ptr.add(i+32) as *const v128);
            let t1 = v128_load(t_ptr.add(i+36) as *const v128);
            let t2 = v128_load(t_ptr.add(i+40) as *const v128);
            let t3 = v128_load(t_ptr.add(i+44) as *const v128);

            sum_tr_2 = f32x4_add(sum_tr_2, f32x4_mul(t0, v_f0));
            sum_tr_2 = f32x4_add(sum_tr_2, f32x4_mul(t1, v_f1));
            sum_tr_2 = f32x4_add(sum_tr_2, f32x4_mul(t2, v_f2));
            sum_tr_2 = f32x4_add(sum_tr_2, f32x4_mul(t3, v_f3));
        }

        // Block 3
        {
            let v_u8 = v128_load(v_ptr.add(i+48) as *const v128);
            let v_lo_16 = i16x8_extend_low_u8x16(v_u8);
            let v_hi_16 = i16x8_extend_high_u8x16(v_u8);
            let sq_lo = i32x4_dot_i16x8(v_lo_16, v_lo_16);
            let sq_hi = i32x4_dot_i16x8(v_hi_16, v_hi_16);
            sum_r_sq_i32_3 = i32x4_add(sum_r_sq_i32_3, i32x4_add(sq_lo, sq_hi));

            let v_f0 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(v_lo_16));
            let v_f1 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(v_lo_16));
            let v_f2 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(v_hi_16));
            let v_f3 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(v_hi_16));

            let t0 = v128_load(t_ptr.add(i+48) as *const v128);
            let t1 = v128_load(t_ptr.add(i+52) as *const v128);
            let t2 = v128_load(t_ptr.add(i+56) as *const v128);
            let t3 = v128_load(t_ptr.add(i+60) as *const v128);

            sum_tr_3 = f32x4_add(sum_tr_3, f32x4_mul(t0, v_f0));
            sum_tr_3 = f32x4_add(sum_tr_3, f32x4_mul(t1, v_f1));
            sum_tr_3 = f32x4_add(sum_tr_3, f32x4_mul(t2, v_f2));
            sum_tr_3 = f32x4_add(sum_tr_3, f32x4_mul(t3, v_f3));
        }

        i += 64;
    }

    // Reduce accumulators
    let sum_r_sq_i32 = i32x4_add(i32x4_add(sum_r_sq_i32_0, sum_r_sq_i32_1), i32x4_add(sum_r_sq_i32_2, sum_r_sq_i32_3));
    let sum_tr = f32x4_add(f32x4_add(sum_tr_0, sum_tr_1), f32x4_add(sum_tr_2, sum_tr_3));


    // Reduction
    // Sum(r^2)
    let r_sq_sum = (i32x4_extract_lane::<0>(sum_r_sq_i32) as f32) +
                   (i32x4_extract_lane::<1>(sum_r_sq_i32) as f32) +
                   (i32x4_extract_lane::<2>(sum_r_sq_i32) as f32) +
                   (i32x4_extract_lane::<3>(sum_r_sq_i32) as f32);
    
    // Sum(T*r)
    let tr_sum = f32x4_extract_lane::<0>(sum_tr) +
                 f32x4_extract_lane::<1>(sum_tr) +
                 f32x4_extract_lane::<2>(sum_tr) +
                 f32x4_extract_lane::<3>(sum_tr);

    let mut final_r_sq = r_sq_sum;
    let mut final_tr = tr_sum;

    // Tail
    while i < dim {
        let r_val = *v_ptr.add(i) as f32;
        let t_val = *t_ptr.add(i);
        final_r_sq += r_val * r_val;
        final_tr += t_val * r_val;
        i += 1;
    }

    // Dist = (1/S^2) * (Sum(T^2) - 2*Sum(T*r) + Sum(r^2))
    // Dist = (1/S^2) * (C - 2B + A)
    (sum_t_sq - 2.0 * final_tr + final_r_sq) * inv_scale_sq
}
