use ordered_float::NotNan;
use std::alloc::{alloc, dealloc, Layout};
use std::cmp::Ordering;

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
        let vectors_bytes = count * dim as usize * 4;

        match posting_codec {
            0 => {
                // raw: [count:u32][row_ids:u32*count][vectors]
                let row_ids_bytes = 4 + count * 4;
                if bytes.len() != row_ids_bytes + vectors_bytes {
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
                    let dist = l2_sq(query, v);
                    let Ok(dist_nn) = NotNan::new(dist) else {
                        continue;
                    };
                    if heap.len() < k {
                        heap.push(Pair { row_id, dist: dist_nn });
                    } else if let Some(worst) = heap.peek() {
                        if dist_nn < worst.dist {
                            let _ = heap.pop();
                            heap.push(Pair { row_id, dist: dist_nn });
                        }
                    }
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
                if bytes.len() != offset + vectors_bytes {
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
                    let dist = l2_sq(query, v);
                    let Ok(dist_nn) = NotNan::new(dist) else {
                        continue;
                    };
                    if heap.len() < k {
                        heap.push(Pair {
                            row_id: cur_row,
                            dist: dist_nn,
                        });
                    } else if let Some(worst) = heap.peek() {
                        if dist_nn < worst.dist {
                            let _ = heap.pop();
                            heap.push(Pair {
                                row_id: cur_row,
                                dist: dist_nn,
                            });
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    let mut out = heap.into_vec();
    out.sort_by(|a, b| a.dist.cmp(&b.dist).then_with(|| a.row_id.cmp(&b.row_id)));
    let out_n = out.len().min(k) as u32;

    let out_words = std::slice::from_raw_parts_mut(out_ptr as *mut u32, (out_cap as usize) * 2);
    for (i, pair) in out.into_iter().take(out_n as usize).enumerate() {
        out_words[i * 2] = pair.row_id;
        out_words[i * 2 + 1] = pair.dist.into_inner().to_bits();
    }
    out_n
}
