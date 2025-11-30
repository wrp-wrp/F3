use core::slice;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Candidate {
    pub id: u32,
    pub dist: f32,
}

// Minimal ABI constants for host/runtime compatibility.
pub const ABI_MAJOR: u16 = 0;
pub const ABI_MINOR: u16 = 1;

#[no_mangle]
pub extern "C" fn FFFUDE_VERSION_0_1() {}

// Simple centroid table packed as contiguous f32 (k * dim).
// Host writes centroids into memory and passes pointer/length.
static mut CENTROIDS: *const f32 = core::ptr::null();
static mut CENTROIDS_LEN: usize = 0;
static mut DIM: usize = 0;
static mut K: usize = 0;

/// Initialize centroids table.
/// input_ptr: *const f32, input_len: total floats (k * dim), kwargs_ptr/len unused.
#[no_mangle]
pub extern "C" fn init_ffi(
    input_ptr: *const u8,
    input_len: u32,
    _kwargs_ptr: *const u8,
    _kwargs_len: u32,
    _out: *mut u8,
) -> i32 {
    unsafe {
        CENTROIDS = input_ptr as *const f32;
        CENTROIDS_LEN = input_len as usize / core::mem::size_of::<f32>();
        // Caller must set DIM/K via kwargs; for simplicity infer a square-ish shape.
        // Here we assume kwargs not provided; host should set DIM/K via globals if needed.
    }
    0
}

/// Host sets dimension and k explicitly.
#[no_mangle]
pub extern "C" fn set_shape(dim: u32, k: u32) {
    unsafe {
        DIM = dim as usize;
        K = k as usize;
    }
}

/// query_ptr: *const f32, dim: u32, k_out: u32, out_ids_ptr: *mut u32, out_dists_ptr: *mut f32
#[no_mangle]
pub extern "C" fn search_ffi(
    query_ptr: *const u8,
    dim: u32,
    k_out: u32,
    out_ids_ptr: *mut u8,
    out_dists_ptr: *mut u8,
) -> i32 {
    unsafe {
        HEAP_OFFSET = 0;
    }
    let dim = dim as usize;
    let k_out = k_out as usize;
    let query = unsafe { slice::from_raw_parts(query_ptr as *const f32, dim) };
    let centroids = unsafe {
        slice::from_raw_parts(CENTROIDS, core::cmp::min(CENTROIDS_LEN, DIM * K))
    };
    let k = unsafe { core::cmp::min(K, centroids.len() / dim) };

    if k == 0 || dim == 0 || query.len() != dim {
        return -1;
    }

    // Naive linear scan over centroids.
    let mut best = [(core::u32::MAX, core::f32::INFINITY); 8];
    for (cid, chunk) in centroids.chunks(dim).take(k).enumerate() {
        let mut dist = 0f32;
        for j in 0..dim {
            let d = query[j] - chunk[j];
            dist += d * d;
        }
        // keep small top-k (k_out <= 8 assumed small)
        let mut pos = None;
        for i in 0..core::cmp::min(k_out, best.len()) {
            if dist < best[i].1 {
                pos = Some(i);
                break;
            }
        }
        if let Some(idx) = pos {
            for shift in (idx + 1..core::cmp::min(k_out, best.len())).rev() {
                best[shift] = best[shift - 1];
            }
            best[idx] = (cid as u32, dist);
        }
    }

    let out_ids = unsafe { slice::from_raw_parts_mut(out_ids_ptr as *mut u32, k_out) };
    let out_dists = unsafe { slice::from_raw_parts_mut(out_dists_ptr as *mut f32, k_out) };
    for i in 0..k_out {
        if i < best.len() {
            out_ids[i] = best[i].0;
            out_dists[i] = best[i].1;
        } else {
            out_ids[i] = core::u32::MAX;
            out_dists[i] = core::f32::INFINITY;
        }
    }
    k_out as i32
}

// Minimal bump allocator to satisfy host runtime expectations.
static mut HEAP: [u8; 64 << 20] = [0; 64 << 20];
static mut HEAP_OFFSET: usize = 0;

#[no_mangle]
pub extern "C" fn alloc(len: u32, _align: u32) -> u32 {
    unsafe {
        let mut start = (HEAP_OFFSET + 7) & !7;
        if start == 0 {
            start = 8;
        }
        let mut end = start + len as usize;
        if end > HEAP.len() {
            start = 8;
            end = start + len as usize;
            if end > HEAP.len() {
                return 0;
            }
        }
        HEAP_OFFSET = end;
        start as u32
    }
}

#[no_mangle]
pub extern "C" fn dealloc(_ptr: u32, _len: u32, _align: u32) {
    // no-op for bump allocator
}

#[no_mangle]
pub extern "C" fn buffer_iterator_next(_iter: *mut u8, _out: *mut u8, _out_buf: *mut u32) {}

#[no_mangle]
pub extern "C" fn buffer_iterator_drop(_iter: *mut u8) {}

#[no_mangle]
pub extern "C" fn buffer_drop(_buf: *mut u8) {}
