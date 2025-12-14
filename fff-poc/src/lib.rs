#![feature(new_range_api)]
#[cfg(not(target_vendor = "apple"))]
use mimalloc::MiMalloc;

pub mod common;
mod compression;
pub mod counter;
pub mod file;
pub mod io;
pub mod options;
pub mod reader;
pub mod writer;

pub mod context;
pub mod decoder;
mod dict;
pub(crate) mod encoder;
// NOTE: `mimalloc` can fail to link on Apple targets in some toolchain / deployment-target
// combinations (e.g. missing `___emutls_get_address`). We keep the system allocator on Apple.
#[cfg(not(target_vendor = "apple"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
