#![feature(new_range_api)]
#[cfg(not(target_os = "macos"))]
use mimalloc::MiMalloc;
#[cfg(target_os = "macos")]
use std::alloc::System;

pub mod common;
mod compression;
pub mod counter;
pub mod file;
pub mod io;
pub mod options;
pub mod reader;
pub mod vector_index;
pub mod writer;

pub mod context;
pub mod decoder;
mod dict;
pub(crate) mod encoder;

#[cfg(not(target_os = "macos"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(target_os = "macos")]
#[global_allocator]
static GLOBAL: System = System;
