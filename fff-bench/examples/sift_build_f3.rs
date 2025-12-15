use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use fff_poc::options::FileWriterOptions;
use fff_poc::writer::FileWriter;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser, Debug)]
struct Args {
    /// Directory containing `learn.fvecs` (see `scripts/download_sift_100k.sh`).
    #[arg(long, default_value = "data/sift")]
    sift_dir: PathBuf,

    /// Max vectors to write (SIFT100K uses 100_000).
    #[arg(long, default_value_t = 100_000)]
    max_vectors: usize,

    /// Output F3 path.
    #[arg(long, default_value = "data/sift/sift100k.f3")]
    out: PathBuf,
}

fn read_fvecs(path: &Path, max_vectors: usize) -> Result<(usize, Vec<f32>)> {
    let f = File::open(path).with_context(|| format!("open fvecs {}", path.display()))?;
    let mut r = BufReader::new(f);

    let dim = r
        .read_i32::<LittleEndian>()
        .with_context(|| format!("read dim from {}", path.display()))? as usize;
    if dim == 0 || dim > 10_000 {
        bail!("invalid dim={dim} in {}", path.display());
    }

    let mut out = Vec::<f32>::new();
    out.reserve(max_vectors.saturating_mul(dim));
    for _ in 0..dim {
        out.push(r.read_f32::<LittleEndian>()?);
    }

    while out.len() / dim < max_vectors {
        let d = match r.read_i32::<LittleEndian>() {
            Ok(d) => d as usize,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        if d != dim {
            bail!("dim mismatch in {}: expected {dim}, got {d}", path.display());
        }
        for _ in 0..dim {
            out.push(r.read_f32::<LittleEndian>()?);
        }
    }
    Ok((dim, out))
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(
        args.out
            .parent()
            .ok_or_else(|| anyhow::anyhow!("invalid out path"))?,
    )?;

    let learn = args.sift_dir.join("learn.fvecs");
    let (dim, vectors) = read_fvecs(&learn, args.max_vectors)
        .with_context(|| format!("read {}", learn.display()))?;
    let n = vectors.len() / dim;
    if n == 0 {
        bail!("no vectors read from {}", learn.display());
    }

    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "v",
            DataType::FixedSizeList(Arc::clone(&item_field), dim as i32),
            false,
        ),
        Field::new("id", DataType::Int32, false),
    ]));
    let v_values = Float32Array::from(vectors);
    let v = FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(v_values), None)?;
    let ids = Int32Array::from((0..n as i32).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(v), Arc::new(ids)])?;

    let f = File::create(&args.out).with_context(|| format!("create {}", args.out.display()))?;
    let mut w = FileWriter::try_new(schema, f, FileWriterOptions::default())
        .map_err(|e| anyhow!("create F3 writer: {e}"))?;
    w.write_batch(&batch)
        .map_err(|e| anyhow!("write batch: {e}"))?;
    w.finish().map_err(|e| anyhow!("finish: {e}"))?;

    println!("wrote: {} (rows={}, dim={})", args.out.display(), n, dim);
    Ok(())
}
