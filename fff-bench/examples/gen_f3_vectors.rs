use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use fff_poc::writer::FileWriter;
use fff_poc::options::FileWriterOptionsBuilder;
use std::fs::File;
use std::sync::Arc;
use rand::prelude::*;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    output: String,

    #[arg(long, default_value_t = 10000)]
    rows: usize,

    #[arg(long, default_value_t = 128)]
    dim: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    let dim = args.dim;
    let rows = args.rows;

    let field = Field::new("vector", DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32
    ), false);
    let schema = Arc::new(Schema::new(vec![field]));

    let mut rng = StdRng::seed_from_u64(42);
    let total_floats = rows * dim;
    let mut values = vec![0.0f32; total_floats];
    for v in values.iter_mut() {
        *v = rng.gen();
    }

    let values_array = Float32Array::from(values);
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values_array),
        None
    )?;

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)])?;

    let file = File::create(&args.output)?;
    let options = FileWriterOptionsBuilder::with_defaults().build();
    let mut writer = FileWriter::try_new(schema, file, options).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    
    writer.write_batch(&batch).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    writer.finish().map_err(|e| anyhow::anyhow!("{:?}", e))?;

    println!("Written {} rows with dim {} to {}", rows, dim, args.output);
    Ok(())
}
