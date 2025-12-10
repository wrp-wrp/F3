use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use fff_poc::{
    options::FileWriterOptions,
    reader::FileReaderV2Builder,
    vector_index::{
        QuantizationMethod, QuantizationSegment, QuantizationSpec, VectorDistanceMetric,
        VectorIndexAlgorithm, VectorIndexConfig,
    },
    writer::FileWriter,
};

#[test]
fn vector_index_metadata_roundtrip() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "values",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))],
    )
    .unwrap();

    let vector_config = VectorIndexConfig {
        index_id: 7,
        column: "values".to_string(),
        algorithm: VectorIndexAlgorithm::BruteForce,
        distance_metric: VectorDistanceMetric::L2,
        priority: 1,
        usage_hint: Some("unit-test".to_string()),
        quantization: QuantizationSpec {
            dimension: 4,
            segments: vec![QuantizationSegment {
                start_dim: 0,
                end_dim: 4,
                method: QuantizationMethod::None,
                bits: 0,
                params: vec![],
            }],
        },
        custom_params: vec![42, 0, 13],
        data: vec![9, 8, 7, 6],
        wasm_module: Some(vec![1, 2, 3, 4]),
    };

    let options = FileWriterOptions::builder()
        .add_vector_index(vector_config)
        .build();

    let temp_file = Arc::new(tempfile::tempfile().unwrap());
    let mut writer = FileWriter::try_new(schema.clone(), temp_file.clone(), options).unwrap();
    writer.write_batch(&batch).unwrap();
    writer.finish().unwrap();

    let mut reader = FileReaderV2Builder::new(temp_file.clone()).build().unwrap();
    let indexes = reader.vector_indexes();
    assert_eq!(indexes.len(), 1);
    let descriptor = &indexes[0];
    assert_eq!(descriptor.index_id, 7);
    assert_eq!(descriptor.column, "values");
    assert_eq!(descriptor.usage_hint.as_deref(), Some("unit-test"));
    assert_eq!(descriptor.custom_params, vec![42, 0, 13]);
    assert_eq!(
        descriptor.quantization.dimension, 4,
        "quantization metadata preserved"
    );

    let data_blob = reader
        .load_vector_index_blob(7)
        .unwrap()
        .expect("vector blob present");
    assert_eq!(data_blob, vec![9, 8, 7, 6]);

    let wasm_blob = reader
        .load_vector_index_wasm(7)
        .unwrap()
        .expect("wasm blob present");
    assert_eq!(wasm_blob, vec![1, 2, 3, 4]);
}
