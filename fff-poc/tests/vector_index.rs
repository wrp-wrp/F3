use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use fff_poc::{
    options::FileWriterOptions,
    reader::FileReaderV2Builder,
    vector_index::{
        encode_bruteforce_index, QuantizationMethod, QuantizationSegment, QuantizationSpec,
        VectorDistanceMetric, VectorIndexAlgorithm, VectorIndexConfig,
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
        vec![Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap();

    let vector_data = encode_bruteforce_index(&vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ])
    .unwrap();

    let vector_config = VectorIndexConfig {
        index_id: 7,
        column: "values".to_string(),
        algorithm: VectorIndexAlgorithm::BruteForce,
        distance_metric: VectorDistanceMetric::L2,
        priority: 1,
        usage_hint: Some("unit-test".to_string()),
        quantization: QuantizationSpec {
            dimension: 2,
            segments: vec![QuantizationSegment {
                start_dim: 0,
                end_dim: 2,
                method: QuantizationMethod::None,
                bits: 0,
                params: vec![],
            }],
        },
        custom_params: vec![42, 0, 13],
        data: vector_data.clone(),
        wasm_module: None,
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
        descriptor.quantization.dimension, 2,
        "quantization metadata preserved"
    );

    let data_blob = reader
        .load_vector_index_blob(7)
        .unwrap()
        .expect("vector blob present");
    assert_eq!(data_blob, vector_data);

    let wasm_blob = reader.load_vector_index_wasm(7).unwrap();
    assert!(
        wasm_blob.is_none(),
        "Wasm blob should be absent for this test"
    );

    // Brute-force KNN search should recover the closest vector exactly (recall@1 = 1.0).
    let query = [0.9_f32, 0.1_f32];
    let results = reader.vector_knn_l2(7, &query, 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].row_id, 1);
    let recall_at_1: f32 = if results[0].row_id == 1 { 1.0 } else { 0.0 };
    assert!(
        (recall_at_1 - 1.0_f32).abs() < f32::EPSILON,
        "recall@1 should be 1.0 for brute-force index"
    );
}
