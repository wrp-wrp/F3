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

    let source_vectors = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
        vec![0.2, 0.9],
        vec![0.9, 0.2],
        vec![0.3, 0.3],
    ];
    let vector_data = encode_bruteforce_index(&source_vectors).unwrap();

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

    // Brute-force KNN search should match ground truth for a variety of queries.
    let queries = vec![[0.9_f32, 0.1_f32], [0.05, 0.05], [0.95, 0.1], [0.25, 0.85]];
    for query in queries {
        let actual = reader.vector_knn_l2(7, &query, 3).unwrap();
        let expected = brute_force_knn(&source_vectors, &query, 3);
        assert_eq!(
            actual.len(),
            expected.len(),
            "result length mismatch for query {:?}",
            query
        );
        for (res, exp) in actual.iter().zip(expected.iter()) {
            assert_eq!(
                res.row_id, exp.0 as u64,
                "row id mismatch for query {:?}",
                query
            );
            assert!(
                (res.distance - exp.1).abs() < 1e-6,
                "distance mismatch for query {:?}",
                query
            );
        }
    }
}

fn brute_force_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, vec)| {
            let dist = vec
                .iter()
                .zip(query.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            (idx, dist)
        })
        .collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    pairs.truncate(k.min(pairs.len()));
    pairs
}
