use std::sync::Arc;

use arrow_array::builder::{Float32Builder, ListBuilder};
use arrow_array::{Array, ArrayRef, Float32Array, Int32Array, ListArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use fff_poc::options::FileWriterOptions;
use fff_poc::reader::FileReaderV2Builder;
use fff_poc::vector_index::{
    QuantizationSpec, VectorDistanceMetric, VectorIndexBuildAlgorithm, VectorIndexBuildConfig,
    VectorIndexAlgorithm,
};
use fff_poc::writer::FileWriter;

#[test]
fn vector_column_with_bruteforce_index_build() {
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

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "vecs",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        ),
        Field::new("values", DataType::Int32, false),
    ]));
    let vec_array = build_list_f32(&source_vectors);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![vec_array, Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80]))],
    )
    .unwrap();

    let build_cfg = VectorIndexBuildConfig {
        index_id: 7,
        column: "vecs".to_string(),
        algorithm: VectorIndexBuildAlgorithm::BruteForce,
        distance_metric: VectorDistanceMetric::L2,
        priority: 0,
        usage_hint: Some("build-brute".to_string()),
        quantization: QuantizationSpec {
            dimension: 0,
            segments: vec![],
        },
        custom_params: vec![],
    };

    let options = FileWriterOptions::builder()
        .add_vector_index_build(build_cfg)
        .build();

    let temp_file = Arc::new(tempfile::tempfile().unwrap());
    let mut writer = FileWriter::try_new(schema.clone(), temp_file.clone(), options).unwrap();
    writer.write_batch(&batch).unwrap();
    writer.finish().unwrap();

    let mut reader = FileReaderV2Builder::new(temp_file.clone()).build().unwrap();
    let indexes = reader.vector_indexes();
    assert_eq!(indexes.len(), 1);
    assert_eq!(indexes[0].index_id, 7);
    assert_eq!(indexes[0].column, "vecs");
    assert_eq!(indexes[0].algorithm, VectorIndexAlgorithm::BruteForce);
    assert_eq!(indexes[0].quantization.dimension, 2);

    let blob = reader.load_vector_index_blob(7).unwrap().unwrap();
    assert!(!blob.is_empty());

    let query = [0.9_f32, 0.1_f32];
    let actual = reader.vector_knn_l2(7, &query, 3).unwrap();
    let expected = brute_force_knn(&source_vectors, &query, 3);
    for (res, exp) in actual.iter().zip(expected.iter()) {
        assert_eq!(res.row_id, exp.0 as u64);
        assert!((res.distance - exp.1).abs() < 1e-6);
    }

    let batches = reader.read_file().unwrap();
    assert_eq!(batches.len(), 1);
    let vecs_back = extract_list_f32(batches[0].column(0));
    assert_eq!(vecs_back, source_vectors);
}

#[test]
fn vector_column_with_hnsw_index_build() {
    let source_vectors = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
        vec![0.2, 0.9],
        vec![0.9, 0.2],
        vec![0.3, 0.3],
        vec![0.8, 0.75],
        vec![0.15, 0.2],
    ];

    let schema = Arc::new(Schema::new(vec![Field::new(
        "vecs",
        DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
        false,
    )]));
    let vec_array = build_list_f32(&source_vectors);
    let batch = RecordBatch::try_new(schema.clone(), vec![vec_array]).unwrap();

    let build_cfg = VectorIndexBuildConfig {
        index_id: 11,
        column: "vecs".to_string(),
        algorithm: VectorIndexBuildAlgorithm::Hnsw {
            max_neighbors: source_vectors.len() - 1,
            ef_search: source_vectors.len(),
        },
        distance_metric: VectorDistanceMetric::L2,
        priority: 0,
        usage_hint: Some("build-hnsw".to_string()),
        quantization: QuantizationSpec {
            dimension: 0,
            segments: vec![],
        },
        custom_params: vec![],
    };

    let options = FileWriterOptions::builder()
        .add_vector_index_build(build_cfg)
        .build();

    let temp_file = Arc::new(tempfile::tempfile().unwrap());
    let mut writer = FileWriter::try_new(schema.clone(), temp_file.clone(), options).unwrap();
    writer.write_batch(&batch).unwrap();
    writer.finish().unwrap();

    let mut reader = FileReaderV2Builder::new(temp_file.clone()).build().unwrap();
    let indexes = reader.vector_indexes();
    assert_eq!(indexes.len(), 1);
    assert_eq!(indexes[0].algorithm, VectorIndexAlgorithm::Hnsw);

    let query = [0.9_f32, 0.1_f32];
    let actual = reader.vector_knn_l2(11, &query, 3).unwrap();
    let expected = brute_force_knn(&source_vectors, &query, 3);
    for (res, exp) in actual.iter().zip(expected.iter()) {
        assert_eq!(res.row_id, exp.0 as u64);
        assert!((res.distance - exp.1).abs() < 1e-6);
    }
}

fn build_list_f32(vectors: &[Vec<f32>]) -> ArrayRef {
    let mut builder = ListBuilder::new(Float32Builder::new());
    for vec in vectors {
        for value in vec {
            builder.values().append_value(*value);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn extract_list_f32(array: &ArrayRef) -> Vec<Vec<f32>> {
    let list = array.as_any().downcast_ref::<ListArray>().unwrap();
    let values = list.values();
    let floats = values.as_any().downcast_ref::<Float32Array>().unwrap();
    let offsets = list.offsets();
    let mut out = Vec::with_capacity(list.len());
    for row in 0..list.len() {
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        let mut vec = Vec::with_capacity(end - start);
        for idx in start..end {
            vec.push(floats.value(idx));
        }
        out.push(vec);
    }
    out
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
