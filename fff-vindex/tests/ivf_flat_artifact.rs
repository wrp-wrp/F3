use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use fff_poc::options::FileWriterOptions;
use fff_poc::writer::FileWriter;
use fff_vindex::artifact::ivf_flat::{
    build_ivf_flat_artifact, build_ivf_flat_artifact_with_options, load_ivf_flat_artifact,
    search_ivf_flat_artifact_native, IvfFlatArtifactBuildOptions, PostingCodec,
};
use fff_vindex::ivf_flat::{build_ivf_flat_sidecar, load_ivf_flat_index, search_ivf_flat, IvfFlatBuildOptions};
use std::fs::File;
use std::sync::Arc;

#[test]
fn ivf_flat_artifact_build_load_search_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let base_path = tmp.path().join("base.f3");

    // Schema: one vector column + one payload column.
    let dim = 2i32;
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "v",
            DataType::FixedSizeList(Arc::clone(&item_field), dim),
            false,
        ),
        Field::new("id", DataType::Int32, false),
    ]));

    // 4 vectors (dim=2). Row 2 is closest to query [2.0, 2.05].
    let v_values = Float32Array::from(vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1]);
    let v = FixedSizeListArray::try_new(item_field, dim, Arc::new(v_values), None).unwrap();
    let ids = Int32Array::from(vec![10, 11, 12, 13]);

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(v), Arc::new(ids)]).unwrap();
    {
        let file = File::create(&base_path).unwrap();
        let mut writer = FileWriter::try_new(schema, file, FileWriterOptions::default()).unwrap();
        writer.write_batch(&batch).unwrap();
        writer.finish().unwrap();
    }

    let build_opts = IvfFlatBuildOptions {
        nlist: 1,
        train_sample: 4,
        seed: 1,
        max_kmeans_iters: 2,
    };
    let index_name = "test_ivf_flat";

    // Build and query the legacy sidecar.
    let sidecar_path =
        build_ivf_flat_sidecar(&base_path, /* vector_leaf_index */ 0, dim as usize, index_name, build_opts.clone())
            .unwrap();
    let sidecar = load_ivf_flat_index(&sidecar_path).unwrap();
    let query = vec![2.0, 2.05];
    let expected = search_ivf_flat(&sidecar, &query, /* k */ 2, /* nprobe */ 1).unwrap();

    // Build and query the artifact.
    let artifact_path =
        build_ivf_flat_artifact(&base_path, /* vector_leaf_index */ 0, dim as usize, index_name, build_opts).unwrap();
    let artifact = load_ivf_flat_artifact(&artifact_path).unwrap();
    artifact.validate_base().unwrap();
    artifact.validate_base_checksums().unwrap();

    let results = search_ivf_flat_artifact_native(&artifact, &query, /* k */ 2, /* nprobe */ 1).unwrap();
    assert_eq!(results.len(), expected.len());
    assert_eq!(results[0].row_id, expected[0].row_id);
    assert!(results[0].distance >= 0.0);
}

#[test]
fn ivf_flat_artifact_rowid_delta_varint_codec_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let base_path = tmp.path().join("base.f3");

    let dim = 2i32;
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "v",
            DataType::FixedSizeList(Arc::clone(&item_field), dim),
            false,
        ),
        Field::new("id", DataType::Int32, false),
    ]));

    let v_values = Float32Array::from(vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1]);
    let v = FixedSizeListArray::try_new(item_field, dim, Arc::new(v_values), None).unwrap();
    let ids = Int32Array::from(vec![10, 11, 12, 13]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(v), Arc::new(ids)]).unwrap();
    {
        let file = File::create(&base_path).unwrap();
        let mut writer = FileWriter::try_new(schema, file, FileWriterOptions::default()).unwrap();
        writer.write_batch(&batch).unwrap();
        writer.finish().unwrap();
    }

    let build_opts = IvfFlatBuildOptions {
        nlist: 1,
        train_sample: 4,
        seed: 1,
        max_kmeans_iters: 2,
    };
    let index_name = "test_ivf_flat_delta";

    let sidecar_path = build_ivf_flat_sidecar(
        &base_path,
        /* vector_leaf_index */ 0,
        dim as usize,
        index_name,
        build_opts.clone(),
    )
    .unwrap();
    let sidecar = load_ivf_flat_index(&sidecar_path).unwrap();
    let query = vec![2.0, 2.05];
    let expected = search_ivf_flat(&sidecar, &query, /* k */ 2, /* nprobe */ 1).unwrap();

    let artifact_path = build_ivf_flat_artifact_with_options(
        &base_path,
        /* vector_leaf_index */ 0,
        dim as usize,
        index_name,
        build_opts,
        IvfFlatArtifactBuildOptions {
            posting_codec: PostingCodec::RowIdDeltaVarintV1,
        },
    )
    .unwrap();
    let artifact = load_ivf_flat_artifact(&artifact_path).unwrap();
    let results = search_ivf_flat_artifact_native(&artifact, &query, /* k */ 2, /* nprobe */ 1).unwrap();
    assert_eq!(results.len(), expected.len());
    assert_eq!(results[0].row_id, expected[0].row_id);
 }
