use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use fff_poc::options::FileWriterOptions;
use fff_poc::reader::{FileReaderV2Builder, Projection, Selection};
use fff_poc::writer::FileWriter;
use fff_vindex::ivf_flat::{
    build_ivf_flat_sidecar, load_ivf_flat_index, search_ivf_flat, IvfFlatBuildOptions,
};
use fff_vindex::manifest::IndexManifest;
use std::fs::File;
use std::sync::Arc;

#[test]
fn ivf_flat_build_load_search_roundtrip() {
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

    // 4 vectors (dim=2). Row 2 is closest to query [2.0, 2.1].
    let v_values = Float32Array::from(vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1]);
    let v = FixedSizeListArray::try_new(
        item_field,
        dim,
        Arc::new(v_values),
        None,
    )
    .unwrap();
    let ids = Int32Array::from(vec![10, 11, 12, 13]);

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(v), Arc::new(ids)]).unwrap();

    {
        let file = File::create(&base_path).unwrap();
        let mut writer = FileWriter::try_new(schema, file, FileWriterOptions::default()).unwrap();
        writer.write_batch(&batch).unwrap();
        writer.finish().unwrap();
    }

    let index_name = "test_ivf_flat";
    let index_path = build_ivf_flat_sidecar(
        &base_path,
        /* vector_leaf_index */ 0,
        dim as usize,
        index_name,
        IvfFlatBuildOptions {
            // Use nlist=1 to make the build deterministic and avoid kmeans sensitivity
            // in a tiny dataset.
            nlist: 1,
            train_sample: 4,
            seed: 1,
            max_kmeans_iters: 2,
        },
    )
    .unwrap();
    assert!(index_path.exists());

    let manifest_path = tmp.path().join("base.f3.vindex").join("manifest.json");
    let manifest = IndexManifest::load(&manifest_path).unwrap();
    let entry = manifest.get_by_name(index_name).unwrap();
    assert_eq!(entry.vector_leaf_index, 0);
    assert_eq!(entry.dim, dim as u32);

    let index = load_ivf_flat_index(&index_path).unwrap();
    assert_eq!(index.dim, dim as usize);
    assert_eq!(index.nlist, 1);
    assert_eq!(index.list_offsets.len(), 2);
    assert_eq!(index.row_ids.len(), 4);
    assert_eq!(index.vectors.len(), 4 * dim as usize);

    // Search top2 then fetch payload rows from the base file using the returned row_ids.
    // Use a query that avoids ties for deterministic ordering.
    let query = vec![2.0, 2.05];
    let results = search_ivf_flat(&index, &query, /* k */ 2, /* nprobe */ 1).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].row_id, 2);
    assert!(results[0].distance >= 0.0);

    let row_indexes: Vec<u64> = results.iter().map(|r| r.row_id as u64).collect();
    let file = File::open(&base_path).unwrap();
    let mut base_reader = FileReaderV2Builder::new(Arc::new(file))
        .with_projections(Projection::All)
        .with_selection(Selection::RowIndexes(row_indexes))
        .build()
        .unwrap();
    let batches = base_reader.read_file().unwrap();
    let schema = batches[0].schema();
    let selected = if batches.len() == 1 {
        batches[0].clone()
    } else {
        arrow::compute::concat_batches(&schema, &batches).unwrap()
    };
    assert_eq!(selected.num_rows(), results.len());

    let ids = selected
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let fetched_ids: Vec<i32> = (0..ids.len()).map(|i| ids.value(i)).collect();
    assert_eq!(fetched_ids, vec![12, 11]);
}
