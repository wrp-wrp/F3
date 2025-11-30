use crate::{
    common::{checksum::ChecksumType, ColumnIndexSequence},
    context::WASMReadingContext,
    counter::EncodingCounter,
    decoder::logical::{create_list_struct_decoder, create_logical_decoder},
    dict::shared_dictionary_cache::SharedDictionaryCache,
    file::footer::{Footer, GroupedColumnMetadata, PostScript},
    io::reader::Reader,
};
#[cfg(feature = "ann-wasm-micro")]
use crate::decoder::logical::advance_column_index;
#[cfg(feature = "ann-wasm-micro")]
use crate::vector::{is_supported_vector, parse_micro_blob};
use arrow::compute::concat;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_buffer::MutableBuffer;
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use fff_core::{
    errors::{Error, Result},
    non_nest_types,
};
use fff_format::File::fff::flatbuf::{self as fb, CompressionType};
use fff_format::{MAGIC, POSTSCRIPT_SIZE};
#[cfg(feature = "ann-wasm-micro")]
use fff_format::vector::VectorColumn;
#[cfg(feature = "ann-wasm-micro")]
use fff_ude_wasm::Runtime;
use std::sync::Arc;
#[cfg(feature = "ann-wasm-micro")]
use std::collections::HashMap;

mod projection;
pub use projection::Projection;
mod selection;
pub use selection::Selection;

mod legacy;
pub use legacy::FileReader;

mod builder;
pub use builder::FileReaderV2Builder;

/// Utility function to get the max size of a Chunk in this FFF file.
pub fn get_max_chunk_size<R: Reader + Clone>(reader: R) -> Result<usize> {
    let file_size = reader.size()?;
    let post_script = read_postscript(&reader, file_size)?;
    let owner = get_metadata_buffer(&reader, &post_script)?;
    let footer = {
        let file_size = reader.size()? as usize;
        Footer::try_new(&owner, file_size, &post_script)
    }?;
    let mut max_size = 0;
    let rg_metas = footer.row_group_metadatas();
    for rg_meta in rg_metas {
        for col_meta in rg_meta.column_metadatas.iter() {
            col_meta.column_chunks().unwrap().iter().for_each(|chunk| {
                // log::error!("chunk size: {}", chunk.size_());
                max_size = std::cmp::max(max_size, chunk.size_() as usize);
            });
        }
    }
    Ok(max_size)
}
/// Utility function to get the average size of all the IOUnits of a specific column in this FFF file.
pub fn get_avg_io_unit_size<R: Reader + Clone>(reader: R, col_idx: usize) -> Result<usize> {
    let file_size = reader.size()?;
    let post_script = read_postscript(&reader, file_size)?;
    let owner = get_metadata_buffer(&reader, &post_script)?;
    let footer = {
        let file_size = reader.size()? as usize;
        Footer::try_new(&owner, file_size, &post_script)
    }?;
    let mut total_size = 0;
    let mut total_count = 0;
    let rg_metas = footer.row_group_metadatas();
    for rg_meta in rg_metas {
        let col_meta = rg_meta.column_metadatas.get(col_idx).unwrap();
        col_meta.column_chunks().unwrap().iter().for_each(|chunk| {
            // log::error!("chunk size: {}", chunk.size_());
            total_size += chunk.size_() as usize;
            total_count += 1;
        });
    }
    Ok(total_size / total_count)
}

pub(crate) struct RowGroupCntNPointer {
    pub(crate) row_count: u32,
    pub(crate) _offset: u64,
    pub(crate) _size: u32,
}

pub struct FileReaderV2<R> {
    reader: R,
    schema: SchemaRef,
    projections: Projection,
    selection: Selection,
    /// Store only the projection of metadata of each row group.
    grouped_column_metadata_buffers: Vec<Vec<Bytes>>,
    row_group_cnt_n_pointers: Vec<RowGroupCntNPointer>,
    /// TODO: remove this Option wrapping when removing V1 reader.
    wasm_context: Option<Arc<WASMReadingContext<R>>>,
    shared_dictionary_cache: Option<SharedDictionaryCache>,
    /// Whether we verify the IOUnit checksum.
    checksum_type: Option<ChecksumType>,
    /// Enable vector micro-index usage (feature gated).
    enable_micro_index: bool,
}

impl<R: Reader + Clone> FileReaderV2<R> {
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn materialize_footer(&self) -> Result<Footer> {
        Footer::try_new_with_projection(
            &self.row_group_cnt_n_pointers,
            self.grouped_column_metadata_buffers
                .iter()
                .map(|c_buffers| {
                    c_buffers
                        .iter()
                        .map(|c_buffer| c_buffer.as_ref())
                        .collect::<Vec<_>>()
                })
                .collect(),
            self.schema.clone(),
        )
    }

    pub fn read_file(&mut self) -> Result<Vec<RecordBatch>> {
        let footer = self.materialize_footer()?;
        let mut reader = self.reader.clone();
        read_file_based_on_footer(
            &mut reader,
            footer,
            &self.projections,
            &self.selection,
            self.wasm_context.clone(),
            self.shared_dictionary_cache.as_ref(),
            self.checksum_type,
        )
    }

    #[allow(clippy::type_complexity)]
    pub fn get_shared_dict_sizes(
        &mut self,
    ) -> Result<(Vec<EncodingCounter>, Vec<Vec<(usize, usize)>>)> {
        let footer = self.materialize_footer()?;
        get_shared_dict_size_based_on_footer(footer, self.shared_dictionary_cache.as_ref().unwrap())
    }

    /// Run a micro-indexed vector search on a FixedSizeList<Float32> column.
    #[cfg(feature = "ann-wasm-micro")]
    pub fn search_vector_column(
        &mut self,
        column_idx: usize,
        query: &[f32],
        k_out: usize,
    ) -> Result<Vec<ArrayRef>> {
        if !self.enable_micro_index {
            return Err(Error::General("micro index is disabled".into()));
        }
        let footer = self.materialize_footer()?;
        if !is_supported_vector(footer.schema().field(column_idx).data_type()) {
            return Err(Error::General("target column is not a supported vector type".into()));
        }
        let mut runtimes: HashMap<(u64, u32), Arc<Runtime>> = HashMap::new();
        let selected_rg_metas = process_selection(&self.selection, footer.row_group_metadatas());
        let mut results = vec![];
        for (rg_meta, _) in selected_rg_metas {
            let column_meta = rg_meta
                .column_metadatas
                .get(column_idx)
                .ok_or_else(|| Error::General("column metadata missing".into()))?;
            let Some(vector_meta_fb) = column_meta.vector_column() else {
                // Fallback to full decode.
                let mut col_idx_seq = ColumnIndexSequence::default();
                for (i, field) in footer.schema().fields().iter().enumerate() {
                    if i == column_idx {
                        let mut decoder = create_logical_decoder(
                            &self.reader,
                            Arc::clone(field),
                            &rg_meta.column_metadatas,
                            &mut col_idx_seq,
                            self.wasm_context.as_ref().map(Arc::clone),
                            self.shared_dictionary_cache.as_ref().unwrap(),
                            self.checksum_type,
                        )?;
                        results.extend(decoder.decode_batch()?);
                        break;
                    }
                    advance_column_index(field.clone(), &mut col_idx_seq)?;
                }
                continue;
            };
            let vector_meta = VectorColumn::from_fb(vector_meta_fb);
            for block in vector_meta.blocks {
                // Fallback to full block if micro-index missing.
                if block.micro_index.aux_size == 0 || block.micro_index.wasm_size == 0 {
                    let mut col_idx_seq = ColumnIndexSequence::default();
                    let mut decoder_opt = None;
                    for (i, field) in footer.schema().fields().iter().enumerate() {
                        if i == column_idx {
                            decoder_opt = Some(create_logical_decoder(
                                &self.reader,
                                Arc::clone(field),
                                &rg_meta.column_metadatas,
                                &mut col_idx_seq,
                                self.wasm_context.as_ref().map(Arc::clone),
                                self.shared_dictionary_cache.as_ref().unwrap(),
                                self.checksum_type,
                            )?);
                            break;
                        }
                        advance_column_index(field.clone(), &mut col_idx_seq)?;
                    }
                    let mut decoder = decoder_opt
                        .ok_or_else(|| Error::General("failed to build decoder".to_string()))?;
                    results.extend(decoder.decode_row_at(
                        block.row_start as usize,
                        block.row_count as usize,
                    )?);
                    continue;
                }
                let mut aux_buf = vec![0u8; block.micro_index.aux_size as usize];
                self.reader
                    .read_exact_at(&mut aux_buf, block.micro_index.aux_offset)?;
                let blob = parse_micro_blob(&aux_buf)?;
                if blob.dim as usize != query.len() {
                    return Err(Error::General(
                        "query dimension does not match micro-index".into(),
                    ));
                }
                let rt_key = (block.micro_index.wasm_offset, block.micro_index.wasm_size);
                let rt = if let Some(rt) = runtimes.get(&rt_key) {
                    Arc::clone(rt)
                } else {
                    let mut wasm_buf = vec![0u8; block.micro_index.wasm_size as usize];
                    self.reader
                        .read_exact_at(&mut wasm_buf, block.micro_index.wasm_offset)?;
                    let rt = Arc::new(
                        Runtime::try_new(&wasm_buf)
                            .map_err(|e| Error::General(e.to_string()))?,
                    );
                    runtimes.insert(rt_key, Arc::clone(&rt));
                    rt
                };
                let mut out_ids = vec![0u8; k_out * std::mem::size_of::<u32>()];
                let mut out_dists = vec![0u8; k_out * std::mem::size_of::<f32>()];
                let centroid_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        blob.centroids.as_ptr() as *const u8,
                        blob.centroids.len() * std::mem::size_of::<f32>(),
                    )
                };
                let query_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        query.as_ptr() as *const u8,
                        query.len() * std::mem::size_of::<f32>(),
                    )
                };
                let _ = rt
                    .call_micro_search(
                        blob.dim,
                        k_out as u32,
                        centroid_bytes,
                        query_bytes,
                        &mut out_ids,
                        &mut out_dists,
                    )
                    .map_err(|e| Error::General(e.to_string()))?;
                let best = u32::from_le_bytes(out_ids[0..4].try_into().unwrap()) as usize;
                let spans = blob.bucket_spans.get(best).cloned().unwrap_or_default();
                if spans.is_empty() {
                    results.push(concat(&[])?);
                    continue;
                }
                let mut per_block_arrays: Vec<ArrayRef> = vec![];
                for (start, len) in spans {
                    let mut col_idx_seq = ColumnIndexSequence::default();
                    let mut decoder_opt = None;
                    for (i, field) in footer.schema().fields().iter().enumerate() {
                        if i == column_idx {
                            decoder_opt = Some(create_logical_decoder(
                                &self.reader,
                                Arc::clone(field),
                                &rg_meta.column_metadatas,
                                &mut col_idx_seq,
                                self.wasm_context.as_ref().map(Arc::clone),
                                self.shared_dictionary_cache.as_ref().unwrap(),
                                self.checksum_type,
                            )?);
                            break;
                        }
                        advance_column_index(field.clone(), &mut col_idx_seq)?;
                    }
                    let mut decoder = decoder_opt
                        .ok_or_else(|| Error::General("failed to build decoder".to_string()))?;
                    let mut decoded = decoder.decode_row_at(
                        block.row_start as usize + start as usize,
                        len as usize,
                    )?;
                    per_block_arrays.append(&mut decoded);
                }
                if per_block_arrays.len() == 1 {
                    results.push(per_block_arrays.remove(0));
                } else {
                    results.push(concat(
                        per_block_arrays
                            .iter()
                            .map(|a| a.as_ref())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )?);
                }
            }
        }
        Ok(results)
    }

    /// Access single row id from a leaf column from potentially nested data
    /// Right now it should only work for List of Struct of Primitives to test the pushdown effects.
    pub fn point_access_list_struct(
        &mut self,
        col_leaf_id: u32,
        col_field: FieldRef,
        row_id: usize,
    ) -> Result<Vec<RecordBatch>> {
        let footer = self.materialize_footer()?;
        let mut reader = self.reader.clone();
        point_access_list_struct(
            &mut reader,
            footer,
            col_leaf_id,
            col_field,
            row_id,
            self.wasm_context.clone(),
            self.shared_dictionary_cache.as_ref(),
        )
    }
}

fn get_metadata_buffer<R: Reader>(reader: &R, post_script: &PostScript) -> Result<MutableBuffer> {
    if post_script.compression != CompressionType::Uncompressed {
        return Err(Error::General("Compression type not supported".to_string()));
    }
    let mut buffer = MutableBuffer::from_len_zeroed(post_script.metadata_size as usize);
    reader.read_exact_at(
        buffer.as_slice_mut(),
        reader.size()? - POSTSCRIPT_SIZE - post_script.metadata_size as u64,
    )?;
    Ok(buffer)
}

fn read_file_based_on_footer<R: Reader>(
    reader: &mut R,
    footer: Footer,
    projections: &Projection,
    selection: &Selection,
    wasm_context: Option<Arc<WASMReadingContext<R>>>,
    shared_dictionary_cache: Option<&SharedDictionaryCache>,
    checksum_type: Option<ChecksumType>,
) -> Result<Vec<RecordBatch>> {
    let shared_dictionary_cache = shared_dictionary_cache.unwrap();
    let mut record_batches = vec![];
    let rg_metas = footer.row_group_metadatas();
    // let projections = projections.map(|vec| vec.iter().map(|v| *v).collect::<HashSet<usize>>());
    let selected_rg_metas = process_selection(selection, rg_metas);
    for (rg_meta, selection_in_rg) in selected_rg_metas {
        let mut column_idx = ColumnIndexSequence::default();
        let mut columns = vec![];
        let mut decode_col = |field: &Arc<Field>| -> Result<()> {
            let mut col_decoder = create_logical_decoder(
                reader,
                Arc::clone(field),
                &rg_meta.column_metadatas,
                &mut column_idx,
                wasm_context.as_ref().map(Arc::clone),
                shared_dictionary_cache,
                checksum_type,
            )?;
            let arrays = if let Selection::RowIndexes(row_indexes) = &selection_in_rg {
                col_decoder.decode_row_at(row_indexes[0] as usize, 1)?
            } else {
                col_decoder.decode_batch()?
            };
            columns.push(arrays);
            Ok(())
        };
        // TODO: needs some magic to handle nested data. Basically needs to go over the schema recursively
        // and figure out which leaf nodes to fetch. Currently projection is only tested on flat data.
        match projections {
            Projection::LeafColumnIndexes(projected_indices) => projected_indices
                .iter()
                .try_for_each(|&v| decode_col(footer.schema().fields().get(v).unwrap()))?,
            Projection::All => {
                for field in footer.schema().fields().iter() {
                    // println!("decode col {field_id}");
                    decode_col(field)?;
                }
            }
        }
        // TODO: vortex may not round-trip out the input Arrow type. https://github.com/spiraldb/vortex/issues/1021
        for i in 0..columns[0].len() {
            let columns_this_batch = columns.iter().map(|c| c[i].clone()).collect::<Vec<_>>();
            record_batches.push(RecordBatch::try_new(
                Schema::new(
                    columns_this_batch
                        .iter()
                        .zip(footer.schema().fields().iter())
                        .map(|(c, f)| Field::new(f.name(), c.data_type().clone(), f.is_nullable()))
                        .collect::<Vec<_>>(),
                )
                .into(),
                columns_this_batch,
            )?);
        }
        // record_batches.push(RecordBatch::try_new(footer.schema().clone(), columns)?);
    }
    Ok(record_batches)
}

#[allow(clippy::type_complexity)]
fn get_shared_dict_size_based_on_footer(
    footer: Footer,
    shared_dictionary_cache: &SharedDictionaryCache,
) -> Result<(Vec<EncodingCounter>, Vec<Vec<(usize, usize)>>)> {
    let rg_metas = footer.row_group_metadatas();
    let mut referenced_dicts: Vec<std::collections::HashSet<u32>> =
        vec![std::collections::HashSet::new(); footer.schema().fields().len()];
    let mut sharing_peers = vec![vec![]; footer.schema().fields().len()];
    let chunk_sizes = shared_dictionary_cache.get_dict_chunk_sizes();
    let dict_ref_to_chunks = shared_dictionary_cache.get_dict_references();
    for rg_meta in rg_metas {
        rg_meta
            .column_metadatas
            .iter()
            .enumerate()
            .for_each(|(field_id, column_meta)| {
                for chunk in column_meta.column_chunks().unwrap().iter() {
                    if let Some(shared_dict_id) = chunk.encoding_as_shared_dictionary() {
                        let shared_dict_id = shared_dict_id.shared_dictionary_idx();
                        referenced_dicts[field_id].insert(shared_dict_id);
                    }
                }
            });
    }
    let mut chunk_referencing_cols =
        vec![std::collections::HashSet::<usize>::new(); chunk_sizes.len()];
    for (col_idx, col_ref_dicts) in referenced_dicts.iter().enumerate() {
        for dict_idx in col_ref_dicts {
            for chunk_idx in &dict_ref_to_chunks[*dict_idx as usize] {
                let ref_cols = &mut chunk_referencing_cols[*chunk_idx];
                if !ref_cols.is_empty() {
                    for peer in ref_cols.iter() {
                        sharing_peers[col_idx].push((*peer, chunk_sizes[*chunk_idx]));
                        sharing_peers[*peer].push((col_idx, chunk_sizes[*chunk_idx]));
                    }
                }
                ref_cols.insert(col_idx);
            }
        }
    }
    let counters = referenced_dicts
        .iter()
        .map(|set| EncodingCounter {
            dict_type: crate::dict::DictionaryTypeOptions::GlobalDictionary,
            dict_size: set
                .iter()
                .map(|dict_idx| {
                    shared_dictionary_cache
                        .get_dict_size(*dict_idx as usize)
                        .unwrap()
                })
                .sum(),
            index_size: 0,
        })
        .collect();
    Ok((counters, sharing_peers))
}

/// Access single row id from a leaf column, in a file with schema List(Struct(_)) where _ is non_nest_type!().
fn point_access_list_struct<R: Reader>(
    reader: &mut R,
    footer: Footer,
    col_leaf_id: u32,
    // The top level column field
    top_col_field: FieldRef,
    row_id: usize,
    wasm_context: Option<Arc<WASMReadingContext<R>>>,
    shared_dictionary_cache: Option<&SharedDictionaryCache>,
) -> Result<Vec<RecordBatch>> {
    let mut record_batches = vec![];
    let shared_dictionary_cache = shared_dictionary_cache.unwrap();
    let rg_metas = footer.row_group_metadatas();
    // let projections = projections.map(|vec| vec.iter().map(|v| *v).collect::<HashSet<usize>>());
    for rg_meta in rg_metas {
        let mut column_idx = ColumnIndexSequence::default();
        let mut columns = vec![];
        // This col_decoder is unfortunately the decoder of List(Struct(_)) type.
        let mut col_decoder = create_list_struct_decoder(
            reader,
            Arc::clone(&top_col_field),
            &rg_meta.column_metadatas,
            &mut column_idx,
            wasm_context.as_ref().map(Arc::clone),
            shared_dictionary_cache,
        )?;
        let arrays = col_decoder.decode_batch_at_with_proj(col_leaf_id as usize, row_id, 1)?;
        columns.push(
            concat(
                arrays
                    .iter()
                    .map(|a| a.as_ref())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap(),
        );
        // TODO: vortex may not round-trip out the input Arrow type.
        record_batches.push(
            RecordBatch::try_new(
                Schema::new(vec![Field::new_list_field(
                    // FIXME:
                    columns[0].data_type().clone(),
                    columns[0].is_nullable(),
                )])
                .into(),
                columns,
            )
            .unwrap(),
        );
        // record_batches.push(RecordBatch::try_new(footer.schema().clone(), columns)?);
    }
    Ok(record_batches)
}

fn collect_stat_for_col(
    field: FieldRef,
    field_id: i32,
    column_metas: &Vec<fb::ColumnMetadata<'_>>,
    column_idx: &mut ColumnIndexSequence,
) -> Result<()> {
    let column_index = column_idx.next_column_index();
    let column_meta = column_metas.get(column_index as usize).unwrap();
    let chunk_size = column_meta.column_chunks().map(|chunks| {
        chunks
            .iter()
            .map(|chunk| chunk.size_() as usize)
            .sum::<usize>()
    });
    println!(
        "Field: {}, Type: {}, id: {}, Column Size: {:?}",
        field.name(),
        field.data_type(),
        field_id,
        chunk_size.unwrap()
    );
    match field.data_type() {
        non_nest_types!() => {}
        DataType::List(child) | DataType::LargeList(child) => {
            collect_stat_for_col(child.clone(), field_id, column_metas, column_idx)?;
        }
        DataType::Struct(child_fields) => {
            for field in child_fields {
                collect_stat_for_col(field.clone(), field_id, column_metas, column_idx)?;
            }
        }
        _ => todo!("Implement logical encoding for field {}", field),
    }
    Ok(())
}

/// Collect size of each physical column in the file.
/// Used for nested data exploring experiment.
pub fn collect_stats(footer: Footer) -> Result<()> {
    let rg_metas = footer.row_group_metadatas();
    // let projections = projections.map(|vec| vec.iter().map(|v| *v).collect::<HashSet<usize>>());
    for rg_meta in rg_metas {
        let mut column_idx = ColumnIndexSequence::default();
        for (field_id, field) in footer.schema().fields().iter().enumerate() {
            collect_stat_for_col(
                Arc::clone(field),
                field_id as i32,
                &rg_meta.column_metadatas,
                &mut column_idx,
            )?;
        }
    }
    Ok(())
}

/// Process Selection and grouped column metadata to produce a vector of tuples,
/// where each tuple contains a row group's metadata and its corresponding adjusted Selection.
///
/// This function handles row index adjustments when some row groups are skipped,
/// ensuring that the selection indices correctly map to the right rows.
///
/// # Arguments
/// * `selection` - The original selection criteria
/// * `grouped_metadata` - A slice of grouped column metadata
///
/// # Returns
/// A vector of tuples where each tuple contains:
/// * The row group metadata
/// * An adjusted Selection specific to that row group
pub fn process_selection<'a>(
    selection: &Selection,
    grouped_metadata: &'a [GroupedColumnMetadata<'a>],
) -> Vec<(&'a GroupedColumnMetadata<'a>, Selection)> {
    match selection {
        Selection::All => {
            // When selecting all rows, simply include all row groups with Selection::All
            grouped_metadata
                .iter()
                .map(|metadata| (metadata, Selection::All))
                .collect()
        }
        Selection::RowIndexes(row_indexes) => {
            // Early return if there are no row indexes
            if row_indexes.is_empty() {
                return vec![];
            }

            // For optimization, we'll work with sorted row indexes
            let mut sorted_indexes = row_indexes.clone();
            sorted_indexes.sort_unstable();

            let mut result = Vec::new();
            let mut cumulative_row_count = 0u64;
            let mut current_idx_pos = 0; // Position in the sorted_indexes array

            // Process each group once, advancing through the sorted indexes
            for metadata in grouped_metadata {
                let row_count = metadata.row_count as u64;
                let start_row = cumulative_row_count;
                let end_row = start_row + row_count;

                // Find all indexes that fall within this group's range
                let mut group_indexes = Vec::new();

                // Skip indexes that are below this group's range
                while current_idx_pos < sorted_indexes.len()
                    && sorted_indexes[current_idx_pos] < start_row
                {
                    current_idx_pos += 1;
                }

                // Collect indexes that fall within this group's range
                while current_idx_pos < sorted_indexes.len()
                    && sorted_indexes[current_idx_pos] < end_row
                {
                    group_indexes.push(sorted_indexes[current_idx_pos] - start_row);
                    current_idx_pos += 1;
                }

                // Only include this group if it contains at least one selected row
                if !group_indexes.is_empty() {
                    result.push((metadata, Selection::RowIndexes(group_indexes)));
                }

                cumulative_row_count = end_row;

                // Early exit if we've processed all row indexes
                if current_idx_pos >= sorted_indexes.len() {
                    break;
                }
            }

            result
        }
    }
}

fn read_postscript<R: Reader + ?Sized>(reader: &R, file_size: u64) -> Result<PostScript> {
    // read postscript from file
    let mut postscript_buffer: [u8; POSTSCRIPT_SIZE as usize] = [0; POSTSCRIPT_SIZE as usize];
    reader.read_exact_at(&mut postscript_buffer, file_size - POSTSCRIPT_SIZE)?;
    if postscript_buffer[postscript_buffer.len() - 2..] != *MAGIC {
        return Err(Error::General("Magic number incorrect".to_string()));
    }
    let metadata_size = LittleEndian::read_u32(&postscript_buffer[0..4]);
    let footer_size = LittleEndian::read_u32(&postscript_buffer[4..8]);
    let footer_compression = postscript_buffer[8];
    let checksum_type = postscript_buffer[9];
    let data_checksum = LittleEndian::read_u64(&postscript_buffer[10..18]);
    let schema_checksum = LittleEndian::read_u64(&postscript_buffer[18..26]);
    let major_version = LittleEndian::read_u16(&postscript_buffer[26..28]);
    let minor_version = LittleEndian::read_u16(&postscript_buffer[28..30]);
    Ok(PostScript {
        metadata_size,
        footer_size,
        compression: footer_compression.into(),
        checksum_type: checksum_type.into(),
        data_checksum,
        schema_checksum,
        major_version,
        minor_version,
    })
}

#[cfg(test)]
mod tests;
