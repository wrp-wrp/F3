# LakeVector Detailed Design: Integration with F3 Architecture

**Status**: Draft
**Target**: `fff` (Future-proof File Format) Ecosystem

---

## 1. Architectural Overview

LakeVector is a set of extensions and conventions built on top of F3, designed to minimize S3 costs and latency for vector retrieval. It is **not** a separate file format, but a specific **configuration** of F3 involving:
1.  **Layout**: Strict physical ordering of data (Sort/Z-Order).
2.  **Metadata**: Custom sections in the F3 Footer (Grid Bitmap).
3.  **Codecs**: Specialized Wasm-based UDEs that implement "Index-as-Codec".

### System Components

*   **LakeVector Writer (Client-side)**: Responsible for `Vector Quantization`, `Scalar Sorting`, `Bitmap Generation`, and `F3 Packaging`.
*   **F3 Storage (S3)**: Standard, static F3 files. No sidecar indices.
*   **LakeVector Reader (Client-side)**: A "Smart Client" that reads F3 Footers first, performs zero-I/O pruning, and then requests specific file ranges.

---

## 2. File Format Specification (F3 Mappings)

We map LakeVector constructs to standard F3 FlatBuffer definitions (`format/File.fbs`).

### 2.1 Global Grid Bitmap -> `Footer.OptionalMetadataSections`

The **Global Grid Bitmap** is a coarse-grained filter used for avoiding complete S3 GET requests.

*   **Location**: `Footer.optional_sections`
*   **Key**: `"LAKEVECTOR_GRID_V1"`
*   **Content**: A serialized compressed bitmap (RoaringBitmap).
    *   *Logical Structure*: A matrix where rows = Cluster IDs, columns = Time Bins (or other primary scalar).
    *   *Physical Structure*: `[ Magic (4B) | N_Rows (4B) | N_Cols (4B) | Compressed_Bitmap_Bytes ]`

### 2.2 Precise Interval Bitmap -> `EncUnit` Metadata

The **Precise Interval Bitmap** allows skipping decoding of specific `EncUnit`s (blocks of 64k rows) even if the file is downloaded.

*   **Location**: Inside the custom `WASMEncoding` header or explicitly passed as `kwargs` to the UDE Init function.
*   **Mechanism**: The `EncUnit` encoding type is set to `CUSTOM_WASM` (or a specific registered ID). The `wasm_args` contain the bitmap.

### 2.3 Neural Compressed Data -> `EncUnit` Payload

The actual vector data is compressed using a neural codec (AutoEncoder/Product Quantization).

*   **Location**: `Chunk.encunits[i].data`
*   **EncodingType**: `CUSTOM_WASM`
*   **Wasm Binary**: Stored in `Footer.WASMBinaries`.

---

## 3. The "Smart Codec" Interface (fff-ude)

We utilize `fff-ude::StatefulWasmDecoder` to implement the "Index-as-Codec" logic.

### 3.1 Interface Definition

```rust
// Pseudo-code implementation of the LakeVector Codec
impl StatefulWasmDecoder for LakeVectorDecoder {
    fn decode(&mut self) -> Result<Option<Box<dyn Iterator<Item = Buffer>>>> {
        // Step 1: Check internal bitmap (Precise Pruning)
        if self.should_skip_current_batch() {
            return Ok(Some(Box::new(std::iter::empty()))); // Return empty, skipping decode
        }
        
        // Step 2: Decode Neural/PQ data
        let decoded_vectors = self.run_neural_decompression()?;
        
        Ok(Some(Box::new(std::iter::once(decoded_vectors))))
    }
}
```

### 3.2 Wasm Interaction
1.  **Reader** loads the Wasm binary from `Footer`.
2.  **Reader** calls `Init(input_buffer, kwargs)`.
    *   `input_buffer`: The compressed vector data (Latent Codes).
    *   `kwargs`: The Precise Bitmap + Query Predicates (passed from the query engine).
3.  **Wasm Module**:
    *   Parses `kwargs` to get the Query Range (e.g., `time > T1`).
    *   Checks `Precise Bitmap` against Query Range.
    *   If Intersection is Empty -> Returns Empty Iterator immediately.
    *   Else -> Decompresses Latent Codes to Float32 vectors.

---

## 4. Implementation Workflows

### 4.1 Writer Workflow (Ingestion)

1.  **Buffer & Cluster**: Accumulate vectors in memory. Run IVF clustering (K-Means) to assign Cluster IDs.
2.  **Sort**: Within each Cluster, sort data by the Primary Scalar (e.g., Timestamp).
3.  **Generate Bitmaps**:
    *   Build `GridBitmap` (Global view).
    *   Build `PreciseBitmap` (Per-EncUnit view).
4.  **Compress**: Run Neural Encoder (floats -> latents).
5.  **Write F3**:
    *   Write `EncUnits` (Latents).
    *   Write `Footer` (Schema + GridBitmap + Wasm Binary).

### 4.2 Reader Workflow (Query)

1.  **Footer Fetch**: `GET` the last ~64KB of the file (Footer).
2.  **Zero-IO Pruning**:
    *   Deserialize `GridBitmap`.
    *   Check `Query(Time) AND Query(ClusterID)` against Grid.
    *   If bit is 0 -> **STOP**. (No further S3 costs).
3.  **Range Request**:
    *   Identify which `RowGroups` / `Chunks` survive the filter.
    *   Issue async `GET` requests for those specific byte ranges.
4.  **Smart Decode**:
    *   Pass `Query Predicates` into the Wasm Decoder via `kwargs`.
    *   Decoder locally prunes `EncUnits` based on `PreciseBitmap`.
    *   Decoder expands surviving vectors.
5.  **Final Verify**: Compute exact distances on returned vectors.

---

## 5. Directory Structure & Modules

We will add the following components to the F3 repository:

*   `lakevector/`: Root for strictly LakeVector-related logic (outside core F3).
    *   `src/writer_extensions.rs`: Helpers for Sorting/Clustering.
    *   `src/bitmap.rs`: RoaringBitmap implementations.
*   `wasm-libs/lakevector-codec/`: The Rust source for the Wasm binary.
    *   Implements `fff_ude::StatefulWasmDecoder`.
    *   Compiles to `lakevector_codec.wasm`.
*   `fff-bench/examples/lakevector_bench.rs`: Specific benchmark runner.

---

## 6. Key Advantages over Standard F3

1.  **Serverless Filter Pushdown**: Filtering logic runs *inside* the codec, driven by the Wasm binary.
2.  **S3-Aware Layout**: Unlike standard Parquet/F3 which might just be columnar, LakeVector enforces *semantic* sorting (IVF + Time) to maximize S3 read contiguity.
3.  **Static Evolution**: We can update the indexing logic (Bitmap format, pruning rules) by shipping a new Wasm binary in the file, without changing the reader code.
