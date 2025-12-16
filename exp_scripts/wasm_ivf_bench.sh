#!/bin/bash
set -e

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

# Build WASM Kernel
echo "Building WASM Kernel..."
cd wasm-libs/ivf-kernel-basic
cargo build --target wasm32-wasip1 --release
cd "$PROJECT_ROOT"

# Build Tools
echo "Building Bench Tools..."
cargo build -p fff-bench --example gen_f3_vectors --release
cargo build -p fff-bench --example bench_ivf_wasm --release

# Generate Data
BASE_DIR="results/ivf_wasm_bench"
mkdir -p "$BASE_DIR"
DATA_FILE="$BASE_DIR/vectors.f3"
WASM_FILE="target/wasm32-wasip1/release/ivf_kernel_basic.wasm"

if [ ! -f "$DATA_FILE" ]; then
    echo "Generating vectors (100k, dim=128)..."
    ./target/release/examples/gen_f3_vectors --output "$DATA_FILE" --rows 100000 --dim 128
else
    echo "Using existing vectors at $DATA_FILE"
fi

# Define codecs to test
codecs=("raw" "row_id_delta_varint_v1" "raw_f16" "row_id_delta_varint_v1_f16")

# Run Benchmark for each codec
echo "mode,nq,nprobe,avg_latency_ms,chunks_fetched,decoded_cache_hits,decode_time_ms,compute_time_ms,codec" > "$BASE_DIR/all_bench_results.csv"

for codec in "${codecs[@]}"; do
    echo "Running Benchmark for codec: $codec"
    INDEX_NAME="idx_ivf_1024_${codec}" # Unique index name per codec
    ./target/release/examples/bench_ivf_wasm \
        --base-path "$DATA_FILE" \
        --index-name "$INDEX_NAME" \
        --wasm-path "$WASM_FILE" \
        --build-if-missing \
        --build-nlist 1024 \
        --dim 128 \
        --nq 100 \
        --k 10 \
        --nprobe 20 \
        --posting-codec "$codec" \
        --csv >> "$BASE_DIR/all_bench_results.csv"
done
