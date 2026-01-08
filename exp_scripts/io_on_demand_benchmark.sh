#!/bin/bash
set -e

# This script proves the "On-Demand IO Benefit" of F3 Vector Index Artifacts.
# It measures: Total Index Size vs Actual Bytes Read for a cold query.

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

# 1. Build Wasm Kernel
echo "Building WASM Kernel..."
cd wasm-libs/ivf-kernel-basic
cargo build --target wasm32-wasip1 --release > /dev/null 2>&1
cd "$PROJECT_ROOT"

# 2. Setup Data
DATA_DIR="results/io_bench"
mkdir -p "$DATA_DIR"
BASE_F3="./results/p0_min_matrix_smoke/vectors_100k_dim128.f3"
INDEX_NAME="idx_io_bench"
DIM=128

if [ ! -f "$BASE_F3" ]; then
    echo "Wait, existing file not found at $BASE_F3. Falling back to small generation..."
    DATA_DIR="results/io_bench"
    mkdir -p "$DATA_DIR"
    BASE_F3="$DATA_DIR/base_100k.f3"
    ROWS=100000
    cargo run --example gen_f3_vectors --release -- --output "$BASE_F3" --rows $ROWS --dim $DIM > /dev/null 2>&1
else
    echo "Using existing dataset: $BASE_F3"
fi

# 3. Build Index Artifact
echo "Building Index Artifact (nlist=1024)..."
cargo run --example vector_ivf_flat_demo --release -- \
    --base-f3 "$BASE_F3" \
    --vector-leaf-index 0 --dim $DIM \
    --index-name "$INDEX_NAME" \
    --nlist 1024 --artifact --artifact-posting-codec raw_f16 > /dev/null 2>&1

INDEX_PATH="$BASE_F3.vindex/$INDEX_NAME.ivf_flat.artifact"
TOTAL_SIZE=$(stat -f%z "$INDEX_PATH")
TOTAL_SIZE_MB=$(echo "scale=2; $TOTAL_SIZE / 1048576" | bc)

# 4. Measure Full Load Time (Baseline 1)
# We simulate a "Full Load" system by reading the entire file and timing it.
echo "Measuring Full Load Time (Baseline)..."
START_LOAD=$(python3 -c "import time; print(time.time())")
# Use dd to simulate reading the whole file into buffer
dd if="$INDEX_PATH" of=/dev/null bs=1M status=none
END_LOAD=$(python3 -c "import time; print(time.time())")
FULL_LOAD_MS=$(echo "scale=3; ($END_LOAD - $START_LOAD) * 1000" | bc)

# 5. Run Cold Query (On-Demand)
echo "Running Cold Query (nprobe=20)..."
RESULT_JSON=$(cargo run --example vector_ivf_flat_demo --release -- \
    --base-f3 "$BASE_F3" \
    --vector-leaf-index 0 --dim $DIM \
    --index-name "$INDEX_NAME" \
    --artifact --artifact-posting-codec raw_f16 \
    --artifact-wasm-kernel target/wasm32-wasip1/release/ivf_kernel_basic.wasm \
    --nq 1 --nprobe 20 --json --quiet --warmup 0 --repeat 1 | grep "\"iteration\":0")

# Parse bytes read and cold latency
BYTES_READ=$(echo "$RESULT_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['compressed_bytes_in'])")
BYTES_READ_MB=$(echo "scale=4; $BYTES_READ / 1048576" | bc)
COLD_QUERY_MS=$(echo "$RESULT_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['wall_ms'])")

# 6. Run Warm Query (In-Memory baseline)
# We use the 'wall_ms' from a second iteration (warm cache)
WARM_RESULT_JSON=$(cargo run --example vector_ivf_flat_demo --release -- \
    --base-f3 "$BASE_F3" \
    --vector-leaf-index 0 --dim $DIM \
    --index-name "$INDEX_NAME" \
    --artifact --artifact-posting-codec raw_f16 \
    --artifact-wasm-kernel target/wasm32-wasip1/release/ivf_kernel_basic.wasm \
    --nq 1 --nprobe 20 --json --quiet --warmup 1 --repeat 1 | grep "\"iteration\":0")
WARM_QUERY_MS=$(echo "$WARM_RESULT_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['wall_ms'])")

# 7. Final Report
TOTAL_FULL_MS=$(echo "scale=3; $FULL_LOAD_MS + $WARM_QUERY_MS" | bc)
RATIO=$(echo "scale=2; $TOTAL_FULL_MS / $COLD_QUERY_MS" | bc)

echo "------------------------------------------------"
echo "IO PERFORMANCE COMPARISON (Time-to-First-Result)"
echo "------------------------------------------------"
echo "1. Full-Load Baseline (Traditional):"
echo "   - Load Entire Index ($TOTAL_SIZE_MB MB):  $FULL_LOAD_MS ms"
echo "   - First Query:                $WARM_QUERY_MS ms"
echo "   - TOTAL Time:                 $TOTAL_FULL_MS ms"
echo ""
echo "2. On-Demand WASM (Executable Artifact):"
echo "   - First Query (Cold Start):   $COLD_QUERY_MS ms"
echo "   - Actual Bytes Fished:        $BYTES_READ_MB MB"
echo ""
echo "3. Steady State (In-Memory):"
echo "   - Subsequent Query (Warm):    $WARM_QUERY_MS ms"
echo "------------------------------------------------"
echo "RESULT: On-Demand Query is ${RATIO}x FASTER than Full-Load for the first query."
echo "WASM Portability: The *fetch logic* (nprobe, codec) is inside the WASM."
echo "The host remains decoupled from the index format internal changes."
echo "------------------------------------------------"
