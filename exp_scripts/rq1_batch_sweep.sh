#!/usr/bin/env bash
set -euo pipefail

# RQ1 evidence sweep:
# - Vary batch size (nq) and measure Wasm overhead amortization and host->wasm copy cost.
#
# Outputs:
# - results/rq1_batch_sweep/all_results.csv
#
# Notes:
# - Reuses `fff-bench/examples/bench_ivf_wasm.rs`, which supports `--build-if-missing`.
# - Builds the Wasm kernel with SIMD by default (set WASM_RUSTFLAGS to override).

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

WASM_RUSTFLAGS="${WASM_RUSTFLAGS:--C target-feature=+simd128}"
WASM_TARGET_DIR="${WASM_TARGET_DIR:-$root/target_rq1_wasm}"
NATIVE_TARGET_DIR="${NATIVE_TARGET_DIR:-}"

out_dir="${1:-$root/results/rq1_batch_sweep}"
mkdir -p "$out_dir"

echo "[1/4] Build wasm kernel (RUSTFLAGS='$WASM_RUSTFLAGS')"
RUSTFLAGS="$WASM_RUSTFLAGS" CARGO_TARGET_DIR="$WASM_TARGET_DIR" \
  cargo build -p ivf-kernel-basic --target wasm32-wasip1 --release
wasm_kernel="$WASM_TARGET_DIR/wasm32-wasip1/release/ivf_kernel_basic.wasm"
echo "wasm_kernel=$wasm_kernel"

echo "[2/4] Build bench tools"
if [[ -n "$NATIVE_TARGET_DIR" ]]; then
  CARGO_TARGET_DIR="$NATIVE_TARGET_DIR" cargo build -p fff-bench --release --example gen_f3_vectors
  CARGO_TARGET_DIR="$NATIVE_TARGET_DIR" cargo build -p fff-bench --release --example bench_ivf_wasm
  gen_vectors="$NATIVE_TARGET_DIR/release/examples/gen_f3_vectors"
  bench="$NATIVE_TARGET_DIR/release/examples/bench_ivf_wasm"
else
  cargo build -p fff-bench --release --example gen_f3_vectors
  cargo build -p fff-bench --release --example bench_ivf_wasm
  gen_vectors="$root/target/release/examples/gen_f3_vectors"
  bench="$root/target/release/examples/bench_ivf_wasm"
fi

echo "[3/4] Ensure dataset"
data_f3="$out_dir/vectors_100k_dim128.f3"
if [[ ! -f "$data_f3" ]]; then
  "$gen_vectors" --output "$data_f3" --rows 100000 --dim 128
fi

echo "[4/4] Run sweep -> $out_dir/all_results.csv"
csv="$out_dir/all_results.csv"
echo "mode,nq,nprobe,avg_latency_ms,chunks_fetched,decoded_cache_hits,decode_time_ms,compute_time_ms,host_copy_time_ms,codec" >"$csv"

# Keep parameters aligned with existing baseline runs unless overridden.
nlist="${NLIST:-1024}"
nprobe="${NPROBE:-20}"
k="${K:-10}"
dim=128
warmup="${WARMUP:-3}"
iters="${ITERS:-10}"

# Batch sizes to sweep (override with NQ_LIST="1 8 32 ...")
NQ_LIST="${NQ_LIST:-1 2 4 8 16 32 64 100}"

# Codecs to include (override with CODECS="raw raw_f16")
CODECS="${CODECS:-raw row_id_delta_varint_v1 raw_f16 row_id_delta_varint_v1_f16}"

for codec in $CODECS; do
  index_name="rq1_idx_${nlist}_${codec}"
  for nq in $NQ_LIST; do
    "$bench" \
      --base-path "$data_f3" \
      --index-name "$index_name" \
      --wasm-path "$wasm_kernel" \
      --build-if-missing \
      --build-nlist "$nlist" \
      --dim "$dim" \
      --nq "$nq" \
      --k "$k" \
      --nprobe "$nprobe" \
      --warmup "$warmup" \
      --iters "$iters" \
      --posting-codec "$codec" \
      --csv >>"$csv"
  done
done

echo "done: $csv"
