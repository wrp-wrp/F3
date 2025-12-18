#!/usr/bin/env bash
set -euo pipefail

# P0: minimal experiment matrix (codec × cache × nq × nprobe) with JSONL output.
#
# Default output:
# - results/p0_min_matrix/runs.jsonl
#
# This script intentionally uses a synthetic dataset to keep it reproducible.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

out_dir="${1:-$root/results/p0_min_matrix}"
mkdir -p "$out_dir"

WASM_RUSTFLAGS="${WASM_RUSTFLAGS:--C target-feature=+simd128}"
WASM_TARGET_DIR="${WASM_TARGET_DIR:-$root/target_p0_matrix_wasm}"

echo "[1/4] Build wasm kernel (RUSTFLAGS='$WASM_RUSTFLAGS')"
RUSTFLAGS="$WASM_RUSTFLAGS" CARGO_TARGET_DIR="$WASM_TARGET_DIR" \
  cargo build -p ivf-kernel-basic --target wasm32-wasip1 --release
wasm_kernel="$WASM_TARGET_DIR/wasm32-wasip1/release/ivf_kernel_basic.wasm"

echo "[2/4] Build tools"
cargo build -p fff-bench --release --example gen_f3_vectors
cargo build -p fff-bench --release --example vector_ivf_flat_demo
gen_vectors="$root/target/release/examples/gen_f3_vectors"
demo="$root/target/release/examples/vector_ivf_flat_demo"

echo "[3/4] Ensure dataset"
data_f3="$out_dir/vectors_100k_dim128.f3"
if [[ ! -f "$data_f3" ]]; then
  "$gen_vectors" --output "$data_f3" --rows 100000 --dim 128
fi

echo "[4/4] Run matrix"
jsonl="$out_dir/runs.jsonl"
: >"$jsonl"

dim=128
leaf=0
nlist="${NLIST:-1024}"
train_sample="${TRAIN_SAMPLE:-51200}"
k="${K:-10}"
repeat="${REPEAT:-5}"
warmup="${WARMUP:-1}"
recall_queries="${RECALL_QUERIES:-10}"

CODECS="${CODECS:-raw raw_f16}"
NQ_LIST="${NQ_LIST:-1 8 100}"
NPROBE_LIST="${NPROBE_LIST:-5 20}"

for codec in $CODECS; do
  for nq in $NQ_LIST; do
    for nprobe in $NPROBE_LIST; do
      index_name="p0_${nlist}_${codec}"
      declare -a wasm_cache_flags=()

      # Native artifact (decoded cache on/off)
      for decoded_cache in true false; do
        "$demo" \
          --base-f3 "$data_f3" \
          --vector-leaf-index "$leaf" \
          --dim "$dim" \
          --index-name "$index_name" \
          --nlist "$nlist" \
          --train-sample "$train_sample" \
          --k "$k" \
          --nq "$nq" \
          --nprobe "$nprobe" \
          --artifact \
          --artifact-posting-codec "$codec" \
          --artifact-native-decoded-cache "$decoded_cache" \
          --warmup "$warmup" \
          --repeat "$repeat" \
          --json \
          --quiet \
          --profile-stages \
          --recall \
          --recall-queries "$recall_queries" \
          >>"$jsonl"
      done

      # Wasm artifact (host chunk cache on/off, host-dist on/off)
      for wasm_cache in true false; do
        for host_dist in false true; do
          wasm_cache_flags=()
          if [[ "$wasm_cache" != true ]]; then
            wasm_cache_flags+=(--artifact-wasm-no-cache)
          fi
          if [[ "$host_dist" == true ]]; then
            wasm_cache_flags+=(--artifact-wasm-host-dist)
          fi
          "$demo" \
            --base-f3 "$data_f3" \
            --vector-leaf-index "$leaf" \
            --dim "$dim" \
            --index-name "$index_name" \
            --nlist "$nlist" \
            --train-sample "$train_sample" \
            --k "$k" \
            --nq "$nq" \
            --nprobe "$nprobe" \
            --artifact-wasm-kernel "$wasm_kernel" \
            --artifact-posting-codec "$codec" \
            ${wasm_cache_flags[@]+"${wasm_cache_flags[@]}"} \
            --warmup "$warmup" \
            --repeat "$repeat" \
            --json \
            --quiet \
            --profile-stages \
            --recall \
            --recall-queries "$recall_queries" \
            >>"$jsonl"
        done
      done
    done
  done
done

echo "done: $jsonl"
