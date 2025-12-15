#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

sift_dir="${SIFT_DIR:-$root/data/sift}"
out_dir="${1:-$root/results/sift_ivf_artifact_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$out_dir"

echo "[1/5] Download SIFT100K into: $sift_dir"
bash "$root/scripts/download_sift_100k.sh" "$sift_dir"

base_f3="$sift_dir/sift100k.f3"
echo "[2/5] Build base F3: $base_f3"
if [[ ! -f "$base_f3" ]]; then
  cargo run -p fff-bench --release --example sift_build_f3 -- \
    --sift-dir "$sift_dir" --max-vectors 100000 --out "$base_f3"
else
  echo "exists: $base_f3"
fi

echo "[3/5] Build Wasm kernel"
cargo build -p ivf-kernel-basic --target wasm32-wasip1 --release
wasm_kernel="$root/target/wasm32-wasip1/release/ivf_kernel_basic.wasm"
echo "wasm: $wasm_kernel"

echo "[4/5] Run experiment matrix -> $out_dir"

common_args=(
  --base-f3 "$base_f3"
  --vector-leaf-index 0
  --dim 128
  --index-name sift_ivf
  --nlist 64
  --train-sample 5000
  --max-kmeans-iters 5
  --seed 1
  --k 10
  --nprobe 16
  --warmup 2
  --repeat 10
  --json
  --quiet
)

run_case() {
  local name="$1"
  shift
  local out="$out_dir/$name.jsonl"
  echo "case: $name -> $out"
  cargo run -p fff-bench --release --example vector_ivf_flat_demo -- \
    "${common_args[@]}" "$@" \
    | tee "$out" >/dev/null
}

# Baselines
run_case "native_sidecar_nq1" --nq 1
run_case "native_sidecar_nq32" --nq 32

run_case "native_artifact_raw_nq32" --artifact --artifact-posting-codec raw --nq 32
run_case "native_artifact_delta_nq32" --artifact --artifact-posting-codec row_id_delta_varint_v1 --nq 32
run_case "native_artifact_raw_f16_nq32" --artifact --artifact-posting-codec raw_f16 --nq 32
run_case "native_artifact_delta_f16_nq32" --artifact --artifact-posting-codec row_id_delta_varint_v1_f16 --nq 32

# Wasm (cache on/off, raw/delta, nq 1 vs 32)
run_case "wasm_artifact_raw_cache_nq32" \
  --artifact-wasm-kernel "$wasm_kernel" --artifact-posting-codec raw --nq 32
run_case "wasm_artifact_delta_cache_nq32" \
  --artifact-wasm-kernel "$wasm_kernel" --artifact-posting-codec row_id_delta_varint_v1 --nq 32
run_case "wasm_artifact_raw_f16_cache_nq32" \
  --artifact-wasm-kernel "$wasm_kernel" --artifact-posting-codec raw_f16 --nq 32
run_case "wasm_artifact_delta_f16_cache_nq32" \
  --artifact-wasm-kernel "$wasm_kernel" --artifact-posting-codec row_id_delta_varint_v1_f16 --nq 32
run_case "wasm_artifact_delta_nocache_nq32" \
  --artifact-wasm-kernel "$wasm_kernel" --artifact-posting-codec row_id_delta_varint_v1 --artifact-wasm-no-cache --nq 32
run_case "wasm_artifact_delta_cache_nq1" \
  --artifact-wasm-kernel "$wasm_kernel" --artifact-posting-codec row_id_delta_varint_v1 --nq 1

echo "[5/5] Done."
echo "results: $out_dir"
