#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

mode="${1:-}"
if [[ "$mode" != "portable" && "$mode" != "simd" ]]; then
  echo "usage: $0 <portable|simd> [out_dir]" >&2
  exit 2
fi

sift_dir="${SIFT_DIR:-$root/data/sift}"
out_dir="${2:-$root/results/sift_ivf_aligned_${mode}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$out_dir"

if [[ "$mode" == "portable" ]]; then
  # Portable baseline: default compiler flags for both native and wasm (no simd128 requirement).
  rustflags_native=""
  rustflags_wasm=""
else
  # Optimized mode: native uses host CPU; wasm enables simd128 if supported by runtime.
  rustflags_native="-C target-cpu=native"
  rustflags_wasm="-C target-feature=+simd128"
fi

cat >"$out_dir/build_config.txt" <<EOF
mode=$mode
rustflags_native=$rustflags_native
rustflags_wasm=$rustflags_wasm
EOF

echo "[1/4] Ensure SIFT100K + base F3"
# Avoid invoking the downloader automatically here (it can be flaky in some environments).
# If you don't have SIFT already, run: `bash scripts/download_sift_100k.sh "$sift_dir"`.
if [[ ! -d "$sift_dir" ]]; then
  mkdir -p "$sift_dir"
fi
base_f3="$sift_dir/sift100k.f3"
if [[ ! -f "$base_f3" ]]; then
  if [[ ! -f "$sift_dir/sift.tar.gz" ]]; then
    echo "missing SIFT data at: $sift_dir (expected sift.tar.gz or sift100k.f3)" >&2
    echo "run: bash scripts/download_sift_100k.sh \"$sift_dir\"" >&2
    exit 1
  fi
  CARGO_TARGET_DIR="$root/target_fair_${mode}_native" RUSTFLAGS="$rustflags_native" \
    cargo run -p fff-bench --release --example sift_build_f3 -- \
      --sift-dir "$sift_dir" --max-vectors 100000 --out "$base_f3"
fi

echo "[2/4] Build Wasm kernel ($mode)"
CARGO_TARGET_DIR="$root/target_fair_${mode}_wasm" RUSTFLAGS="$rustflags_wasm" \
  cargo build -p ivf-kernel-basic --target wasm32-wasip1 --release
wasm_kernel="$root/target_fair_${mode}_wasm/wasm32-wasip1/release/ivf_kernel_basic.wasm"

echo "[3/4] Run aligned matrix ($mode) -> $out_dir"

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
  --nq 32
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
  CARGO_TARGET_DIR="$root/target_fair_${mode}_native" RUSTFLAGS="$rustflags_native" \
    cargo run -p fff-bench --release --example vector_ivf_flat_demo -- \
      "${common_args[@]}" "$@" \
      | tee "$out" >/dev/null
}

# Notes:
# - Warm: decoded caches enabled (native decoded cache + wasm decoded cache) and host chunk cache enabled.
# - Cold: decoded caches disabled and host chunk cache disabled.

# f32 (raw)
run_case "native_f32_warm" --artifact --artifact-posting-codec raw \
  --artifact-native-decoded-cache=true --artifact-native-clear-each-iter=false
run_case "native_f32_cold" --artifact --artifact-posting-codec raw \
  --artifact-native-decoded-cache=false --artifact-native-clear-each-iter=false

run_case "wasm_f32_warm" --artifact-wasm-kernel "$wasm_kernel" --artifact-posting-codec raw \
  --artifact-wasm-decoded-cache-bytes 201326592
run_case "wasm_f32_cold" --artifact-wasm-kernel "$wasm_kernel" --artifact-posting-codec raw \
  --artifact-wasm-no-cache --artifact-wasm-decoded-cache-bytes 0

# f16 (raw_f16)
run_case "native_f16_warm" --artifact --artifact-posting-codec raw_f16 \
  --artifact-native-decoded-cache=true --artifact-native-clear-each-iter=false
run_case "native_f16_cold" --artifact --artifact-posting-codec raw_f16 \
  --artifact-native-decoded-cache=false --artifact-native-clear-each-iter=false

run_case "wasm_f16_warm" --artifact-wasm-kernel "$wasm_kernel" --artifact-posting-codec raw_f16 \
  --artifact-wasm-decoded-cache-bytes 201326592
run_case "wasm_f16_cold" --artifact-wasm-kernel "$wasm_kernel" --artifact-posting-codec raw_f16 \
  --artifact-wasm-no-cache --artifact-wasm-decoded-cache-bytes 0

echo "[4/4] Summary"
python3 "$root/scripts/summarize_ivf_jsonl.py" "$out_dir"
echo "results: $out_dir"
