#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

sift_dir="${SIFT_DIR:-$root/data/sift}"
out_dir="${1:-$root/results/sift_ivf_aligned_profile_strict_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$out_dir"

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
  cargo run -p fff-bench --release --example sift_build_f3 -- \
    --sift-dir "$sift_dir" --max-vectors 100000 --out "$base_f3"
fi

echo "[2/4] Build Wasm kernel"
WASM_RUSTFLAGS="${WASM_RUSTFLAGS:-}"
if [[ -n "$WASM_RUSTFLAGS" ]]; then
  echo "WASM_RUSTFLAGS=$WASM_RUSTFLAGS"
fi
wasm_target_dir="${WASM_TARGET_DIR:-$root/target_profile_strict_wasm}"
RUSTFLAGS="$WASM_RUSTFLAGS" CARGO_TARGET_DIR="$wasm_target_dir" \
  cargo build -p ivf-kernel-basic --target wasm32-wasip1 --release
wasm_kernel="$wasm_target_dir/wasm32-wasip1/release/ivf_kernel_basic.wasm"

echo "[3/4] Run strict stage-profile matrix -> $out_dir"

NATIVE_RUSTFLAGS="${NATIVE_RUSTFLAGS:-}"
if [[ -n "$NATIVE_RUSTFLAGS" ]]; then
  echo "NATIVE_RUSTFLAGS=$NATIVE_RUSTFLAGS"
fi
cat >"$out_dir/build_config.txt" <<EOF
NATIVE_RUSTFLAGS=$NATIVE_RUSTFLAGS
WASM_RUSTFLAGS=$WASM_RUSTFLAGS
WASM_TARGET_DIR=$wasm_target_dir
EOF

echo "[3a/4] Build native runner"
native_target_dir="${NATIVE_TARGET_DIR:-$root/target_profile_strict_native}"
RUSTFLAGS="$NATIVE_RUSTFLAGS" CARGO_TARGET_DIR="$native_target_dir" \
  cargo build -p fff-bench --release --example vector_ivf_flat_demo
native_bin="$native_target_dir/release/examples/vector_ivf_flat_demo"
if [[ ! -x "$native_bin" ]]; then
  echo "native runner not found: $native_bin" >&2
  exit 1
fi

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
  --profile-stages
)

run_case() {
  local name="$1"
  shift
  local out="$out_dir/$name.jsonl"
  echo "case: $name -> $out"
  "$native_bin" "${common_args[@]}" "$@" \
    | tee "$out" >/dev/null
}

# Strict alignment notes for profiling:
# - Stage profiling adds overhead; we still keep cold/warm knobs identical to `run_sift_ivf_aligned.sh`.
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

echo "[4/4] Summaries"
python3 "$root/scripts/summarize_ivf_jsonl.py" "$out_dir"
echo "results: $out_dir"
