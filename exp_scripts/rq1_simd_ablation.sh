#!/usr/bin/env bash
set -euo pipefail

# RQ1 SIMD ablation:
# - Run the same batch sweep twice:
#   1) wasm kernel compiled with +simd128
#   2) wasm kernel compiled without simd128
#
# Outputs:
# - results/rq1_simd_ablation/simd/all_results.csv
# - results/rq1_simd_ablation/nosimd/all_results.csv
#
# Then summarize:
# - python3 scripts/compare_rq1_simd_ablation.py results/rq1_simd_ablation

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

base_out="${1:-$root/results/rq1_simd_ablation}"

echo "[1/3] Run SIMD build sweep"
mkdir -p "$base_out/simd"
WASM_RUSTFLAGS="-C target-feature=+simd128" \
WASM_TARGET_DIR="$root/target_rq1_simd_wasm" \
NATIVE_TARGET_DIR="" \
CODECS="${CODECS:-raw}" \
bash "$root/exp_scripts/rq1_batch_sweep.sh" "$base_out/simd"

echo "[2/3] Run no-SIMD build sweep"
mkdir -p "$base_out/nosimd"
WASM_RUSTFLAGS="-C target-feature=-simd128" \
WASM_TARGET_DIR="$root/target_rq1_nosimd_wasm" \
NATIVE_TARGET_DIR="" \
CODECS="${CODECS:-raw}" \
bash "$root/exp_scripts/rq1_batch_sweep.sh" "$base_out/nosimd"

echo "[3/3] Done"
echo "run: python3 scripts/compare_rq1_simd_ablation.py \"$base_out\""
