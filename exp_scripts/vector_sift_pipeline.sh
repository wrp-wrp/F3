#!/usr/bin/env bash
# Runs the vector index recall pipeline against the SIFT1M dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sift1m}"
TRAIN_SIZE="${TRAIN_SIZE:-20000}"
QUERY_SIZE="${QUERY_SIZE:-100}"
K="${K:-5}"
ALGORITHM="${ALGORITHM:-wasm}"
HNSW_M="${HNSW_M:-48}"
HNSW_EF_SEARCH="${HNSW_EF_SEARCH:-200}"
OUTPUT_BLOB="${OUTPUT_BLOB:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
COMPARE_NATIVE="${COMPARE_NATIVE:-1}"
WASM_MODULE="${WASM_MODULE:-${REPO_ROOT}/target/wasm32-wasip1/release/vector_hnsw_wasm.wasm}"
ARCHIVE_URL="${ARCHIVE_URL:-ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz}"

mkdir -p "${DATA_DIR}"

SIFT_BASE="${DATA_DIR}/sift_base.fvecs"
SIFT_QUERY="${DATA_DIR}/sift_query.fvecs"
SIFT_GT="${DATA_DIR}/sift_groundtruth.ivecs"

if [[ ! -f "${SIFT_BASE}" || ! -f "${SIFT_QUERY}" || ! -f "${SIFT_GT}" ]]; then
  echo "[vector-sift] Downloading SIFT1M archive..."
  TMP_ARCHIVE="$(mktemp "${DATA_DIR}/sift.XXXXXX.tar.gz")"
  curl -L "${ARCHIVE_URL}" -o "${TMP_ARCHIVE}"
  TMP_DIR="$(mktemp -d "${DATA_DIR}/sift_tmp.XXXXXX")"
  tar -xzf "${TMP_ARCHIVE}" -C "${TMP_DIR}"
  if [[ -d "${TMP_DIR}/sift" ]]; then
    mv "${TMP_DIR}/sift/"* "${DATA_DIR}/"
  else
    mv "${TMP_DIR}/"* "${DATA_DIR}/"
  fi
  rm -rf "${TMP_DIR}" "${TMP_ARCHIVE}"
fi

if [[ ! -f "${SIFT_BASE}" || ! -f "${SIFT_QUERY}" || ! -f "${SIFT_GT}" ]]; then
  echo "SIFT files are still missing under ${DATA_DIR}"
  exit 1
fi

if [[ "${ALGORITHM}" == "wasm" ]]; then
  rustup target add wasm32-wasip1 >/dev/null 2>&1 || true
  echo "[vector-sift] Building vector-hnsw Wasm module..."
  (cd "${REPO_ROOT}" && cargo build -p vector-hnsw-wasm --release --target wasm32-wasip1 >/dev/null)
  if [[ ! -f "${WASM_MODULE}" ]]; then
    echo "Failed to find Wasm module at ${WASM_MODULE}"
    exit 1
  fi
fi

CMD=(cargo run --release -p fff-poc --example vector_sift_pipeline -- --dataset-dir "${DATA_DIR}" --train-size "${TRAIN_SIZE}" --query-size "${QUERY_SIZE}" --k "${K}" --algorithm "${ALGORITHM}" --hnsw-m "${HNSW_M}" --hnsw-ef-search "${HNSW_EF_SEARCH}")

if [[ -n "${OUTPUT_BLOB}" ]]; then
  CMD+=(--output-blob "${OUTPUT_BLOB}")
fi

if [[ "${ALGORITHM}" == "wasm" ]]; then
  CMD+=(--wasm-module "${WASM_MODULE}")
  if [[ "${COMPARE_NATIVE}" == "1" ]]; then
    CMD+=(--compare-native)
  fi
fi

if [[ -n "${OUTPUT_FILE}" ]]; then
  CMD+=(--output-file "${OUTPUT_FILE}")
fi

echo "[vector-sift] Running: ${CMD[*]}"
pushd "${REPO_ROOT}" >/dev/null
"${CMD[@]}"
popd >/dev/null
