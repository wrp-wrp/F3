#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${1:-$root/data/sift}"

mkdir -p "$out_dir"

# Source listed from http://corpus-texmex.irisa.fr/ (TEXMEX / IRISA).
base_url="ftp://ftp.irisa.fr/local/texmex/corpus"

download() {
  local rel="$1"
  local dst="$2"
  if [[ -f "$dst" ]]; then
    echo "exists: $dst"
    return 0
  fi
  echo "downloading: $base_url/$rel -> $dst"
  curl -L --fail --retry 5 --retry-delay 2 -o "$dst.part" "$base_url/$rel"
  mv "$dst.part" "$dst"
}

archive="$out_dir/sift.tar.gz"
download "sift.tar.gz" "$archive"

# Extract only the pieces we need for a 100K run.
tar -xzf "$archive" -C "$out_dir" \
  sift/sift_learn.fvecs \
  sift/sift_query.fvecs \
  sift/sift_groundtruth.ivecs

# Convenience names expected by tests.
ln -sf "$out_dir/sift/sift_learn.fvecs" "$out_dir/learn.fvecs"
ln -sf "$out_dir/sift/sift_query.fvecs" "$out_dir/query.fvecs"
ln -sf "$out_dir/sift/sift_groundtruth.ivecs" "$out_dir/groundtruth.ivecs"

echo "done: $out_dir"
