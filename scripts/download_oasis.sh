#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://download.nrg.wustl.edu/data"
OUT_DIR="${1:-data/OASIS}"

mkdir -p "${OUT_DIR}"

for i in {1..12}; do
  fname="oasis_cross-sectional_disc${i}.tar.gz"
  url="${BASE_URL}/${fname}"
  echo "Downloading ${url} -> ${OUT_DIR}/${fname}"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "${OUT_DIR}/${fname}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "${OUT_DIR}/${fname}" "${url}"
  else
    echo "Error: need wget or curl" >&2
    exit 1
  fi
done

echo "Done."
