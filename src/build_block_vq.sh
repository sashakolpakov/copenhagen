#!/bin/bash
# Build the block-VQ pybind module into python/core/block_vq.so (macOS / Linux).
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
PYINC=$(python3 -m pybind11 --includes)
OUT="$ROOT/python/core/block_vq.so"
CXX=${CXX:-g++}
COMMON="-O3 -march=native -shared -std=c++17 -fPIC $PYINC"

case "$(uname -s)" in
  Darwin) $CXX $COMMON block_quant_py.cpp -o "$OUT" -undefined dynamic_lookup ;;
  *)      $CXX $COMMON block_quant_py.cpp -o "$OUT" ;;
esac
echo "built: $OUT"
