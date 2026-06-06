#!/bin/bash
# Build the block-VQ pybind module into python/core/block_vq.so
cd "$(dirname "$0")"
PYINC=$(python3 -m pybind11 --includes)
OUT="$(cd .. && pwd)/python/core/block_vq.so"
c++ -O3 -march=native -shared -std=c++17 -fPIC $PYINC \
    block_quant_py.cpp -o "$OUT" -undefined dynamic_lookup
echo "built: $OUT"
