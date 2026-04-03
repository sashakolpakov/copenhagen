#!/bin/bash
# Copenhagen: Quantum-Inspired Dynamic Index with BLAS acceleration

cd "$(dirname "$0")"

PYBIND11_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

OUTPUT="/Users/sasha/copenhagen/python/core/copenhagen.so"

g++ -O3 -march=armv8-a -shared -std=c++17 -fPIC -DUSE_ACCELERATE \
    -I"$PYBIND11_INC" \
    -I"$PYTHON_INC" \
    dynamic_ivf.cpp -o "$OUTPUT" \
    -framework Accelerate \
    -undefined dynamic_lookup

echo "Copenhagen index built: $OUTPUT"
