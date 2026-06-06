#!/bin/bash
# Copenhagen — build the dynamic IVF C++ extension (portable: macOS / Linux).
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

PYBIND11_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
OUTPUT="$ROOT/python/core/copenhagen.so"

CXX=${CXX:-g++}
COMMON="-O3 -march=native -shared -std=c++17 -fPIC -I$PYBIND11_INC -I$PYTHON_INC"

case "$(uname -s)" in
  Darwin)
    # macOS: Apple Accelerate BLAS; -undefined dynamic_lookup for Python symbols.
    $CXX $COMMON -DUSE_ACCELERATE dynamic_ivf.cpp -o "$OUTPUT" \
        -framework Accelerate -undefined dynamic_lookup
    ;;
  *)
    # Linux: OpenBLAS (apt install libopenblas-dev). cblas.h must be on the
    # include path; add it explicitly if your distro puts it elsewhere.
    BLAS_INC=""
    for d in /usr/include /usr/include/x86_64-linux-gnu /usr/include/openblas; do
      [ -f "$d/cblas.h" ] && BLAS_INC="-I$d" && break
    done
    $CXX $COMMON $BLAS_INC -DUSE_OPENBLAS dynamic_ivf.cpp -o "$OUTPUT" \
        -lopenblas -lpthread
    ;;
esac

echo "Copenhagen index built: $OUTPUT"
