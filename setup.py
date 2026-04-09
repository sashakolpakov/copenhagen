"""
Build the Copenhagen C++ extension (copenhagen._copenhagen_ext).

BLAS backend detection order:
  macOS   → Apple Accelerate  (always available, zero extra dependency)
  Linux   → OpenBLAS  (via pkg-config or common prefix search)
          → Intel MKL (via MKLROOT env var)
          → native tiled SIMD fallback (no external BLAS needed)
  Windows → Intel MKL (via MKLROOT) or native fallback
"""

import os
import platform
import shutil
import subprocess
from setuptools import setup, Extension


# ── BLAS detection ────────────────────────────────────────────────────────────

def _detect_blas():
    """Return (defines, include_dirs, link_args) for the best available BLAS."""
    sys = platform.system()

    # ── macOS: Accelerate is always present ───────────────────────────────────
    if sys == "Darwin":
        return (
            [("USE_ACCELERATE", None)],
            [],
            ["-framework", "Accelerate"],
        )

    # ── Linux / Windows: try OpenBLAS via pkg-config ─────────────────────────
    try:
        cflags = subprocess.check_output(
            ["pkg-config", "--cflags-only-I", "openblas"],
            stderr=subprocess.DEVNULL,
        ).decode().split()
        libs = subprocess.check_output(
            ["pkg-config", "--libs", "openblas"],
            stderr=subprocess.DEVNULL,
        ).decode().split()
        inc = [f[2:] for f in cflags if f.startswith("-I")]
        print("Copenhagen: found OpenBLAS via pkg-config")
        return ([("USE_OPENBLAS", None)], inc, libs)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # ── Linux: try common OpenBLAS install prefixes ───────────────────────────
    for prefix in ["/usr", "/usr/local", "/opt/homebrew", "/opt/local",
                   "/usr/local/opt/openblas"]:
        header = os.path.join(prefix, "include", "openblas", "cblas.h")
        if not os.path.exists(header):
            header = os.path.join(prefix, "include", "cblas.h")
        if os.path.exists(header):
            inc_dir = os.path.dirname(header)
            lib_dir = os.path.join(prefix, "lib")
            print(f"Copenhagen: found OpenBLAS at {prefix}")
            return (
                [("USE_OPENBLAS", None)],
                [inc_dir],
                [f"-L{lib_dir}", "-lopenblas"],
            )

    # ── Intel MKL: check MKLROOT env var ─────────────────────────────────────
    mklroot = os.environ.get("MKLROOT", "")
    if mklroot:
        inc_dir = os.path.join(mklroot, "include")
        lib_dir = os.path.join(mklroot, "lib", "intel64")
        if os.path.exists(inc_dir):
            print(f"Copenhagen: found Intel MKL at {mklroot}")
            mkl_libs = [
                f"-L{lib_dir}",
                "-lmkl_rt",
                "-lpthread", "-lm", "-ldl",
            ]
            return ([("USE_MKL", None)], [inc_dir], mkl_libs)

    # ── Fallback: no external BLAS ────────────────────────────────────────────
    print("Copenhagen: no external BLAS found — using built-in scalar kernel")
    return ([], [], [])


# ── Extension build ───────────────────────────────────────────────────────────

def _ext_modules():
    try:
        import pybind11
    except ImportError:
        return []

    extra_compile = ["-O3", "-std=c++17", "-fvisibility=hidden"]
    extra_link    = []

    # COPENHAGEN_ARCH controls the CPU tuning level:
    #   "native"   – optimise for this exact machine (fastest, not portable)
    #   "avx2"     – AVX2 + FMA baseline, default for PyPI wheels
    #                (Intel Haswell 2013+, AMD Ryzen 2017+; ~95 % of x86 machines)
    #   "baseline" – plain x86-64 / arm64, scalar fallback only
    arch_env = os.environ.get("COPENHAGEN_ARCH", "avx2")

    if platform.system() == "Darwin":
        machine = platform.machine()
        is_apple_clang = not (
            shutil.which("gcc") and "Apple" not in subprocess.getoutput("gcc --version")
        )
        if machine == "arm64":
            # NEON is always available on arm64; -mcpu selects pipeline tuning.
            if arch_env == "native":
                extra_compile += ["-mcpu=native"]
            elif arch_env != "baseline":
                extra_compile += ["-mcpu=apple-m1"]   # covers all Apple Silicon
        else:  # x86_64
            if arch_env == "native":
                flag = "-mcpu=native" if is_apple_clang else "-march=native"
                extra_compile += [flag]
            elif arch_env != "baseline":
                extra_compile += ["-march=x86-64-v3"]  # AVX2 + FMA
    elif platform.system() == "Linux":
        if arch_env == "native":
            extra_compile += ["-march=native"]
        elif arch_env != "baseline":
            extra_compile += ["-march=x86-64-v3"]      # AVX2 + FMA
    elif platform.system() == "Windows":
        if arch_env == "native":
            extra_compile = ["/O2", "/std:c++17"]      # MSVC has no native-march
        elif arch_env == "baseline":
            extra_compile = ["/O2", "/std:c++17"]
        else:
            extra_compile = ["/O2", "/std:c++17", "/arch:AVX2"]

    blas_defines, blas_inc, blas_link = _detect_blas()

    return [
        Extension(
            name="python.core.copenhagen",
            sources=["src/dynamic_ivf.cpp"],
            include_dirs=[pybind11.get_include()] + blas_inc,
            define_macros=blas_defines,
            language="c++",
            extra_compile_args=extra_compile,
            extra_link_args=extra_link + blas_link,
        )
    ]


setup(
    ext_modules=_ext_modules(),
    extras_require={
        # pip install copenhagen[gpu]
        # Installs PyTorch CPU wheel; users on CUDA/MPS should install torch
        # themselves first (vendor-specific wheels from pytorch.org).
        "gpu": ["torch"],
    },
)
