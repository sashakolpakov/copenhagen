# Copenhagen benchmark image — self-contained: BLAS, FAISS, HNSW, TurboVec,
# the C++ extension, and the block-VQ module, ready to run the full matrix.
FROM python:3.11-bookworm

# System build deps + OpenBLAS for the C++ extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libopenblas-dev curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain — fallback so `pip install turbovec` can build from source on
# platforms without a prebuilt wheel (the image stays self-contained either way).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Python deps. abi3 wheels are used when available; Rust covers the rest.
RUN pip install --no-cache-dir \
        numpy pybind11 faiss-cpu hnswlib h5py matplotlib turbovec maturin sphinx

WORKDIR /copenhagen
COPY . /copenhagen

# Build the dynamic IVF extension + the block-VQ pybind module (-march=native →
# targets the build host's CPU; build the image on the box you run it on).
RUN bash src/build.sh && bash src/build_block_vq.sh

# Datasets and results are expected on mounted volumes (see run_benchmarks.sh):
#   -v $PWD/data:/copenhagen/data  -v $PWD/benchmarks/results:/copenhagen/benchmarks/results
ENTRYPOINT ["python3", "benchmarks/reproduce.py"]
CMD ["--jobs", "6"]
