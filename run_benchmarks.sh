#!/bin/bash
# run_benchmarks.sh — push-button, self-contained benchmark matrix in Docker.
#
# From a fresh `git clone`, on a Linux box:
#     bash run_benchmarks.sh            # full matrix, --jobs 6
#     bash run_benchmarks.sh 12         # full matrix, --jobs 12 (e.g. 30-vCPU box)
#     bash run_benchmarks.sh 12 --quick # fast smoke
#
# It installs Docker if missing, builds the image, downloads all datasets into a
# persistent ./data volume (cached across runs), runs every benchmark × method
# concurrently, and writes benchmarks/results/REPORT_latest.md.
set -e
cd "$(dirname "$0")"

JOBS="${1:-6}"; shift || true
EXTRA_ARGS=("$@")
IMAGE=copenhagen-bench

# 1. Ensure Docker.
if ! command -v docker >/dev/null 2>&1; then
  echo "==> Docker not found; installing (sudo)…"
  sudo apt-get update -y && sudo apt-get install -y docker.io
  sudo systemctl enable --now docker || true
fi
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# 2. Persistent host dirs (datasets cached, results/figures extracted).
#    The repo ships `data` as a symlink to a sibling checkout; replace it with a
#    real directory so it can be bind-mounted.
[ -L data ] && rm -f data
mkdir -p data benchmarks/results figures

# 3. Build the image (BLAS, FAISS, HNSW, TurboVec, the C++ extension).
echo "==> Building image $IMAGE …"
$DOCKER build -t "$IMAGE" .

# 4. Run the full matrix. Datasets download into the mounted ./data on first run.
echo "==> Running benchmark matrix (--jobs $JOBS ${EXTRA_ARGS[*]}) …"
$DOCKER run --rm \
  -v "$PWD/data:/copenhagen/data" \
  -v "$PWD/benchmarks/results:/copenhagen/benchmarks/results" \
  -v "$PWD/figures:/copenhagen/figures" \
  "$IMAGE" --jobs "$JOBS" "${EXTRA_ARGS[@]}"

echo
echo "==> Done. Report:    benchmarks/results/REPORT_latest.md"
echo "==> Per-run logs:    benchmarks/results/logs/"
echo "==> Figures:         figures/"
