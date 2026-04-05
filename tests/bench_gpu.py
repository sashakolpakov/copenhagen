"""bench_gpu.py — GPU performance benchmarks for CopenhagenIndex.

Measures:
  1. Insert throughput: GPU vs CPU (vectors/second)
  2. Data transfer overhead: host→device and device→host round-trip
  3. Assignment compute time: torch.mm centroid distance on device vs cblas on CPU
  4. Pinning overhead: one-time cost to copy centroids to device

Run directly:
    python tests/bench_gpu.py

Or via pytest (skips if torch absent):
    pytest tests/bench_gpu.py -v -s
"""

import time
import sys
from pathlib import Path

import importlib.util

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from core import CopenhagenIndex

HAS_TORCH = importlib.util.find_spec("torch") is not None

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
]

SCALES = [10_000, 50_000, 100_000]


def _device():
    if not HAS_TORCH:
        return None
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync(device):
    """Block until device ops complete (needed for accurate timing)."""
    if not HAS_TORCH:
        return
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _trained_index(device, d=64, n_clusters=32, seed=0):
    """Return an index that has been trained (centroids pinned) but is otherwise empty."""
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((1000, d)).astype(np.float32)
    idx = CopenhagenIndex(dim=d, n_clusters=n_clusters, nprobe=8, soft_k=2, device=device)
    idx.add(train)          # first add trains + pins centroids
    return idx, rng


# ---------------------------------------------------------------------------
# 1. Insert throughput
# ---------------------------------------------------------------------------

def _measure_insert_throughput(device, n=50_000, d=64, n_clusters=32, repeats=3):
    results = []
    for _ in range(repeats):
        idx, rng = _trained_index(device, d=d, n_clusters=n_clusters)
        batch = rng.standard_normal((n, d)).astype(np.float32)

        _sync(device)
        t0 = time.perf_counter()
        idx.add(batch)
        _sync(device)
        elapsed = time.perf_counter() - t0
        results.append(n / elapsed)

    return sorted(results)[len(results) // 2]   # median


def test_insert_throughput(capsys):
    device = _device()
    with capsys.disabled():
        print(f"\n--- Insert throughput (d=64, k=32) ---")
        print(f"  {'n':>8}   {'CPU (v/s)':>14}   {'GPU '+str(device)+' (v/s)':>16}   {'speedup':>8}")
        print(f"  {'-'*8}   {'-'*14}   {'-'*16}   {'-'*8}")
    for n in SCALES:
        gpu_vps = _measure_insert_throughput(device, n=n)
        cpu_vps = _measure_insert_throughput(None, n=n)
        speedup = gpu_vps / cpu_vps
        with capsys.disabled():
            print(f"  {n:>8,}   {cpu_vps:>14,.0f}   {gpu_vps:>16,.0f}   {speedup:>7.2f}x")
        assert gpu_vps > cpu_vps * 0.25, (
            f"n={n}: GPU insert ({gpu_vps:.0f} v/s) is more than 4x slower than CPU ({cpu_vps:.0f} v/s)"
        )


# ---------------------------------------------------------------------------
# 2. Data transfer: host → device → host round-trip
# ---------------------------------------------------------------------------

def _measure_transfer(device, n=50_000, d=64, repeats=5):
    """Time a full host→device→host round trip for an (n, d) float32 matrix."""
    import torch
    data = np.random.default_rng(1).standard_normal((n, d)).astype(np.float32)
    results = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        t = torch.from_numpy(data).to(device)
        _sync(device)
        back = t.cpu().numpy()
        _sync(device)
        elapsed = time.perf_counter() - t0
        results.append(elapsed)
        del t, back

    median = sorted(results)[len(results) // 2]
    nbytes = data.nbytes
    return median, nbytes


def test_data_transfer(capsys):
    device = _device()
    with capsys.disabled():
        print(f"\n--- Data transfer round-trip (d=64, device={device}) ---")
        print(f"  {'n':>8}   {'MB':>6}   {'ms':>8}   {'GB/s':>8}")
        print(f"  {'-'*8}   {'-'*6}   {'-'*8}   {'-'*8}")
    for n in SCALES:
        elapsed, nbytes = _measure_transfer(device, n=n)
        gb_per_s = (nbytes / elapsed) / 1e9
        with capsys.disabled():
            print(f"  {n:>8,}   {nbytes/1e6:>6.1f}   {elapsed*1000:>8.2f}   {gb_per_s:>8.2f}")
        assert elapsed < 10.0, f"n={n}: transfer took {elapsed:.2f}s — unusually slow"


# ---------------------------------------------------------------------------
# 3. Assignment compute: torch.mm vs numpy on host
# ---------------------------------------------------------------------------

def _measure_assignment(device, n=50_000, d=64, k=32, repeats=5):
    import torch
    rng = np.random.default_rng(2)
    vectors_np = rng.standard_normal((n, d)).astype(np.float32)
    centroids_np = rng.standard_normal((k, d)).astype(np.float32)

    # GPU path
    v_t = torch.from_numpy(vectors_np).to(device)
    c_t = torch.from_numpy(centroids_np).to(device)
    _sync(device)

    gpu_times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        v_sq = (v_t * v_t).sum(dim=1, keepdim=True)
        c_sq = (c_t * c_t).sum(dim=1, keepdim=True).T
        dists = v_sq + c_sq - 2.0 * torch.mm(v_t, c_t.T)
        _ = torch.topk(dists, 2, dim=1, largest=False).indices.cpu().numpy()
        _sync(device)
        gpu_times.append(time.perf_counter() - t0)

    # CPU path (numpy)
    cpu_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        v_sq = (vectors_np ** 2).sum(axis=1, keepdims=True)
        c_sq = (centroids_np ** 2).sum(axis=1, keepdims=True).T
        dists = v_sq + c_sq - 2.0 * (vectors_np @ centroids_np.T)
        np.argpartition(dists, 2, axis=1)[:, :2]
        cpu_times.append(time.perf_counter() - t0)

    return (sorted(gpu_times)[len(gpu_times) // 2],
            sorted(cpu_times)[len(cpu_times) // 2])


def test_assignment_compute(capsys):
    device = _device()
    with capsys.disabled():
        print(f"\n--- Centroid assignment compute (d=64, k=32) ---")
        print(f"  {'n':>8}   {'CPU (ms)':>10}   {'GPU '+str(device)+' (ms)':>14}   {'speedup':>8}")
        print(f"  {'-'*8}   {'-'*10}   {'-'*14}   {'-'*8}")
    for n in SCALES:
        gpu_ms, cpu_ms = _measure_assignment(device, n=n)
        gpu_ms *= 1000
        cpu_ms *= 1000
        speedup = cpu_ms / gpu_ms
        with capsys.disabled():
            print(f"  {n:>8,}   {cpu_ms:>10.2f}   {gpu_ms:>14.2f}   {speedup:>7.2f}x")
        assert gpu_ms > 0 and cpu_ms > 0


# ---------------------------------------------------------------------------
# 4. Centroid pinning overhead
# ---------------------------------------------------------------------------

def _measure_pinning(device, d=64, k=32, repeats=10):
    import torch
    centroids_np = np.random.default_rng(3).standard_normal((k, d)).astype(np.float32)
    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        c = torch.from_numpy(np.ascontiguousarray(centroids_np)).to(device)
        _sync(device)
        times.append(time.perf_counter() - t0)
        del c
    return sorted(times)[len(times) // 2]


def test_pinning_overhead(capsys):
    device = _device()
    elapsed_ms = _measure_pinning(device) * 1000

    with capsys.disabled():
        print(f"\n--- Centroid pinning overhead (k=32, d=64) ---")
        print(f"  Device: {device}")
        print(f"  Pinning time (median): {elapsed_ms:.3f} ms")

    # Pinning 32*64*4 = 8 KB should never take more than 100 ms
    assert elapsed_ms < 100, f"Centroid pinning took {elapsed_ms:.1f} ms — unexpectedly slow"


# ---------------------------------------------------------------------------
# 5. Batch search throughput (QPS)
# ---------------------------------------------------------------------------

def _measure_search_qps(device, n_index=50_000, d=64, n_clusters=32,
                         nprobe=8, batch_sizes=(1, 10, 100, 1000), repeats=3):
    """Return dict batch_size → QPS for batch search."""
    idx, rng = _built_index_for_search(device, n_index, d, n_clusters)
    queries_large = rng.standard_normal((max(batch_sizes), d)).astype(np.float32)

    results = {}
    for bs in batch_sizes:
        q = queries_large[:bs]
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            idx.search_batch(q, k=10)
            times.append(time.perf_counter() - t0)
        median = sorted(times)[len(times) // 2]
        results[bs] = bs / median
    return results


def _built_index_for_search(device, n, d, n_clusters, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype(np.float32)
    idx = CopenhagenIndex(dim=d, n_clusters=n_clusters, nprobe=8, soft_k=2, device=device)
    idx.add(data)
    return idx, rng


def test_search_throughput(capsys):
    batch_sizes = (1, 10, 100, 1000)
    cpu_qps = _measure_search_qps(None)
    gpu_qps = _measure_search_qps(_device())

    with capsys.disabled():
        print(f"\n--- Batch search QPS (n=50k, d=64, k=32, nprobe=8) ---")
        print(f"  {'batch':>6}   {'CPU QPS':>12}   {'GPU QPS':>12}   {'speedup':>8}")
        print(f"  {'-'*6}   {'-'*12}   {'-'*12}   {'-'*8}")
        for bs in batch_sizes:
            cq = cpu_qps[bs]
            gq = gpu_qps[bs]
            print(f"  {bs:>6}   {cq:>12,.0f}   {gq:>12,.0f}   {gq/cq:>7.2f}x")

    # Batch search should not be catastrophically slower than single-query
    assert cpu_qps[1000] > cpu_qps[1] * 0.5


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not HAS_TORCH:
        print("torch not installed — skipping GPU benchmarks")
        sys.exit(0)

    device = _device()
    print(f"Device: {device}  d=64  k=32\n")

    pin_ms = _measure_pinning(device) * 1000
    print(f"[pinning]  centroid pin (k=32, d=64): {pin_ms:.3f} ms\n")

    header = f"  {'n':>8}   {'CPU':>14}   {'GPU ('+device+')':>14}   {'speedup':>8}"
    sep    = f"  {'-'*8}   {'-'*14}   {'-'*14}   {'-'*8}"

    print("[transfer] host→device→host round-trip")
    print(f"  {'n':>8}   {'MB':>6}   {'ms':>8}   {'GB/s':>8}")
    print(f"  {'-'*8}   {'-'*6}   {'-'*8}   {'-'*8}")
    for n in SCALES:
        elapsed, nbytes = _measure_transfer(device, n=n)
        print(f"  {n:>8,}   {nbytes/1e6:>6.1f}   {elapsed*1000:>8.2f}   {nbytes/elapsed/1e9:>8.2f}")

    print(f"\n[assign]   centroid distance compute (ms)")
    print(header); print(sep)
    for n in SCALES:
        gms, cms = _measure_assignment(device, n=n)
        print(f"  {n:>8,}   {cms*1000:>14.2f}   {gms*1000:>14.2f}   {cms/gms:>7.2f}x")

    print(f"\n[insert]   end-to-end insert throughput (vectors/s)")
    print(header); print(sep)
    for n in SCALES:
        gvps = _measure_insert_throughput(device, n=n)
        cvps = _measure_insert_throughput(None,   n=n)
        print(f"  {n:>8,}   {cvps:>14,.0f}   {gvps:>14,.0f}   {gvps/cvps:>7.2f}x")

    batch_sizes = (1, 10, 100, 1000)
    cpu_qps = _measure_search_qps(None)
    gpu_qps = _measure_search_qps(device)
    print(f"\n[search]   batch search QPS (n=50k, d=64, k=32, nprobe=8)")
    print(f"  {'batch':>6}   {'CPU QPS':>12}   {'GPU ('+device+') QPS':>16}   {'speedup':>8}")
    print(f"  {'-'*6}   {'-'*12}   {'-'*16}   {'-'*8}")
    for bs in batch_sizes:
        cq = cpu_qps[bs]
        gq = gpu_qps[bs]
        print(f"  {bs:>6}   {cq:>12,.0f}   {gq:>16,.0f}   {gq/cq:>7.2f}x")
