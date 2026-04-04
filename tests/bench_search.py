"""bench_search.py — CPU search throughput benchmarks for CopenhagenIndex.

Measures batch search QPS across scales and batch sizes.
The inverted-cluster-scan path (search_batch) groups queries by probed cluster
and issues one GEMM per cluster; single-query search uses precomputed norms.

Run directly:
    python tests/bench_search.py

Or via pytest:
    pytest tests/bench_search.py -v -s
"""

import time
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from core import CopenhagenIndex

N_SCALES   = [10_000, 50_000, 100_000]
BATCH_SIZES = [1, 10, 100, 1000]
D          = 64
N_CLUSTERS = 32
NPROBE     = 8


def _build_index(n, d=D, n_clusters=N_CLUSTERS, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype(np.float32)
    idx = CopenhagenIndex(dim=d, n_clusters=n_clusters, nprobe=NPROBE, soft_k=2)
    idx.add(data)
    return idx, rng


def _qps(idx, queries, repeats=5):
    nq = len(queries)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        if nq == 1:
            idx.search(queries[0], k=10)
        else:
            idx.search_batch(queries, k=10)
        times.append(time.perf_counter() - t0)
    return nq / sorted(times)[len(times) // 2]


# ---------------------------------------------------------------------------
# 1. Batch search QPS across scales
# ---------------------------------------------------------------------------

def test_batch_search_qps(capsys):
    with capsys.disabled():
        print(f"\n--- Batch search QPS (d={D}, k={N_CLUSTERS}, nprobe={NPROBE}) ---")
        for n in N_SCALES:
            idx, rng = _build_index(n)
            queries = rng.standard_normal((1000, D)).astype(np.float32)

            row = f"  n={n:>7,}  "
            for bs in BATCH_SIZES:
                q = queries[:bs]
                qps = _qps(idx, q)
                row += f"  bs={bs}: {qps:>8,.0f}"
            print(row)

    # Sanity: batch of 1000 should achieve ≥100 QPS on any reasonable machine
    idx, rng = _build_index(50_000)
    queries = rng.standard_normal((1000, D)).astype(np.float32)
    qps = _qps(idx, queries)
    assert qps >= 100, f"Batch-1000 QPS too low: {qps:.0f}"


# ---------------------------------------------------------------------------
# 2. Single vs batch: batching should not degrade QPS
# ---------------------------------------------------------------------------

def test_batch_vs_single(capsys):
    idx, rng = _build_index(50_000)
    q1    = rng.standard_normal((1,    D)).astype(np.float32)
    q1000 = rng.standard_normal((1000, D)).astype(np.float32)

    qps1    = _qps(idx, q1)
    qps1000 = _qps(idx, q1000)

    with capsys.disabled():
        print(f"\n--- Single vs batch QPS (n=50k, d={D}, nprobe={NPROBE}) ---")
        print(f"  batch=1:    {qps1:>8,.0f} QPS")
        print(f"  batch=1000: {qps1000:>8,.0f} QPS")
        print(f"  ratio: {qps1000/qps1:.2f}x")

    # batch of 1000 should not be worse than single-query
    assert qps1000 >= qps1 * 0.5


# ---------------------------------------------------------------------------
# 3. Insert throughput across scales (CPU baseline)
# ---------------------------------------------------------------------------

def _insert_throughput(n, d=D, n_clusters=N_CLUSTERS, repeats=3):
    times = []
    for _ in range(repeats):
        rng = np.random.default_rng(0)
        idx = CopenhagenIndex(dim=d, n_clusters=n_clusters, nprobe=NPROBE, soft_k=2)
        train = rng.standard_normal((1000, d)).astype(np.float32)
        idx.add(train)
        batch = rng.standard_normal((n, d)).astype(np.float32)
        t0 = time.perf_counter()
        idx.add(batch)
        times.append(n / (time.perf_counter() - t0))
    return sorted(times)[len(times) // 2]


def test_insert_throughput(capsys):
    with capsys.disabled():
        print(f"\n--- CPU insert throughput (d={D}, k={N_CLUSTERS}) ---")
        print(f"  {'n':>8}   {'vectors/s':>14}")
        print(f"  {'-'*8}   {'-'*14}")
    for n in N_SCALES:
        vps = _insert_throughput(n)
        with capsys.disabled():
            print(f"  {n:>8,}   {vps:>14,.0f}")
        assert vps > 100_000, f"n={n}: insert throughput too low ({vps:.0f} v/s)"


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"d={D}  n_clusters={N_CLUSTERS}  nprobe={NPROBE}\n")

    print("[insert]  CPU insert throughput (vectors/s)")
    print(f"  {'n':>8}   {'vectors/s':>14}")
    print(f"  {'-'*8}   {'-'*14}")
    for n in N_SCALES:
        vps = _insert_throughput(n)
        print(f"  {n:>8,}   {vps:>14,.0f}")

    print(f"\n[search]  batch search QPS  (rows = index size, cols = batch size)")
    hdr = f"  {'n':>8}"
    for bs in BATCH_SIZES:
        hdr += f"   {'bs='+str(bs):>10}"
    print(hdr)
    print("  " + "-" * (8 + 13 * len(BATCH_SIZES)))
    for n in N_SCALES:
        idx, rng = _build_index(n)
        queries = rng.standard_normal((max(BATCH_SIZES), D)).astype(np.float32)
        row = f"  {n:>8,}"
        for bs in BATCH_SIZES:
            qps = _qps(idx, queries[:bs])
            row += f"   {qps:>10,.0f}"
        print(row)
