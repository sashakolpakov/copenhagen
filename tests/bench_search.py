"""bench_search.py — Search throughput benchmark: Copenhagen vs FAISS IVF.

Compares single-query and batch search QPS across index sizes.

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

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from ampi import AMPIAffineFanIndex
    HAS_AMPI = True
except ImportError:
    HAS_AMPI = False

N_SCALES   = [10_000, 50_000, 100_000]
BATCH_SIZES = [1, 10, 100, 1000]
D          = 64
N_CLUSTERS = 32
NPROBE     = 8


def _build_cph(n, d=D, n_clusters=N_CLUSTERS, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype(np.float32)
    idx = CopenhagenIndex(dim=d, n_clusters=n_clusters, nprobe=NPROBE, soft_k=2)
    idx.add(data)
    return idx, rng


def _build_faiss_ivf(n, d=D, n_clusters=N_CLUSTERS, seed=0):
    if not HAS_FAISS:
        return None, None
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype(np.float32)
    q   = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFFlat(q, d, n_clusters, faiss.METRIC_L2)
    idx.train(data)
    idx.add(data)
    idx.nprobe = NPROBE
    return idx, rng


def _build_faiss_hnsw(n, d=D, M=32, seed=0):
    """FAISS HNSW — no nprobe tuning needed; ef_search controls recall/speed trade-off."""
    if not HAS_FAISS:
        return None, None
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype(np.float32)
    idx = faiss.IndexHNSWFlat(d, M)
    idx.add(data)
    return idx, rng


def _build_ampi(n, d=D, n_clusters=N_CLUSTERS, seed=0):
    if not HAS_AMPI:
        return None, None
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype(np.float32)
    nlist = max(n_clusters, int(1.5 * n ** 0.5))
    idx = AMPIAffineFanIndex(data, nlist=nlist, num_fans=8, cone_top_k=1)
    return idx, rng


def _qps_cph(idx, queries, repeats=5):
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


def _qps_faiss(idx, queries, repeats=5):
    nq = len(queries)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        idx.search(queries, 10)
        times.append(time.perf_counter() - t0)
    return nq / sorted(times)[len(times) // 2]


def _qps_ampi(idx, queries, repeats=5):
    nq = len(queries)
    times = []
    probes = max(8, int(1.5 * nq ** 0.5))
    for _ in range(repeats):
        t0 = time.perf_counter()
        for q in queries:
            idx.query(q, k=10, probes=probes, fan_probes=8, window_size=20)
        times.append(time.perf_counter() - t0)
    return nq / sorted(times)[len(times) // 2]


# ---------------------------------------------------------------------------
# 1. Search QPS: Copenhagen vs FAISS IVF / HNSW / AMPI
# ---------------------------------------------------------------------------

def test_search_qps_vs_faiss(capsys):
    with capsys.disabled():
        print(f"\n--- Search QPS: Copenhagen vs FAISS IVF / HNSW"
              f"{' / AMPI' if HAS_AMPI else ''} "
              f"(d={D}, n_clusters={N_CLUSTERS}, nprobe={NPROBE}, bs=1000) ---")
        cols = ["cph", "faiss-ivf", "faiss-hnsw"]
        if HAS_AMPI:
            cols.append("ampi")
        print("  " + f"{'n':>8}  " + "  ".join(f"{c:>12}" for c in cols))
        print("  " + "-" * (10 + 14 * len(cols)))

        for n in N_SCALES:
            cph, rng = _build_cph(n)
            q1k = rng.standard_normal((1000, D)).astype(np.float32)
            row = {"cph": _qps_cph(cph, q1k)}

            if HAS_FAISS:
                fi, _  = _build_faiss_ivf(n)
                fh, _  = _build_faiss_hnsw(n)
                row["faiss-ivf"]  = _qps_faiss(fi, q1k)
                row["faiss-hnsw"] = _qps_faiss(fh, q1k)

            if HAS_AMPI:
                am, _ = _build_ampi(n)
                # AMPI: use smaller query set — it's single-query, so 1000 queries is slow
                q50 = rng.standard_normal((50, D)).astype(np.float32)
                row["ampi"] = _qps_ampi(am, q50)

            line = f"  {n:>8,}  "
            for c in cols:
                line += f"  {row.get(c, 0):>12,.0f}"
            print(line)

    cph, rng = _build_cph(50_000)
    q = rng.standard_normal((1000, D)).astype(np.float32)
    assert _qps_cph(cph, q) >= 100


# ---------------------------------------------------------------------------
# 2. Single vs batch: batching should not degrade QPS
# ---------------------------------------------------------------------------

def test_batch_vs_single(capsys):
    cph, rng = _build_cph(50_000)
    q1    = rng.standard_normal((1,    D)).astype(np.float32)
    q1000 = rng.standard_normal((1000, D)).astype(np.float32)

    qps1    = _qps_cph(cph, q1)
    qps1000 = _qps_cph(cph, q1000)

    with capsys.disabled():
        print(f"\n--- Single vs batch QPS (n=50k, d={D}, nprobe={NPROBE}) ---")
        print(f"  batch=1:    {qps1:>8,.0f} QPS")
        print(f"  batch=1000: {qps1000:>8,.0f} QPS")
        print(f"  ratio: {qps1000/qps1:.2f}x")

    assert qps1000 >= qps1 * 0.5


# ---------------------------------------------------------------------------
# 3. Insert throughput: Copenhagen vs FAISS IVF
# ---------------------------------------------------------------------------

def _insert_throughput_cph(n, d=D, n_clusters=N_CLUSTERS, repeats=3):
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


def _insert_throughput_faiss(n, d=D, n_clusters=N_CLUSTERS, repeats=3):
    if not HAS_FAISS:
        return None
    times = []
    for _ in range(repeats):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((n, d)).astype(np.float32)
        q   = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFFlat(q, d, n_clusters, faiss.METRIC_L2)
        idx.train(data)
        batch = rng.standard_normal((n, d)).astype(np.float32)
        t0 = time.perf_counter()
        idx.add(batch)
        times.append(n / (time.perf_counter() - t0))
    return sorted(times)[len(times) // 2]


def test_insert_throughput(capsys):
    with capsys.disabled():
        print(f"\n--- Insert throughput: Copenhagen vs FAISS IVF (d={D}, n_clusters={N_CLUSTERS}) ---")
        hdr = f"  {'n':>8}   {'cph (vec/s)':>14}"
        if HAS_FAISS:
            hdr += f"  {'faiss (vec/s)':>14}  {'ratio':>6}"
        print(hdr)
        print("  " + "-" * (8 + 40))

    for n in N_SCALES:
        cph_vps = _insert_throughput_cph(n)
        with capsys.disabled():
            if HAS_FAISS:
                f_vps = _insert_throughput_faiss(n)
                ratio = cph_vps / f_vps
                print(f"  {n:>8,}   {cph_vps:>14,.0f}  {f_vps:>14,.0f}  {ratio:>5.1f}x")
            else:
                print(f"  {n:>8,}   {cph_vps:>14,.0f}")
        assert cph_vps > 100_000, f"n={n}: insert throughput too low ({cph_vps:.0f} v/s)"


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"d={D}  n_clusters={N_CLUSTERS}  nprobe={NPROBE}\n")

    print("[insert]  Copenhagen vs FAISS IVF throughput (vectors/s)")
    hdr = f"  {'n':>8}   {'cph':>14}"
    if HAS_FAISS:
        hdr += f"  {'faiss':>14}  {'ratio':>6}"
    print(hdr)
    print("  " + "-" * (8 + 40))
    for n in N_SCALES:
        cph_vps = _insert_throughput_cph(n)
        if HAS_FAISS:
            f_vps = _insert_throughput_faiss(n)
            print(f"  {n:>8,}   {cph_vps:>14,.0f}  {f_vps:>14,.0f}  {cph_vps/f_vps:>5.1f}x")
        else:
            print(f"  {n:>8,}   {cph_vps:>14,.0f}")

    print(f"\n[search]  Copenhagen vs FAISS IVF QPS")
    print(f"  {'n':>8}   {'cph bs=1':>10}  {'cph bs=1000':>12}  "
          f"{'faiss bs=1':>11}  {'faiss bs=1000':>14}  {'ratio (1k)':>10}")
    print("  " + "-" * 75)
    for n in N_SCALES:
        cph, rng = _build_cph(n)
        q1  = rng.standard_normal((1,    D)).astype(np.float32)
        q1k = rng.standard_normal((1000, D)).astype(np.float32)
        c1  = _qps_cph(cph, q1)
        c1k = _qps_cph(cph, q1k)
        if HAS_FAISS:
            fi, _  = _build_faiss(n)
            f1  = _qps_faiss(fi, q1)
            f1k = _qps_faiss(fi, q1k)
            print(f"  {n:>8,}   {c1:>10,.0f}  {c1k:>12,.0f}  "
                  f"{f1:>11,.0f}  {f1k:>14,.0f}  {c1k/f1k:>9.2f}x")
        else:
            print(f"  {n:>8,}   {c1:>10,.0f}  {c1k:>12,.0f}  (faiss not installed)")
