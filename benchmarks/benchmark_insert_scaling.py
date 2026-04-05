"""benchmark_insert_scaling.py — Insert cost and recall vs dataset size.

Compares Copenhagen (O(1) amortized insert) vs HNSW (O(log n) insert):
  - insert cost (µs/vector) at each scale
  - recall@10 after all inserts

Dataset: SIFT-128 (real L2 vectors, d=128).

Usage:
    python benchmarks/benchmark_insert_scaling.py
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from core import CopenhagenIndex

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

D          = 128
N_CLUSTERS = 64
NPROBE     = 8
REPEATS    = 3
BATCH      = 1_000
N_QUERIES  = 200
K          = 10

SCALES = [5_000, 10_000, 25_000, 50_000, 100_000]

DATA_PATH  = Path(__file__).parent.parent / "data/sift/sift-128-euclidean.hdf5"


def load_sift():
    with h5py.File(DATA_PATH) as f:
        train = f["train"][:max(SCALES) + BATCH].astype(np.float32)
        test  = f["test"][:N_QUERIES].astype(np.float32)
    return train, test


def _median(xs):
    return sorted(xs)[len(xs) // 2]


def _brute_knn(data, queries, k):
    q2 = np.sum(queries**2, axis=1)[:, None]
    d2 = np.sum(data**2,    axis=1)[None, :]
    D2 = q2 + d2 - 2.0 * (queries @ data.T)
    return np.argsort(D2, axis=1)[:, :k]


def _recall(gt, found):
    hits = sum(len(set(g.tolist()) & set(f.tolist())) for g, f in zip(gt, found))
    return hits / (len(gt) * K)


def measure_cph(data, queries, n):
    """Returns (µs/insert, recall@10) for a CPH index of size n."""
    base  = data[:n]
    batch = data[n:n + BATCH]

    # insert timing
    times = []
    for _ in range(REPEATS):
        idx = CopenhagenIndex(dim=D, n_clusters=N_CLUSTERS, nprobe=NPROBE, soft_k=2)
        idx.add(base)
        t0 = time.perf_counter()
        idx.add(batch)
        times.append((time.perf_counter() - t0) * 1e6 / BATCH)
    us = _median(times)

    # recall on index of size n (without the extra batch)
    idx = CopenhagenIndex(dim=D, n_clusters=N_CLUSTERS, nprobe=NPROBE, soft_k=2)
    idx.add(base)
    gt = _brute_knn(base, queries, K)
    found = np.array([idx.search(q, k=K)[0] for q in queries])
    rec = _recall(gt, found)

    return us, rec


def measure_faiss_ivf(data, queries, n):
    """Returns (µs/insert, recall@10) for FAISS IVF with online add (no retrain)."""
    if not HAS_FAISS:
        return None, None
    base  = data[:n]
    batch = data[n:n + BATCH]

    # train centroids on base, then time online adds
    times = []
    for _ in range(REPEATS):
        quantizer = faiss.IndexFlatL2(D)
        idx = faiss.IndexIVFFlat(quantizer, D, N_CLUSTERS, faiss.METRIC_L2)
        idx.train(base)
        idx.add(base)
        t0 = time.perf_counter()
        idx.add(batch)
        times.append((time.perf_counter() - t0) * 1e6 / BATCH)
    us = _median(times)

    # recall on base
    quantizer = faiss.IndexFlatL2(D)
    idx = faiss.IndexIVFFlat(quantizer, D, N_CLUSTERS, faiss.METRIC_L2)
    idx.nprobe = NPROBE
    idx.train(base)
    idx.add(base)
    gt = _brute_knn(base, queries, K)
    _, I = idx.search(queries, K)
    rec = _recall(gt, I)

    return us, rec


def measure_hnsw(data, queries, n):
    """Returns (µs/insert, recall@10) for an HNSW index of size n."""
    if not HAS_FAISS:
        return None, None
    base  = data[:n]
    batch = data[n:n + BATCH]

    # insert timing (single-vector to measure per-insert cost at size n)
    times = []
    for _ in range(REPEATS):
        idx = faiss.IndexHNSWFlat(D, 32)
        idx.add(base)
        t0 = time.perf_counter()
        for vec in batch:
            idx.add(vec.reshape(1, -1))
        times.append((time.perf_counter() - t0) * 1e6 / BATCH)
    us = _median(times)

    # recall
    idx = faiss.IndexHNSWFlat(D, 32)
    idx.hnsw.efSearch = 64
    idx.add(base)
    gt = _brute_knn(base, queries, K)
    _, I = idx.search(queries, K)
    rec = _recall(gt, I)

    return us, rec


if __name__ == "__main__":
    print(f"Loading SIFT-128...")
    train, queries = load_sift()
    print(f"Loaded {len(train):,} train vectors, {len(queries)} queries\n")

    print(f"d={D}  k={N_CLUSTERS}  nprobe={NPROBE}  batch={BATCH:,}  queries={N_QUERIES}\n")

    hdr = f"  {'n':>8}   {'CPH µs/vec':>12}   {'CPH R@10':>10}"
    if HAS_FAISS:
        hdr += f"   {'IVF µs/vec':>12}   {'IVF R@10':>10}"
        hdr += f"   {'HNSW µs/vec':>12}   {'HNSW R@10':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results = []
    for n in SCALES:
        cph_us, cph_rec = measure_cph(train, queries, n)
        row = {"n": n, "cph_us": cph_us, "cph_rec": cph_rec}
        line = f"  {n:>8,}   {cph_us:>12.2f}   {cph_rec:>10.3f}"

        if HAS_FAISS:
            f_us, f_rec = measure_faiss_ivf(train, queries, n)
            row["ivf_us"] = f_us
            row["ivf_rec"] = f_rec
            line += f"   {f_us:>12.2f}   {f_rec:>10.3f}"

            h_us, h_rec = measure_hnsw(train, queries, n)
            row["hnsw_us"] = h_us
            row["hnsw_rec"] = h_rec
            line += f"   {h_us:>12.2f}   {h_rec:>10.3f}"

        print(line, flush=True)
        results.append(row)

    first, last = results[0], results[-1]
    scale = last["n"] / first["n"]
    print(f"\n  CPH insert cost change:  {last['cph_us']/first['cph_us']:.2f}x over {scale:.0f}x scale-up  → O(1)")
    if HAS_FAISS and first.get("ivf_us") and last.get("ivf_us"):
        print(f"  IVF insert cost change:  {last['ivf_us']/first['ivf_us']:.2f}x over {scale:.0f}x scale-up  → O(1)")
    if HAS_FAISS and first.get("hnsw_us") and last.get("hnsw_us"):
        print(f"  HNSW insert cost change: {last['hnsw_us']/first['hnsw_us']:.2f}x over {scale:.0f}x scale-up  → O(log n) + cache effects")
