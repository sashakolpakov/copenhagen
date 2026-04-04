"""
benchmark_drift_streaming.py — Gradual distribution drift over many batches
────────────────────────────────────────────────────────────────────────────
Unlike benchmark_drift.py (which dumps all OOD data in one shot), this
benchmark simulates a realistic streaming scenario: Fashion-MNIST vectors
arrive in small batches over time while the index stays live.

Compared methods:
  FAISS add-only  — standard IVF, train on MNIST, add Fashion incrementally
  Copenhagen base — fixed centroids, soft_k=1 (no splits)
  Copenhagen best — adaptive splits + soft_k=2

For each batch we record recall@10 on Fashion queries, so you can see:
  - FAISS degrades monotonically as Fashion clusters bloat
  - Copenhagen base similarly degrades (no splits to fix imbalance)
  - Copenhagen best stabilises after the first few splits fire

Run:
    python3 benchmarks/benchmark_drift_streaming.py
    python3 benchmarks/benchmark_drift_streaming.py --full
"""

import sys, os, time, json, argparse
from pathlib import Path
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from python.core import DynamicIVF

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("WARNING: faiss not installed — FAISS row skipped")

_DATA_DIR    = _REPO_ROOT / "data"
MNIST_PATH   = _DATA_DIR / "MNIST"   / "mnist-784-euclidean.hdf5"
FASHION_PATH = _DATA_DIR / "fashion-mnist" / "fashion-mnist-784-euclidean.hdf5"


def load(quick=True):
    import h5py
    with h5py.File(MNIST_PATH, 'r') as f:
        mt = np.array(f['train'], dtype=np.float32)
        mq = np.array(f['test'],  dtype=np.float32)
    with h5py.File(FASHION_PATH, 'r') as f:
        ft = np.array(f['train'], dtype=np.float32)
        fq = np.array(f['test'],  dtype=np.float32)
    if quick:
        mt, mq = mt[:10_000], mq[:200]
        ft, fq = ft[:5_000],  fq[:200]
    return mt, mq, ft, fq


def brute_knn(data, queries, k):
    if HAS_FAISS:
        idx = faiss.IndexFlatL2(data.shape[1])
        idx.add(data)
        _, I = idx.search(queries, k)
        return I
    dists = np.linalg.norm(queries[:, None] - data[None], axis=-1)
    return np.argsort(dists, axis=1)[:, :k]


def recall_k(approx_ids, exact_ids):
    hits = sum(len(set(a.tolist()) & set(e.tolist()))
               for a, e in zip(approx_ids, exact_ids))
    return hits / (len(exact_ids) * len(exact_ids[0]))


def run_streaming(mnist_train, fashion_train, fashion_test,
                  n_clusters, nprobe, soft_k, split_threshold,
                  batch_size, n_batches, k=10):
    """
    Stream fashion_train in n_batches of batch_size vectors.
    Record recall after each batch against the ground truth at that point.
    """
    dim = mnist_train.shape[1]

    idx = DynamicIVF(dim, n_clusters, nprobe, 0, 8, 256, soft_k)
    idx.split_threshold = split_threshold
    idx.train(mnist_train)

    recalls = []
    n_clusters_history = []
    inserted = 0

    for b in range(n_batches):
        batch = fashion_train[b * batch_size:(b + 1) * batch_size]
        if len(batch) == 0:
            break
        idx.insert_batch(batch)
        inserted += len(batch)

        # Ground truth: brute force on everything inserted so far
        all_data = np.vstack([mnist_train, fashion_train[:inserted]])
        gt = brute_knn(all_data, fashion_test, k)

        I = np.array([idx.search(q, k, nprobe)[0] for q in fashion_test])
        recalls.append(round(recall_k(I, gt), 4))
        n_clusters_history.append(idx.get_stats()["n_clusters"])

    return recalls, n_clusters_history


def run_faiss_streaming(mnist_train, fashion_train, fashion_test,
                        n_clusters, nprobe, batch_size, n_batches, k=10):
    if not HAS_FAISS:
        return [], []
    dim = mnist_train.shape[1]
    q   = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFFlat(q, dim, n_clusters, faiss.METRIC_L2)
    idx.train(mnist_train)
    idx.add(mnist_train)
    idx.nprobe = nprobe

    recalls = []
    inserted = 0
    for b in range(n_batches):
        batch = fashion_train[b * batch_size:(b + 1) * batch_size]
        if len(batch) == 0:
            break
        idx.add(batch)
        inserted += len(batch)

        all_data = np.vstack([mnist_train, fashion_train[:inserted]])
        gt = brute_knn(all_data, fashion_test, k)
        _, I = idx.search(fashion_test, k)
        recalls.append(round(recall_k(I, gt), 4))

    return recalls, [n_clusters] * len(recalls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='Use full dataset')
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from download_data import ensure_datasets
    ensure_datasets(["mnist", "fashion"])

    mt, mq, ft, fq = load(quick=not args.full)
    print("Copenhagen — Gradual Streaming Drift Benchmark")
    print("=" * 60)
    print(f"  MNIST train={len(mt):,}  Fashion train={len(ft):,}  queries={len(fq):,}")

    N_CLUSTERS = 32
    NPROBE     = 4
    BATCH_SIZE = 500
    N_BATCHES  = len(ft) // BATCH_SIZE

    print(f"  Config: {N_CLUSTERS} clusters, nprobe={NPROBE}, "
          f"batches={N_BATCHES}×{BATCH_SIZE}\n")

    configs = [
        (1, 9999.0, "Copenhagen baseline (no splits, soft_k=1)"),
        (2,  3.0,   "Copenhagen best     (splits + soft_k=2)"),
    ]

    all_recalls = {}

    if HAS_FAISS:
        print("  Running FAISS add-only...")
        r, _ = run_faiss_streaming(mt, ft, fq, N_CLUSTERS, NPROBE, BATCH_SIZE, N_BATCHES)
        all_recalls["faiss_add"] = r
        print(f"    recall at batch 1/{N_BATCHES}: {r[0]:.4f}  "
              f"final: {r[-1]:.4f}")

    for soft_k, threshold, tag in configs:
        print(f"  Running {tag}...")
        r, nc = run_streaming(mt, ft, fq, N_CLUSTERS, NPROBE,
                              soft_k, threshold, BATCH_SIZE, N_BATCHES)
        all_recalls[tag] = r
        splits_fired = nc[-1] - N_CLUSTERS
        print(f"    recall at batch 1/{N_BATCHES}: {r[0]:.4f}  "
              f"final: {r[-1]:.4f}  splits: +{splits_fired}")

    # Summary table
    print()
    print("=" * 60)
    print("RECALL@10 OVER STREAMING BATCHES")
    print(f"  {'method':<42} {'start':>6}  {'mid':>6}  {'final':>6}")
    print(f"  {'-'*42} {'------':>6}  {'-----':>6}  {'-----':>6}")
    for name, r in all_recalls.items():
        if not r:
            continue
        mid = r[len(r) // 2]
        label = name if not name.startswith("Copenhagen") else name
        print(f"  {label:<42} {r[0]:>6.4f}  {mid:>6.4f}  {r[-1]:>6.4f}")

    os.makedirs(_REPO_ROOT / "results", exist_ok=True)
    out = _REPO_ROOT / "results" / "drift_streaming.json"
    with open(out, 'w') as fh:
        json.dump(all_recalls, fh, indent=2)
    print(f"\n  Results → {out}")


if __name__ == '__main__':
    main()
