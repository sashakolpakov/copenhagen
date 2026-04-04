"""
benchmark_drift.py — Copenhagen's core scenario
─────────────────────────────────────────────────
THE scenario this system was built for:

  Train on MNIST digits (handwritten numbers, 784d).
  Then insert Fashion-MNIST (clothing images, same dim, completely different manifold).

Why this is hard for standard IVF:
  - Centroids are trained on digit shapes (curvy strokes, sparse pixels)
  - Fashion images (shirts, shoes) live in completely different parts of 784d space
  - They all land in the few MNIST Voronoi cells that happen to be "least wrong"
  - Those cells bloat to 10–30x normal size
  - nprobe=4 scans only 4/32 = 12% of the index, missing most of the fashion data

Methods compared:
  FAISS IVF add-only  — standard FAISS, just add vectors (no retrain)
  FAISS IVF rebuild   — retrain on all data (correct but expensive)
  AMPI                — adaptive multi-projection index with drift detection
                        (optional: pip install git+https://github.com/sashakolpakov/ampi)
  Copenhagen baseline — fixed centroids, soft_k=1
  Copenhagen adaptive — cluster splitting when size > 3× mean
  Copenhagen soft_k=2 — each vector in 2 nearest clusters
  Copenhagen best     — adaptive + soft_k=2

Run (quick, ~2 min):
    python3 benchmarks/benchmark_drift.py

Run (full data, ~15 min):
    python3 benchmarks/benchmark_drift.py --full
"""

import sys, os, time, json, argparse
import numpy as np
import h5py

sys.path.insert(0, '/Users/sasha/copenhagen')
from python.core import DynamicIVF

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("WARNING: faiss not installed — FAISS rows skipped")

try:
    from ampi import AMPIAffineFanIndex
    HAS_AMPI = True
except ImportError:
    HAS_AMPI = False
    print("WARNING: ampi not installed — AMPI row skipped")
    print("  To enable: pip install git+https://github.com/sashakolpakov/ampi")

MNIST_PATH   = '/Users/sasha/copenhagen/data/MNIST/mnist-784-euclidean.hdf5'
FASHION_PATH = '/Users/sasha/copenhagen/data/fashion-mnist/fashion-mnist-784-euclidean.hdf5'


# ── Data ──────────────────────────────────────────────────────────────────────

def load(quick=True):
    with h5py.File(MNIST_PATH, 'r') as f:
        mt = np.array(f['train'], dtype=np.float32)
        mq = np.array(f['test'],  dtype=np.float32)
    with h5py.File(FASHION_PATH, 'r') as f:
        ft = np.array(f['train'], dtype=np.float32)
        fq = np.array(f['test'],  dtype=np.float32)
    if quick:
        mt, mq = mt[:20_000], mq[:500]
        ft, fq = ft[:10_000], fq[:500]
    return mt, mq, ft, fq


# ── Recall helpers ────────────────────────────────────────────────────────────

def _brute_knn(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact k-NN via FAISS FlatL2 (avoids O(n²) Python loop)."""
    if HAS_FAISS:
        dim = data.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(data)
        _, I = idx.search(queries, k)
        return I  # shape (nq, k)
    # Fallback: brute force numpy
    dists = np.linalg.norm(queries[:, None, :] - data[None, :, :], axis=-1)  # nq × n
    return np.argsort(dists, axis=1)[:, :k]


def recall_k(approx_ids, exact_ids) -> float:
    """Recall@k: fraction of exact neighbors found in approx results."""
    hits = 0
    for a, e in zip(approx_ids, exact_ids):
        hits += len(set(a.tolist()) & set(e.tolist()))
    return hits / (len(exact_ids) * len(exact_ids[0]))


# ── Runners ───────────────────────────────────────────────────────────────────

def run_faiss(mnist_train, mnist_test, fashion_train, fashion_test,
              n_clusters, nprobe, k=10):
    if not HAS_FAISS:
        return {}
    dim = mnist_train.shape[1]
    all_data = np.vstack([mnist_train, fashion_train])

    # Ground truth: brute force on all data
    gt_fashion = _brute_knn(all_data, fashion_test, k)
    gt_mnist   = _brute_knn(all_data, mnist_test,   k)

    results = {}

    # Add-only: train on MNIST, add fashion without retrain
    q   = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFFlat(q, dim, n_clusters, faiss.METRIC_L2)
    idx.train(mnist_train)
    idx.add(mnist_train)
    t0 = time.perf_counter()
    idx.add(fashion_train)
    add_ms = (time.perf_counter() - t0) * 1000
    idx.nprobe = nprobe
    _, I_f = idx.search(fashion_test, k)
    _, I_m = idx.search(mnist_test, k)
    results["faiss_add"] = {
        "tag": "FAISS IVF  add-only (no retrain)",
        "recall_fashion": round(recall_k(I_f, gt_fashion), 4),
        "recall_mnist":   round(recall_k(I_m, gt_mnist), 4),
        "insert_ms":      round(add_ms, 0),
        "splits": 0, "imbalance": None,
    }

    # Full rebuild
    t0 = time.perf_counter()
    q2   = faiss.IndexFlatL2(dim)
    idx2 = faiss.IndexIVFFlat(q2, dim, n_clusters, faiss.METRIC_L2)
    idx2.train(all_data)
    idx2.add(all_data)
    rebuild_ms = (time.perf_counter() - t0) * 1000
    idx2.nprobe = nprobe
    _, I_f2 = idx2.search(fashion_test, k)
    _, I_m2 = idx2.search(mnist_test, k)
    results["faiss_rebuild"] = {
        "tag": "FAISS IVF  full rebuild",
        "recall_fashion": round(recall_k(I_f2, gt_fashion), 4),
        "recall_mnist":   round(recall_k(I_m2, gt_mnist), 4),
        "insert_ms":      round(rebuild_ms, 0),
        "splits": 0, "imbalance": None,
    }

    return results


def run_ampi(mnist_train, mnist_test, fashion_train, fashion_test,
             n_clusters, nprobe, k=10):
    if not HAS_AMPI:
        return {}
    dim = mnist_train.shape[1]
    all_data = np.vstack([mnist_train, fashion_train])

    gt_fashion = _brute_knn(all_data, fashion_test, k)
    gt_mnist   = _brute_knn(all_data, mnist_test,   k)

    # AMPI parameter tuning for 784d drift scenario:
    #   nlist ~ 1.5 × sqrt(n_train)  — AMPI's own recommendation; 32 (IVF default)
    #     is far too few for this dataset size (20k→ ~212, 60k→ ~367)
    #   num_fans=16  — works well for 784d per AMPI BENCHMARKS.md
    #   cone_top_k=2 — soft assignment improves boundary recall (like our soft_k)
    #   probes=16    — more cluster probes to cover the drifted region
    #   fan_probes=16 — probe all fans per cluster (was 4, starving recall)
    #   window_size=50 — AMPI BENCHMARKS.md recommends ~9; 50 is conservative
    n_train = len(mnist_train)
    ampi_nlist = max(n_clusters, int(1.5 * n_train ** 0.5))
    NUM_FANS   = 16
    AMPI_PROBES     = max(nprobe * 4, 16)
    AMPI_FAN_PROBES = NUM_FANS          # probe all fans per cluster
    AMPI_WINDOW     = 50

    idx = AMPIAffineFanIndex(mnist_train, nlist=ampi_nlist, num_fans=NUM_FANS, cone_top_k=2)

    t0 = time.perf_counter()
    idx.batch_add(fashion_train)
    add_ms = (time.perf_counter() - t0) * 1000

    def _query(q):
        _, _, ids = idx.query(q, k=k, probes=AMPI_PROBES,
                              fan_probes=AMPI_FAN_PROBES, window_size=AMPI_WINDOW)
        return ids.tolist()

    I_f = np.array([_query(q) for q in fashion_test])
    I_m = np.array([_query(q) for q in mnist_test])

    tag = (f"AMPI               "
           f"(nlist={ampi_nlist}, fans={NUM_FANS}, probes={AMPI_PROBES})")
    return {
        "ampi": {
            "tag": tag,
            "recall_fashion": round(recall_k(I_f, gt_fashion), 4),
            "recall_mnist":   round(recall_k(I_m, gt_mnist), 4),
            "insert_ms":      round(add_ms, 0),
            "splits": "auto", "imbalance": None,
        }
    }


def run_copenhagen(mnist_train, mnist_test, fashion_train, fashion_test,
                   n_clusters, nprobe, soft_k, split_threshold, tag, k=10):
    dim = mnist_train.shape[1]
    all_data = np.vstack([mnist_train, fashion_train])

    gt_fashion = _brute_knn(all_data, fashion_test, k)
    gt_mnist   = _brute_knn(all_data, mnist_test,   k)

    idx = DynamicIVF(dim, n_clusters, nprobe, 0, 8, 256, soft_k)
    idx.split_threshold = split_threshold
    idx.train(mnist_train)
    s0 = idx.get_stats()

    t0 = time.perf_counter()
    idx.insert_batch(fashion_train)
    ins_ms = (time.perf_counter() - t0) * 1000
    s1 = idx.get_stats()

    I_f = np.array([[ids for ids in [idx.search(q, k, nprobe)[0]]][0]
                    for q in fashion_test])
    I_m = np.array([[ids for ids in [idx.search(q, k, nprobe)[0]]][0]
                    for q in mnist_test])

    splits = s1['n_clusters'] - s0['n_clusters']
    imbal  = round(s1['max_cluster_size'] / max(s1['mean_cluster_size'], 1), 1)

    return {
        "tag": tag,
        "recall_fashion": round(recall_k(I_f, gt_fashion), 4),
        "recall_mnist":   round(recall_k(I_m, gt_mnist), 4),
        "insert_ms":      round(ins_ms, 0),
        "splits": splits, "imbalance": imbal,
        "max_cluster": s1['max_cluster_size'],
        "mean_cluster": round(s1['mean_cluster_size'], 1),
        "soft_k": soft_k, "split_threshold": split_threshold,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='Use full dataset (~15 min)')
    args = ap.parse_args()

    print("Copenhagen — MNIST → Fashion-MNIST Distribution Drift")
    print("=" * 62)

    mt, mq, ft, fq = load(quick=not args.full)
    print(f"  MNIST   train={len(mt):>6,}  queries={len(mq):>4,}")
    print(f"  Fashion train={len(ft):>6,}  queries={len(fq):>4,}  dim={mt.shape[1]}")

    N_CLUSTERS = 32
    NPROBE     = 4
    print(f"\n  Config: {N_CLUSTERS} clusters, nprobe={NPROBE} "
          f"(scanning {NPROBE/N_CLUSTERS*100:.0f}% of index per query)\n")

    all_results = {}

    # FAISS
    print("  Running FAISS baselines...")
    faiss_res = run_faiss(mt, mq, ft, fq, N_CLUSTERS, NPROBE)
    all_results.update(faiss_res)

    # AMPI
    if HAS_AMPI:
        print("  Running AMPI...")
        ampi_res = run_ampi(mt, mq, ft, fq, N_CLUSTERS, NPROBE)
        all_results.update(ampi_res)

    # Copenhagen variants
    cph_configs = [
        (1, 9999.0, "Copenhagen baseline  (fixed, soft_k=1)"),
        (1,  3.0,   "Copenhagen adaptive  (splits, soft_k=1)"),
        (2, 9999.0, "Copenhagen soft_k=2  (fixed, soft_k=2)"),
        (2,  3.0,   "Copenhagen best      (splits + soft_k=2)"),
    ]
    for soft_k, threshold, tag in cph_configs:
        print(f"  Running {tag}...")
        r = run_copenhagen(mt, mq, ft, fq, N_CLUSTERS, NPROBE,
                           soft_k, threshold, tag)
        all_results[tag] = r

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 44
    print("\n" + "=" * 78)
    print("RESULTS  (recall@10 vs brute-force ground truth on all data)")
    print("-" * 78)
    print(f"  {'method':<{W}} {'max/mean':>8}  {'fashion':>8}  {'mnist':>8}  {'insert':>9}")
    print(f"  {'-'*W} {'--------':>8}  {'-------':>8}  {'------':>8}  {'---------':>9}")

    def fmt_imbal(r):
        if r.get("imbalance") is None:
            return "     —"
        return f"{r['imbalance']:>5.1f}x"

    def fmt_splits(r):
        s = r.get("splits", 0)
        if s == "auto": return " (adaptive)"
        if s == 0:      return ""
        return f" (+{s} splits)"

    # Print rows in order
    order = ["faiss_add", "faiss_rebuild", "ampi",
             "Copenhagen baseline  (fixed, soft_k=1)",
             "Copenhagen adaptive  (splits, soft_k=1)",
             "Copenhagen soft_k=2  (fixed, soft_k=2)",
             "Copenhagen best      (splits + soft_k=2)"]

    sep_printed = False
    for key in order:
        if key not in all_results:
            continue
        r = all_results[key]
        # Print separator before Copenhagen rows
        if key.startswith("Copenhagen") and not sep_printed:
            print(f"  {'·'*W}")
            sep_printed = True
        print(f"  {r['tag']:<{W}} {fmt_imbal(r)}  "
              f"  {r['recall_fashion']:.4f}  "
              f"  {r['recall_mnist']:.4f}  "
              f"  {r['insert_ms']:>6.0f}ms{fmt_splits(r)}")

    # ── Key numbers ───────────────────────────────────────────────────────────
    print()
    cph_best = all_results.get("Copenhagen best      (splits + soft_k=2)", {})
    cph_base = all_results.get("Copenhagen baseline  (fixed, soft_k=1)", {})
    faiss_rb = all_results.get("faiss_rebuild", {})
    faiss_ao = all_results.get("faiss_add", {})

    if faiss_rb and cph_best:
        speedup = faiss_rb['insert_ms'] / max(cph_best['insert_ms'], 1)
        recall_delta = (cph_best['recall_fashion'] - faiss_rb['recall_fashion']) * 100
        print(f"  Copenhagen best  vs  FAISS rebuild: "
              f"{speedup:.1f}x faster insert, recall {recall_delta:+.2f}pp")
    if faiss_ao and cph_base:
        recall_delta = (cph_base['recall_fashion'] - faiss_ao['recall_fashion']) * 100
        print(f"  Copenhagen base  vs  FAISS add-only: recall {recall_delta:+.2f}pp")

    os.makedirs('/Users/sasha/copenhagen/results', exist_ok=True)
    out = '/Users/sasha/copenhagen/results/drift.json'
    with open(out, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n  Results → {out}")


if __name__ == '__main__':
    main()
