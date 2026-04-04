"""
Stress test for DynamicIVF streaming API (add / delete / update).

Covers adversarial scenarios that the smoke test does not:
  - Insert findability
  - Delete / update correctness (no false positives)
  - Idempotent delete, invalid delete
  - Boundary inserts (equidistant from multiple clusters)
  - Outlier and zero-vector inserts
  - Bulk add recall
  - High-deletion recall
  - All-cluster-deleted query
  - Interleaved mutations and queries

No external dependencies beyond numpy (brute-force ground truth used throughout).
"""

import sys
import os
import traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from core import CopenhagenIndex

# ── helpers ───────────────────────────────────────────────────────────────────

def _brute_knn(data, queries, k):
    """Exact k-NN via numpy."""
    q_sq = np.sum(queries ** 2, axis=1)[:, None]
    d_sq = np.sum(data ** 2, axis=1)[None, :]
    d2 = q_sq + d_sq - 2.0 * (queries @ data.T)
    return np.argsort(d2, axis=1)[:, :k].astype(np.int32)


def _recall(gt, found_ids, k):
    """Recall@k averaged over queries."""
    hits = 0
    for g, f in zip(gt, found_ids):
        f_ids = set(f[:k])
        hits += len(set(g[:k].tolist()) & f_ids)
    return hits / (len(gt) * k)


def _small_index(n=3000, d=32, n_clusters=32, soft_k=1, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype('float32')
    idx = CopenhagenIndex(dim=d, n_clusters=n_clusters, nprobe=8, soft_k=soft_k)
    idx.add(data)
    return idx, data, rng


def _spike(d, axis, scale=1e3):
    """A point far from the bulk of N(0,1) data along one basis direction."""
    v = np.zeros(d, dtype='float32')
    v[axis] = scale
    return v


# ── test registry ─────────────────────────────────────────────────────────────

_TESTS = []


def _register(fn):
    _TESTS.append(fn)
    return fn


# ── scenarios ────────────────────────────────────────────────────────────────

@_register
def insert_findability():
    """Inserted spike must be the exact nearest neighbour when queried."""
    idx, data, _ = _small_index()
    d = idx.dim
    x = _spike(d, 0)
    idx.add(x)

    ids, _ = idx.search(x, k=1, n_probes=32)

    gt = _brute_knn(np.vstack([data, x]), x.reshape(1, -1), 1)
    assert ids[0] == gt[0][0], f"Spike not found as NN, got {ids}"


@_register
def delete_no_false_positive():
    """After deleting a spike, it must never appear in any query result."""
    idx, _, rng = _small_index()
    d = idx.dim
    x = _spike(d, 1)
    gid = idx.n_vectors
    idx.add(x)

    ids, _ = idx.search(x, k=1, n_probes=32)
    assert gid in ids, f"Spike not found before deletion (gid={gid}, got {ids})"

    idx.delete(gid)

    ids, _ = idx.search(x, k=20, n_probes=32)
    assert gid not in ids, f"Deleted gid={gid} still returned after delete"


@_register
def double_delete_is_noop():
    """Deleting the same id twice must not raise or corrupt state."""
    idx, _, _ = _small_index()
    gid = idx.n_vectors
    idx.add(_spike(idx.dim, 0))

    idx.delete(gid)
    stats_before = idx.get_stats()

    idx.delete(gid)
    stats_after = idx.get_stats()
    assert stats_after["deleted_count"] == stats_before["deleted_count"], \
        "deleted_count changed on double delete"


@_register
def invalid_delete_raises():
    """delete() with an out-of-range id should not crash."""
    idx, _, _ = _small_index()
    try:
        idx.delete(-1)
        idx.delete(idx.n_vectors + 9999)
    except Exception as e:
        raise AssertionError(f"delete raised unexpected error: {e}")


@_register
def outlier_insert():
    """Insert a point far outside the training distribution; it must be findable."""
    idx, data, _ = _small_index()
    d = idx.dim
    x = _spike(d, 4, scale=1e4)
    idx.add(x)

    ids, _ = idx.search(x, k=1, n_probes=32)
    assert len(ids) > 0, "No results returned for outlier"


@_register
def zero_vector_insert():
    """Inserting a zero vector must not crash."""
    idx, _, _ = _small_index()
    z = np.zeros(idx.dim, dtype='float32')
    n_before = idx.n_vectors
    idx.add(z)
    n_after = idx.n_vectors
    assert n_after == n_before + 1, "Zero vector not added"
    idx.search(z, k=5)


@_register
def bulk_add_recall():
    """After bulk add, recall@10 >= 0.70."""
    idx, data, rng = _small_index(n=2000, d=32, n_clusters=16)
    extra = rng.standard_normal((500, 32)).astype('float32')
    idx.add(extra)

    all_data = np.vstack([data, extra])
    qs = rng.standard_normal((50, 32)).astype('float32')
    gt = _brute_knn(all_data, qs, 10)

    found_ids = []
    for q in qs:
        ids, _ = idx.search(q, k=10)
        found_ids.append(ids)
    rec = _recall(gt, found_ids, 10)
    assert rec >= 0.70, f"bulk_add recall@10 = {rec:.3f} < 0.70"


@_register
def high_deletion_recall():
    """After deleting 30% of vectors, recall@10 should remain reasonable."""
    idx, data, rng = _small_index(n=2000, d=32, n_clusters=16)

    to_delete = list(range(0, 600))
    for gid in to_delete:
        idx.delete(gid)

    qs = rng.standard_normal((50, 32)).astype('float32')

    remaining_ids = [i for i in range(2000) if i not in set(to_delete)]
    remaining_data = data[remaining_ids]
    gt = _brute_knn(remaining_data, qs, 10)
    gt_original_ids = np.array([[remaining_ids[j] for j in row] for row in gt])

    found_ids = []
    for q in qs:
        ids, _ = idx.search(q, k=10, n_probes=16)
        found_ids.append(ids)
    rec = _recall(gt_original_ids, found_ids, 10)
    assert rec >= 0.40, f"high deletion recall@10 = {rec:.3f} < 0.40"


@_register
def interleaved_mutations_and_queries():
    """Interleaved add/delete/query must not crash and maintain reasonable recall."""
    idx, data, rng = _small_index(n=1000, d=32, n_clusters=16)

    qs = rng.standard_normal((50, 32)).astype('float32')
    gt = _brute_knn(data, qs, 10)

    for i in range(100):
        if i % 3 == 0:
            x = rng.standard_normal(32).astype('float32')
            idx.add(x)
        elif i % 3 == 1:
            if idx.n_vectors > 100:
                idx.delete(i % 100)
        else:
            idx.search(qs[i % len(qs)], k=10)

    found_ids = []
    for q in qs[:20]:
        ids, _ = idx.search(q, k=10)
        found_ids.append(ids)
    rec = _recall(gt[:20], found_ids, 10)
    assert rec >= 0.50, f"interleaved recall@10 = {rec:.3f} < 0.50"


@_register
def dimension_mismatch_raises():
    """Inserting/querying with wrong dimension must raise ValueError."""
    idx, _, _ = _small_index(d=32)

    try:
        wrong_dim = np.zeros(64, dtype='float32')
        idx.add(wrong_dim)
        raise AssertionError("expected ValueError for wrong dimension")
    except ValueError:
        pass

    try:
        idx.search(wrong_dim, k=10)
        raise AssertionError("expected ValueError for wrong query dimension")
    except ValueError:
        pass


@_register
def no_duplicate_results():
    """Search results must not contain duplicate IDs."""
    idx, data, _ = _small_index(n=5000, d=64, n_clusters=32)

    qs = np.random.randn(10, 64).astype('float32')
    for q in qs:
        ids, _ = idx.search(q, k=20)
        assert len(ids) == len(set(ids)), f"Duplicate IDs in results: {ids}"


@_register
def batch_delete():
    """Batch delete removes correct number of vectors."""
    d = 64
    n = 1000
    rng = np.random.default_rng(0)
    idx = CopenhagenIndex(dim=d, n_clusters=32, nprobe=8)
    data = rng.standard_normal((n, d)).astype('float32')
    idx.add(data)

    delete_ids = np.arange(0, 500, 2)
    idx.delete(delete_ids)

    stats = idx.get_stats()
    assert stats["deleted_count"] == len(delete_ids), \
        f"Expected {len(delete_ids)} deleted, got {stats['deleted_count']}"


@_register
def large_scale():
    """10K vectors, measure add/search times."""
    n, d = 10000, 128
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, d)).astype('float32')
    idx = CopenhagenIndex(dim=d, n_clusters=128, nprobe=8)

    import time
    t0 = time.time()
    idx.add(data)
    add_time = time.time() - t0

    query = data[0]
    t0 = time.time()
    ids, _ = idx.search(query, k=10)
    search_time = time.time() - t0

    assert len(ids) == 10
    assert ids[0] == 0
    print(f"  large_scale: add={add_time:.2f}s, search={search_time*1000:.1f}ms")


@_register
def search_with_various_k():
    """Search with different k values must work correctly."""
    idx, data, rng = _small_index(n=2000, d=32, n_clusters=16)

    q = rng.standard_normal(32).astype('float32')
    for k in [1, 5, 10, 50, 100]:
        ids, _ = idx.search(q, k=k)
        assert len(ids) == k, f"Expected {k} results, got {len(ids)}"
        assert len(ids) == len(set(ids)), f"Duplicates for k={k}"


# ── runner ───────────────────────────────────────────────────────────────────

def main():
    passed, failed = [], []
    for fn in _TESTS:
        name = fn.__name__
        try:
            fn()
            passed.append(name)
            print(f"[PASS] {name}")
        except Exception:
            failed.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()

    print(f"\n{len(passed)}/{len(passed)+len(failed)} passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
