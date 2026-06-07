"""End-to-end validation of the SIMD fast-scan TQ path.

On synthetic Gaussian data, absolute recall is low (distance concentration in
high dim) — even an exact IVF scan caps around 0.4 at these probe counts. So we
test the fast-scan path by PARITY against the exact (`quant="none"`) IVF on the
same data and config: the 4-bit blocked SIMD path should track exact recall
closely, and must never resurface a deleted id. This isolates "is the kernel +
blocked layout correct" from "is the dataset hard". Real-dataset recall numbers
come from the benchmark matrix, not here.

Run: python -m pytest tests/test_fastscan_recall.py -v -s
"""
import numpy as np
import pytest
from python.core import CopenhagenIndex


def _ground_truth(data, queries, k):
    gt = np.empty((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        gt[i] = np.argsort(np.sum((data - q) ** 2, axis=1))[:k]
    return gt


def _recall(index, queries, gt, k):
    ids, _ = index.search_batch(queries, k=k)
    hits = sum(len(set(np.asarray(ids[i]).tolist()) & set(gt[i][:k].tolist()))
               for i in range(len(queries)))
    return hits / (len(queries) * k)


def _build(dim, data, **kw):
    idx = CopenhagenIndex(dim=dim, n_clusters=64, nprobe=12, **kw)
    idx.add(data)
    return idx


@pytest.mark.parametrize("dim,bits", [(128, 4), (768, 4), (256, 3), (256, 2)])
def test_fastscan_tracks_exact(dim, bits):
    """Blocked SIMD fast-scan (bits<=4) recall must track the exact IVF path."""
    rng = np.random.default_rng(0)
    n, nq, k = 20000, 200, 10
    data = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((nq, dim)).astype(np.float32)
    gt = _ground_truth(data, queries, k)

    r_exact = _recall(_build(dim, data, quant="none"), queries, gt, k)
    r_tq = _recall(_build(dim, data, quant="tq", tq_bits=bits), queries, gt, k)
    print(f"\n  dim={dim} bits={bits}: exact={r_exact:.3f} fastscan={r_tq:.3f} "
          f"(delta {r_tq - r_exact:+.3f})")
    # Quantization can only lose a little vs exact; fewer bits loses a bit more.
    floor = 0.08 if bits >= 4 else 0.15
    assert r_tq > r_exact - floor, f"fastscan {r_tq} trails exact {r_exact} by too much"


def test_blocked_vs_scalar_parity():
    """bits=4 (blocked SIMD kernel) vs bits=5 (row-major scalar score_ip):
    both are TQ; the 4-bit blocked path must not trail the 5-bit scalar path by
    more than the bit-depth difference warrants."""
    rng = np.random.default_rng(1)
    dim, n, nq, k = 256, 20000, 200, 10
    data = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((nq, dim)).astype(np.float32)
    gt = _ground_truth(data, queries, k)

    r_blocked = _recall(_build(dim, data, quant="tq", tq_bits=4), queries, gt, k)
    r_scalar = _recall(_build(dim, data, quant="tq", tq_bits=5), queries, gt, k)
    print(f"\n  recall: blocked(bits=4)={r_blocked:.3f}  scalar(bits=5)={r_scalar:.3f}")
    assert r_blocked > r_scalar - 0.08, f"blocked {r_blocked} << scalar {r_scalar}"


def test_recall_survives_churn():
    """Delete (tombstones + blocked-layout compaction) then insert (growth +
    splits), and confirm the fast-scan path still tracks an exact index built on
    the surviving vectors, and never returns a deleted id."""
    rng = np.random.default_rng(2)
    dim, n, nq, k = 128, 24000, 200, 10
    data = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((nq, dim)).astype(np.float32)

    idx = CopenhagenIndex(dim=dim, n_clusters=32, nprobe=10, quant="tq", tq_bits=4)
    idx.add(data[:12000])
    dead = np.arange(0, 4000, dtype=np.int32)
    idx.delete(dead)
    idx.add(data[12000:])

    dead_set = set(dead.tolist())
    alive = np.array([i for i in range(n) if i not in dead_set])
    gt_alive = _ground_truth(data[alive], queries, k)
    gt = alive[gt_alive]  # map back to original ids

    ids, _ = idx.search_batch(queries, k=k)
    hits = 0
    for i in range(nq):
        got = set(np.asarray(ids[i]).tolist())
        assert not (got & dead_set), "deleted id resurfaced!"
        hits += len(got & set(gt[i].tolist()))
    r = idx_recall = hits / (nq * k)

    # Exact reference on surviving vectors at the same config.
    exact = CopenhagenIndex(dim=dim, n_clusters=32, nprobe=10, quant="none")
    exact.add(data[alive])
    eids, _ = exact.search_batch(queries, k=k)
    ehits = sum(len(set(np.asarray(eids[i]).tolist()) & set(gt_alive[i].tolist()))
                for i in range(nq))
    r_exact = ehits / (nq * k)
    print(f"\n  post-churn recall@{k}: fastscan={r:.3f} exact={r_exact:.3f} "
          f"(n_vectors={idx.n_vectors})")
    assert r > r_exact - 0.10, f"post-churn fastscan {r} trails exact {r_exact}"
