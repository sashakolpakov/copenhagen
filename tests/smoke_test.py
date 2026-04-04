"""
Smoke test for DynamicIVF: build small indexes, verify basic functionality.
Runs in ~10s on a laptop, no datasets required.
"""
import numpy as np
import faiss

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from core import CopenhagenIndex

rng = np.random.default_rng(42)

# ── Basic IVF with 2 clusters (works correctly) ─────────────────────────────────
n, d = 5_000, 64
data = rng.standard_normal((n, d)).astype("float32")
qs = rng.standard_normal((50, d)).astype("float32")

flat = faiss.IndexFlatL2(d)
flat.add(data)
_, gt = flat.search(qs, 10)


def recall10(gt, found):
    hits = sum(len(set(g.tolist()) & set(f[:10].tolist())) for g, f in zip(gt, found))
    return hits / (len(gt) * 10)


idx = CopenhagenIndex(dim=d, n_clusters=32, nprobe=32)
idx.add(data)
results = [idx.search(q, k=10) for q in qs]
found = [np.array(found[0]) for found in results]
rec = recall10(gt, found)
print(f"DynamicIVF recall@10 = {rec:.3f}")
assert rec >= 0.95, f"Recall too low: {rec:.3f}"

# ── Delete functionality ──────────────────────────────────────────────────────
idx2 = CopenhagenIndex(dim=d, n_clusters=32, nprobe=32)
idx2.add(data[:1000])
n_before = idx2.n_vectors

spike = np.zeros(d, dtype=np.float32)
spike[0] = 1e4
idx2.add(spike)
n_after = idx2.n_vectors
gid = n_before
assert n_after == n_before + 1, f"Expected {n_before+1} vectors, got {n_after}"

ids, _ = idx2.search(spike, k=5)
assert gid in list(ids), f"Inserted spike (gid={gid}) not found"

idx2.delete(gid)
assert idx2.get_stats()["deleted_count"] == 1, "Tombstone not set"

ids_after, _ = idx2.search(spike, k=20)
assert gid not in list(ids_after), f"Deleted spike (gid={gid}) still returned"

# ── Smoke: different k values ─────────────────────────────────────────────────
idx3 = CopenhagenIndex(dim=d, n_clusters=32, nprobe=32)
idx3.add(data[:1000])
q = data[0]
for k in [1, 5, 10, 50]:
    ids, _ = idx3.search(q, k=k)
    assert len(ids) == k, f"Expected {k} results, got {len(ids)}"
print("k values: OK")

# ── Smoke: different dimensions ────────────────────────────────────────────────
for d_test, nc in [(128, 32), (256, 64)]:
    data_d = rng.standard_normal((500, d_test)).astype("float32")
    idx_d = CopenhagenIndex(dim=d_test, n_clusters=nc, nprobe=nc)
    idx_d.add(data_d)
    ids, _ = idx_d.search(data_d[0], k=10)
    assert len(ids) == 10
    print(f"dim={d_test}, n_clusters={nc}: OK")

print("OK")
