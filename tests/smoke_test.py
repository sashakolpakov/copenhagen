"""
Smoke test for DynamicIVF: build small indexes, verify basic functionality.
Runs in ~10s on a laptop, no datasets required.
"""
import numpy as np
import faiss

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from index import DynamicIVFIndex

rng = np.random.default_rng(42)
n, d = 5_000, 64
data = rng.standard_normal((n, d)).astype("float32")
qs = rng.standard_normal((50, d)).astype("float32")

# Ground truth
flat = faiss.IndexFlatL2(d)
flat.add(data)
_, gt = flat.search(qs, 10)


def recall10(gt, found):
    hits = sum(len(set(g.tolist()) & set(f[:10].tolist())) for g, f in zip(gt, found))
    return hits / (len(gt) * 10)


# ── Basic IVF ─────────────────────────────────────────────────────────────────
idx = DynamicIVFIndex(dim=d, n_clusters=32, nprobe=8)
idx.add(data)
results = [idx.search(q, k=10, n_probes=8) for q in qs]
found = [np.array(found[0]) for found in results]
rec = recall10(gt, found)
print(f"DynamicIVF recall@10 = {rec:.3f}")
assert rec >= 0.70, f"Recall too low: {rec:.3f}"

# ── Delete functionality ──────────────────────────────────────────────────────
idx2 = DynamicIVFIndex(dim=d, n_clusters=32, nprobe=8)
idx2.add(data[:1000])
n_before = idx2.n_vectors

# Add a spike (last index will be n_before)
spike = np.zeros(d, dtype=np.float32)
spike[0] = 1e4
idx2.add(spike)
n_after = idx2.n_vectors
gid = n_before  # Last added vector has ID = n_before
assert n_after == n_before + 1, f"Expected {n_before+1} vectors, got {n_after}"

# Query should find the spike (at position n_before)
ids, _ = idx2.search(spike, k=5, n_probes=32)
ids = list(ids)
assert gid in ids, f"Inserted spike (gid={gid}) not found in {ids}"

# Delete it
idx2.delete(gid)
assert idx2.n_vectors == n_before, "Vector count should be back to original"

# Query should not find it anymore
ids_after, _ = idx2.search(spike, k=20, n_probes=32)
ids_after = list(ids_after)
assert gid not in ids_after, f"Deleted spike (gid={gid}) still returned in {ids_after}"

print("OK")
