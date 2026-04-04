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

# ── soft_k=2 basic smoke ──────────────────────────────────────────────────────
idx_sk = CopenhagenIndex(dim=64, n_clusters=16, nprobe=8, soft_k=2)
idx_sk.add(rng.standard_normal((500, 64)).astype("float32"))
ids_sk, _ = idx_sk.search(rng.standard_normal(64).astype("float32"), k=10)
assert len(ids_sk) == len(set(ids_sk)), "soft_k=2: duplicate IDs in results"
print("soft_k=2: OK")

# ── search_batch ──────────────────────────────────────────────────────────────
idx_b = CopenhagenIndex(dim=64, n_clusters=16, nprobe=8)
idx_b.add(rng.standard_normal((500, 64)).astype("float32"))
batch_qs = rng.standard_normal((5, 64)).astype("float32")
all_ids, all_dists = idx_b.search_batch(batch_qs, k=10)
assert len(all_ids) == 5, f"search_batch: expected 5 id lists, got {len(all_ids)}"
assert len(all_dists) == 5, f"search_batch: expected 5 dist lists, got {len(all_dists)}"
for ids_r, dists_r in zip(all_ids, all_dists):
    assert len(ids_r) == 10
    assert len(dists_r) == 10
print("search_batch: OK")

# ── brute_force_search ────────────────────────────────────────────────────────
bf_data = rng.standard_normal((200, 64)).astype("float32")
idx_bf = CopenhagenIndex(dim=64, n_clusters=8, nprobe=8)
idx_bf.add(bf_data)
q_bf = bf_data[0]
bf_ids, bf_dists = idx_bf.brute_force_search(q_bf, k=5)
assert bf_ids[0] == 0, f"brute_force_search: nearest to data[0] should be itself, got {bf_ids[0]}"
assert len(bf_ids) == 5
print("brute_force_search: OK")

# ── save / load roundtrip ─────────────────────────────────────────────────────
import tempfile, os
save_data = rng.standard_normal((300, 64)).astype("float32")
idx_save = CopenhagenIndex(dim=64, n_clusters=16, nprobe=8, soft_k=2)
idx_save.add(save_data)
idx_save.delete(0)
idx_save.delete(1)
with tempfile.TemporaryDirectory() as tmpdir:
    idx_save.save(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "clusters.npz"))
    assert os.path.exists(os.path.join(tmpdir, "metadata.json"))
    idx_load = CopenhagenIndex.load(tmpdir)
assert idx_load.n_vectors == idx_save.n_vectors, "save/load: n_vectors mismatch"
assert idx_load.get_stats()["deleted_count"] == 2, "save/load: deleted_ids not preserved"
q_sl = rng.standard_normal(64).astype("float32")
ids_orig, _ = idx_save.search(q_sl, k=10)
ids_load, _ = idx_load.search(q_sl, k=10)
assert 0 not in ids_load and 1 not in ids_load, "save/load: deleted IDs returned after reload"
print("save/load: OK")

# ── compact ───────────────────────────────────────────────────────────────────
idx_c = CopenhagenIndex(dim=64, n_clusters=8, nprobe=8)
idx_c.add(rng.standard_normal((200, 64)).astype("float32"))
for gid in range(50):
    idx_c.delete(gid)
assert idx_c.get_stats()["deleted_count"] == 50
idx_c.compact()
assert idx_c.get_stats()["deleted_count"] == 0, "compact: deleted_count not cleared"
ids_c, _ = idx_c.search(rng.standard_normal(64).astype("float32"), k=10)
assert not any(i < 50 for i in ids_c), "compact: deleted ID returned after compact"
print("compact: OK")

# ── get_stats fields ──────────────────────────────────────────────────────────
idx_gs = CopenhagenIndex(dim=32, n_clusters=8, nprobe=4)
idx_gs.add(rng.standard_normal((100, 32)).astype("float32"))
stats = idx_gs.get_stats()
for field in ("n_vectors", "n_clusters", "deleted_count", "max_cluster_size",
              "mean_cluster_size", "dim", "soft_k"):
    assert field in stats, f"get_stats: missing field '{field}'"
assert stats["n_vectors"] == 100
assert stats["n_clusters"] == 8
assert stats["deleted_count"] == 0
print("get_stats: OK")

# ── __repr__ ──────────────────────────────────────────────────────────────────
r = repr(idx_gs)
assert "CopenhagenIndex" in r and "dim=32" in r, f"__repr__ unexpected: {r}"
print("__repr__: OK")

# ── writable attributes ───────────────────────────────────────────────────────
idx_wa = CopenhagenIndex(dim=32, n_clusters=8, nprobe=4)
idx_wa.add(rng.standard_normal((100, 32)).astype("float32"))
idx_wa._index.split_threshold = 2.0
idx_wa._index.soft_k = 2
idx_wa._index.max_split_iters = 5
assert idx_wa._index.split_threshold == 2.0
assert idx_wa._index.soft_k == 2
assert idx_wa._index.max_split_iters == 5
print("writable attributes: OK")

# ── use_pq=True basic smoke ───────────────────────────────────────────────────
pq_data = rng.standard_normal((500, 64)).astype("float32")
idx_pq = CopenhagenIndex(dim=64, n_clusters=16, nprobe=8, use_pq=True, pq_m=8)
idx_pq.add(pq_data)
ids_pq, dists_pq = idx_pq.search(pq_data[0], k=10)
assert len(ids_pq) == 10, f"PQ search: expected 10 results, got {len(ids_pq)}"
assert 0 in ids_pq, "PQ search: nearest to data[0] should include itself"
print("use_pq=True: OK")

# ── mmap basic smoke ──────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as mmap_dir:
    idx_mm = CopenhagenIndex(dim=64, n_clusters=8, nprobe=8, use_mmap=True, mmap_dir=mmap_dir)
    mmap_data = rng.standard_normal((300, 64)).astype("float32")
    idx_mm.add(mmap_data)
    ids_mm, _ = idx_mm.search(mmap_data[0], k=5)
    assert 0 in ids_mm, "mmap: nearest to data[0] should include itself"

    # save / reload
    import tempfile as _tf
    with _tf.TemporaryDirectory() as save_dir:
        idx_mm.save(save_dir)
        import os as _os
        assert _os.path.exists(_os.path.join(save_dir, "clusters_meta.npz"))
        assert not _os.path.exists(_os.path.join(save_dir, "clusters.npz")), \
            "mmap save should not write clusters.npz"
        idx_mm2 = CopenhagenIndex.load(save_dir)
    ids_mm2, _ = idx_mm2.search(mmap_data[0], k=5)
    assert 0 in ids_mm2, "mmap load: nearest to data[0] should include itself"
print("mmap: OK")

print("ALL OK")
