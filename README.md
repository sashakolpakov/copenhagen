# Copenhagen

Copenhagen is a fully dynamic approximate nearest neighbor (ANN) index based on
Inverted File Index (IVF) that handles continuous inserts and deletes without
ever rebuilding. Standard ANN indexes (FAISS IVF, HNSW) assume a static or
slowly-changing dataset; when data drifts — new product SKUs, freshly-published
articles, continuous message embeddings — their centroids no longer represent
the full data manifold, clusters grow massively imbalanced, and recall collapses
for new-data queries. Copenhagen solves this with three complementary mechanisms:
O(1) tombstone deletion with lazy compaction, soft multi-cluster assignment to
maintain recall near Voronoi boundaries, and adaptive cluster splitting that adds
new centroids only where distribution drift has overloaded an existing cell —
all without an offline rebuild step. The design is inspired by the logarithmic-method
and deletion-only structures from arXiv:2604.00271 ("Engineering Fully Dynamic
Convex Hulls", IT University of Copenhagen).

---

## When to use Copenhagen

Copenhagen is the right tool when **insert throughput matters more than knowing
why the distribution shifted**.

**E-commerce product catalog** — 10k new SKUs per hour, the recommendation index
must stay live. At 0.4µs/vector insert, you never go offline to rebuild. FAISS IVF
requires a full retrain (minutes) to recover recall after a catalog expansion;
Copenhagen handles it incrementally with adaptive cluster splitting.

**News feed / real-time content index** — articles are indexed as they are published,
stale content is deleted as it ages out. FAISS IVF has no real delete primitive on
`IndexIVFFlat` — you need a full rebuild or a lossy `remove_ids` workaround.
Copenhagen's O(1) tombstone delete compacts lazily at search time: 0.3µs per delete,
zero leaked results.

**Chat / document embedding cache** — millions of message embeddings accumulate
continuously. soft_k=2 keeps recall high at Voronoi boundaries without any offline
step. Storage overhead is exactly 2× vs standard IVF.

**When NOT to use Copenhagen** — if you need to detect *which part* of the
distribution drifted, or *in what direction*, Copenhagen only tells you a cluster
grew too large and splits it. For drift geometry and directional signals, prefer
[AMPI](https://github.com/sashakolpakov/ampi), which tracks each cluster's
subspace via an Oja sketch and alerts when the drift angle exceeds a threshold.

---

## Quick start

### Install (pip — pre-built wheel)

```bash
pip install copenhagen-ann
```

### Build from source

```bash
cd src && bash build.sh
```

The build requires a C++17 compiler and Apple Accelerate (macOS) or OpenBLAS (Linux).
The resulting shared library is placed at `python/core/copenhagen.so`.

### Minimal Python example

```python
import numpy as np
from python.core import CopenhagenIndex

dim       = 128
n_clusters = 32

# 1. Create and train the index
idx    = CopenhagenIndex(dim, n_clusters, nprobe=4, soft_k=2)
train  = np.random.randn(10_000, dim).astype(np.float32)
idx._index.train(train)

# 2. Insert a batch of vectors
vecs = np.random.randn(5_000, dim).astype(np.float32)
idx.add(vecs)                          # insert_batch under the hood

# 3. Search
query       = np.random.randn(dim).astype(np.float32)
ids, dists  = idx.search(query, k=10)

# 4. Delete by ID
idx.delete(ids[0])                     # O(1) tombstone

# 5. Compact (flush all tombstones immediately)
idx.compact()                          # also auto-triggers at 10 % churn

# 6. Save and reload
idx.save("my_index/")
idx2 = CopenhagenIndex.load("my_index/")
```

---

## Key features

- **Soft assignments** — each vector is indexed in its top-`soft_k` nearest
  clusters, improving recall near Voronoi boundaries without any query-side change.
- **Adaptive cluster splitting** — when drift causes a cell to exceed
  `mean_size × split_threshold`, mini k-means (k=2) splits it and adds a new
  centroid; no full rebuild needed.
- **O(1) tombstone delete** — `delete_vector` marks an ID in an unordered set;
  physical eviction happens lazily during search via a single in-place pass.
- **Auto-compact** — `compact()` is called automatically by `insert_batch` when
  deleted IDs exceed 10 % of `n_vectors`; call it manually for bulk deletes.
- **Save / load (.npz)** — `save(path)` writes `clusters.npz` and
  `metadata.json`; `CopenhagenIndex.load(path)` restores the full index state
  including tombstones and `id_to_location`.
- **GPU acceleration** — `CopenhagenIndex(device="cuda"|"mps"|"cpu")` offloads
  centroid distance computation via `torch.mm`; PyTorch dispatches to cuBLAS,
  rocBLAS, or Metal MPS transparently. Centroids are pinned once at train time;
  only argmin indices are transferred back. Install `torch` to enable.
- **BLAS-accelerated** — all dense distance computation uses `cblas_sgemm`
  via Apple Accelerate; cluster storage is 32-byte aligned for AVX/NEON
  cache efficiency.
- **Correct move semantics** — `Cluster` and `PQCluster` carry explicit move
  constructors, preventing double-free when `std::vector` reallocates during
  `emplace_back` on cluster splits.

---

## Benchmark results

Full results with tables and run instructions: **[BENCHMARKS.md](BENCHMARKS.md)**

Headline numbers:

- **Streaming churn (30% delete/round, n=50k, 10 rounds)**: CPH holds 0.93–0.95
  recall@10 while HNSW+filter collapses to 0.46 and FAISS IVF+filter to 0.66.
  HNSW+rebuild and IVF+rebuild restore recall each round but cost 10–20k and
  100–200k effective inserts/s respectively; CPH runs at 1M+ inserts/s and
  1M+ deletes/s throughout.

- **Distribution drift (MNIST → Fashion-MNIST, 784d)**: CPH soft_k=2 matches
  FAISS full-rebuild recall (0.99) at 3.6× faster insert (45 ms vs 164 ms).

- **Tombstone delete**: 0.6–1.1 µs per delete, zero leaked results.
  FAISS IVF has no native delete; equivalent via rebuild costs ~88 ms.

---

## Parameters

| Parameter         | Default | Effect                                                             |
|-------------------|---------|--------------------------------------------------------------------|
| `soft_k`          | 1       | Clusters per vector at index time (1 = standard IVF, 2–3 = better recall near boundaries) |
| `split_threshold` | 3.0     | Split a cluster when its live size exceeds `mean_size × threshold` |
| `max_split_iters` | 10      | Mini k-means iterations inside `split_cluster`                     |
| `nprobe`          | 1       | Number of clusters probed per query                                |

`split_threshold` and `soft_k` are writable Python attributes on the index object:

```python
idx._index.split_threshold = 2.5
idx._index.soft_k          = 2
```

---

## Building from source

```bash
cd src && bash build.sh
```

Requirements: C++17, Apple Accelerate (macOS) or OpenBLAS (Linux). The build
script compiles `dynamic_ivf.cpp` and deposits `copenhagen.so` into
`python/core/`.

---

## Repository layout

```
src/                  C++ extension (dynamic_ivf.cpp, build.sh)
python/core/          Python wrapper (CopenhagenIndex), __init__.py, copenhagen.so
benchmarks/           Drift, MNIST, and cumulative-update benchmark scripts
data/                 Dataset storage (SIFT, MNIST, Fashion-MNIST)
results/              Benchmark output (CSV / plots)
```

---

## Credits

- **arXiv:2604.00271** — van der Hoog, Reinstädtler, Rotenberg (IT University of
  Copenhagen). The logarithmic-method bucket structure, deletion-only hull analysis,
  and tombstone correctness argument directly informed the Copenhagen design.
- **AMPI** ([github.com/sashakolpakov/ampi](https://github.com/sashakolpakov/ampi))
  — Adaptive multi-projection index with per-cluster Oja drift detection. Used as
  a comparison baseline in the drift benchmarks; recommended when directional drift
  signals are needed rather than pure insert throughput.
- **FAISS** (Facebook AI Research) — IVF baseline for all benchmarks.
