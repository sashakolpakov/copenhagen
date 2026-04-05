# Copenhagen

A fully dynamic approximate nearest neighbor (ANN) index built on Inverted File Index (IVF). Handles continuous inserts and deletes without ever rebuilding.

Standard ANN indexes (FAISS IVF, HNSW) assume a static or slowly-changing dataset. When data drifts — new product SKUs, freshly-published articles, continuous message embeddings — their centroids no longer represent the full data manifold, clusters grow massively imbalanced, and recall collapses for new-data queries. Copenhagen solves this with three mechanisms that compose:

- **O(1) amortized insert** — per-vector cost is constant in n (fixed clusters k, dimension d); stays flat from 5k to 100k+ vectors on real data while HNSW insert cost grows 30× over the same range
- **O(1) tombstone delete** with lazy compaction at search time
- **Soft multi-cluster assignment** (`soft_k`) for recall near Voronoi boundaries
- **Adaptive cluster splitting** — adds new centroids only where drift has overloaded an existing cell, no offline rebuild

The design draws on the logarithmic-method bucket structure and tombstone correctness argument from [arXiv:2604.00271](https://arxiv.org/abs/2604.00271) ("Engineering Fully Dynamic Convex Hulls", IT University of Copenhagen).

---

## When to use Copenhagen

**Use Copenhagen when insert throughput and live deletes matter more than explaining why the distribution shifted.**

| Scenario | Why Copenhagen |
|---|---|
| E-commerce catalog — 10k new SKUs/hour | 0.4 µs/vector insert, never goes offline; adaptive splits recover recall after catalog expansions |
| News / real-time content index | O(1) tombstone delete (0.3 µs/delete, zero leaked results); FAISS IVF has no native delete primitive |
| Chat / document embedding cache | `soft_k=2` keeps recall high at Voronoi boundaries; storage overhead is exactly 2× standard IVF |
| Streaming with gradual drift | Recall stays stable as new-distribution vectors arrive in batches; FAISS degrades monotonically |

**When NOT to use Copenhagen** — if you need to detect *where* or *in what direction* the distribution drifted. Copenhagen only tells you a cluster grew too large and splits it. For directional drift signals, use [AMPI](https://github.com/sashakolpakov/ampi), which tracks per-cluster subspace drift via Oja sketches.

---

## Quick start

### Build from source

```bash
cd src && bash build.sh
```

Requires C++17 and Apple Accelerate (macOS) or OpenBLAS (Linux). Deposits `copenhagen.so` into `python/core/`.

### Basic usage

```python
import numpy as np
from python.core import CopenhagenIndex

idx = CopenhagenIndex(dim=128, n_clusters=32, nprobe=4, soft_k=2)

# Train + insert in one call — first add() trains on the batch, subsequent adds insert
train = np.random.randn(10_000, 128).astype(np.float32)
idx.add(train)

# Insert more (incremental, no rebuild)
new_vecs = np.random.randn(5_000, 128).astype(np.float32)
idx.add(new_vecs)

# Search
query = np.random.randn(128).astype(np.float32)
ids, dists = idx.search(query, k=10)

# Delete by ID (O(1) tombstone)
idx.delete(ids[0])

# Compact tombstones immediately (also auto-fires at 10% churn)
idx.compact()

# Save / load
idx.save("my_index/")
idx2 = CopenhagenIndex.load("my_index/")
```

### Streaming insert

For large streams that shouldn't be materialised in memory:

```python
def vector_generator():
    for chunk in fetch_from_kafka():
        yield chunk.astype(np.float32)   # shape (dim,) or (n, dim)

first_id, last_id = idx.insert_stream(vector_generator(), chunk_size=1000)
```

### GPU acceleration

```python
# Offloads centroid distance computation via torch.mm (cuBLAS / Metal MPS)
idx = CopenhagenIndex(dim=128, n_clusters=32, device="cuda")   # NVIDIA
idx = CopenhagenIndex(dim=128, n_clusters=32, device="mps")    # Apple M-series
```

Centroids are pinned on device once at train time; only `(n, soft_k)` argmin indices transfer back per batch. Search stays on CPU (single-query centroid ranking is too small to benefit from device offload).

---

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `n_clusters` | — | Number of IVF clusters (Voronoi cells). Rule of thumb: `sqrt(n_vectors)` |
| `nprobe` | 1 | Clusters scanned per query. Higher = better recall, slower search |
| `soft_k` | 1 | Clusters each vector is indexed in. `soft_k=2` matches FAISS rebuild recall on drift benchmarks |
| `split_threshold` | 3.0 | Split a cluster when `live_size > mean_size × threshold` |
| `max_split_iters` | 10 | Mini k-means iterations inside `split_cluster` |
| `use_pq` | False | Product Quantization for compressed storage and faster approximate search |
| `use_mmap` | False | Memory-map cluster storage for indexes larger than RAM |

`split_threshold` and `soft_k` are writable on the index object at any time:

```python
idx._index.split_threshold = 2.5   # more aggressive splits
idx._index.soft_k = 2
```

---

## Monitoring

```python
# Global stats
stats = idx.get_stats()
# {'n_vectors': 15000, 'n_clusters': 33, 'deleted_count': 12, ...}

# Per-cluster breakdown
for cs in idx.get_cluster_stats():
    print(cs['cluster_id'], cs['live_size'], cs['physical_size'], cs['last_split_round'])
    # last_split_round = -1 for training-time clusters
    # last_split_round = N for clusters created by the Nth insert_batch call
```

`get_cluster_stats()` returns a list of dicts with `cluster_id`, `live_size`, `physical_size`, `centroid` (numpy array), and `last_split_round`. Useful for surfacing which clusters are overloaded and which have recently split.

---

## Benchmark results

Full results and run instructions: **[BENCHMARKS.md](BENCHMARKS.md)**

### Distribution drift — MNIST → Fashion-MNIST (784d, quick mode)

Train on MNIST (20k vectors), insert Fashion-MNIST (10k vectors) without retraining. Fashion vectors land in wrong Voronoi cells; clusters bloat 3–4×.

| Method | Fashion recall@10 | MNIST recall@10 | Insert time |
|---|---|---|---|
| FAISS IVF add-only | 0.953 | 0.973 | 21 ms |
| FAISS IVF full rebuild | 0.989 | 0.974 | 119 ms |
| AMPI (nlist=212, fans=16, probes=16) | **0.997** | 0.974 | 7,063 ms |
| Copenhagen baseline (soft_k=1, no splits) | 0.962 | 0.980 | 29 ms |
| Copenhagen soft_k=2 (fixed) | 0.989 | **0.994** | 65 ms |
| Copenhagen best (soft_k=2 + splits) | 0.979 | 0.993 | 168 ms |

Copenhagen soft_k=2 matches FAISS full rebuild on fashion recall (0.989 vs 0.989) at 1.8× faster insert (65 ms vs 119 ms), with +2.0pp MNIST recall. Copenhagen best is ~42× faster to insert than AMPI at a cost of 1.8pp fashion recall. AMPI is the right call when drift recall is the primary metric and insert latency is not a constraint.

### Gradual streaming drift (500 vectors/batch × 10 batches)

| Method | Recall at batch 1 | Mid | Final |
|---|---|---|---|
| FAISS add-only | 0.946 | 0.977 | 0.980 |
| Copenhagen baseline | 0.916 | 0.959 | 0.969 |
| Copenhagen best (soft_k=2 + splits) | **0.972** | **0.986** | **0.987** |

Copenhagen best leads FAISS from the very first batch (+2.6pp) and finishes +0.7pp ahead.

### Insert scaling — O(1) vs O(log n) (SIFT-128, d=128)

| n       | CPH µs/vec | CPH R@10 | FAISS IVF µs/vec | IVF R@10 | HNSW µs/vec | HNSW R@10 |
|---------|-----------|----------|-----------------|----------|------------|----------|
| 5,000   | 1.16      | 0.993    | 0.79            | 0.951    | 280        | 1.000    |
| 25,000  | 1.12      | 0.997    | 0.64            | 0.972    | 1,092      | 1.000    |
| 100,000 | 1.08      | 0.996    | 0.67            | 0.985    | 8,419      | 0.997    |

CPH and FAISS IVF both show flat insert cost (O(1) in n). HNSW grows 30× over the same range. HNSW starts at 240× the CPH insert cost and reaches 7,800× at n=100k. CPH recall (0.993–0.998) exceeds FAISS IVF (0.951–0.985) at the same nprobe on this dataset.

### Static recall vs HNSW (SIFT-100k, d=128)

On a static dataset HNSW dominates: at R@10 ≈ 0.97, HNSW M=32 ef=32 runs at 5,803 QPS vs CPH n=32 nprobe=4 at 3,661 QPS. CPH's advantage is insert cost (240–7,800× cheaper per vector at n=5k–100k, see insert scaling above) and O(1) delete — HNSW has neither.

Full recall/QPS table: [BENCHMARKS.md §8](BENCHMARKS.md).

### Streaming churn (30% delete/round, n=50k, 10 rounds)

| Method | Recall@10 | Inserts/s | Deletes/s |
|---|---|---|---|
| HNSW + filter | 0.47 | — | — |
| HNSW + rebuild | 0.95 | ~10k | — |
| Copenhagen | **0.93–0.95** | **~900k** | **~1M** |

Full tables in [BENCHMARKS.md](BENCHMARKS.md).

### Tombstone delete

0.3 µs per delete, zero leaked results. FAISS IVF has no native delete primitive; equivalent via rebuild costs ~11 ms for 10k vectors.

---

## Repository layout

```
src/                  C++ extension (dynamic_ivf.cpp, build.sh)
python/core/          Python wrapper (CopenhagenIndex), __init__.py, copenhagen.so
benchmarks/           Benchmark scripts:
  benchmark_drift.py              MNIST → Fashion drift (one-shot + --per-cluster)
  benchmark_drift_streaming.py    Gradual streaming drift (batches over time)
  benchmark_hnsw_churn.py         30% churn vs HNSW
  benchmark_ivf_churn.py          30% churn vs FAISS IVF
  benchmark_vs_faiss.py           Static recall and throughput
tests/                smoke_test.py, stress_test.py, test_gpu.py, bench_gpu.py, bench_search.py
                      GPU and FAISS tests must run separately: pytest -m gpu / pytest -m faiss
data/                 Dataset storage (MNIST, Fashion-MNIST, SIFT)
results/              Benchmark output (JSON)
```

---

## Credits

- **arXiv:2604.00271** — van der Hoog, Reinstädtler, Rotenberg (IT University of Copenhagen). The logarithmic-method bucket structure, quarter-full invariant, and tombstone correctness argument for ranked queries directly informed the Copenhagen design.
- **FAISS** (Facebook AI Research) — IVF baseline for all benchmarks.
- **AMPI** ([github.com/sashakolpakov/ampi](https://github.com/sashakolpakov/ampi)) — Adaptive multi-projection index with per-cluster Oja drift detection. Recommended when directional drift signals are needed rather than pure insert throughput; used as comparison baseline in drift benchmarks.
