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
| E-commerce catalog — 10k new SKUs/hour | 1.3-1.8 µs/vector insert, never goes offline; adaptive splits recover recall after catalog expansions |
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
| `quant` | `"none"` | Quantized scan mode. Use `quant="tq"` for TurboQuant |
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
| FAISS IVF add-only | 0.9548 | 0.9736 | 31 ms |
| FAISS IVF full rebuild | 0.9892 | 0.9738 | 226 ms |
| Copenhagen baseline (soft_k=1, no splits) | 0.9488 | 0.9766 | 57 ms |
| Copenhagen adaptive (splits, soft_k=1) | 0.9236 | 0.9602 | 100 ms |
| Copenhagen soft_k=2 (fixed) | 0.9864 | **0.9936** | 101 ms |
| Copenhagen best (soft_k=2 + splits) | 0.9864 | **0.9936** | 102 ms |

On the latest full Linux run, Copenhagen best is 2.2x faster to insert than FAISS full rebuild (102 ms vs 226 ms) while trailing by just 0.28pp on Fashion recall and leading by 1.98pp on MNIST recall. `soft_k=2` is doing most of the work here; adaptive splitting did not fire in this run. AMPI was not installed on this host, so it is omitted from the table.

### Gradual streaming drift (500 vectors/batch × 10 batches)

| Method | Recall at batch 1 | Mid | Final |
|---|---|---|---|
| FAISS add-only | 0.946 | 0.977 | 0.980 |
| Copenhagen baseline | 0.921 | 0.958 | 0.964 |
| Copenhagen best (soft_k=2 + splits) | **0.972** | **0.988** | **0.990** |

Copenhagen best leads FAISS from the first batch (+2.65pp) and finishes +1.05pp ahead.

### Insert scaling — O(1) vs O(log n) (SIFT-128, d=128)

| n       | CPH µs/vec | CPH R@10 | FAISS IVF µs/vec | IVF R@10 | HNSW µs/vec | HNSW R@10 |
|---------|-----------|----------|-----------------|----------|------------|----------|
| 5,000   | 1.18      | 0.991    | 0.50            | 0.951    | 75.96      | 1.000    |
| 25,000  | 1.39      | 0.996    | 0.59            | 0.970    | 214.41     | 1.000    |
| 100,000 | 1.82      | 0.999    | 0.54            | 0.985    | 512.70     | 0.995    |

CPH and FAISS IVF both stay effectively flat in insert cost. In the latest run CPH grows 1.54x over a 20x scale-up, IVF 1.08x, while HNSW grows 6.75x. CPH recall (0.991-0.999) still exceeds IVF (0.951-0.985) at the same `nprobe` on this dataset.

### Static recall vs HNSW (SIFT-100k, d=128)

On a static dataset HNSW still dominates the recall/QPS frontier: at R@10 ~= 0.97, HNSW `M=32 ef=32` runs at 11,633 QPS vs Copenhagen `n_clusters=32 nprobe=4` at 2,975 QPS. Copenhagen's advantage is insert cost and O(1) delete, not peak static-search efficiency.

Full recall/QPS table: [BENCHMARKS.md §8](BENCHMARKS.md).

### Streaming churn (30% delete/round, n=50k, 10 rounds)

| Method | Recall@10 | Inserts/s | Deletes/s |
|---|---|---|---|
| FAISS IVF + filter | 0.642 | — | — |
| FAISS IVF + rebuild | 0.803 | ~339k | — |
| HNSW + filter | 0.276 | — | — |
| HNSW + rebuild | 0.916 | ~8.3k | — |
| Copenhagen | **0.937** | **~839k** | **~1.49M** |

By round 10 (~93% cumulative churn), Copenhagen stays near rebuild-level recall while deleting about 180x faster than HNSW's rebuild throughput and inserting about 2.5x faster than FAISS IVF rebuild. Full per-round tables are in [BENCHMARKS.md](BENCHMARKS.md).

### Tombstone delete

0.3 µs per delete, zero leaked results. FAISS IVF has no native delete primitive; equivalent via rebuild costs ~11 ms for 10k vectors.

---

## Compression: TurboQuant + block VQ (vs TurboVec)

Copenhagen wins **dynamics**; [TurboVec](https://github.com/RyanCodrai/turbovec)
— the [TurboQuant](https://arxiv.org/abs/2504.19874) scalar quantizer — wins
**compression**. The aim is to have both in one index. We reimplemented
TurboQuant in C++ and added a **block / sub-vector vector-quantization**
front-end that fixes TurboQuant's low-dimension weakness. Full derivation and
references: [docs/source/theory.rst](docs/source/theory.rst).

**Head-to-head on normalized SIFT-128 (50k), recall@10 vs bytes/vector:**

| Index | recall@10 | bytes/vec | compression |
|---|---|---|---|
| Copenhagen float | 0.9971 | 512 | 1.0× |
| Copenhagen TQ-4bit (SIMD fast-scan) | 0.8770 | 584 | 0.9× |
| TurboVec 4-bit | 0.8476 | 68 | 7.5× |
| TurboVec 2-bit | 0.6266 | 36 | 14.2× |
| Copenhagen-TQ block VQ `B=2` | 0.9070 | 72 | 7.1× |
| Copenhagen-TQ block VQ `B=4` | 0.7002 | 40 | 12.8× |

Copenhagen's in-index TQ path retains the float32 vectors (for exact rerank,
deletes, and splits), so it trades memory for high recall and a fast SIMD scan
rather than for compression — the memory win is the block-VQ path. The removed
IVFPQ path was dominated on both axes; rationale and evidence in
[IVFPQ.md](IVFPQ.md). TurboQuant is the replacement.

**Scoring is SIMD-accelerated.** At `tq_bits<=4` (the default) the per-candidate
TurboQuant scorer runs as a nibble-LUT *fast-scan* kernel — codes stored
block-transposed, 32 vectors scored per dimension with one `vqtbl1q_u8` (NEON) /
`pshufb` (AVX2), then the exact rerank resolves the 8-bit LUT. It is ~10–25×
faster than the scalar loop, bit-exact with it, and recall-preserving; an
unrecognized ISA falls back to a scalar-block kernel over the same layout. See
[BENCHMARKS.md §6](BENCHMARKS.md).

**Why TurboVec struggles at low dimension — and what we do about it.** After a
random rotation, a unit vector's coordinates are Beta-distributed and, crucially,
*negatively correlated* by the constraint Σxᵢ²=1 — an `O(1/d)` effect that is
negligible at d=1536 but dominant at d=128. Scalar (per-coordinate) quantization
assumes independence and throws this structure away. **Block VQ quantizes B
coordinates jointly** ("adaptive binning"), recovering it. At matched bytes/vector
(synthetic, raw recall@10, no rerank):

| d | rate | scalar TurboQuant | **block VQ** | Δ |
|---|---|---|---|---|
| 128 | 2-bit | 0.4581 | **0.5806** | **+12.3 pp** |
| 128 | 4-bit | 0.8183 | **0.8412** | **+2.3 pp** |
| 768 | 2-bit | 0.5990 | **0.7288** | **+13.0 pp** |
| 768 | 4-bit | 0.8700 | **0.8956** | **+2.6 pp** |

The gain is largest exactly where scalar TurboQuant is weakest. We also
implemented the ScaNN anisotropic loss and found it *does not help here* — the
length-renormalization already corrects the parallel residual it targets (see
the theory docs for the full negative result).

> **Status.** `quant="tq"` is now the live integrated compressed-search mode in
> [`src/dynamic_ivf.cpp`](src/dynamic_ivf.cpp). For the removed compressed path
> and why it was culled, see [IVFPQ.md](IVFPQ.md). The next compression work is
> block-VQ / E8 / OPQ on top of the TurboQuant path. See
> [QUANTIZATION_GOAL.md](QUANTIZATION_GOAL.md).

### Reproduce everything

**Push-button (Docker, self-contained)** — from a fresh clone on a Linux box:

```bash
bash run_benchmarks.sh 12        # installs Docker if needed, builds the image,
                                 # downloads all datasets, runs every benchmark ×
                                 # method on 12 concurrent jobs → results/REPORT_latest.md
```

**Or directly:**

```bash
python3 benchmarks/reproduce.py --jobs 6   # build, deps, download, run, report
python3 benchmarks/benchmark_vs_turbovec.py            # Copenhagen vs TurboVec, 3 axes
python3 benchmarks/benchmark_ivf_churn.py --with-turbovec   # TurboVec as a churn column
```

Full benchmark + reproduction guide: [docs/source/benchmarks.rst](docs/source/benchmarks.rst).

---

## Repository layout

```
src/                  C++ extension + quantizers
  dynamic_ivf.cpp                 The dynamic IVF index (insert/delete/split, BLAS)
  turbo_quant.hpp                 Scalar TurboQuant (rotation, Lloyd–Max, TQ+, renorm)
  block_quant.hpp                 Block/sub-vector VQ + ScaNN anisotropic codebook
  tq_standalone_test.cpp          Scalar-TQ microbenchmark (diagnostic harness)
  tq_block_test.cpp               Block VQ vs scalar at matched bytes
  tq_aniso_test.cpp               Anisotropic eta sweep
python/core/          Python wrapper (CopenhagenIndex), __init__.py, copenhagen.so
benchmarks/           Benchmark scripts:
  reproduce.py                    One-shot: build, deps, download, run, report
  benchmark_vs_turbovec.py        Copenhagen vs TurboVec — compression/recall/dynamics
  _turbovec_runner.py             Shared TurboVec baseline adapter
  benchmark_drift.py              MNIST → Fashion drift (one-shot + --per-cluster)
  benchmark_drift_streaming.py    Gradual streaming drift (batches over time)
  benchmark_hnsw_churn.py         30% churn vs HNSW          (+ --with-turbovec)
  benchmark_ivf_churn.py          30% churn vs FAISS IVF     (+ --with-turbovec)
  benchmark_vs_faiss.py           Static recall and throughput
docs/source/          Sphinx docs: theory.rst (math + references), benchmarks.rst
tests/                smoke_test.py, stress_test.py, test_gpu.py, bench_gpu.py, bench_search.py
                      GPU and FAISS tests must run separately: pytest -m gpu / pytest -m faiss
data/                 Dataset storage (MNIST, Fashion-MNIST, SIFT)
results/              Benchmark output (JSON)
```

---

## Credits

- **arXiv:2604.00271** — van der Hoog, Reinstädtler, Rotenberg (IT University of Copenhagen). The logarithmic-method bucket structure, quarter-full invariant, and tombstone correctness argument for ranked queries directly informed the Copenhagen design.
- **FAISS** (Facebook AI Research) — IVF family baseline for the benchmarks.
- **TurboQuant** (Zandieh et al., *Online Vector Quantization with Near-optimal Distortion Rate*, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), Google & NYU, ICLR 2026) — the scalar quantizer at the core of the compressed path: random rotation → Beta-concentrated coordinates → per-coordinate optimal scalar quantizer, with a QJL residual stage. Reimplemented in C++ here and extended with block VQ.
- **TurboVec** ([github.com/RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec)) — Ryan Codrai's implementation of TurboQuant; compression baseline in the benchmarks.
- **RaBitQ** (Gao & Long, SIGMOD 2024) — length-renormalized unbiased inner-product estimator underlying the per-vector scale.
- **ScaNN** (Guo et al., ICML 2020) — anisotropic quantization loss (implemented and evaluated; see theory docs).
- **Product Quantization** (Jégou et al., TPAMI 2011) — the sub-vector codebook + LUT layout block VQ adopts.
- **AMPI** ([github.com/sashakolpakov/ampi](https://github.com/sashakolpakov/ampi)) — Adaptive multi-projection index with per-cluster Oja drift detection. Recommended when directional drift signals are needed rather than pure insert throughput; used as comparison baseline in drift benchmarks.
