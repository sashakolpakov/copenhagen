# Copenhagen — TODO

## Priority 0: Correctness / Limits

### Tombstone compaction trigger  *(partially done)*
`deleted_ids` grows unboundedly. Two properties are now enforced:

- **Split decisions use live counts** (done): `cluster_live_count[]` tracks
  live vectors per cluster, excluding tombstones. `rebalance_if_needed()` uses
  this, so dead vectors no longer trigger spurious splits or mask real imbalance.
- **`compact_cluster()` does not touch `vec_sum`** (done): centroids are frozen;
  subtracting dead vectors from `vec_sum` would imply a drifting centroid, which
  violates the design. `cluster_live_count` is the correct accounting.

**Still needed**: when `deleted_ids.size() > n_vectors * 0.1` (10% churn), run
`compact_all()` — a full pass over all clusters to physically evict tombstones and
then clear `deleted_ids`. Without this, `deleted_ids.count()` inside `compact_cluster`
degrades from O(1) toward O(n) as the hash set grows. Expose as `idx.compact()` for
manual control; auto-trigger at end of `insert_batch`.

**Files**: `src/dynamic_ivf.cpp` — add `compact_all()`, call from `insert_batch`.

---

## Priority 1: Infrastructure

### Memory-mapped cluster storage
Replace `aligned_alloc` per cluster with mmap-backed files so the index
can exceed available RAM. The OS handles paging transparently.

**Plan**:
- Add `use_mmap: bool` flag + `mmap_dir: str` to `DynamicIVF`
- In `Cluster`: track `mmap_fd`, `mmap_size`; replace `aligned_alloc` with
  `ftruncate` + `mmap(MAP_SHARED)`
- On growth: `ftruncate` to 2× → `munmap` → `mmap` (macOS has no `mremap`)
- Per-cluster file: `<mmap_dir>/cluster_<N>_vectors.mmap`
- Expose via `DynamicIVF(... use_mmap=False, mmap_dir="")`
- `save()` already works; mmap files ARE the persistence layer when enabled

**Files**: `src/dynamic_ivf.cpp` (Cluster struct, init, add_vector, destructor),
`python/core/__init__.py` (constructor, save/load)

---

## Priority 2: Search Quality

### OPQ (Optimized Product Quantization)
When `soft_k > 1`, vectors span multiple cluster boundaries. A learned rotation
before PQ reduces quantization error. Standard PQ ignores inter-subspace
correlations; OPQ finds the rotation that minimizes them.

**Plan**: Learn `d×d` rotation matrix during `train()` via alternating optimization
(Ge et al. 2013). Apply rotation before PQ encode; apply inverse before distances.

### Auto-tune `split_threshold`
Currently a manual hyperparameter (default 3.0). Could be set empirically by
tracking recall degradation on a held-out query set sampled during `insert_batch`.
If recall drops >2pp per batch, lower the threshold; if splits fire too often
(>5% of inserts trigger a split), raise it.

### Batch BLAS in `split_cluster`
Mini k-means inside `split_cluster` uses scalar loops over `max_split_iters=10`.
For large clusters (>10k vectors), replace assign/update with `cblas_sgemm`.

---

## Priority 3: Benchmarking

### Full-scale drift benchmark (`--full` flag)
`benchmark_drift.py --full` uses 60k MNIST + 60k Fashion. Currently too slow
because AMPI takes ~70s for 60k inserts. Either time-limit AMPI or run it
separately.

### Continuous streaming drift
Current benchmarks insert all OOD data at once. Real systems see gradual drift
(e.g., 1k new-distribution vectors per batch over 50 batches). Show recall
over time as drift accumulates: FAISS degrades monotonically, Copenhagen
stabilizes after the first few splits.

### Per-cluster recall breakdown
After drift, show recall separately for in-distribution vs OOD queries, and
per-cluster query volumes. Makes the imbalance story visual.

---

## Priority 4: API / Usability

### `CopenhagenIndex.save()` compression
Currently writes one `.npy` per cluster — for 10k clusters this is 10k files.
Replace with a single `.npz` archive or a binary pack file.

### Streaming insert API
`insert_stream(generator)` — accept a Python generator yielding `(id, vector)`
pairs so the caller doesn't need to materialize the full batch in memory.

### `get_cluster_stats()` per-cluster breakdown
Return per-cluster `(size, centroid, drift_direction)` for monitoring and
visualization. Analogous to AMPI's Oja sketch output.

---

## Notes / Won't Do

- **Full logarithmic-method bucket IVF**: would give O(n log n log log n)
  amortized insert but requires 5× more memory and a much more complex split/merge
  protocol. Not worth it given soft_k=2 already matches FAISS rebuild recall.
- **CUDA support**: FAISS GPU is the right tool for GPU workloads.
  Copenhagen targets CPU-only real-time insert scenarios.
