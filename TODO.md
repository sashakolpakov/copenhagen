# Copenhagen — TODO

## Completed

- **Tombstone deletion** — O(1) `delete_vector`, lazy `compact_cluster` at
  search time, auto `compact_all` when tombstones exceed 10% of n_vectors.
  `cluster_live_count` decremented on delete; split decisions use live counts,
  not physical sizes.

- **Soft assignments (soft_k)** — each vector indexed in its top-`soft_k`
  nearest clusters. `id_to_location` tracks all slot positions; dedup pass at
  search time. soft_k=2 matches FAISS full-rebuild recall at 3.6× faster insert
  on the MNIST→Fashion drift benchmark.

- **Adaptive cluster splitting** — `rebalance_if_needed()` checks live counts
  against `mean × split_threshold` after each `insert_batch`; `split_cluster`
  runs mini k-means (k=2) on live vectors only, drops tombstones, grows the
  centroid array in place.

- **Correct move semantics** — explicit move constructors on `Cluster` and
  `PQCluster` prevent double-free when `std::vector` reallocates on splits.

- **Memory-mapped cluster storage** — `use_mmap=True` + `mmap_dir` path swaps
  `aligned_alloc` for `mmap(MAP_SHARED)` backed files; OS handles paging for
  indexes larger than RAM.

- **Save / load** — `save(path)` writes `clusters.npz` + `metadata.json`;
  `CopenhagenIndex.load(path)` restores centroids, cluster vectors/ids,
  tombstone set, `id_to_location`, and live counts.

- **GPU acceleration (PyTorch)** — `CopenhagenIndex(device="cuda"|"mps"|"cpu")`
  offloads centroid distance computation to device via `torch.mm`. Centroids are
  pinned once at train time; insert passes `n × d` vectors host→device and pulls
  back only `n × soft_k` argmin indices. Dynamic parts (tombstones,
  `id_to_location`, splits) stay on CPU. Benchmarked on MPS: 3.5–4× speedup on
  assignment compute; end-to-end insert throughput is slightly lower than CPU at
  current batch sizes (bookkeeping dominates). Tests: `tests/test_gpu.py`,
  `tests/bench_gpu.py`.

---

## Priority 1: Search Quality

### Batch BLAS in `split_cluster`
Mini k-means inside `split_cluster` uses scalar loops over `max_split_iters=10`.
For large clusters (>10k vectors), replace assign/update steps with `cblas_sgemm`.
**Files**: `src/dynamic_ivf.cpp` → `split_cluster()`.

### Auto-tune `split_threshold`
Currently a manual hyperparameter (default 3.0). Candidate approach: track recall
on a held-out query sample during `insert_batch`; lower threshold if recall drops
>2pp per batch, raise it if splits fire on >5% of inserts.

### OPQ (Optimized Product Quantization)
When `soft_k > 1`, vectors span multiple cluster boundaries. A learned rotation
before PQ reduces quantization error. Learn `d×d` rotation matrix during `train()`
via alternating optimization (Ge et al. 2013); apply before PQ encode and invert
before distance computation.

---

## Priority 2: Benchmarking

### Continuous streaming drift
Current `benchmark_drift.py` inserts all OOD data in one batch. Real systems
see gradual drift (e.g., 1k new-distribution vectors per batch over 50 batches).
Show recall over time as drift accumulates: FAISS degrades monotonically,
Copenhagen stabilizes after the first few splits.

### Per-cluster recall breakdown
After drift, show recall separately for in-distribution vs OOD queries, and
per-cluster query volumes. Makes the imbalance story visual.

### Full-scale drift benchmark
`benchmark_drift.py --full` uses 60k MNIST + 60k Fashion. Currently slow because
AMPI takes ~70s for 60k inserts. Either time-limit AMPI or run it separately.

---

## Priority 3: API / Usability

### Streaming insert API
`insert_stream(generator)` — accept a Python generator yielding `(id, vector)`
pairs so the caller doesn't need to materialize the full batch in memory.

### `get_cluster_stats()` per-cluster breakdown
Return per-cluster `(live_size, centroid, last_split_round)` for monitoring and
visualization. Analogous to AMPI's Oja sketch output.

---

## Won't Do

- **Full logarithmic-method bucket IVF**: would give O(n log n log log n)
  amortized insert but requires 5× more memory and a much more complex
  split/merge protocol. soft_k=2 already matches FAISS rebuild recall.
- **GPU search path**: single-query centroid ranking is too small to benefit
  from device offload. At current batch sizes, CPU bookkeeping dominates
  end-to-end insert time; GPU assignment compute wins at large n but the
  transfer overhead limits net throughput gain.
