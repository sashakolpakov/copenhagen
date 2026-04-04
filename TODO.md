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
  centroid array in place. Assignment step uses `blas_l2_distances` over a
  contiguous `live_vecs` matrix — O(n·d) BLAS replaces O(n·d) scalar loops
  per iteration (significant for large clusters at high `dim`).

- **Correct move semantics** — explicit move constructors on `Cluster` and
  `PQCluster` prevent double-free when `std::vector` reallocates on splits.

- **Memory-mapped cluster storage** — `use_mmap=True` + `mmap_dir` path swaps
  `aligned_alloc` for `mmap(MAP_SHARED)` backed files; OS handles paging for
  indexes larger than RAM.

- **Save / load** — `save(path)` writes `clusters.npz` + `metadata.json`;
  `CopenhagenIndex.load(path)` restores centroids, cluster vectors/ids,
  tombstone set, `id_to_location`, and live counts.

- **`get_cluster_stats()`** — Returns a list of per-cluster dicts:
  `cluster_id`, `live_size`, `physical_size`, `centroid` (numpy array),
  `last_split_round` (-1 for training-time clusters, else the
  `rebalance_if_needed` round that created it). Exposed from C++ via pybind11
  and forwarded through `CopenhagenIndex`. Useful for monitoring imbalance and
  visualising split history.

- **`insert_stream(generator, chunk_size=1000)`** — Python-layer streaming
  insert: accepts a generator of `(dim,)` or `(n, dim)` numpy arrays,
  accumulates up to `chunk_size` vectors, then calls `add()` per chunk.
  Returns `(first_id, last_id)` range of auto-assigned IDs. Memory-bounded
  for large streams.

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

## Priority 2: Benchmarking  ✓ Done

- **Continuous streaming drift** — `benchmarks/benchmark_drift_streaming.py`:
  inserts Fashion-MNIST in batches of 500 and records recall@10 after each.
  Shows FAISS degrading monotonically vs Copenhagen stabilising after splits fire.

- **Per-cluster recall breakdown** — `benchmark_drift.py --per-cluster`:
  prints per-cluster table (live_size, physical_size, last_split_round, query
  hits) for in-dist (MNIST) and OOD (Fashion) queries after drift.

- **Full-scale drift benchmark** — `benchmark_drift.py --full` now accepts
  `--ampi-timeout <seconds>` (default 30s) to cap AMPI's insert phase.

---

## Priority 3: API / Usability  ✓ Done

### Streaming insert API  ✓ Done

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
