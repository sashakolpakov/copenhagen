# Copenhagen — Approach, Novelty, and Prior Art

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

## What problem this solves

Standard ANN indexes (FAISS IVF, HNSW) assume a static or slowly-changing dataset.
In practice, data drifts: you train a recommender on a catalog of 10k products,
then add 200k new SKUs over the next quarter. The index trained on the original
data degrades silently — the centroids no longer represent the full data manifold,
clusters become massively imbalanced, and recall collapses for new-data queries.

The standard fix is a full rebuild. This is O(n·k·d·iterations) — minutes to
hours for large indexes, often done offline nightly. Copenhagen aims to handle
this incrementally, in real time, without a rebuild.

---

## What the Copenhagen paper actually does

**Paper**: "Engineering Fully Dynamic Convex Hulls" (arXiv:2604.00271)
**Authors**: van der Hoog, Reinstädtler, Rotenberg — IT University of Copenhagen
**Venue**: arXiv, March 2026

### Core contribution

A practical, fully dynamic convex hull data structure with:
- **O(n log n log log n)** amortized update time (insert + delete)
- **O(log² n)** query time for non-decomposable point-location queries
- Empirically the fastest method on update-heavy workloads

### How it works (briefly)

Two classical techniques combined:

1. **Logarithmic method (Overmars, 1981)**: Partition data into `O(log n)` buckets
   of sizes `2^i`. Insert → merge smallest full buckets into next bucket.
   Each point participates in `O(log n)` merges → amortized efficiency.
   Within each bucket, run a static algorithm (here: Graham's scan in linear time).

2. **Deletion-only convex hull (Chazelle 1985; Hershberger-Suri 1992)**: Within
   each bucket, maintain a decremental convex hull structure (doubly linked list,
   never rotated). Deletions are handled by this structure without tombstoning.

**Key insight on tombstones**: The paper proves tombstoning *does not work* for
convex hulls because the query answer is Boolean (is point q inside CH?), not
cumulative. You can't subtract tombstone contributions from a yes/no answer.

### Is it original?

The individual techniques (logarithmic method, deletion-only hulls) are classical
(1981–1992). The contribution is:
- Correctly combining them for fully dynamic convex hulls (non-trivial — the
  "quarter-full invariant" and the merge procedure require new analysis)
- Engineering a robust, practical implementation that handles real-world degeneracies
  (duplicate coordinates) — existing open-source implementations all crash on these
- Demonstrating that a vector-based (no tree rotations) implementation is faster
  than the classical CQ-Tree by an order of magnitude in update speed

---

## What our algorithm does that is "in the spirit of" the paper

### 1. Tombstoning for deletion (but it *does* work here)

The paper explicitly shows tombstoning fails for convex hulls. However, for ANN
search, the query returns *ranked distances* (cumulative), not a Boolean. Tombstoned
vectors can be filtered from the candidate list. We exploit this difference:

```cpp
void delete_vector(int id) {
    deleted_ids.insert(id);  // O(1); physical removal happens lazily during search
}
```

**This is exactly the tombstone approach the paper rules out — but for a different
query type where it legitimately works.** The paper's analysis clarifies *why* it
works here: recall is a count (cumulative), not a Boolean.

### 2. Logarithmic-method spirit: local recomputation instead of global rebuild

The paper's key insight is: don't rebuild the whole structure when data changes —
rebuild only the affected bucket (O(bucket_size) instead of O(n)).

Our `split_cluster` does the same thing: when distribution drift causes a cluster
to overflow, we don't retrain all `n_clusters` centroids — we run mini k-means
only on the overloaded cluster (O(cluster_size · d · iterations) instead of
O(n · n_clusters · d · iterations)).

This is more limited than the full logarithmic method (we don't have the
O(log n log log n) amortized guarantee), but it's practical and effective.

### 3. The quarter-full invariant → split threshold

The paper uses a "quarter-full invariant" on buckets to guarantee amortized
merge costs. We use an analogous threshold: split when `cluster_size > mean * 3`.
This bounds the imbalance ratio and keeps per-cluster search cost predictable.

---

## What is genuinely novel in our approach

### 1. Soft assignments for IVF (indexing each vector in K clusters)

Standard IVF gives each vector exactly one cluster assignment (hard Voronoi).
Vectors near cell boundaries have lower recall because they're only in one
candidate pool. We index each vector in its `top-K` nearest clusters:

```
soft_k=1  →  standard IVF (baseline)
soft_k=2  →  +2.7pp recall@10 on Fashion-MNIST drift, same insert speed
soft_k=3  →  +28.8pp recall@10 on random synthetic data
```

**Benchmark result**: soft_k=2 matches FAISS full-rebuild recall (0.989) at
3.6x faster insert time (45ms vs 164ms) on the MNIST→Fashion drift scenario.

Multi-probe search (querying multiple clusters at query time) is related but
different: it doesn't help vectors that aren't indexed near the boundary.
Soft assignments at *index time* are more effective for boundary recall.

### 2. Adaptive cluster splitting for distribution drift

When new data from an unseen distribution arrives, it concentrates in the
few existing clusters that happen to be "least wrong." Those cells bloat,
and `nprobe` queries miss most of the new data.

We detect this — `max_cluster_size > mean * split_threshold` — and split
the overloaded cell via mini k-means (k=2). A new centroid is added to
represent the new data region. No full rebuild, no offline step.

**Benchmark result**: 4 initial clusters grow to 5 after inserting out-of-
distribution data; max cluster size reduced from 475 to controlled levels.

### 3. Correct move semantics for dynamic cluster vectors

The original codebase used `std::vector<Cluster>` with raw pointer members
but no user-defined move constructor. `emplace_back` during splits would
trigger reallocation, move-constructing clusters with copied (not transferred)
raw pointers, then calling destructors on the moved-from objects — double free.
We add proper move constructors to `Cluster` and `PQCluster`.

---

## How we compare to AMPI

**AMPI** (https://github.com/Frikallo/axiom, originally sashakolpakov/ampi) uses
a fundamentally different approach: affine fan cones within each cluster, sorted
projection arrays, and an Oja subspace sketch for drift detection. AMPI detects
drift per-cluster and triggers a local cone refresh only for affected clusters.

**On the MNIST→Fashion drift benchmark**:
- AMPI (tuned): `nlist=~212`, `num_fans=16`, `probes=16`, `fan_probes=16`, `window_size=50`
- Copenhagen soft_k=2: 0.99 recall@10, 45ms insert

Earlier runs with `nlist=32` (matching IVF n_clusters) and `fan_probes=4` produced
0.71 recall — this was a misconfiguration. AMPI's own BENCHMARKS.md recommends
`nlist ≈ 1.5×sqrt(n)` and probing all fans (`fan_probes=num_fans`). The benchmark
now uses these corrected values; see `run_ampi()` in `benchmarks/benchmark_drift.py`.

The two approaches are complementary: AMPI has a more principled drift detection
mechanism (Oja sketch tracks the direction of drift); Copenhagen's splitting is
simpler but immediately effective for the imbalance problem.

---

## Credits

- **arXiv:2604.00271** — Logarithmic method + deletion-only structures.
  The tombstone analysis and quarter-full invariant directly informed our design.

- **Axiom / AMPI** (https://github.com/Frikallo/axiom) — Adaptive multi-projection
  index with per-cluster Oja drift detection. Comparison and design contrast above.

- **FAISS** (Facebook AI Research) — IVF baseline for all benchmarks.

- **Product Quantization** — Jégou, Douze, Schmid (2011).
