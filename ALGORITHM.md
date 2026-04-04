# Copenhagen Algorithm

## Overview

Copenhagen is a fully dynamic approximate nearest neighbor (ANN) index
based on IVF (Inverted File Index). It supports O(1) inserts, O(1) deletes,
and handles distribution drift via adaptive cluster splitting.

Inspired by arXiv:2604.00271 "Engineering Fully Dynamic Convex Hulls"
(van der Hoog, Reinstädtler, Rotenberg — IT University of Copenhagen).

---

## 1. Index Structure

```
Copenhagen Index:
├── centroids (n_clusters × d)         — frozen after training; grows on splits
├── clusters (n_clusters):
│   ├── vectors (size × d)             — live + tombstoned vectors, 32-byte aligned
│   ├── ids     (size,)                — original vector IDs
│   └── vec_sum (d,)                   — running sum; NOT used as centroid proxy
│                                        (centroids are frozen; vec_sum is vestigial)
├── cluster_live_count (n_clusters,)   — live vectors per cluster (tombstones excluded)
│                                        used by rebalance_if_needed(), not clusters[c].size
├── deleted_ids: unordered_set<int>    — tombstone set; cleared by compact_all()
├── id_to_location: vector<vector<pair<int,int>>>
│   └── id → [(cluster_id, position), ...]   — one entry per soft_k slot
└── Optional PQ codebook
```

---

## 2. Training

```python
centroids = kmeans(vectors, k_clusters, n_iterations=25)  # BLAS-accelerated
for v in vectors:
    top_k = argsort(distance(v, centroids))[:soft_k]
    for c in top_k:
        cluster[c].add(v)
        cluster_live_count[c] += 1
# centroids frozen after this point
```

---

## 3. Insert — O(n·k·d) via BLAS, O(1) per vector assignment

```python
dists = gemm(vectors, centroids)              # BLAS: n × n_clusters distances
for v in vectors:
    top_k = argsort(dists[v])[:soft_k]        # top-soft_k clusters
    for c in top_k:
        cluster[c].append(v)
        cluster_live_count[c] += 1
        id_to_location[v.id].append((c, pos))
n_vectors += len(vectors)
rebalance_if_needed()                         # split overloaded clusters

# Auto-compact if tombstone load exceeds 10%:
if deleted_ids.size() > n_vectors / 10:
    compact_all()
```

**Stable distributions**: centroids never updated → O(1) per vector.
**Drifting distributions**: `rebalance_if_needed` adaptively adds centroids.

---

## 4. Delete — O(1) tombstone + O(soft_k) live-count update

```python
deleted_ids.add(id)
for (cluster_id, pos) in id_to_location[id]:
    cluster_live_count[cluster_id] -= 1
```

**Why tombstoning works here** (unlike convex hulls in the paper):
ANN returns cumulative ranked distances — deleted vectors are filtered
from the candidate list. Convex hull queries return Boolean answers, so
tombstones cannot be combined across buckets.

**Why we do NOT update vec_sum on delete**: centroids are frozen after
training. Subtracting dead vectors from vec_sum would imply a drifting
centroid, violating the design. `cluster_live_count` is the correct
accounting for split decisions; vec_sum is unused for centroid computation.

**Lazy compaction**: during `search()`, when a probed cluster contains
tombstoned vectors, `compact_cluster()` runs a single in-place pass to
remove them before the BLAS scan. Subsequent searches see a clean cluster.

**Auto-compact**: `insert_batch` triggers `compact_all()` when
`deleted_ids.size() > n_vectors / 10`, clearing the hash set and keeping
`compact_cluster`'s O(1) lookup cost.

---

## 5. Search

```python
cent_dists = blas_l2(query, centroids)        # BLAS
top_c = argsort(cent_dists)[:nprobe]

for c in top_c:
    if any(id in deleted_ids for id in cluster[c].ids):
        compact_cluster(c)                    # lazy eviction

candidates = concat(cluster[c].vectors for c in top_c)
exact_dists = blas_l2(query, candidates)

if soft_k > 1:
    candidates = deduplicate_by_id(candidates)

return top_k_by_distance(candidates, k)
```

---

## 6. Cluster Splitting (Distribution Drift)

After `insert_batch`, `rebalance_if_needed()` uses **live counts** (not
physical cluster sizes) to detect imbalance:

```python
def rebalance_if_needed():
    total_live = sum(cluster_live_count)
    mean  = total_live / n_clusters
    threshold = mean * split_threshold          # default: 3.0x mean
    for c in range(current_n_clusters):         # only existing clusters
        if cluster_live_count[c] > threshold:
            split_cluster(c)

def split_cluster(c):
    # 1. Save c's live data to temp buffer (before emplace_back invalidates refs)
    # 2. Mini k-means (k=2, 10 iters) on saved live vectors → two new centroids
    # 3. Expand centroids array: c keeps centroid_0, new_c gets centroid_1
    # 4. clusters.emplace_back() — safe because data is already saved
    # 5. cluster_live_count.push_back(0); cluster_live_count[c] = 0
    # 6. Redistribute saved vectors → update id_to_location, increment live counts
    # Tombstoned vectors are dropped during redistribution (not re-inserted)
```

**Why using live counts matters**: a cluster with 1000 physical slots but
800 tombstones has effective size 200. Splitting it would create two nearly-
empty clusters and waste a centroid. Using `cluster_live_count` prevents this.

**Why this solves drift**: new data from unseen distributions falls into the
nearest existing Voronoi cell. When too many live vectors pile up (imbalance
> 3x), that cell is split and a new centroid represents the new region.

---

## 7. Soft Assignments

Each vector is stored in its `soft_k` nearest clusters:

- **soft_k=1** (default): standard IVF, backward compatible
- **soft_k=2**: vector in 2 clusters; recall at Voronoi boundaries improves significantly
- **soft_k=3**: near-perfect recall in most configurations

At search time, deduplication ensures each vector appears at most once in
results. Storage overhead is exactly `soft_k × n_vectors` physical slots.

---

## 8. Persistence (Save / Load)

```python
idx.save("path/")   # writes clusters.npz + metadata.json
idx = CopenhagenIndex.load("path/")
```

`clusters.npz` contains: centroids + per-cluster vectors and ids in a single
archive. `metadata.json` stores scalar state, `deleted_ids`, and
`id_to_location`. On load, live counts are recomputed from the restored
cluster arrays and tombstone set.

---

## 9. Compaction

```python
idx.compact()   # evict all tombstones, clear deleted_ids
```

Called automatically when tombstone load exceeds 10% of `n_vectors`.
Call manually after a bulk delete to reclaim memory immediately.

---

## 10. Complexity

| Operation      | Standard IVF        | Copenhagen (this)                        |
|---------------|---------------------|------------------------------------------|
| Insert (n)    | O(nd) rebuild       | O(n·k·d) BLAS, O(n·soft_k) append       |
| Delete        | O(nd) rebuild       | O(soft_k) live-count update + tombstone  |
| Compact       | —                   | O(cluster_size) lazy, or O(n) full pass  |
| Search        | O(nprobe·M·d) BLAS  | same + O(n·soft_k) dedup pass            |
| Split trigger | —                   | O(n_clusters) check + O(m·d·iters) split |

Where m = live cluster size, iters = max_split_iters (default 10).

---

## 11. Implementation Notes

- All dense distance computation via `cblas_sgemm` (Apple Accelerate)
- All cluster storage is 32-byte aligned for AVX/NEON cache efficiency
- `Cluster` and `PQCluster` have explicit move constructors to prevent
  double-free when `std::vector` reallocates during `emplace_back` in splits
- `id_to_location` maps ID → `vector<pair<cluster_id, position>>` supporting
  soft_k entries per vector and position updates after lazy compaction
- `cluster_live_count` is decremented in `delete_vector` via `id_to_location`
  (O(soft_k) per delete) and rebuilt in full after `restore_state`
- `split_threshold`, `soft_k`, `max_split_iters` are Python-settable attributes

---

## 12. References

- arXiv:2604.00271 — Logarithmic method + deletion-only structures (inspiration)
- FAISS (Facebook AI) — IVF baseline comparison
- AMPI (https://github.com/sashakolpakov/ampi) — Adaptive multi-projection index
- Product Quantization — Jégou et al. (2011)
