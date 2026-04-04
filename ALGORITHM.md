# Copenhagen — Algorithm, Novelty, and Prior Art

Copenhagen is a fully dynamic approximate nearest neighbor (ANN) index built on
IVF (Inverted File Index). It supports O(1) inserts, O(1) deletes, and handles
distribution drift via adaptive cluster splitting — without ever doing a full
index rebuild.

---

## 1. What the Copenhagen Paper Actually Contributes

**Paper**: "Engineering Fully Dynamic Convex Hulls" (arXiv:2604.00271)  
**Authors**: van der Hoog, Reinstädtler, Rotenberg — IT University of Copenhagen  
**Published**: March 2026

The paper solves a different problem — fully dynamic convex hulls in computational
geometry — using two classical techniques:

1. **Logarithmic method (Overmars, 1981)**: Partition data into `O(log n)` buckets
   of sizes `2^i`. Insert → fill the smallest empty bucket. When a bucket fills,
   merge it with the next. Each point participates in `O(log n)` merges →
   O(n log n log log n) amortized insert. Within each bucket, run a static
   algorithm (Graham's scan in linear time).

2. **Deletion-only convex hull (Chazelle 1985; Hershberger-Suri 1992)**: Within
   each bucket, maintain a decremental structure that handles deletions without
   tombstoning.

**Key result on tombstones**: The paper proves tombstoning *does not work* for
convex hulls because the hull query is Boolean (is point q inside the hull?).
You cannot subtract tombstone contributions from a yes/no answer.

**What is new in the paper**: correctly combining these classical techniques for
the fully dynamic case, proving the "quarter-full invariant" needed for amortized
guarantees, and shipping an implementation that handles real-world degeneracies
(duplicate coordinates) that crash all existing open-source implementations.

**What the paper does not do**: it does not propose a dynamic ANN index. It does
not address IVF, soft assignments, cluster splitting, or distribution drift.

---

## 2. What We Took From the Paper

### 2.1 Tombstoning works here — for a different reason

The paper's tombstone analysis cuts both ways. It shows tombstoning fails for
Boolean queries but implicitly identifies *why it would work* for cumulative
(ranked) queries. ANN search returns ranked distances, not a Boolean answer.
Tombstoned vectors can be filtered from the candidate list without corrupting
the ranking of live vectors.

This is the direct theoretical justification for our O(1) delete:

```cpp
void delete_vector(int id) {
    deleted_ids.insert(id);           // O(1); filter at search time
    for (auto [c, pos] : id_to_location[id])
        cluster_live_count[c]--;      // O(soft_k)
}
```

### 2.2 Local recomputation instead of global rebuild

The logarithmic method's core idea is: when data changes, rebuild only the
affected bucket (O(bucket_size)), not the whole structure (O(n)).

Our `split_cluster` applies the same logic to IVF: when drift causes one cluster
to overflow, run mini k-means only on that cluster (O(m·d·iters) where m is the
live cluster size), not a full retrain of all `n_clusters` centroids
(O(n·n_clusters·d·iters)).

We do **not** claim the O(log n log log n) amortized guarantee — we have no
bucket hierarchy and no merge step. The analogy is conceptual, not formal.

### 2.3 Imbalance threshold ≈ quarter-full invariant

The paper's quarter-full invariant bounds merge cost by ensuring no bucket is
too empty when a merge triggers. Our `split_threshold` (default: 3× mean live
cluster size) bounds imbalance for the same practical reason: predictable
per-cluster search cost.

---

## 3. What Is Novel Engineering (Our Contribution)

These are engineering contributions, not mathematical novelty. We make no claim
to new asymptotic bounds or new theory.

### 3.1 Soft assignments at index time

Standard IVF assigns each vector to exactly one cluster (hard Voronoi). Vectors
near cell boundaries have systematically lower recall because they only appear in
one candidate pool. We index each vector in its `soft_k` nearest clusters:

```
soft_k=1  →  standard IVF (backward compatible)
soft_k=2  →  vector in 2 nearest clusters; recall at boundaries improves
soft_k=3  →  near-perfect recall in most configurations
```

**Benchmark (MNIST→Fashion drift, n=30k, d=784, nprobe=4)**:
- soft_k=1: 0.47 recall@10 (fashion queries)
- soft_k=2: 0.99 recall@10, 45ms insert — matches FAISS full-rebuild recall at
  3.6× faster insert

Multi-probe search (querying extra clusters at search time) is related but
different: it helps at query time but doesn't index the vector in the boundary
cluster, so recall at index time is still limited. Soft assignment at *index
time* is more robust for streaming inserts where query-time cost matters.

Storage overhead: exactly `soft_k × n_vectors` physical slots, no more.

At search time, a deduplication pass ensures each ID appears at most once in
results even when it was found via multiple cluster scans.

### 3.2 Adaptive cluster splitting for distribution drift

When new data from an unseen distribution arrives, it concentrates in the few
existing clusters that happen to be geometrically nearest. Those cells bloat;
with a fixed `nprobe`, queries miss most of the new data.

We detect this via **live counts** (not physical cluster sizes — a cluster with
1000 slots and 800 tombstones has effective size 200 and should not be split):

```python
def rebalance_if_needed():
    mean = sum(cluster_live_count) / n_clusters
    for c in range(n_clusters):
        if cluster_live_count[c] > mean * split_threshold:
            split_cluster(c)

def split_cluster(c):
    # mini k-means (k=2) on live vectors of cluster c only
    # adds one new centroid; redistributes vectors; drops tombstones
```

No full rebuild, no offline step. The centroid array grows; `nprobe` stays
constant (or can be scaled proportionally if clusters double).

Using live counts for the split decision is a concrete implementation choice
that matters: a physically large but mostly-deleted cluster should not trigger
a split that wastes a centroid on dead space.

### 3.3 BLAS-accelerated batch insert path

Centroid distance computation during insert is the inner loop. For a batch of
`n` vectors and `k` centroids in `d` dimensions, computing all pairwise
distances is an `(n × d) · (d × k)` matrix multiply — a single `cblas_sgemm`
call. This gives ~60–100× throughput improvement over a Python loop:

```
CPH insert throughput:  ~1,000,000 vectors/s
HNSW insert throughput: ~10,000–20,000 vectors/s
```

HNSW graph construction cannot be batched this way (each insert depends on the
graph state after the previous one). IVF with frozen centroids can be.

### 3.4 Correct move semantics for dynamic cluster storage

When `split_cluster` calls `clusters.emplace_back()`, `std::vector` may
reallocate. If `Cluster` holds raw pointers (aligned memory, PQ codebook) and
has no user-defined move constructor, the compiler-generated one copies the
pointer and both old and new objects call `free()` on it — double free.

We add explicit move constructors to `Cluster` and `PQCluster` that null out
the source pointer after transfer. This is a correctness fix that enabled the
dynamic split feature; without it, any split crashes.

---

## 4. Index Structure

```
CopenhagenIndex:
├── centroids (n_clusters × d)         — frozen after training; grows on splits
├── clusters (n_clusters):
│   ├── vectors (size × d)             — live + tombstoned; 32-byte aligned
│   ├── ids     (size,)                — original vector IDs
│   └── vec_sum (d,)                   — vestigial; not used for centroid proxy
├── cluster_live_count (n_clusters,)   — live vectors per cluster (excl. tombstones)
├── deleted_ids: unordered_set<int>    — tombstone set; cleared by compact_all()
├── id_to_location: vector<vector<pair<int,int>>>
│   └── id → [(cluster_id, position), ...]   — one entry per soft_k slot
└── Optional PQ codebook
```

---

## 5. Operations

### Training

```python
centroids = kmeans(vectors, k=n_clusters, iters=25)   # BLAS-accelerated
for v in vectors:
    top_k = argsort(distance(v, centroids))[:soft_k]
    for c in top_k:
        cluster[c].add(v); cluster_live_count[c] += 1
# centroids frozen after this point
```

### Insert — O(n·k·d) BLAS, O(n·soft_k) append

```python
dists = gemm(vectors, centroids)              # single sgemm call
for v in vectors:
    top_k = argsort(dists[v])[:soft_k]
    for c in top_k:
        cluster[c].append(v)
        cluster_live_count[c] += 1
        id_to_location[v.id].append((c, pos))
n_vectors += len(vectors)
rebalance_if_needed()
if deleted_ids.size() > n_vectors / 10:
    compact_all()                             # auto-compact at 10% tombstone load
```

### Delete — O(soft_k)

```python
deleted_ids.add(id)
for (c, pos) in id_to_location[id]:
    cluster_live_count[c] -= 1
```

**Lazy compaction**: during `search()`, when a probed cluster contains
tombstones, `compact_cluster()` runs one in-place pass before the BLAS scan.
Subsequent searches on that cluster see no dead vectors.

### Search

```python
cent_dists = blas_l2(query, centroids)
top_c = argsort(cent_dists)[:nprobe]

for c in top_c:
    if cluster[c] has tombstones:
        compact_cluster(c)                    # lazy eviction

candidates = concat(cluster[c].vectors for c in top_c)
exact_dists = blas_l2(query, candidates)

if soft_k > 1:
    candidates = deduplicate_by_id(candidates)

return top_k_by_distance(candidates, k)
```

---

## 6. Complexity

| Operation     | Standard IVF             | Copenhagen                              |
|---------------|--------------------------|-----------------------------------------|
| Insert (n)    | O(nd) rebuild            | O(n·k·d) BLAS, O(n·soft_k) append      |
| Delete        | O(nd) rebuild            | O(soft_k) live-count update + tombstone |
| Compact       | —                        | O(cluster_size) lazy or O(n) full pass  |
| Search        | O(nprobe·M·d) BLAS       | same + O(n·soft_k) dedup               |
| Split trigger | —                        | O(n_clusters) check + O(m·d·iters)     |

Where m = live cluster size, iters = max_split_iters (default 10).

---

## 7. Comparison to HNSW, FAISS IVF, and AMPI

**vs HNSW**: HNSW graph construction is not batchable — each insert requires
updating the graph for all its neighbours. Under 30% per-round churn
(n_init=50k, 10 rounds), HNSW+filter recall collapses to 0.46 at 93% churn
because deleted nodes clog traversal paths. HNSW+rebuild holds recall (0.95+)
but requires a full graph rebuild every round (~8k–10k effective inserts/s).
Copenhagen (nprobe=32/64): 0.93–0.95 recall, 1M+ inserts/s, 1M+ deletes/s.

**vs FAISS IVFFlat**: FAISS has no native delete — tombstone+filter or full
rebuild are the only options. With the same 64 clusters and nprobe=32, FAISS
IVF+filter degrades from 0.80 to 0.66 recall over 10 churn rounds; Copenhagen
holds 0.93–0.95. FAISS IVF+rebuild restores recall each round but costs
100–200k inserts/s effective (full retrain).

**vs FAISS IVFPQ**: Product quantization reduces memory (32 bytes/vec at M=32
vs 512 for float32) at the cost of recall. At the same 64 clusters / nprobe=32,
IVFPQ achieves 0.42–0.47 recall@10 under churn. Copenhagen full precision:
0.93–0.95.

**vs AMPI**: AMPI uses affine fan cones with an Oja subspace sketch for
per-cluster drift detection — a more principled geometric approach. It detects
*which direction* a cluster drifted and triggers a local cone refresh.
Copenhagen's splitting is coarser (size threshold only, no directional signal)
but simpler and immediately effective for the imbalance problem.
For drift geometry or directional alerts, AMPI is more appropriate.

---

## 8. GPU Portability

Copenhagen's two computationally expensive operations are both dense matrix
multiplies expressible as `cblas_sgemm`:

- **Insert** centroid assignment: `(n × d) · (d × k)` distance matrix
- **Search** cluster scan: `(1 × d) · (d × m)` candidate distances

Both map mechanically to `cublasSgemm`. The dynamic parts — tombstone set,
`id_to_location`, cluster append, split logic, soft-assignment dedup — are
cheap CPU work with no natural GPU equivalent, and can remain on host.

This makes a GPU acceleration path structurally simpler than FAISS GPU (which
needs bespoke CUDA kernels for inverted list management) or HNSW GPU (graph
traversal). No custom CUDA kernels are required; only the gemm calls move to
device.

---

## 9. Implementation Notes

- Dense distance computation via `cblas_sgemm` (Apple Accelerate / OpenBLAS / MKL)
- Cluster storage 32-byte aligned for AVX/NEON cache efficiency
- Explicit move constructors on `Cluster` and `PQCluster` (see §3.4)
- `id_to_location` maps ID → `vector<pair<cluster_id, position>>` supporting
  `soft_k` entries and position updates after lazy compaction
- `split_threshold`, `soft_k`, `max_split_iters` are Python-settable attributes

---

## 9. References

- arXiv:2604.00271 — van der Hoog, Reinstädtler, Rotenberg. "Engineering Fully
  Dynamic Convex Hulls." IT University of Copenhagen, 2026.
- FAISS — Johnson, Douze, Jégou. Facebook AI Research.
- AMPI — https://github.com/sashakolpakov/ampi
- Product Quantization — Jégou, Douze, Schmid. IEEE TPAMI, 2011.
- Logarithmic method — Overmars, 1981.
