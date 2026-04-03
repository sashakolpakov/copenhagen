# Ideas from Dynamic Convex Hulls Paper (arXiv:2604.00271) for IVF Improvement

## Paper Summary
This paper (van der Hoog, Reinstädtler, Rotenberg - IT University of Copenhagen) presents a fully dynamic convex hull algorithm using the **logarithmic method** combined with deletion-only data structures. The key insight: partition data into buckets of sizes 2^i, merge them incrementally.

**Core technique**: The logarithmic method by Overmars - partition across buckets Bi of size 2^i, merge when full, maintain deletion-only structures in each bucket.

---

## 1. Logarithmic Method for IVF Partition Updates

**From the paper (Section 3)**: The logarithmic method partitions data into buckets that grow exponentially (B1: 2^0, B2: 2^1, B3: 2^2, ...). When inserting, find largest j where B1...Bj are full, merge them into Bj+1.

**Application to IVF**:
- Instead of re-clustering ALL points when centroids shift, maintain IVF partitions as buckets
- New vectors go into small buckets; when bucket fills, merge with neighboring bucket and re-cluster only that merged subset
- Each vector participates in O(log n) merges over lifetime - same amortized analysis applies
- This is essentially **incremental re-clustering** rather than full rebuild

**Challenge**: Convex hulls can be computed in linear time when sorted; IVF clustering isn't as clean.

---

## 2. Bucket-Merge with Local Re-clustering

**From the paper**: Merge(j) procedure takes sorted points from B1...Bj, does O(j)-way merge using a loser tree, then constructs the data structure in linear time.

**Application to IVF**:
- When merging two IVF clusters (buckets), only re-cluster the merged set of vectors
- Use a "loser tree" approach to efficiently merge sorted lists of vectors
- After merge, recompute centroids for the new bucket only
- Keep invariant: merged bucket is at most half-full to minimize future merges

**⚠️ CRITICAL ISSUE - Voronoi vs Convex Hull**:
In convex hulls, merging two buckets just combines point sets - no external effects.
In IVF, clusters are Voronoi cells - they partition ALL of space based on centroids.
When merging cluster A + B → AB with new centroid C:
- C's Voronoi cell is DIFFERENT from A's + B's combined
- C now competes with ALL neighboring centroids
- Vectors from neighbors may now belong to AB, and vice versa
- This is a GLOBAL effect from a LOCAL merge - breaks the paper's independence assumption!

**Possible fixes**:
1. **Soft assignments [BEST]**: Vectors can belong partially to multiple clusters
   - Instead of hard assign to one cluster, store membership scores across k nearest centroids
   - At query time, compute distance to each centroid, use as weight
   - Solves boundary drift: if P now belongs to C after merge, but we keep old soft assignment (P belonged to A and D), P still gets searched via D's list
   - Bonus: **improves recall** - query near cluster boundary can retrieve from multiple clusters, sorted by combined score
2. **Accept drift**: Let boundaries be slightly wrong, fix gradually during queries
3. **Full neighbor check**: After merge, check all neighbors and repatriate vectors (expensive)
4. **Cascading merges**: If AB steals from neighbor C, C might need to merge with its neighbor

---

## 2.1 Why Voronoi Merges Are Harder Than Hull Merges

**Convex hull case** (from paper):
- Each bucket Bi maintains its own convex hull independently
- Merging just combines two point sets, computes new hull
- No effect on other buckets - points outside the hull are irrelevant

**IVF/Voronoi case**:
- Each cluster's "hull" is defined by WHOLE SPACE partition
- Cluster A owns all points closer to centroid A than any other centroid
- Merging A+B doesn't just combine points - it creates new partition
- The new centroid C might now own points that were in neighbor D's region
- A point that was in cluster D might now be closer to C

**Concrete example**:
- 3 clusters: A at (0,0), B at (10,0), D at (5,10)
- Point P at (5,1) belongs to B (closest to (10,0))
- Merge A+B → new centroid C at (5,0)
- Now P at (5,1) is closest to C, not B - P must move from D to C's cluster!
- D's cluster now has "hole" where C expanded into it

---

## 2.2 Soft Assignment - Details & Recall Benefits

**How it works**:
- Each vector v stores (cluster_id, distance_to_centroid) for TOP-K nearest centroids
- K typically 2-4, can be tuned
- When merging A+B → C, old assignments persist:
  - Vectors that were in A still have (A, d_A) in their soft list
  - They also get (C, d_C) computed once
  - No need to migrate vectors from neighboring clusters

**Query time**:
1. Compute query q's distance to all centroids → sorted list
2. For top-M clusters (M could equal K), scan their inverted lists
3. For each vector found, combine: vector_score + cluster_weight(q, cluster_id)
4. Sort final results by combined score

**Recall benefits** (as user noted):
- Standard IVF: only search clusters where centroid is close to q
- Soft assignment: can search clusters where q is "borderline" - improves recall at boundaries
- If cluster A's centroid is 0.9 away, cluster B is 1.0 away, but query is 0.85 from A → search B too for extra recall
- Similar to "IVF+F" or "multi-probe" approaches

**Trade-offs**:
- More clusters to scan per query (M vs 1)
- More vectors in inverted lists (duplicates across clusters)
- Storage overhead: store K distances per vector instead of 1

---

## 3. Implementation Details

### 3.1 Data Structures

```
SoftAssignment:
  vector_id: int
  assignments: List[(cluster_id, distance_to_centroid)]  // sorted by distance, size <= K
  
IVFIndex:
  centroids: List[ndarray]  // current centroids, size = n_clusters
  buckets: List[Bucket]    // logarithmic method buckets
  bucket_i_vectors: Dict[int, List[SoftAssignment]]
  
Bucket:
  vectors: List[SoftAssignment]
  cluster_ids: Set[int]    // which clusters have vectors in this bucket
  size: int                // number of vectors
```

### 3.2 Insertion Algorithm

```python
def insert(vector: ndarray):
    # Step 1: Find K nearest centroids
    distances = [(i, euclidean(vector, centroids[i])) for i in range(n_clusters)]
    distances.sort(key=lambda x: x[1])
    top_k = distances[:K]
    
    # Step 2: Create soft assignment
    assignment = SoftAssignment(
        vector_id=next_id(),
        assignments=top_k
    )
    
    # Step 3: Find which bucket to insert into
    # (Logarithmic method - find largest j where B1...Bj are full)
    j = find_largest_full_bucket()
    
    # Step 4: Add to bucket j
    buckets[j].add(assignment)
    
    # Step 5: If bucket j is full, trigger merge
    if buckets[j].is_full():
        merge_buckets(j)
```

### 3.3 Merge Algorithm (with soft assignment handling)

```python
def merge_buckets(j: int):
    # Clear buckets 1...j, get all vectors
    all_vectors = []
    for i in range(1, j+1):
        all_vectors.extend(buckets[i].vectors)
        buckets[i].clear()
    
    # Add the new vector that triggered this
    all_vectors.append(new_vector)
    
    # Re-cluster: run k-means on merged set
    # BUT: also update centroids for ALL clusters (affects neighbors!)
    new_centroids = kmeans(all_vectors, n_clusters=2^j)  # doubled
    
    # Key: DON'T migrate vectors from other buckets!
    # Keep their old soft assignments, they still get searched
    # Just add new centroid to global centroids list
    
    # Add to merged bucket
    buckets[j+1].vectors = all_vectors
    
    # The quarter-full invariant: don't overfill
    ensure_quarter_full(buckets[j+1])
```

### 3.4 Query-Time Scoring Function

This is the critical part for recall. Need to think through combining cluster weights with vector distances.

**Option 1: Additive**
```
score(v, q) = vector_distance(v, q) + alpha * cluster_weight(v, q)
```

**Option 2: Multiplicative**
```
score(v, q) = vector_distance(v, q) * (1 + beta * cluster_weight(v, q))
```

**Option 3: Tiered scan**
```
# First, compute query distances to all centroids
centroid_distances = [euclidean(q, c) for c in centroids]
sorted_centroids = argsort(centroid_distances)

# Scan clusters in order
# For each cluster, scan its vectors and collect top-K results
# But also scan "nearby" clusters (within delta of closest centroid)
```

**Recommended: Option 3 with soft assignment weighting**

```python
def query(q: ndarray, n_results: int, n_probes: int = 4):
    # Step 1: Get centroid distances
    centroid_distances = [(i, euclidean(q, centroids[i])) for i in range(n_clusters)]
    centroid_distances.sort(key=lambda x: x[1])
    
    # Step 2: Choose clusters to scan
    # - Always scan top-M by centroid distance
    # - Also scan clusters where q is close to boundary (within factor of closest)
    min_dist = centroid_distances[0][1]
    boundary_clusters = [
        c for c, d in centroid_distances 
        if d < min_dist * 1.5  # 50% further than closest
    ]
    clusters_to_scan = set([c for c, _ in centroid_distances[:n_probes]] + boundary_clusters)
    
    # Step 3: Scan each cluster's vectors
    candidates = []
    for cluster_id in clusters_to_scan:
        for assignment in bucket_vectors[cluster_id]:
            # Combine: vector's distance to q + cluster's affinity from assignment
            # Soft assignment has: (cluster_id, distance_to_cluster_centroid)
            
            # If this is the cluster we're currently scanning
            if assignment.cluster_id == cluster_id:
                # Use actual vector distance
                v_dist = euclidean(q, assignment.vector)
            else:
                # This vector is in this cluster due to soft assignment from another cluster
                # Its cluster_weight = how close q is to THIS cluster's centroid
                v_dist = euclidean(q, assignment.vector)
            
            # Combine scores
            # For soft assignment: weight = 1 / (1 + distance_to_centroid)
            cluster_weight = 1.0 / (1.0 + assignment.distance_to_centroid)
            combined_score = v_dist - alpha * cluster_weight  # lower is better
            
            candidates.append((assignment.vector_id, combined_score))
    
    # Step 4: Get top-N
    candidates.sort(key=lambda x: x[1])
    return candidates[:n_results]
```

### 3.5 Tuning Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| K (soft assignment size) | 2-4 | More clusters per vector = more storage, better recall |
| n_probes | 4-8 | How many clusters to scan at query time |
| alpha | 0.1-0.5 | How much cluster affinity weights vs vector distance |
| bucket_size | 1024 | From paper - affects merge frequency |
| merge_threshold | 0.25 (quarter-full) | How empty to keep buckets after merge |

### 3.6 Storage Overhead

- Standard IVF: 1x storage (each vector in 1 cluster)
- Soft assignment K=2: ~1.5-1.8x (some vectors in 2 clusters)
- Soft assignment K=4: ~2-3x

But! Can compress: since clusters are contiguous in embedding space, many vectors share cluster combinations. Could use cluster_id index into precomputed weight table instead of storing float distances.

---

## 4. Complexity Analysis for IVF with Soft Assignment

| Operation | Standard IVF | This approach |
|-----------|--------------|---------------|
| Insert | O(d) assign + O(1) add | O(d·n_clusters) to find top-K centroids + O(1) add |
| Merge | O(n·d) full rebuild | O(2^j·d) for merged bucket only |
| Query | O(n_probes·scan) | O(n_probes·scan·K) |
| Storage | O(n) | O(n·K) |

**Key insight**: Query time increases but we get better recall AND we get dynamic updates that don't require full rebuilds. Tradeoff may be worth it.

---

## 5. Open Questions

1. **How to handle centroids changing globally?**
   - When we merge A+B → C with new centroid, other centroids D,E,F stay same
   - But vectors that were assigned to A with soft assignment (A, D_A) now have new (C, D_C)
   - Do we recompute all soft assignments periodically? Or just let them drift?

2. **When to shrink K?**
   - As data grows, some vectors might belong to very stable clusters
   - Could reduce K over time for storage efficiency

3. **Cluster topology changes**
   - If we merge A+B but also need to split a large cluster, how to coordinate?
   - Paper's logarithmic method only does merges, not splits

4. **Practical implementation**
   - Can we use FAISS's existing IVF and just add soft assignment layer on top?
   - Yes: use faiss.IndexIVF but modify search to use soft assignment scores

---

## 3. Deletion-Only Data Structures for IVF Clusters

**From the paper**: They use [10, 24] deletion-only convex hull structures that don't rotate trees. Implemented as vectors for efficiency.

**Application to IVF**:
- Each IVF cluster (Voronoi cell) can be maintained with a deletion-only structure
- When a vector is deleted from a cluster, don't immediately recompute - just mark it
- Periodically clean up "tombstones" during bucket merges
- The paper proves this gives O(log n log log n) amortized updates

**Note on tombstones**: The paper mentions tombstones don't work for convex hulls (Boolean answer vs cumulative). For IVF with nearest-neighbor queries, tombstones might work better since you're counting distances.

---

## 4. CH-Tree Structure for Hierarchical IVF

**From the paper**: CH-Tree is a balanced binary tree storing points in leaves sorted by x-coordinate. Each node stores the "bridge" between left and right sub-hulls.

**Application to IVF**:
- Similar hierarchical structure could work for IVF - each node stores the "boundary" between two cluster partitions
- Query: to find if point is in convex hull, check if it lies below some bridge; for IVF, check if point is closer to centroid A or B
- This gives a way to incrementally update cluster boundaries without full recomputation

---

## 5. Point Location Query Adaptation

**From the paper**: For point-in-hull queries, they compute tangents for each bucket and combine results. If q is outside all bucket hulls, compute X = {all tangent endpoints} and check if q ∈ CH(X).

**Application to IVF - Finding which cluster a point belongs to**:
- Instead of computing nearest centroid by brute force, use hierarchical approach
- At each tree level, determine which side of the partition the query falls into
- Similar to binary search but on cluster boundaries
- Could reduce "find cluster" from O(k) to O(log k) where k = number of clusters

---

## 6. The "Quarter-Full" Invariant

**From the paper**: After merge, ensure bucket is at most quarter-full (or at least half-full). This guarantees Ω(2^j) new insertions before next Merge(j), giving good amortized costs.

**Application to IVF**:
- When merging clusters, don't fill the new cluster completely
- Leave headroom for new insertions before next re-cluster
- Tune the threshold: quarter-full vs half-full affects merge frequency vs cluster fullness tradeoff

---

## 7. Practical Implementation Notes from Paper

The paper mentions these practical challenges:
- **Vector-based implementation**: Avoid tree rotations and pointer traversals for speed
- **Loser tree for merging**: Balanced tree with O(log j) update per element during j-way merge
- **Handling degeneracies**: Real data has duplicate x-coordinates; handle by keeping max-y only

**For IVF**: Similar concerns - avoid pointer-heavy structures, use contiguous memory, handle duplicate vectors.

---

## 8. Efficiency Analysis from Paper

**Theorem 1 from paper**: The procedure takes amortized O(n log n log log n) time for n updates.

**For IVF**:
- Each vector participates in O(log n) bucket merges
- Each merge does O(size) work but sizes double, so total is O(n log n)
- Extra log log n factor from the merge complexity
- This is much better than rebuilding entire IVF index O(nd) each time

---

## 9. Experiments - Key Findings for IVF

**From Section 4**: The paper compares against Semi-Static (rebuild on update), CH_Tree, CQ_Tree, DPCH.

**Key findings**:
1. **Sparse hulls** (Box, Bell): Semi-Static wins because most updates don't change the hull
2. **Dense hulls** (Disk, Circle): Their FDH method wins 3x faster than naive rebuild
3. **Mixed workloads**: FDH remains preferred until query:update ratio exceeds 10:1

**For IVF**: Similar - if data distribution is stable, occasional rebuild is fine. But for drifting data or high update rates, incremental bucket merging wins.

---

## 10. Practical Insights for IVF Implementation

**Bucket size tuning**: Paper used bucket size 32 and 1024 - 1024 was slightly better. IVF cluster sizes would need similar tuning.

**Robustness**: Real-world data has duplicate coordinates - they handled by keeping max-y only. For IVF, handle duplicate vectors similarly.

**Query-update tradeoff**: FDH is slower on queries (O(log² n) vs O(log n)) but much faster on updates. For IVF, similar - might need to scan more clusters but cluster updates are cheaper.

---

## Summary: Which Ideas Are Practical?

| Idea | Feasibility | Notes |
|------|-------------|-------|
| Bucket-based incremental clustering | Medium | Need to adapt cluster merging algorithm |
| Local re-clustering on merge | High | Only re-cluster merged buckets, not all |
| Deletion-only cluster structures | Medium | Tombstones probably work for IVF (cumulative queries) |
| Hierarchical cluster trees | High | Similar to existing IVF-flat approaches |
| Quarter-full invariant | Medium | Tuning parameter for merge frequency |
| Loser tree for merging | High | Efficient j-way merge for sorted vectors |

**Most practical first step**: Implement bucket-based IVF where new vectors go into small buckets, merge buckets when they fill, re-cluster only merged content.

**Expected improvement**: O(n log n log log n) amortized updates vs O(nd) full rebuilds - huge for high update rates.