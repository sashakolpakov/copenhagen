# Copenhagen Status

## Honest Assessment

**Both FAISS and Copenhagen degrade with dynamic updates** - they just degrade differently.

- **FAISS PQ**: Quantization error inherent to PQ (31% recall drop on SIFT)
- **Copenhagen**: Centroid drift from incremental updates (29% recall drop)

**Copenhagen's real advantage**: Fast O(1) inserts. FAISS requires full rebuild.
**Copenhagen's limitation**: Need periodic retraining to maintain recall.

## Benchmarks

### SIFT 50k (d=128, 256 clusters, nprobe=16, trained on all data)

| Metric | FAISS Exact | FAISS PQ | Copenhagen |
|--------|-------------|----------|------------|
| Recall@10 | 0.926 | 0.660 | 0.950 |
| Query (ms) | 0.055 | 0.072 | 0.141 |

### After Dynamic Updates (train 50k, add 150k more)

| Metric | FAISS | Copenhagen |
|--------|-------|------------|
| Degradation | 31% (PQ error) | 29% (centroid drift) |

**Both degrade similarly. Difference is FAISS requires rebuild, Copenhagen doesn't.**

## What's Working

- ✅ Matches FAISS exact recall when trained on all data
- ✅ Fast O(1) inserts (vs FAISS full rebuild)
- ✅ BLAS-accelerated training and search
- ✅ PQ + re-ranking for memory efficiency

## What's Not Working (Yet)

- ❌ Centroid drift from incremental updates
- ❌ Online centroid adaptation (mathematically causes orphaning)

## What's Next

1. **Periodic retrain() method** - User calls when ready, O(n) but keeps exact centroids
2. **Cluster balancing** - Ensure vectors evenly distributed
3. **OPQ (Optimized PQ)** - Rotation before quantization
4. **Batch search optimization** - Currently not well optimized

## Architecture Notes

```
Insert: O(1) - add to nearest cluster, update centroid sum
Search: O(nprobe * cluster_size) - BLAS L2 distances
Retrain: O(n) - recompute centroids from scratch (user-triggered)
```
