#!/usr/bin/env python3
"""
Benchmark: DynamicIVF vs FAISS IVF
Tests incremental inserts and compares recall/QPS with FAISS static index.
"""

import numpy as np
import time
import faiss
import sys
sys.path.insert(0, 'python')
from index import DynamicIVFIndex

np.random.seed(42)

def make_gaussian(n=10000, d=128):
    return np.random.randn(n, d).astype(np.float32)


def benchmark_dynamic_vs_faiss():
    print("=" * 70)
    print("DYNAMIC IVF vs FAISS IVF BENCHMARK")
    print("=" * 70)
    
    # Load data
    print("\nLoading MNIST data...")
    try:
        import h5py
        with h5py.File('data/MNIST/mnist-784-euclidean.hdf5', 'r') as f:
            full_data = f['train'][:]
            test = f['test'][:]
    except Exception as e:
        print(f"Could not load MNIST: {e}")
        print("Using synthetic Gaussian data instead...")
        full_data = make_gaussian(20000, 128)
        test = make_gaussian(1000, 128)
    
    d = full_data.shape[1]
    print(f"Dataset: {full_data.shape[0]} vectors, dim={d}")
    
    # Split into train and incremental
    n_train = 50000
    n_incremental = 10000
    n_test = min(1000, len(test))
    
    train_data = full_data[:n_train]
    incremental_data = full_data[n_train:n_train + n_incremental]
    queries = test[:n_test]
    
    # Ground truth
    print("Computing ground truth...")
    dists = np.sqrt(((queries[:, None] - train_data[None]) ** 2).sum(axis=2))
    gt_indices = np.argsort(dists, axis=1)[:, :10]
    
    # Parameters to test
    n_clusters = 256
    nprobe = 8
    
    print(f"\n{'='*70}")
    print(f"TEST 1: STATIC COMPARISON (n_clusters={n_clusters}, nprobe={nprobe})")
    print(f"{'='*70}")
    
    # FAISS static
    print("\nBuilding FAISS IVF index...")
    t0 = time.perf_counter()
    faiss_idx = faiss.IndexFlatL2(d)
    quantizer = faiss.IndexIVFFlat(faiss_idx, d, n_clusters)
    quantizer.train(train_data)
    quantizer.add(train_data)
    quantizer.nprobe = nprobe
    faiss_build_time = time.perf_counter() - t0
    print(f"FAISS build time: {faiss_build_time:.2f}s")
    
    # DynamicIVF (batch load like FAISS)
    print("\nBuilding DynamicIVF index (batch)...")
    t0 = time.perf_counter()
    dyn_idx = DynamicIVFIndex(dim=d, n_clusters=n_clusters, nprobe=nprobe)
    dyn_idx.add(train_data)
    dyn_build_time = time.perf_counter() - t0
    print(f"DynamicIVF build time: {dyn_build_time:.2f}s")
    
    # Search benchmark
    print("\nSearching...")
    n_q = len(queries)
    warmup = 10
    
    for _ in range(warmup):
        faiss_idx.search(queries[:1], 10)
        dyn_idx.search(queries[:1], 10)
    
    t0 = time.perf_counter()
    faiss_results = faiss_idx.search(queries, 10)
    faiss_qps = len(queries) / (time.perf_counter() - t0)
    
    t0 = time.perf_counter()
    dyn_results = [dyn_idx.search(q, 10) for q in queries]
    dyn_qps = len(queries) / (time.perf_counter() - t0)
    
    # Compute recalls
    faiss_ids = faiss_results[1]
    dyn_ids = np.array([r[0] for r in dyn_results], dtype=np.int64)
    
    faiss_recall = np.mean([
        len(set(faiss_ids[i][:10]) & set(gt_indices[i])) / 10 
        for i in range(n_q)
    ])
    dyn_recall = np.mean([
        len(set(dyn_ids[i]) & set(gt_indices[i])) / 10 
        for i in range(n_q)
    ])
    
    print(f"\n  FAISS:        QPS={faiss_qps:.1f}, Recall@10={faiss_recall:.4f}")
    print(f"  DynamicIVF:    QPS={dyn_qps:.1f}, Recall@10={dyn_recall:.4f}")
    
    print(f"\n{'='*70}")
    print(f"TEST 2: INCREMENTAL UPDATES")
    print(f"{'='*70}")
    
    # FAISS: need to rebuild for incremental updates (expensive)
    print("\nFAISS incremental update: REBUILD entire index (expensive)...")
    t0 = time.perf_counter()
    combined_data = np.vstack([train_data, incremental_data])
    faiss_idx2 = faiss.IndexFlatL2(d)
    quantizer2 = faiss.IndexIVFFlat(faiss_idx2, d, n_clusters)
    quantizer2.train(combined_data)
    quantizer2.add(combined_data)
    quantizer2.nprobe = nprobe
    faiss_rebuild_time = time.perf_counter() - t0
    print(f"FAISS rebuild time: {faiss_rebuild_time:.2f}s")
    
    # DynamicIVF: incremental add (cheap)
    print("\nDynamicIVF incremental update: ADD vectors...")
    t0 = time.perf_counter()
    dyn_idx.add(incremental_data)
    dyn_add_time = time.perf_counter() - t0
    print(f"DynamicIVF add time: {dyn_add_time:.2f}s")
    print(f"Speedup: {faiss_rebuild_time/dyn_add_time:.1f}x faster!")
    
    # Search after incremental
    t0 = time.perf_counter()
    dyn_results2 = [dyn_idx.search(q, 10) for q in queries]
    dyn_qps2 = len(queries) / (time.perf_counter() - t0)
    dyn_ids2 = np.array([r[0] for r in dyn_results2], dtype=np.int64)
    dyn_recall2 = np.mean([
        len(set(dyn_ids2[i]) & set(gt_indices[i])) / 10 
        for i in range(n_q)
    ])
    
    print(f"\nDynamicIVF after incremental:")
    print(f"  QPS={dyn_qps2:.1f}, Recall@10={dyn_recall2:.4f}")
    
    print(f"\n{'='*70}")
    print(f"TEST 3: DELETES")
    print(f"{'='*70}")
    
    # Delete some vectors
    n_delete = 5000
    delete_ids = np.random.choice(len(train_data), n_delete, replace=False)
    
    print(f"\nDeleting {n_delete} vectors...")
    
    # FAISS: rebuild
    t0 = time.perf_counter()
    mask = np.ones(len(combined_data), dtype=bool)
    mask[delete_ids] = False
    remaining_data = combined_data[mask]
    faiss_idx3 = faiss.IndexFlatL2(d)
    quantizer3 = faiss.IndexIVFFlat(faiss_idx3, d, n_clusters)
    quantizer3.train(remaining_data)
    quantizer3.add(remaining_data)
    quantizer3.nprobe = nprobe
    faiss_delete_rebuild = time.perf_counter() - t0
    print(f"FAISS rebuild after delete: {faiss_delete_rebuild:.2f}s")
    
    # DynamicIVF: delete
    t0 = time.perf_counter()
    dyn_idx.delete(delete_ids)
    dyn_delete_time = time.perf_counter() - t0
    print(f"DynamicIVF delete time: {dyn_delete_time:.2f}s")
    print(f"Speedup: {faiss_delete_rebuild/dyn_delete_time:.1f}x faster!")
    
    print(f"\n{'='*70}")
    print(f"TEST 4: TUNING FOR BETTER RECALL/QPS")
    print(f"{'='*70}")
    
    # Test different nprobe values
    print("\nTuning nprobe for DynamicIVF...")
    for np_ in [4, 8, 16, 32]:
        dyn_idx.nprobe = np_
        t0 = time.perf_counter()
        res = [dyn_idx.search(q, 10) for q in queries[:100]]
        qps = 100 / (time.perf_counter() - t0)
        ids = np.array([r[0] for r in res], dtype=np.int64)
        recall = np.mean([
            len(set(ids[i]) & set(gt_indices[i])) / 10 
            for i in range(100)
        ])
        print(f"  nprobe={np_:3d}: QPS={qps:7.1f}, Recall@10={recall:.4f}")
    
    # Test with more clusters
    print("\nTrying more clusters...")
    dyn_idx2 = DynamicIVFIndex(dim=d, n_clusters=512, nprobe=16)
    dyn_idx2.add(train_data)
    
    t0 = time.perf_counter()
    res = [dyn_idx2.search(q, 10) for q in queries[:100]]
    qps = 100 / (time.perf_counter() - t0)
    ids = np.array([r[0] for r in res], dtype=np.int64)
    recall = np.mean([
        len(set(ids[i]) & set(gt_indices[i])) / 10 
        for i in range(100)
    ])
    print(f"  n_clusters=512, nprobe=16: QPS={qps:7.1f}, Recall@10={recall:.4f}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"""
Static FAISS vs DynamicIVF:
  - FAISS:  {faiss_qps:.1f} QPS, {faiss_recall:.4f} recall
  - DynamicIVF: {dyn_qps:.1f} QPS, {dyn_recall:.4f} recall

Dynamic Updates:
  - FAISS rebuild: {faiss_rebuild_time:.2f}s
  - DynamicIVF add: {dyn_add_time:.4f}s
  - Speedup: {faiss_rebuild_time/dyn_add_time:.0f}x

Deletes:
  - FAISS rebuild: {faiss_delete_rebuild:.2f}s
  - DynamicIVF delete: {dyn_delete_time:.4f}s
  - Speedup: {faiss_delete_rebuild/dyn_delete_time:.0f}x
""")


if __name__ == "__main__":
    benchmark_dynamic_vs_faiss()
