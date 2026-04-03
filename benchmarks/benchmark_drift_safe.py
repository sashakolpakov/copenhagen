#!/usr/bin/env python3
"""
Memory-efficient benchmark for Copenhagen centroid drift.
Focuses on testing that recall stays stable with inserts (no centroid drift).
"""

import numpy as np
import time
import sys
import h5py
import gc

sys.path.insert(0, 'python/core')
from copenhagen import DynamicIVF

import faiss

def load_sift_subset(n_train=100000, n_test=1000):
    """Load SIFT subset to avoid OOM."""
    print(f"Loading SIFT subset (train={n_train}, test={n_test})...")
    with h5py.File('data/sift/sift-128-euclidean.hdf5', 'r') as f:
        X_train = f['train'][:n_train].astype(np.float32)
        X_test = f['test'][:n_test].astype(np.float32)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test

def brute_force_gt_chunked(X_train, X_test, k=10, chunk_size=10000):
    """Compute ground truth in chunks to save memory."""
    print(f"Computing ground truth (chunked, {len(X_test)} queries)...")
    n_test = len(X_test)
    gt = np.zeros((n_test, k), dtype=np.int32)
    
    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        queries = X_test[start:end]
        
        dists = np.sum((queries[:, None, :] - X_train[None, :, :]) ** 2, axis=2)
        gt[start:end] = np.argsort(dists, axis=1)[:, :k]
        
        if start % 10000 == 0:
            print(f"  Processed {end}/{n_test} queries...")
        gc.collect()
    
    return gt

def recall_at_k(gt, results, k=10):
    """Compute recall@k."""
    total = 0
    for i in range(len(gt)):
        total += len(set(results[i][:k]) & set(gt[i]))
    return total / (len(gt) * k)

def benchmark_centroid_drift(n_initial=50000, n_inserts=50000, n_clusters=256, nprobe=16):
    """Test centroid drift - recall should stay stable with fixed centroids."""
    d = 128
    print(f"\n{'='*60}")
    print(f"CENTROID DRIFT TEST")
    print(f"  Initial: {n_initial}, Insert: {n_inserts}, Clusters: {n_clusters}")
    print(f"{'='*60}")
    
    X_train, X_test = load_sift_subset(n_train=n_initial + n_inserts, n_test=1000)
    
    X_initial = X_train[:n_initial]
    X_new = X_train[n_initial:]
    queries = X_test[:100]
    
    print(f"\nComputing ground truth on FULL data ({n_initial + n_inserts} vectors)...")
    gt_full = brute_force_gt_chunked(X_train, queries, k=10)
    
    print("\nBuilding Copenhagen index...")
    t0 = time.perf_counter()
    cop_idx = DynamicIVF(dim=d, n_clusters=n_clusters, nprobe=nprobe, use_pq=0)
    cop_idx.train(X_initial)
    build_time = time.perf_counter() - t0
    print(f"  Build time: {build_time:.2f}s")
    
    print("\nInitial recall (before inserts)...")
    results = [cop_idx.search(q, 10, nprobe)[0] for q in queries]
    initial_recall = recall_at_k(gt_full, results)
    print(f"  Recall@10: {initial_recall:.4f}")
    
    print(f"\nInserting {n_inserts} vectors...")
    t0 = time.perf_counter()
    cop_idx.insert_batch(X_new)
    insert_time = time.perf_counter() - t0
    print(f"  Insert time: {insert_time:.2f}s ({insert_time/n_inserts*1000:.4f}ms/vector)")
    
    print("\nRecall AFTER inserts (GT on full data)...")
    results = [cop_idx.search(q, 10, nprobe)[0] for q in queries]
    final_recall = recall_at_k(gt_full, results)
    print(f"  Recall@10: {final_recall:.4f}")
    
    drift = initial_recall - final_recall
    print(f"\n  Drift: {drift:.4f} (positive = recall dropped)")
    
    print("\nComparing to FAISS rebuild...")
    t0 = time.perf_counter()
    faiss_idx = faiss.IndexFlatL2(d)
    faiss_ivf = faiss.IndexIVFFlat(faiss_idx, d, n_clusters)
    faiss_ivf.train(X_train)
    faiss_ivf.add(X_train)
    faiss_ivf.nprobe = nprobe
    faiss_build = time.perf_counter() - t0
    print(f"  FAISS rebuild time: {faiss_build:.2f}s")
    
    print("\nFAISS recall (on full data)...")
    faiss_results = [faiss_ivf.search(q.reshape(1, -1), 10)[0] for q in queries]
    faiss_recall = recall_at_k(gt_full, faiss_results)
    print(f"  FAISS Recall@10: {faiss_recall:.4f}")
    
    return {
        'initial_recall': initial_recall,
        'final_recall': final_recall,
        'faiss_recall': faiss_recall,
        'drift': drift,
        'insert_time': insert_time,
        'faiss_build': faiss_build,
    }

def main():
    print("="*60)
    print("COPENHAGEN CENTROID DRIFT BENCHMARK")
    print("="*60)
    
    gc.collect()
    
    result = benchmark_centroid_drift(
        n_initial=50000,
        n_inserts=50000,
        n_clusters=256,
        nprobe=16
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Copenhagen initial recall: {result['initial_recall']:.4f}")
    print(f"  Copenhagen final recall:   {result['final_recall']:.4f}")
    print(f"  FAISS recall:              {result['faiss_recall']:.4f}")
    print(f"  Drift (initial - final):   {result['drift']:.4f}")
    print(f"  Copenhagen insert:         {result['insert_time']:.2f}s")
    print(f"  FAISS rebuild:             {result['faiss_build']:.2f}s")
    print(f"  Speedup:                   {result['faiss_build']/result['insert_time']:.1f}x")
    
    if result['drift'] < 0.01:
        print("\n✓ PASS: Centroid drift is minimal (< 1%)")
    else:
        print(f"\n✗ FAIL: Centroid drift detected ({result['drift']*100:.2f}%)")

if __name__ == "__main__":
    main()