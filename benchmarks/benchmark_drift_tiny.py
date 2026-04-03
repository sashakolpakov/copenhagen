#!/usr/bin/env python3
"""
Memory-efficient centroid drift test - SMALL dataset.
"""

import numpy as np
import time
import gc
import sys

sys.path.insert(0, 'python/core')
from copenhagen import DynamicIVF

import faiss

def load_sift_small(n_train=20000, n_test=100):
    import h5py
    print(f"Loading SIFT (train={n_train}, test={n_test})...")
    with h5py.File('data/sift/sift-128-euclidean.hdf5', 'r') as f:
        X_train = f['train'][:n_train].astype(np.float32)
        X_test = f['test'][:n_test].astype(np.float32)
    return X_train, X_test

def brute_force_gt(X_train, X_test, k=10):
    print(f"Computing GT ({len(X_test)} queries)...")
    n_test = len(X_test)
    gt = np.zeros((n_test, k), dtype=np.int32)
    for i in range(n_test):
        q = X_test[i]
        dists = np.sum((X_train - q)**2, axis=1)
        gt[i] = np.argpartition(dists, k-1)[:k]
    return gt

def recall_at_k(gt, results, k=10):
    total = 0
    for i in range(len(gt)):
        if isinstance(results[i], tuple):
            result_ids = list(results[i][0][:k])
        else:
            r = results[i]
            if r.ndim > 1:
                r = r[0]
            result_ids = [int(x) for x in r[:k]]
        gt_set = set(int(x) for x in gt[i])
        total += len(set(result_ids) & gt_set)
    return total / (len(gt) * k)

def main():
    print("="*50)
    print("CENTROID DRIFT TEST")
    print("="*50)
    
    d = 128
    n_initial = 10000
    n_inserts = 10000
    n_clusters = 128
    nprobe = 8
    
    X_train, X_test = load_sift_small(n_train=n_initial + n_inserts, n_test=100)
    X_initial = X_train[:n_initial]
    X_new = X_train[n_initial:]
    queries = X_test[:50]
    
    # GT on initial
    print("\nGT on initial data...")
    gt_initial = brute_force_gt(X_initial, queries, k=10)
    
    # GT on full
    print("GT on full data...")
    gt_full = brute_force_gt(X_train, queries, k=10)
    gc.collect()
    
    # ===== Copenhagen =====
    print("\nBuilding Copenhagen...")
    t0 = time.perf_counter()
    cop_idx = DynamicIVF(dim=d, n_clusters=n_clusters, nprobe=nprobe, use_pq=0)
    cop_idx.train(X_initial)
    cop_train_time = time.perf_counter() - t0
    print(f"  Train time: {cop_train_time:.2f}s")
    
    # Initial recall
    print("\nInitial recall...")
    results = [cop_idx.search(q, 10, nprobe) for q in queries]
    initial_recall = recall_at_k(gt_initial, results)
    print(f"  Recall@10: {initial_recall:.4f}")
    
    # Insert new vectors - measure TIME
    print(f"\nInserting {n_inserts} vectors...")
    t0 = time.perf_counter()
    cop_idx.insert_batch(X_new)
    cop_insert_time = time.perf_counter() - t0
    print(f"  Insert time: {cop_insert_time:.2f}s ({cop_insert_time/n_inserts*1000:.4f}ms/vec)")
    
    # Final recall
    print("\nFinal recall (GT on FULL)...")
    results = [cop_idx.search(q, 10, nprobe) for q in queries]
    final_recall = recall_at_k(gt_full, results)
    print(f"  Recall@10: {final_recall:.4f}")
    
    # ===== FAISS =====
    print("\nFAISS build time...")
    t0 = time.perf_counter()
    faiss_idx = faiss.IndexFlatL2(d)
    faiss_ivf = faiss.IndexIVFFlat(faiss_idx, d, n_clusters)
    faiss_ivf.train(X_train)
    faiss_ivf.add(X_train)
    faiss_ivf.nprobe = nprobe
    faiss_build_time = time.perf_counter() - t0
    print(f"  Build time: {faiss_build_time:.2f}s ({faiss_build_time/n_inserts*1000:.4f}ms/vec)")
    
    print("FAISS search...")
    faiss_results = []
    for q in queries:
        D, I = faiss_ivf.search(q.reshape(1, -1), 10)
        faiss_results.append(I)
    print(f"  Sample FAISS result: {faiss_results[0]}")
    
    faiss_recall = recall_at_k(gt_full, faiss_results)
    print(f"  Recall@10: {faiss_recall:.4f}")
    
    # ===== Results =====
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Copenhagen insert: {cop_insert_time:.2f}s ({cop_insert_time/n_inserts*1000:.4f}ms/vec)")
    print(f"FAISS rebuild:     {faiss_build_time:.2f}s ({faiss_build_time/n_inserts*1000:.4f}ms/vec)")
    print(f"Speedup:           {faiss_build_time/cop_insert_time:.1f}x")
    print()
    print(f"Copenhagen final: {final_recall:.4f}")
    print(f"FAISS:            {faiss_recall:.4f}")
    print(f"Diff:             {abs(final_recall - faiss_recall):.4f}")
    
    if abs(final_recall - faiss_recall) < 0.05:
        print("\n✓ PASS: Copenhagen matches FAISS recall")
    else:
        print(f"\n✗ FAIL")

if __name__ == "__main__":
    main()