#!/usr/bin/env python3
"""Proper centroid drift benchmark - measure recall against same GT."""

import numpy as np
import time
import gc
import sys

sys.path.insert(0, 'python/core')
from copenhagen import DynamicIVF
import faiss

def load_sift(n_train=120000, n_test=200):
    import h5py
    print(f"Loading SIFT (train={n_train}, test={n_test})...")
    with h5py.File('data/sift/sift-128-euclidean.hdf5', 'r') as f:
        X_train = f['train'][:n_train].astype(np.float32)
        X_test = f['test'][:n_test].astype(np.float32)
    return X_train, X_test

def brute_force_gt(X_train, X_test, k=10):
    print(f"Computing GT ({len(X_test)} queries)...")
    gt = np.zeros((len(X_test), k), dtype=np.int32)
    for i in range(len(X_test)):
        dists = np.sum((X_train - X_test[i])**2, axis=1)
        gt[i] = np.argpartition(dists, k-1)[:k]
    return gt

def recall_at_k(gt, results, k=10):
    total = 0
    for i in range(len(gt)):
        ids = results[i][1] if len(results[i]) == 2 and isinstance(results[i][0], np.ndarray) else results[i][0]
        if hasattr(ids, 'flatten'):
            ids_int = [int(round(x)) for x in ids.flatten()[:k]]
        else:
            ids_int = [int(round(x)) for x in ids[:k]]
        gt_set = set(int(x) for x in gt[i])
        total += len(set(ids_int) & gt_set)
    return total / (len(gt) * k)

def main():
    print("="*60)
    print("CENTROID DRIFT BENCHMARK (Proper Test)")
    print("="*60)
    
    d = 128
    n_initial = 50000
    n_inserts = 50000
    n_clusters = 256
    nprobe = 16
    
    X_train, X_test = load_sift(n_train=n_initial + n_inserts, n_test=200)
    X_initial = X_train[:n_initial]
    X_new = X_train[n_initial:]
    queries = X_test[:100]
    
    # GT on INITIAL data only - for initial recall
    print("\nGT on initial data only...")
    gt_initial = brute_force_gt(X_initial, queries, k=10)
    
    # GT on FULL data - for final recall
    print("GT on full data...")
    gt_full = brute_force_gt(X_train, queries, k=10)
    gc.collect()
    
    # Copenhagen
    print(f"\n[1] Copenhagen: train on {n_initial} vectors")
    t0 = time.perf_counter()
    cop_idx = DynamicIVF(dim=d, n_clusters=n_clusters, nprobe=nprobe, use_pq=0)
    cop_idx.train(X_initial)
    cop_train = time.perf_counter() - t0
    print(f"    Train time: {cop_train:.2f}s")
    
    # Initial recall - GT on initial data
    print(f"\n[2] Initial recall (GT on {n_initial} vectors)")
    results = [cop_idx.search(q, 10, nprobe) for q in queries]
    initial_recall = recall_at_k(gt_initial, results)
    print(f"    Recall@10: {initial_recall:.4f}")
    
    # Insert new vectors
    print(f"\n[3] Insert {n_inserts} new vectors")
    t0 = time.perf_counter()
    cop_idx.insert_batch(X_new)
    cop_insert = time.perf_counter() - t0
    print(f"    Insert time: {cop_insert:.2f}s ({cop_insert/n_inserts*1000:.4f}ms/vec)")
    
    # Final recall - GT on FULL data
    print(f"\n[4] Final recall (GT on {n_initial + n_inserts} vectors)")
    results = [cop_idx.search(q, 10, nprobe) for q in queries]
    final_recall = recall_at_k(gt_full, results)
    print(f"    Recall@10: {final_recall:.4f}")
    
    # FAISS on full data for comparison
    print(f"\n[5] FAISS: rebuild on full {n_initial + n_inserts} vectors")
    t0 = time.perf_counter()
    faiss_idx = faiss.IndexFlatL2(d)
    faiss_ivf = faiss.IndexIVFFlat(faiss_idx, d, n_clusters)
    faiss_ivf.train(X_train)
    faiss_ivf.add(X_train)
    faiss_ivf.nprobe = nprobe
    faiss_build = time.perf_counter() - t0
    print(f"    Build time: {faiss_build:.2f}s ({faiss_build/n_inserts*1000:.4f}ms/vec)")
    
    results = [faiss_ivf.search(q.reshape(1,-1), 10) for q in queries]
    faiss_recall = recall_at_k(gt_full, results)
    print(f"    Recall@10: {faiss_recall:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Initial recall (GT on init): {initial_recall:.4f}")
    print(f"  Final recall (GT on full):   {final_recall:.4f}")
    print(f"  FAISS recall (GT on full):   {faiss_recall:.4f}")
    print(f"  Copenhagen vs FAISS diff:    {abs(final_recall - faiss_recall):.4f}")
    print()
    print(f"  Copenhagen insert: {cop_insert:.2f}s ({cop_insert/n_inserts*1000:.4f}ms/vec)")
    print(f"  FAISS rebuild:    {faiss_build:.2f}s ({faiss_build/n_inserts*1000:.4f}ms/vec)")
    if cop_insert < faiss_build:
        print(f"  Speedup:          {faiss_build/cop_insert:.1f}x faster")
    else:
        print(f"  Slower:          {cop_insert/faiss_build:.1f}x")
    
    if abs(final_recall - faiss_recall) < 0.02:
        print("\n✓ PASS: Copenhagen matches FAISS recall")
    else:
        print(f"\n✗ FAIL: Recall diff = {abs(final_recall - faiss_recall)*100:.1f}%")

if __name__ == "__main__":
    main()