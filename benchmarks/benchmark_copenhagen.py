#!/usr/bin/env python3
"""
Comprehensive benchmark for Copenhagen vs FAISS
Tests: recall, query speed, insert speed, centroid drift over time
"""

import numpy as np
import time
import sys
import h5py

sys.path.insert(0, 'python/core')
from copenhagen import DynamicIVF

import faiss

def load_dataset(name, n_train=None):
    """Load dataset from HDF5."""
    paths = {
        'sift': 'data/sift/sift-128-euclidean.hdf5',
        'fashion': 'data/fashion-mnist/fashion-mnist-784-euclidean.hdf5',
        'mnist': 'data/MNIST/mnist-784-euclidean.hdf5',
    }
    
    print(f"Loading {name}...")
    with h5py.File(paths[name], 'r') as f:
        X_train_full = f['train'][:].astype(np.float32)
        X_test = f['test'][:].astype(np.float32)
    
    if n_train and n_train < len(X_train_full):
        X_train = X_train_full[:n_train]
    else:
        X_train = X_train_full
        n_train = len(X_train_full)
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, n_train

def brute_force_gt(X_train, X_test, k=10):
    """Compute ground truth using brute force."""
    print(f"Computing ground truth ({len(X_test)} queries)...")
    dists = np.sum((X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=2)
    gt = np.argsort(dists, axis=1)[:, :k]
    return gt

def recall_at_k(gt, results, k=10):
    """Compute recall@k."""
    total = 0
    for i in range(len(gt)):
        total += len(set(results[i][:k]) & set(gt[i]))
    return total / (len(gt) * k)

def benchmark_recall_and_speed(name, X_train, X_test, gt, n_clusters=256, nprobe=16):
    """Benchmark recall and query speed."""
    d = X_train.shape[1]
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name} (d={d}, n_clusters={n_clusters}, nprobe={nprobe})")
    print(f"{'='*60}")
    
    # FAISS
    print("\nBuilding FAISS...")
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    faiss_idx = faiss.IndexIVFFlat(quantizer, d, n_clusters)
    faiss_idx.train(X_train)
    faiss_idx.add(X_train)
    faiss_idx.nprobe = nprobe
    faiss_build_time = time.perf_counter() - t0
    print(f"  FAISS build: {faiss_build_time:.2f}s")
    
    # Copenhagen
    print("Building Copenhagen...")
    t0 = time.perf_counter()
    cop_idx = DynamicIVF(dim=d, n_clusters=n_clusters, nprobe=nprobe, use_pq=0)
    cop_idx.train(X_train)
    cop_build_time = time.perf_counter() - t0
    print(f"  Copenhagen build: {cop_build_time:.2f}s")
    
    # Search benchmark
    print("\nSearch benchmark (100 queries, warmup=10)...")
    queries = X_test[:100]
    
    # Warmup
    for q in queries[:10]:
        faiss_idx.search(q.reshape(1, -1), 10)
        cop_idx.search(q, 10, nprobe)
    
    # FAISS
    t0 = time.perf_counter()
    faiss_results = []
    for q in queries:
        D, I = faiss_idx.search(q.reshape(1, -1), 10)
        faiss_results.append(I[0])
    faiss_search_time = (time.perf_counter() - t0) / len(queries) * 1000
    
    # Copenhagen
    t0 = time.perf_counter()
    cop_results = []
    for q in queries:
        I, D = cop_idx.search(q, 10, nprobe)
        cop_results.append(I)
    cop_search_time = (time.perf_counter() - t0) / len(queries) * 1000
    
    # Recall
    faiss_recall = recall_at_k(gt[:100], faiss_results)
    cop_recall = recall_at_k(gt[:100], cop_results)
    
    print(f"\nResults:")
    print(f"  FAISS:       recall@10={faiss_recall:.4f}, query={faiss_search_time:.3f}ms")
    print(f"  Copenhagen:   recall@10={cop_recall:.4f}, query={cop_search_time:.3f}ms")
    print(f"  Recall diff: {abs(faiss_recall - cop_recall):.4f}")
    
    return {
        'name': name,
        'faiss_recall': faiss_recall,
        'faiss_query_ms': faiss_search_time,
        'faiss_build_s': faiss_build_time,
        'cop_recall': cop_recall,
        'cop_query_ms': cop_search_time,
        'cop_build_s': cop_build_time,
    }

def benchmark_insert_speed(name, X_initial, X_new, X_test, gt_initial, n_clusters=256, nprobe=16):
    """Benchmark insert speed and centroid drift."""
    d = X_initial.shape[1]
    print(f"\n{'='*60}")
    print(f"INSERT BENCHMARK: {name}")
    print(f"{'='*60}")
    
    # Train on initial data
    print(f"\nTraining on {len(X_initial)} vectors...")
    cop_idx = DynamicIVF(dim=d, n_clusters=n_clusters, nprobe=nprobe, use_pq=0)
    cop_idx.train(X_initial)
    
    # Compute ground truth on initial data
    dists = np.sum((X_test[:, None, :] - X_initial[None, :, :]) ** 2, axis=2)
    gt_initial = np.argsort(dists, axis=1)[:, :10]
    
    # Initial recall
    queries = X_test[:50]
    results = [cop_idx.search(q, 10, nprobe)[0] for q in queries]
    initial_recall = recall_at_k(gt_initial, results)
    print(f"  Initial recall@10: {initial_recall:.4f}")
    
    # Insert new vectors
    print(f"\nInserting {len(X_new)} vectors...")
    t0 = time.perf_counter()
    cop_idx.insert_batch(X_new)
    insert_time = time.perf_counter() - t0
    insert_per_vector = insert_time / len(X_new) * 1000
    print(f"  Total time: {insert_time:.3f}s")
    print(f"  Per vector: {insert_per_vector:.4f}ms")
    
    # Compute ground truth on full data
    X_full = np.vstack([X_initial, X_new])
    dists = np.sum((X_test[:, None, :] - X_full[None, :, :]) ** 2, axis=2)
    gt_full = np.argsort(dists, axis=1)[:, :10]
    
    # Recall after inserts
    results = [cop_idx.search(q, 10, nprobe)[0] for q in queries]
    full_recall = recall_at_k(gt_full, results)
    print(f"\n  Recall@10 after inserts (GT on full data): {full_recall:.4f}")
    print(f"  (This is the relevant metric - GT should be on indexed data)")
    
    # Compare to FAISS rebuild
    print(f"\nFAISS rebuild time (for comparison)...")
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    faiss_idx = faiss.IndexIVFFlat(quantizer, d, n_clusters)
    faiss_idx.train(X_full)
    faiss_idx.add(X_full)
    faiss_idx.nprobe = nprobe
    faiss_rebuild_time = time.perf_counter() - t0
    print(f"  FAISS rebuild: {faiss_rebuild_time:.3f}s")
    print(f"  Copenhagen speedup: {faiss_rebuild_time/insert_time:.1f}x faster")
    
    return {
        'name': name,
        'initial_recall': initial_recall,
        'full_recall': full_recall,
        'insert_time_s': insert_time,
        'insert_per_vector_ms': insert_per_vector,
        'faiss_rebuild_s': faiss_rebuild_time,
    }

def benchmark_centroid_drift(X, X_test, n_initial=50000, nprobe=16):
    """Test centroid drift over time with incremental inserts."""
    d = X.shape[1]
    n_clusters = 512
    
    print(f"\n{'='*60}")
    print(f"CENTROID DRIFT TEST")
    print(f"{'='*60}")
    print(f"Initial: {n_initial}, Total: {len(X)}, n_clusters={n_clusters}")
    
    # Compute ground truth on full data
    dists = np.sum((X_test[:, None, :] - X[None, :, :]) ** 2, axis=2)
    gt = np.argsort(dists, axis=1)[:, :10]
    
    queries = X_test[:50]
    
    cop_idx = DynamicIVF(dim=d, n_clusters=n_clusters, nprobe=nprobe, use_pq=0)
    cop_idx.train(X[:n_initial])
    
    # Initial recall
    results = [cop_idx.search(q, 10, nprobe)[0] for q in queries]
    initial_recall = recall_at_k(gt[:50], results)
    print(f"\nInitial recall@10 (n={n_initial}): {initial_recall:.4f}")
    
    # Add vectors in batches and measure recall
    batch_sizes = [50000, 100000, 200000]
    results_log = [('init', n_initial, initial_recall)]
    
    for n_total in batch_sizes:
        if n_total <= n_initial:
            continue
        n_new = n_total - n_initial
        cop_idx.insert_batch(X[n_initial:n_total])
        
        results = [cop_idx.search(q, 10, nprobe)[0] for q in queries]
        recall = recall_at_k(gt[:50], results)
        results_log.append((f'+{n_new}', n_total, recall))
        print(f"After {n_total} vectors: recall@10={recall:.4f}")
        n_initial = n_total
    
    return results_log

def main():
    print("="*60)
    print("COPENHAGEN vs FAISS BENCHMARK")
    print("="*60)
    
    # Load datasets
    X_sift_train, X_sift_test, _ = load_dataset('sift', n_train=50000)
    X_fashion_train, X_fashion_test, _ = load_dataset('fashion', n_train=10000)
    
    # Ground truth
    gt_sift = brute_force_gt(X_sift_train, X_sift_test[:50])
    gt_fashion = brute_force_gt(X_fashion_train, X_fashion_test[:50])
    
    results = []
    
    # Benchmark 1: Recall and speed
    results.append(benchmark_recall_and_speed('SIFT', X_sift_train, X_sift_test, gt_sift))
    results.append(benchmark_recall_and_speed('FashionMNIST', X_fashion_train, X_fashion_test, gt_fashion))
    
    # Benchmark 2: Insert speed
    results.append(benchmark_insert_speed(
        'SIFT', 
        X_sift_train[:50000], 
        X_sift_train[50000:100000],
        X_sift_test,
        gt_sift
    ))
    
    # Benchmark 3: Centroid drift
    drift_results = benchmark_centroid_drift(X_sift_train, X_sift_test)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n| Metric | FAISS | Copenhagen |")
    print("|--------|-------|------------|")
    for r in results:
        if 'faiss_recall' in r:
            print(f"| {r['name']} Recall@10 | {r['faiss_recall']:.3f} | {r['cop_recall']:.3f} |")
            print(f"| {r['name']} Query (ms) | {r['faiss_query_ms']:.3f} | {r['cop_query_ms']:.3f} |")
        if 'insert_per_vector_ms' in r:
            print(f"| {r['name']} Insert (ms/vec) | {r['faiss_rebuild_s']*1000/len(X_sift_train):.4f} | {r['insert_per_vector_ms']:.4f} |")
    
    print("\nCentroid Drift (recall@10 after inserts, GT on full data):")
    for label, n, recall in drift_results:
        print(f"  {label:>10}: n={n:>6}, recall={recall:.4f}")

if __name__ == "__main__":
    main()
