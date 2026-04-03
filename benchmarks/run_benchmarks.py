#!/usr/bin/env python3
"""
Comprehensive benchmark suite for Copenhagen - Dynamic IVF Index
Compares against FAISS IVF with JSON output and plots.
"""

import sys
import os
import time
import json
from datetime import datetime
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from core import CopenhagenIndex

import numpy as np
import h5py
import faiss

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_mnist(limit=50000):
    """Load MNIST dataset."""
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'MNIST', 'mnist-784-euclidean.hdf5')
    with h5py.File(path, 'r') as f:
        train = f['train'][:limit].astype(np.float32)
        test = f['test'][:1000].astype(np.float32)
    return train, test


def benchmark_search_speed(datasets, nprobe_range):
    """Benchmark search speed vs FAISS."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "name": "search_speed_comparison",
        "data": []
    }
    
    for name, train, test in datasets:
        print(f"\n--- Dataset: {name} ---")
        
        d = train.shape[1]
        n = len(train)
        n_queries = min(200, len(test))
        queries = test[:n_queries]
        
        # Ground truth (subset for speed)
        print("  Computing ground truth...")
        gt_data = train[:10000] if len(train) > 10000 else train
        dists = np.sqrt(((queries[:, None] - gt_data[None]) ** 2).sum(axis=2))
        gt = np.argsort(dists, axis=1)[:, :10]
        
        for nlist in [64, 128, 256]:
            # FAISS
            print(f"  Building FAISS (nlist={nlist})...")
            t0 = time.time()
            quantizer = faiss.IndexFlatL2(d)
            faiss_idx = faiss.IndexIVFFlat(quantizer, d, nlist)
            faiss_idx.train(train)
            faiss_idx.add(train)
            faiss_idx.nprobe = 8
            faiss_build = time.time() - t0
            
            t0 = time.time()
            for q in queries:
                faiss_idx.search(q.reshape(1, -1), 10)
            faiss_time = (time.time() - t0) / n_queries * 1000
            faiss_qps = 1000 / faiss_time
            
            # Copenhagen
            print(f"  Building Copenhagen (nlist={nlist})...")
            t0 = time.time()
            cph_idx = CopenhagenIndex(dim=d, n_clusters=nlist, nprobe=8)
            cph_idx.add(train)
            cph_build = time.time() - t0
            
            t0 = time.time()
            for q in queries:
                cph_idx.search(q, 10)
            cph_time = (time.time() - t0) / n_queries * 1000
            cph_qps = 1000 / cph_time
            
            # Compute recalls
            faiss_recalls, cph_recalls = [], []
            for i in range(n_queries):
                faiss_r = faiss_idx.search(queries[i].reshape(1, -1), 10)
                cph_r = cph_idx.search(queries[i], 10)
                
                faiss_recalls.append(len(set(faiss_r[1][0]) & set(gt[i])) / 10)
                cph_recalls.append(len(set(cph_r[0]) & set(gt[i])) / 10)
            
            faiss_recall = np.mean(faiss_recalls)
            cph_recall = np.mean(cph_recalls)
            
            print(f"    FAISS:      {faiss_qps:>8.1f} QPS, {faiss_recall:.4f} recall, build={faiss_build:.2f}s")
            print(f"    Copenhagen: {cph_qps:>8.1f} QPS, {cph_recall:.4f} recall, build={cph_build:.2f}s")
            
            results["data"].append({
                "dataset": name,
                "nlist": nlist,
                "faiss": {
                    "qps": faiss_qps,
                    "recall": faiss_recall,
                    "build_time": faiss_build
                },
                "copenhagen": {
                    "qps": cph_qps,
                    "recall": cph_recall,
                    "build_time": cph_build
                }
            })
    
    return results


def benchmark_dynamic_updates(train, incremental, nlist=128):
    """Benchmark incremental insert and delete."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "name": "dynamic_updates",
        "data": {}
    }
    
    d = train.shape[1]
    n_inc = len(incremental)
    
    # FAISS incremental (rebuild)
    print("\n--- Dynamic Updates ---")
    
    print(f"  FAISS incremental (rebuild)...")
    t0 = time.time()
    quantizer = faiss.IndexFlatL2(d)
    faiss_idx = faiss.IndexIVFFlat(quantizer, d, nlist)
    faiss_idx.train(train)
    faiss_idx.add(train)
    faiss_idx.nprobe = 8
    initial_build = time.time() - t0
    
    t0 = time.time()
    combined = np.vstack([train, incremental])
    quantizer2 = faiss.IndexFlatL2(d)
    faiss_idx2 = faiss.IndexIVFFlat(quantizer2, d, nlist)
    faiss_idx2.train(combined)
    faiss_idx2.add(combined)
    faiss_idx2.nprobe = 8
    faiss_incremental = time.time() - t0
    print(f"    Initial: {initial_build:.3f}s, Incremental rebuild: {faiss_incremental:.3f}s")
    
    # Copenhagen incremental
    print(f"  Copenhagen incremental...")
    t0 = time.time()
    cph_idx = CopenhagenIndex(dim=d, n_clusters=nlist, nprobe=8)
    cph_idx.add(train)
    cph_initial = time.time() - t0
    
    t0 = time.time()
    cph_idx.add(incremental)
    cph_incremental = time.time() - t0
    print(f"    Initial: {cph_initial:.3f}s, Incremental add: {cph_incremental:.3f}s")
    
    # Delete benchmark
    n_delete = min(5000, len(train) // 10)
    delete_ids = np.arange(0, n_delete)
    
    print(f"  Delete {n_delete} vectors...")
    
    t0 = time.time()
    mask = np.ones(len(combined), dtype=bool)
    mask[delete_ids] = False
    remaining = combined[mask]
    quantizer3 = faiss.IndexFlatL2(d)
    faiss_idx3 = faiss.IndexIVFFlat(quantizer3, d, nlist)
    faiss_idx3.train(remaining)
    faiss_idx3.add(remaining)
    faiss_idx3.nprobe = 8
    faiss_delete = time.time() - t0
    
    t0 = time.time()
    cph_idx.delete(delete_ids)
    cph_delete = time.time() - t0
    
    print(f"    FAISS rebuild: {faiss_delete:.3f}s, Copenhagen delete: {cph_delete:.3f}s")
    
    results["data"] = {
        "initial_build": {"faiss": initial_build, "copenhagen": cph_initial},
        "incremental_insert": {"faiss": faiss_incremental, "copenhagen": cph_incremental, "speedup": faiss_incremental / cph_incremental},
        "delete": {"faiss": faiss_delete, "copenhagen": cph_delete, "speedup": faiss_delete / cph_delete}
    }
    
    return results


def benchmark_nprobe_recall(train, test, nlist_range=[32, 64, 128, 256]):
    """Benchmark recall vs nprobe."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "name": "nprobe_recall",
        "data": {}
    }
    
    d = train.shape[1]
    n_queries = min(100, len(test))
    queries = test[:n_queries]
    
    print("\n--- Recall vs nprobe ---")
    
    # Ground truth
    gt_data = train[:10000] if len(train) > 10000 else train
    dists = np.sqrt(((queries[:, None] - gt_data[None]) ** 2).sum(axis=2))
    gt = np.argsort(dists, axis=1)[:, :10]
    
    for nlist in nlist_range:
        print(f"  nlist={nlist}...")
        
        # FAISS
        quantizer = faiss.IndexFlatL2(d)
        faiss_idx = faiss.IndexIVFFlat(quantizer, d, nlist)
        faiss_idx.train(train)
        faiss_idx.add(train)
        
        # Copenhagen
        cph_idx = CopenhagenIndex(dim=d, n_clusters=nlist, nprobe=1)
        cph_idx.add(train)
        
        faiss_recalls = {1: [], 4: [], 8: [], 16: []}
        cph_recalls = {1: [], 4: [], 8: [], 16: []}
        
        for np_ in [1, 4, 8, 16]:
            faiss_idx.nprobe = np_
            
            for i in range(n_queries):
                faiss_r = faiss_idx.search(queries[i].reshape(1, -1), 10)
                cph_r = cph_idx.search(queries[i], 10, n_probes=np_)
                
                faiss_recalls[np_].append(len(set(faiss_r[1][0]) & set(gt[i])) / 10)
                cph_recalls[np_].append(len(set(cph_r[0]) & set(gt[i])) / 10)
        
        results["data"][f"nlist_{nlist}"] = {
            "faiss": {k: float(np.mean(v)) for k, v in faiss_recalls.items()},
            "copenhagen": {k: float(np.mean(v)) for k, v in cph_recalls.items()}
        }
        
        print(f"    FAISS:      R@10 = {[f'{np.mean(faiss_recalls[np_]):.3f}' for np_ in [1,4,8,16]]}")
        print(f"    Copenhagen: R@10 = {[f'{np.mean(cph_recalls[np_]):.3f}' for np_ in [1,4,8,16]]}")
    
    return results


def create_plots(results_dir):
    """Create matplotlib plots from benchmark results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Find result files
        import glob
        result_files = glob.glob(os.path.join(results_dir, '*.json'))
        
        for rf in result_files:
            with open(rf, 'r') as f:
                data = json.load(f)
            
            if data.get('name') == 'search_speed_comparison':
                create_search_plot(data, rf)
            elif data.get('name') == 'nprobe_recall':
                create_recall_plot(data, rf)
            elif data.get('name') == 'dynamic_updates':
                create_update_plot(data, rf)
        
        print(f"\nPlots saved to: {FIGURES_DIR}")
        
    except ImportError:
        print("\nmatplotlib not available, skipping plots")


def create_search_plot(data, source_file):
    """Create search speed comparison plot."""
    import matplotlib.pyplot as plt
    
    datasets = {}
    for entry in data['data']:
        name = entry['dataset']
        if name not in datasets:
            datasets[name] = {'nlist': [], 'faiss_qps': [], 'cph_qps': [], 'faiss_recall': [], 'cph_recall': []}
        datasets[name]['nlist'].append(entry['nlist'])
        datasets[name]['faiss_qps'].append(entry['faiss']['qps'])
        datasets[name]['cph_qps'].append(entry['copenhagen']['qps'])
        datasets[name]['faiss_recall'].append(entry['faiss']['recall'])
        datasets[name]['cph_recall'].append(entry['copenhagen']['recall'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, d in datasets.items():
        axes[0].plot(d['nlist'], d['faiss_qps'], 'o-', label=f'{name} FAISS')
        axes[0].plot(d['nlist'], d['cph_qps'], 's--', label=f'{name} Copenhagen')
        axes[1].plot(d['nlist'], d['faiss_recall'], 'o-', label=f'{name} FAISS')
        axes[1].plot(d['nlist'], d['cph_recall'], 's--', label=f'{name} Copenhagen')
    
    axes[0].set_xlabel('Number of clusters')
    axes[0].set_ylabel('QPS')
    axes[0].set_title('Search Speed Comparison')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Number of clusters')
    axes[1].set_ylabel('Recall@10')
    axes[1].set_title('Recall Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, 'search_speed_comparison.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Saved: {out_file}")


def create_recall_plot(data, source_file):
    """Create recall vs nprobe plot."""
    import matplotlib.pyplot as plt
    
    nlist_keys = sorted(data['data'].keys(), key=lambda x: int(x.split('_')[1]))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D']
    
    for i, nlist_key in enumerate(nlist_keys):
        nlist_data = data['data'][nlist_key]
        nlist = nlist_key.split('_')[1]
        
        np_values = sorted(nlist_data['faiss'].keys())
        
        faiss_recalls = [nlist_data['faiss'][str(np_)] for np_ in np_values]
        cph_recalls = [nlist_data['copenhagen'][str(np_)] for np_ in np_values]
        
        ax.plot(np_values, faiss_recalls, f'{markers[i % len(markers)]}-', 
                color=colors[i % len(colors)], label=f'FAISS nlist={nlist}')
        ax.plot(np_values, cph_recalls, f'{markers[i % len(markers)]}--', 
                color=colors[i % len(colors)], alpha=0.5, label=f'Copenhagen nlist={nlist}')
    
    ax.set_xlabel('nprobe')
    ax.set_ylabel('Recall@10')
    ax.set_title('Recall vs nprobe')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, 'recall_vs_nprobe.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Saved: {out_file}")


def create_update_plot(data, source_file):
    """Create dynamic update comparison plot."""
    import matplotlib.pyplot as plt
    
    update_data = data['data']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    labels = ['FAISS', 'Copenhagen']
    
    # Incremental insert
    ax = axes[0]
    inc_faiss = update_data['incremental_insert']['faiss']
    inc_cph = update_data['incremental_insert']['copenhagen']
    speedup = update_data['incremental_insert']['speedup']
    
    bars = ax.bar(labels, [inc_faiss, inc_cph], color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Incremental Insert\n(speedup: {speedup:.1f}x)')
    for bar, val in zip(bars, [inc_faiss, inc_cph]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}s',
                ha='center', va='bottom')
    
    # Delete
    ax = axes[1]
    del_faiss = update_data['delete']['faiss']
    del_cph = update_data['delete']['copenhagen']
    speedup = update_data['delete']['speedup']
    
    bars = ax.bar(labels, [del_faiss, del_cph], color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Delete 5000 vectors\n(speedup: {speedup:.1f}x)')
    for bar, val in zip(bars, [del_faiss, del_cph]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}s',
                ha='center', va='bottom')
    
    # Initial build
    ax = axes[2]
    init_faiss = update_data['initial_build']['faiss']
    init_cph = update_data['initial_build']['copenhagen']
    
    bars = ax.bar(labels, [init_faiss, init_cph], color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Time (s)')
    ax.set_title('Initial Build Time')
    for bar, val in zip(bars, [init_faiss, init_cph]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, 'dynamic_updates.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Saved: {out_file}")


def run_benchmarks(args):
    """Run all benchmarks."""
    print("=" * 60)
    print("COPENHAGEN BENCHMARK SUITE")
    print("=" * 60)
    
    # Load data
    print("\nLoading datasets...")
    datasets = []
    
    if args.mnist or args.all:
        try:
            print("  Loading MNIST...")
            mnist_train, mnist_test = load_mnist(limit=50000)
            datasets.append(("MNIST", mnist_train, mnist_test))
            print(f"    MNIST: {mnist_train.shape}")
        except Exception as e:
            print(f"    Failed to load MNIST: {e}")
    
    if args.synthetic or args.all:
        print("  Generating synthetic data...")
        synth_train = np.random.randn(20000, 128).astype(np.float32)
        synth_test = np.random.randn(1000, 128).astype(np.float32)
        datasets.append(("Synthetic", synth_train, synth_test))
        print(f"    Synthetic: {synth_train.shape}")
    
    if not datasets:
        print("No datasets loaded. Use --all or --mnist --synthetic")
        return
    
    # Run benchmarks
    if args.search or args.all:
        print("\n" + "=" * 60)
        print("BENCHMARK: Search Speed")
        print("=" * 60)
        results = benchmark_search_speed(datasets, [64, 128, 256])
        out_file = os.path.join(RESULTS_DIR, f'search_speed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_file}")
    
    if args.nprobe or args.all:
        print("\n" + "=" * 60)
        print("BENCHMARK: Recall vs nprobe")
        print("=" * 60)
        for name, train, test in datasets:
            results = benchmark_nprobe_recall(train, test)
            out_file = os.path.join(RESULTS_DIR, f'nprobe_recall_{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {out_file}")
    
    if args.dynamic or args.all:
        if datasets:
            name, train, _ = datasets[0]
            try:
                incremental = np.random.randn(10000, train.shape[1]).astype(np.float32)
                results = benchmark_dynamic_updates(train, incremental)
                out_file = os.path.join(RESULTS_DIR, f'dynamic_updates_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                with open(out_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {out_file}")
            except Exception as e:
                print(f"Dynamic benchmark failed: {e}")
    
    # Create plots
    if args.plots:
        print("\n" + "=" * 60)
        print("Generating plots...")
        print("=" * 60)
        create_plots(RESULTS_DIR)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copenhagen Benchmark Suite')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--search', action='store_true', help='Search speed benchmark')
    parser.add_argument('--nprobe', action='store_true', help='Recall vs nprobe benchmark')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic updates benchmark')
    parser.add_argument('--plots', action='store_true', help='Generate plots from results')
    parser.add_argument('--mnist', action='store_true', help='Use MNIST dataset')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    
    args = parser.parse_args()
    
    if not any([args.all, args.search, args.nprobe, args.dynamic, args.plots]):
        args.all = True
        args.mnist = True
        args.synthetic = True
        args.plots = True
    
    run_benchmarks(args)
