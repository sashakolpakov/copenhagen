"""benchmark_vs_hnsw.py — Copenhagen vs HNSW: recall@10 / QPS tradeoff.

Copenhagen is tuned via (n_clusters, nprobe, soft_k).
HNSW is tuned via (M, efSearch).

Both are evaluated on the same dataset and queries; results are printed
as a recall@10 / QPS table and saved as a Pareto-frontier figure.

Dataset: SIFT-128 (default — real L2 vectors, large enough to stress HNSW graph).

Usage
  python benchmark_vs_hnsw.py           # SIFT-128 (default)
  python benchmark_vs_hnsw.py gauss     # 10k Gaussian d=128
  python benchmark_vs_hnsw.py mnist
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import faiss

from _bench_common import (
    K, K_MAX, DATA_DIR,
    make_gaussian, load_hdf5,
    print_table_header, run_evaluation,
    build_dynamic_ivf_configs,
    FAMILY_STYLE_BASE, save_figures,
    ensure_datasets,
)

_FAMILY_STYLE = {
    **FAMILY_STYLE_BASE,
    "HNSW": dict(color="#d62728", marker="s", ls="-", lw=2, s=49),
    "Copenhagen": dict(color="#1f77b4", marker="o", ls="-", lw=2, s=49),
}

# HNSW variants: (M, [efSearch values])
_HNSW_CONFIGS = [
    (16,  [16, 32, 64, 128]),
    (32,  [16, 32, 64, 128, 256]),
    (64,  [32, 64, 128]),
]


def _build_hnsw(data, M, ef_construction=200):
    n, d = data.shape
    idx = faiss.IndexHNSWFlat(d, M)
    idx.hnsw.efConstruction = ef_construction
    idx.add(data)
    return idx


def make_hnsw_query(idx, ef_search):
    def fn(q):
        idx.hnsw.efSearch = ef_search
        dists, ids = idx.search(q[None], K_MAX)
        return (ids[0], dists[0])
    return fn


def run(dataset_name, data, queries, gt):
    n, d = data.shape
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {dataset_name}  --  n={n:,}  d={d}  queries={len(queries)}  k={K}")
    print(sep)

    # Build HNSW indexes
    hnsw_configs = []
    for M, ef_list in _HNSW_CONFIGS:
        print(f"  Building HNSW M={M}...", end=" ", flush=True)
        t0 = time.perf_counter()
        hnsw_idx = _build_hnsw(data, M)
        print(f"{time.perf_counter() - t0:.2f}s")
        for ef in ef_list:
            label = f"HNSW M={M} ef={ef}"
            hnsw_configs.append((label, make_hnsw_query(hnsw_idx, ef)))

    # Build Copenhagen indexes
    cph_configs = build_dynamic_ivf_configs(data, queries, gt)

    all_configs = hnsw_configs + cph_configs

    print_table_header()
    return run_evaluation(all_configs, queries, gt, data, n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="*", default=["sift"],
                    help="gauss mnist fashion sift  (default: sift)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    targets = set(args.dataset)
    if "all" in targets:
        targets = {"gauss", "mnist", "fashion", "sift"}

    ensure_datasets(targets, force=args.force)
    all_results = []

    if "gauss" in targets:
        data, queries, gt = make_gaussian(n=10_000, d=128)
        all_results.append(("Gaussian 10k  d=128",
                             run("Gaussian 10k  d=128", data, queries, gt)))

    if "mnist" in targets:
        path = DATA_DIR / "MNIST/mnist-784-euclidean.hdf5"
        if path.exists():
            data, queries, gt = load_hdf5(path)
            all_results.append(("MNIST 60k  d=784",
                                 run("MNIST 60k  d=784", data, queries, gt)))

    if "fashion" in targets:
        path = DATA_DIR / "fashion-mnist/fashion-mnist-784-euclidean.hdf5"
        if path.exists():
            data, queries, gt = load_hdf5(path)
            all_results.append(("Fashion-MNIST 60k  d=784",
                                 run("Fashion-MNIST 60k  d=784", data, queries, gt)))

    if "sift" in targets:
        path = DATA_DIR / "sift/sift-128-euclidean.hdf5"
        if path.exists():
            data, queries, gt = load_hdf5(path, n_train=100_000)
            all_results.append(("SIFT 100k  d=128",
                                 run("SIFT 100k  d=128", data, queries, gt)))

    print()
    save_figures(all_results, family_style=_FAMILY_STYLE, suffix="_vs_hnsw")
