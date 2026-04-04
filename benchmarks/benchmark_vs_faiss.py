"""benchmark_vs_faiss.py — DynamicIVF vs FAISS: recall@1/10/100 / QPS across index variants.

Datasets
  gauss   : 10 000 iid N(0,1) vectors, d=128  (synthetic)
  mnist   : data/MNIST/mnist-784-euclidean.hdf5
  fashion : data/fashion-mnist/fashion-mnist-784-euclidean.hdf5
  sift    : data/sift/sift-128-euclidean.hdf5

Usage
  python benchmark_vs_faiss.py                      # gauss only (default)
  python benchmark_vs_faiss.py gauss mnist fashion
  python benchmark_vs_faiss.py all
"""

import argparse
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
    "Copenhagen": dict(color="#1f77b4", marker="o", ls="-", lw=2),
}


_FAISS_RAM_LIMIT = 1.5e9


def _try_build_faiss(data):
    """Build FAISS Flat + IVF indexes; return (flat, ivf, nlist) or None on OOM."""
    n, d = data.shape
    est_ram = n * d * 4
    if est_ram > _FAISS_RAM_LIMIT:
        gb = est_ram / 1e9
        print(f"  [skip FAISS] estimated RAM {gb:.1f} GB > {_FAISS_RAM_LIMIT/1e9:.1f} GB limit")
        return None

    try:
        flat = faiss.IndexFlatL2(d)
        flat.add(data)

        nlist = max(16, int(np.sqrt(n)))
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_L2)
        ivf.train(data)
        ivf.add(data)
        return flat, ivf, nlist
    except (MemoryError, RuntimeError) as exc:
        print(f"  [skip FAISS] OOM during build: {exc}")
        return None


def run(dataset_name, data, queries, gt, metric='l2'):
    n, d = data.shape
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {dataset_name}  --  n={n:,}  d={d}  queries={len(queries)}  k={K}")
    print(sep)

    faiss_result = _try_build_faiss(data)

    faiss_configs = []
    if faiss_result is not None:
        flat, ivf, nlist = faiss_result
        faiss_configs.append((
            "Flat L2",
            lambda q: flat.search(q[None], K_MAX)[1][0],
        ))
        for nprobe in [1, 5, 10, 25, 50]:
            if nprobe > nlist:
                continue
            ivf.nprobe = nprobe
            label = f"IVF nprobe={nprobe}"
            faiss_configs.append((
                label,
                lambda q, p=nprobe: (_set_nprobe(ivf, p) or flat.search(q[None], K_MAX)[1][0]),
            ))

    def make_flat_query(flat_idx):
        def fn(q):
            return [(i, 0.0) for i in flat_idx.search(q[None], K_MAX)[1][0]]
        return fn

    dynamic_configs = build_dynamic_ivf_configs(data, queries, gt)

    exact_labels = ("Flat L2",) if faiss_result is not None else ()
    print_table_header()

    all_configs = []
    if faiss_result is not None:
        flat, ivf, nlist = faiss_result
        all_configs.append(("Flat L2", make_flat_query(flat)))
        for nprobe in [1, 5, 10, 25]:
            if nprobe > nlist:
                continue
            ivf.nprobe = nprobe
            label = f"IVF nprobe={nprobe}"
            all_configs.append((label, make_ivf_query(ivf)))

    all_configs.extend(dynamic_configs)

    return run_evaluation(
        all_configs,
        queries, gt, data, n,
        exact_labels=exact_labels,
    )


def _set_nprobe(index, nprobe):
    index.nprobe = nprobe


def make_ivf_query(ivf_index):
    def fn(q):
        return [(i, 0.0) for i in ivf_index.search(q[None], K_MAX)[1][0]]
    return fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="*", default=["gauss"],
                    help="gauss mnist fashion sift all  (default: gauss)")
    ap.add_argument("--force", action="store_true",
                    help="re-download dataset files even if they already exist")
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
            data, queries, gt = load_hdf5(path)
            all_results.append(("SIFT 1M  d=128",
                                 run("SIFT 1M  d=128", data, queries, gt)))

    print()
    save_figures(all_results, family_style=_FAMILY_STYLE, suffix="_vs_faiss")
