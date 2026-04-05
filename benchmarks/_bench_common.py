"""_bench_common.py — shared constants, dataset loaders, and evaluation helpers.

Imported by benchmark_vs_faiss.py.
"""

import sys
import time
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from core import CopenhagenIndex as DynamicIVFIndex


# ── constants ─────────────────────────────────────────────────────────────────

K             = 10
K_MAX         = 100
N_QUERIES     = 200
WARMUP        = 10
MAX_CAND_FRAC = 0.25
CAND_SAMPLE   = 10
QUICK_SAMPLE  = 10
PARETO_TOL    = 0.03

_REPO_ROOT    = Path(__file__).parent.parent
DATA_DIR      = _REPO_ROOT / "data"
FIGURES_DIR   = _REPO_ROOT / "figures"


# ── datasets ──────────────────────────────────────────────────────────────────

def _sort_gt(data, queries, gt):
    """Sort each gt row by ascending L2 distance."""
    sorted_gt = np.empty_like(gt)
    for i, (q, row) in enumerate(zip(queries, gt)):
        diffs = data[row] - q
        dists = np.einsum('ij,ij->i', diffs, diffs)
        sorted_gt[i] = row[np.argsort(dists)]
    return sorted_gt


def _brute_knn(data, queries, k):
    """Exact k-NN via numpy."""
    q_sq = np.sum(queries ** 2, axis=1)[:, None]
    d_sq = np.sum(data ** 2, axis=1)[None, :]
    d2 = q_sq + d_sq - 2.0 * (queries @ data.T)
    return np.argsort(d2, axis=1)[:, :k].astype(np.int32)


def make_gaussian(n=10_000, d=128, seed=42):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype(np.float32)
    qs = rng.standard_normal((N_QUERIES, d)).astype(np.float32)
    gt = _brute_knn(data, qs, K_MAX)
    return data, qs, _sort_gt(data, qs, gt).astype(np.int32)


def load_hdf5(path, n_train=None, normalize=False, mmap_dir=None):
    """Load an ANN-benchmark HDF5 file and return (data, queries, gt)."""
    import os
    with h5py.File(path) as f:
        n_total = f["train"].shape[0]
        d = f["train"].shape[1]
        n = n_total if n_train is None else min(n_train, n_total)
        queries = f["test"][:N_QUERIES].astype(np.float32)
        if "neighbors" in f and n >= n_total:
            gt = f["neighbors"][:N_QUERIES, :K_MAX].astype(np.int32)
            precomputed_gt = True
        else:
            precomputed_gt = False

        if mmap_dir is not None:
            os.makedirs(mmap_dir, exist_ok=True)
            fpath = os.path.join(mmap_dir, "_train_data.dat")
            data = np.memmap(fpath, mode="w+", dtype="float32", shape=(n, d))
            chunk = 50_000
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                data[start:end] = f["train"][start:end].astype(np.float32)
            data.flush()
        else:
            data = f["train"][:n].astype(np.float32)

    if normalize:
        data = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-10)
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)

    if not precomputed_gt:
        gt = _brute_knn(data, queries, K_MAX)
        gt = _sort_gt(data, queries, gt).astype(np.int32)
    return data, queries, gt


# ── evaluation ────────────────────────────────────────────────────────────────

def recall(gt, approx, k=K):
    """Recall@k: fraction of true top-k neighbours found."""
    hits = sum(
        len(set(g[:k].tolist()) & set(a[:k].tolist()))
        for g, a in zip(gt, approx)
    )
    return hits / (len(gt) * k)


def avg_cands(query_fn, queries):
    return int(np.mean([
        len(query_fn(q)) if isinstance(query_fn(q), list) else 1000
        for q in queries[:CAND_SAMPLE]
    ]))


def approx_ratio(data, queries, gt_indices, found_indices):
    """Mean ratio: avg distance to found / avg distance to true."""
    ratios = []
    for q, true_idx, found_idx in zip(queries, gt_indices, found_indices):
        true_d = np.sqrt(np.sum((data[true_idx] - q) ** 2, axis=1)).mean()
        found_d = np.sqrt(np.sum((data[found_idx[:K]] - q) ** 2, axis=1)).mean()
        ratios.append(found_d / (true_d + 1e-10))
    return float(np.mean(ratios))


def _pareto_group(label):
    """Broad family group for Pareto pruning."""
    parts = label.split()
    return parts[0]


def evaluate(label, query_fn, queries, gt, data, n, pareto=None, exact_labels=()):
    """Run one benchmark configuration and return result dict."""
    is_exact = label in exact_labels

    for q in queries[:WARMUP]:
        query_fn(q)

    t0 = time.perf_counter()
    results = [query_fn(q) for q in queries]
    ms = (time.perf_counter() - t0) / len(queries) * 1e3

    raw = [np.array(res[0], dtype=np.int64) for res in results]
    padded = np.array([np.pad(ix[:K_MAX], (0, max(0, K_MAX - len(ix))), constant_values=-1) for ix in raw])

    rec1 = recall(gt, padded, 1)
    rec10 = recall(gt, padded, K)
    rec100 = recall(gt, padded, K_MAX)
    ratio = approx_ratio(data, queries, gt[:, :K], padded)

    cands = avg_cands(query_fn, queries)
    r = dict(label=label, recall=rec10, recall1=rec1, recall100=rec100,
             ratio=ratio, ms=ms, qps=1e3 / ms, cands=cands)

    cands_str = f"{cands:>7,}" if cands < n else f"{'n':>7}"
    print(f"  {label:<38}  {rec1:>6.3f}  {rec10:>6.3f}  {rec100:>6.3f}"
          f"  {ratio:>10.4f}  {1e3/ms:>8.1f}  {ms:>7.3f}  {cands_str}",
          flush=True)
    return r


def print_table_header():
    hdr = (f"  {'Method':<38}  {'R@1':>6}  {'R@10':>6}  {'R@100':>6}"
           f"  {'dist ratio':>10}  {'QPS':>8}  {'ms/q':>7}  {'cands':>7}")
    print()
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))


def run_evaluation(configs, queries, gt, data, n, exact_labels=()):
    """Evaluate all configs and return result rows."""
    rows = []
    pareto = {}
    for label, fn in configs:
        r = evaluate(label, fn, queries, gt, data, n, pareto=pareto, exact_labels=exact_labels)
        if r is not None:
            rows.append(r)
    return rows


# ── figures ───────────────────────────────────────────────────────────────────

FAMILY_STYLE_BASE = {
    "Flat L2":  dict(color="black", marker="*", ls="none", s=120, zorder=5),
    "Copenhagen": dict(color="#1f77b4", marker="o", ls="-", lw=2, s=49),
}


def _family(label):
    parts = label.split()
    return parts[0]


def _pareto_frontier(xs, ys):
    """Return indices of Pareto-optimal points."""
    pts = sorted(enumerate(zip(xs, ys)), key=lambda t: t[1][0])
    frontier = []
    best_y = -np.inf
    for idx, (x, y) in pts:
        if y > best_y:
            best_y = y
            frontier.append(idx)
    return frontier


def save_figures(all_results, family_style=None, suffix=""):
    """Save PNG per dataset into FIGURES_DIR."""
    if family_style is None:
        family_style = FAMILY_STYLE_BASE
    FIGURES_DIR.mkdir(exist_ok=True)

    for dataset_name, rows in all_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_r1, ax_r10, ax_qps, ax_ratio = axes.flat

        for r in rows:
            fam = _family(r["label"])
            style = family_style.get(fam, dict(color="gray", marker="o", ls="-"))
            xs = [r["cands"]]
            for ax, yval, ylabel in [
                (ax_r1, r["recall1"], "Recall@1"),
                (ax_r10, r["recall"], "Recall@10"),
                (ax_qps, r["qps"], "QPS"),
            ]:
                ax.scatter(xs, [yval], label=r["label"], **style)

        for ax, ylabel, title in [
            (ax_r1, "Recall@1", "Recall@1 vs Candidates"),
            (ax_r10, "Recall@10", "Recall@10"),
            (ax_qps, "QPS", "QPS"),
        ]:
            ax.set_xscale("log")
            ax.set_xlabel("Candidates")
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_title(title)

        ax_ratio.set_xlabel("Recall@10")
        ax_ratio.set_ylabel("Dist ratio")
        ax_ratio.axhline(1.0, color="black", lw=0.8, ls=":")
        ax_ratio.grid(True, alpha=0.3)

        fig.tight_layout()
        slug = dataset_name.split()[0].lower()
        out = FIGURES_DIR / f"{slug}{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> saved {out}")


# ── index builder ─────────────────────────────────────────────────────────────

def build_dynamic_ivf_configs(data, queries, gt):
    """Build DynamicIVF index configs with different parameters."""
    n, d = data.shape
    configs = []

    n_clusters_options = [16, 32, 64, 128]
    nprobe_options = [1, 4, 8]

    for nc in n_clusters_options:
        for np_ in nprobe_options:
            if nc > n:
                continue

            print(f"  Building Copenhagen n_clusters={nc} nprobe={np_}...", end=" ", flush=True)
            t0 = time.perf_counter()
            idx = DynamicIVFIndex(dim=d, n_clusters=nc, nprobe=np_)
            idx.add(data)
            print(f"{time.perf_counter() - t0:.2f}s")

            def make_query_fn(index, nc, np_):
                def fn(q):
                    return index.search(q, k=K_MAX, n_probes=np_)
                return fn

            label = f"Copenhagen n_clusters={nc} nprobe={np_}"
            configs.append((label, make_query_fn(idx, nc, np_)))

    return configs


from download_data import ensure_datasets, DATASETS  # noqa: F401  re-exported for callers
