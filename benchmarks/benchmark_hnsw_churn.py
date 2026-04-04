"""benchmark_hnsw_churn.py — Copenhagen vs FAISS HNSW under streaming churn.

Scenario
--------
Start with N=10 000 vectors (d=128).  Run ROUNDS rounds, each:
  1. Insert BATCH_INSERT new vectors.
  2. Delete the oldest DELETE_FRAC fraction of the current live corpus.
  3. Measure recall@10 on 200 random queries against the live ground truth.

Three strategies
----------------
Copenhagen          O(1) tombstone delete; lazy compact; splits handle drift.
HNSW + filter       Delete = tombstone mask; search post-filters dead IDs.
                    No graph rebuild → recall degrades as deleted nodes clog
                    traversal paths.
HNSW + rebuild      Full IndexHNSWFlat rebuild from live vectors every round.
                    Recall stays stable; cost is O(n log n) per round.

Outputs
-------
  Console table: recall@10, insert throughput, delete throughput, search QPS
  figures/hnsw_churn.png: recall@10 and QPS over rounds

Usage
-----
  python benchmarks/benchmark_hnsw_churn.py
  python benchmarks/benchmark_hnsw_churn.py --rounds 20 --n 20000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from core import CopenhagenIndex

# ── defaults ──────────────────────────────────────────────────────────────────
N_INIT        = 50_000
D             = 128
ROUNDS        = 10
BATCH_INSERT  = 1_000
DELETE_FRAC   = 0.30
N_QUERIES     = 200
K             = 10
HNSW_M        = 32      # HNSW graph degree
HNSW_EF_BUILD = 64
HNSW_EF_SEARCH = 64
CPH_N_CLUSTERS      = 64
CPH_NPROBE          = 32
CPH_SOFT_K          = 2
CPH_SPLIT_THRESHOLD = 3.0   # split a cluster when it grows to 3× the mean size

FIGURES_DIR = Path(__file__).parent.parent / "figures"


# ── ground truth ──────────────────────────────────────────────────────────────

def brute_knn(data, queries, k):
    """Exact k-NN via numpy (no FAISS dependency for GT)."""
    q2 = np.sum(queries ** 2, axis=1)[:, None]
    d2 = np.sum(data    ** 2, axis=1)[None, :]
    cross = queries @ data.T
    dists = q2 + d2 - 2.0 * cross
    return np.argsort(dists, axis=1)[:, :k].astype(np.int32)


def recall_at_k(gt_ids, found_ids, k=K):
    hits = sum(
        len(set(g[:k].tolist()) & set(list(f)[:k]))
        for g, f in zip(gt_ids, found_ids)
    )
    return hits / (len(gt_ids) * k)


# ── Copenhagen wrapper ────────────────────────────────────────────────────────

class CopenhagenRunner:
    def __init__(self, d, n_clusters, nprobe, soft_k, split_threshold=CPH_SPLIT_THRESHOLD):
        self.idx = CopenhagenIndex(dim=d, n_clusters=n_clusters,
                                   nprobe=nprobe, soft_k=soft_k)
        # Enable adaptive cluster splitting so centroids track the live corpus.
        self.idx._index.split_threshold = split_threshold
        self.d   = d
        self._nprobe_frac = nprobe / n_clusters  # maintain search ratio as splits add clusters
        self._live_ids = set()

    def add(self, vecs):
        """Insert and return the new global IDs assigned."""
        start_id = self.idx.n_vectors
        self.idx.add(vecs)
        new_ids = list(range(start_id, self.idx.n_vectors))
        self._live_ids.update(new_ids)
        return new_ids

    def delete(self, ids):
        for gid in ids:
            self.idx.delete(gid)
            self._live_ids.discard(gid)
        # Compact immediately so dead vectors don't clog cluster scans during search.
        self.idx.compact()

    def search(self, q, k=K):
        # Scale nprobe with the current cluster count so the search-ratio stays constant
        # even after splits have added new centroids.
        n_clusters = self.idx._index.get_stats()["n_clusters"]
        nprobe = max(1, round(n_clusters * self._nprobe_frac))
        ids, _ = self.idx.search(q, k=k, n_probes=nprobe)
        return ids

    def n_live(self):
        return len(self._live_ids)


# ── HNSW + filter (tombstone mask, no rebuild) ────────────────────────────────

class HNSWFilterRunner:
    """Wraps IndexHNSWFlat; deletes are tombstones filtered at query time.

    HNSW graph traversal still visits deleted nodes internally — the filter
    only removes them from the returned result list.  When many nodes are
    deleted the graph cannot re-route, so recall degrades.
    """

    def __init__(self, d, M, ef_construction, ef_search):
        self._d          = d
        self._M          = M
        self._ef_c       = ef_construction
        self._ef_s       = ef_search
        self._index      = self._new_hnsw(d, M, ef_construction, ef_search)
        self._deleted    = set()
        self._next_id    = 0
        self._id_map     = {}   # internal_pos → global_id

    @staticmethod
    def _new_hnsw(d, M, ef_c, ef_s):
        idx = faiss.IndexHNSWFlat(d, M)
        idx.hnsw.efConstruction = ef_c
        idx.hnsw.efSearch       = ef_s
        return idx

    def add(self, vecs):
        n = len(vecs)
        new_ids = list(range(self._next_id, self._next_id + n))
        self._index.add(vecs)
        for i, gid in enumerate(new_ids):
            self._id_map[self._next_id + i] = gid
        self._next_id += n
        return new_ids

    def delete(self, ids):
        self._deleted.update(ids)

    def search(self, q, k=K):
        # Over-fetch to compensate for filtered-out deleted vectors.
        # Cap at 10× k to avoid QPS collapse at high churn obscuring graph quality.
        fetch = min(k * 10, self._index.ntotal)
        fetch = max(fetch, k)
        _, I  = self._index.search(q[None], fetch)
        live  = [self._id_map.get(int(i), -1) for i in I[0]
                 if i >= 0 and self._id_map.get(int(i), -1) not in self._deleted]
        return np.array(live[:k], dtype=np.int32)

    def n_live(self):
        return self._index.ntotal - len(self._deleted)


# ── HNSW + rebuild (rebuild from live vectors every round) ────────────────────

class HNSWRebuildRunner:
    """Full IndexHNSWFlat rebuild from scratch after every delete batch."""

    def __init__(self, d, M, ef_construction, ef_search):
        self._d       = d
        self._M       = M
        self._ef_c    = ef_construction
        self._ef_s    = ef_search
        self._live    = {}   # global_id → vector
        self._next_id = 0
        self._index   = None
        self._id_list = []   # index position → global_id

    def add(self, vecs):
        new_ids = list(range(self._next_id, self._next_id + len(vecs)))
        for gid, v in zip(new_ids, vecs):
            self._live[gid] = v
        self._next_id += len(vecs)
        return new_ids

    def delete(self, ids):
        for gid in ids:
            self._live.pop(gid, None)

    def rebuild(self):
        """Rebuild HNSW from all live vectors."""
        idx = faiss.IndexHNSWFlat(self._d, self._M)
        idx.hnsw.efConstruction = self._ef_c
        idx.hnsw.efSearch       = self._ef_s
        if self._live:
            self._id_list = list(self._live.keys())
            mat = np.stack([self._live[i] for i in self._id_list]).astype(np.float32)
            idx.add(mat)
        else:
            self._id_list = []
        self._index = idx

    def search(self, q, k=K):
        if self._index is None or self._index.ntotal == 0:
            return np.array([], dtype=np.int32)
        _, I = self._index.search(q[None], k)
        return np.array([self._id_list[i] for i in I[0] if i >= 0], dtype=np.int32)

    def n_live(self):
        return len(self._live)


# ── simulation ────────────────────────────────────────────────────────────────

def run_simulation(rng, n_init, d, rounds, batch_insert, delete_frac, n_queries, k):
    print(f"\nn_init={n_init:,}  d={d}  rounds={rounds}"
          f"  batch_insert={batch_insert:,}  delete_frac={delete_frac:.0%}"
          f"  k={k}\n")

    # Initial data
    data_init = rng.standard_normal((n_init, d)).astype(np.float32)
    queries   = rng.standard_normal((n_queries, d)).astype(np.float32)

    # Build all three runners
    cph   = CopenhagenRunner(d, CPH_N_CLUSTERS, CPH_NPROBE, CPH_SOFT_K)
    hnsw_f = HNSWFilterRunner(d, HNSW_M, HNSW_EF_BUILD, HNSW_EF_SEARCH)
    hnsw_r = HNSWRebuildRunner(d, HNSW_M, HNSW_EF_BUILD, HNSW_EF_SEARCH)

    print("Building initial indexes …")
    t0 = time.perf_counter()
    init_ids_cph   = cph.add(data_init)
    init_ids_hnsw_f = hnsw_f.add(data_init)
    init_ids_hnsw_r = hnsw_r.add(data_init)
    hnsw_r.rebuild()
    t_build = time.perf_counter() - t0
    print(f"  build time: {t_build:.2f}s")

    live_ids_cph    = list(init_ids_cph)
    live_ids_hnsw_f = list(init_ids_hnsw_f)
    live_ids_hnsw_r = list(init_ids_hnsw_r)

    # All vectors ever inserted (for ground-truth lookups)
    all_vecs = list(data_init)

    header = (f"\n{'Round':>6}  {'Live':>6}  {'Del%':>5}  "
              f"{'CPH R@10':>9}  {'HNSW+filter R@10':>17}  {'HNSW+rebuild R@10':>18}  "
              f"{'CPH ins/s':>10}  {'HNSW-r ins/s':>13}  "
              f"{'CPH del/s':>10}  "
              f"{'CPH QPS':>8}  {'HNSW-f QPS':>11}  {'HNSW-r QPS':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    history = []
    total_deleted = 0

    for rnd in range(rounds + 1):
        # ── Ground truth: exact NN over live vectors in Copenhagen ──
        if live_ids_cph:
            live_mat = np.stack([all_vecs[i] for i in live_ids_cph]).astype(np.float32)
            gt_local = brute_knn(live_mat, queries, k)
            gt_global = np.array([[live_ids_cph[j] for j in row] for row in gt_local])
        else:
            gt_global = np.zeros((n_queries, k), dtype=np.int32)

        # ── Evaluate recall ──
        def eval_recall(runner):
            found = [runner.search(q, k=k) for q in queries]
            return recall_at_k(gt_global, found, k)

        def eval_qps(runner):
            t = time.perf_counter()
            for q in queries:
                runner.search(q, k=k)
            return len(queries) / (time.perf_counter() - t)

        rec_cph    = eval_recall(cph)
        rec_hnsw_f = eval_recall(hnsw_f)
        rec_hnsw_r = eval_recall(hnsw_r)
        qps_cph    = eval_qps(cph)
        qps_hnsw_f = eval_qps(hnsw_f)
        qps_hnsw_r = eval_qps(hnsw_r)

        pct_del = total_deleted / max(1, total_deleted + len(live_ids_cph))
        history.append(dict(
            round=rnd,
            n_live=len(live_ids_cph),
            pct_del=pct_del,
            rec_cph=rec_cph,
            rec_hnsw_f=rec_hnsw_f,
            rec_hnsw_r=rec_hnsw_r,
            qps_cph=qps_cph,
            qps_hnsw_f=qps_hnsw_f,
            qps_hnsw_r=qps_hnsw_r,
        ))

        if rnd == rounds:
            break

        # ── Insert new batch ──
        new_vecs = rng.standard_normal((batch_insert, d)).astype(np.float32)
        start_global = len(all_vecs)
        all_vecs.extend(new_vecs)

        t0 = time.perf_counter()
        new_ids_cph = cph.add(new_vecs)
        t_ins_cph = time.perf_counter() - t0

        t0 = time.perf_counter()
        new_ids_hnsw_f = hnsw_f.add(new_vecs)
        _                = None
        new_ids_hnsw_r = hnsw_r.add(new_vecs)
        t_ins_hnsw_r = time.perf_counter() - t0  # rebuild runner tracks inserts only

        live_ids_cph.extend(new_ids_cph)
        live_ids_hnsw_f.extend(new_ids_hnsw_f)
        live_ids_hnsw_r.extend(new_ids_hnsw_r)

        ins_per_s_cph    = batch_insert / t_ins_cph
        ins_per_s_hnsw_r = batch_insert / t_ins_hnsw_r

        # ── Delete oldest DELETE_FRAC of live corpus ──
        n_delete = max(1, int(len(live_ids_cph) * delete_frac))
        to_delete = live_ids_cph[:n_delete]
        total_deleted += n_delete

        t0 = time.perf_counter()
        cph.delete(to_delete)
        t_del_cph = time.perf_counter() - t0

        hnsw_f.delete(to_delete)
        hnsw_r.delete(to_delete)

        live_ids_cph    = live_ids_cph[n_delete:]
        live_ids_hnsw_f = live_ids_hnsw_f[n_delete:]
        live_ids_hnsw_r = live_ids_hnsw_r[n_delete:]

        del_per_s_cph = n_delete / t_del_cph

        # ── Rebuild HNSW (this is the expensive step for hnsw_r) ──
        hnsw_r.rebuild()

        print(f"  {rnd+1:>5}  {len(live_ids_cph):>6,}  {pct_del:>4.0%}  "
              f"{rec_cph:>9.3f}  {rec_hnsw_f:>17.3f}  {rec_hnsw_r:>18.3f}  "
              f"{ins_per_s_cph:>10,.0f}  {ins_per_s_hnsw_r:>13,.0f}  "
              f"{del_per_s_cph:>10,.0f}  "
              f"{qps_cph:>8,.0f}  {qps_hnsw_f:>11,.0f}  {qps_hnsw_r:>11,.0f}")

    return history


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(history, suffix=""):
    rounds = [h["round"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── recall@10 over rounds ──
    ax = axes[0]
    ax.plot(rounds, [h["rec_cph"]    for h in history],
            color="#1f77b4", marker="o", lw=2, label="Copenhagen")
    ax.plot(rounds, [h["rec_hnsw_r"] for h in history],
            color="#2ca02c", marker="s", lw=2, ls="--", label="HNSW + rebuild")
    ax.plot(rounds, [h["rec_hnsw_f"] for h in history],
            color="#d62728", marker="^", lw=2, ls=":", label="HNSW + filter")
    ax.set_xlabel("Round")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Recall@10 under {DELETE_FRAC:.0%} churn per round")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── QPS over rounds ──
    ax = axes[1]
    ax.plot(rounds, [h["qps_cph"]    for h in history],
            color="#1f77b4", marker="o", lw=2, label="Copenhagen")
    ax.plot(rounds, [h["qps_hnsw_r"] for h in history],
            color="#2ca02c", marker="s", lw=2, ls="--", label="HNSW + rebuild")
    ax.plot(rounds, [h["qps_hnsw_f"] for h in history],
            color="#d62728", marker="^", lw=2, ls=":", label="HNSW + filter")
    ax.set_xlabel("Round")
    ax.set_ylabel("QPS")
    ax.set_title("Search QPS over rounds")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Streaming churn: n_init={N_INIT:,}, d={D}, "
        f"+{BATCH_INSERT:,}/round, delete {DELETE_FRAC:.0%} oldest",
        fontsize=11,
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / f"hnsw_churn{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=N_INIT,       help="Initial corpus size")
    ap.add_argument("--rounds",  type=int, default=ROUNDS,        help="Number of streaming rounds")
    ap.add_argument("--insert",  type=int, default=BATCH_INSERT,  help="Vectors inserted per round")
    ap.add_argument("--delete",  type=float, default=DELETE_FRAC, help="Fraction deleted per round")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    N_INIT       = args.n
    ROUNDS       = args.rounds
    BATCH_INSERT = args.insert
    DELETE_FRAC  = args.delete

    rng = np.random.default_rng(args.seed)
    history = run_simulation(rng, N_INIT, D, ROUNDS, BATCH_INSERT, DELETE_FRAC,
                             N_QUERIES, K)
    plot(history)
