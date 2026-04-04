"""benchmark_ivf_churn.py — Copenhagen vs FAISS IVF under streaming churn.

Scenario
--------
Start with N=50 000 vectors (d=128).  Run ROUNDS rounds, each:
  1. Insert BATCH_INSERT new vectors.
  2. Delete the oldest DELETE_FRAC fraction of the current live corpus.
  3. Measure recall@10 on 200 random queries against the live ground truth.

Three strategies
----------------
Copenhagen          O(1) tombstone delete; lazy compact; adaptive splits.
FAISS IVF + filter  Same n_clusters / nprobe as CPH.  Delete = tombstone mask;
                    search over-fetches and filters dead IDs.  No retrain →
                    recall degrades as dead vectors clog cluster lists.
FAISS IVF + rebuild Full retrain from live vectors every round.  Recall stays
                    stable; cost is O(n log n) per round.

Usage
-----
  python benchmarks/benchmark_ivf_churn.py
  python benchmarks/benchmark_ivf_churn.py --rounds 20 --n 100000
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

CPH_N_CLUSTERS      = 64
CPH_NPROBE          = 32
CPH_SOFT_K          = 2
CPH_SPLIT_THRESHOLD = 3.0

PQ_M     = 32   # number of PQ sub-quantizers  (d must be divisible by M)
PQ_NBITS = 8    # bits per sub-quantizer code   (256 centroids per subspace)

FIGURES_DIR = Path(__file__).parent.parent / "figures"


# ── ground truth ──────────────────────────────────────────────────────────────

def brute_knn(data, queries, k):
    q2    = np.sum(queries ** 2, axis=1)[:, None]
    d2    = np.sum(data    ** 2, axis=1)[None, :]
    cross = queries @ data.T
    return np.argsort(q2 + d2 - 2.0 * cross, axis=1)[:, :k].astype(np.int32)


def recall_at_k(gt_ids, found_ids, k=K):
    hits = sum(
        len(set(g[:k].tolist()) & set(list(f)[:k]))
        for g, f in zip(gt_ids, found_ids)
    )
    return hits / (len(gt_ids) * k)


# ── Copenhagen ────────────────────────────────────────────────────────────────

class CopenhagenRunner:
    def __init__(self, d, n_clusters, nprobe, soft_k, split_threshold):
        self.idx = CopenhagenIndex(dim=d, n_clusters=n_clusters,
                                   nprobe=nprobe, soft_k=soft_k)
        self.idx._index.split_threshold = split_threshold
        self._nprobe_frac = nprobe / n_clusters
        self._live_ids = set()

    def add(self, vecs):
        start_id = self.idx.n_vectors
        self.idx.add(vecs)
        new_ids = list(range(start_id, self.idx.n_vectors))
        self._live_ids.update(new_ids)
        return new_ids

    def delete(self, ids):
        for gid in ids:
            self.idx.delete(gid)
            self._live_ids.discard(gid)
        self.idx.compact()

    def search(self, q, k=K):
        n_clusters = self.idx._index.get_stats()["n_clusters"]
        nprobe = max(1, round(n_clusters * self._nprobe_frac))
        ids, _ = self.idx.search(q, k=k, n_probes=nprobe)
        return ids

    def n_live(self):
        return len(self._live_ids)


# ── FAISS IVF + filter ────────────────────────────────────────────────────────

class FAISSIVFFilterRunner:
    """IndexIVFFlat trained once; deletes are tombstones filtered at query time.

    Same n_clusters / nprobe as Copenhagen — isolates the algorithmic
    contribution of CPH's splits, soft_k, and compact from parameter tuning.
    """

    def __init__(self, d, n_clusters, nprobe):
        self._d          = d
        self._n_clusters = n_clusters
        self._nprobe     = nprobe
        self._index      = None
        self._deleted    = set()
        self._next_id    = 0
        self._id_map     = {}

    def _new_ivf(self, train_data):
        q   = faiss.IndexFlatL2(self._d)
        idx = faiss.IndexIVFFlat(q, self._d, self._n_clusters, faiss.METRIC_L2)
        idx.train(train_data)
        idx.nprobe = self._nprobe
        return idx

    def add(self, vecs):
        n = len(vecs)
        new_ids = list(range(self._next_id, self._next_id + n))
        if self._index is None:
            self._index = self._new_ivf(vecs)
        self._index.add(vecs)
        for i, gid in enumerate(new_ids):
            self._id_map[self._next_id + i] = gid
        self._next_id += n
        return new_ids

    def delete(self, ids):
        self._deleted.update(ids)

    def search(self, q, k=K):
        fetch = min(k * 10, self._index.ntotal)
        fetch = max(fetch, k)
        _, I  = self._index.search(q[None], fetch)
        live  = [self._id_map.get(int(i), -1) for i in I[0]
                 if i >= 0 and self._id_map.get(int(i), -1) not in self._deleted]
        return np.array(live[:k], dtype=np.int32)

    def n_live(self):
        return self._index.ntotal - len(self._deleted)


# ── FAISS IVF + rebuild ───────────────────────────────────────────────────────

class FAISSIVFRebuildRunner:
    """IndexIVFFlat retrained from live vectors every round.

    Same n_clusters / nprobe as Copenhagen — shows the recall ceiling for
    IVF at these parameters, and the cost of achieving it via full retrain.
    """

    def __init__(self, d, n_clusters, nprobe):
        self._d          = d
        self._n_clusters = n_clusters
        self._nprobe     = nprobe
        self._live       = {}
        self._next_id    = 0
        self._index      = None
        self._id_list    = []

    def _new_ivf(self, train_data):
        q   = faiss.IndexFlatL2(self._d)
        idx = faiss.IndexIVFFlat(q, self._d, self._n_clusters, faiss.METRIC_L2)
        idx.train(train_data)
        idx.nprobe = self._nprobe
        return idx

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
        if not self._live:
            self._index  = None
            self._id_list = []
            return
        self._id_list = list(self._live.keys())
        mat = np.stack([self._live[i] for i in self._id_list]).astype(np.float32)
        self._index = self._new_ivf(mat)
        self._index.add(mat)

    def search(self, q, k=K):
        if self._index is None or self._index.ntotal == 0:
            return np.array([], dtype=np.int32)
        _, I = self._index.search(q[None], k)
        return np.array([self._id_list[i] for i in I[0] if i >= 0], dtype=np.int32)

    def n_live(self):
        return len(self._live)


# ── FAISS IVFPQ + filter ──────────────────────────────────────────────────────

class FAISSIVFPQFilterRunner:
    """IndexIVFPQ trained once; deletes are tombstones filtered at query time.

    PQ compresses each vector to M×nbits bits (M=8,nbits=8 → 8 bytes vs 512
    for float32), trading recall for memory.  No native delete → same tombstone
    approach as IVFFlat+filter.
    """

    def __init__(self, d, n_clusters, nprobe, pq_m, pq_nbits):
        self._d          = d
        self._n_clusters = n_clusters
        self._nprobe     = nprobe
        self._pq_m       = pq_m
        self._pq_nbits   = pq_nbits
        self._index      = None
        self._deleted    = set()
        self._next_id    = 0
        self._id_map     = {}

    def _new_ivfpq(self, train_data):
        q   = faiss.IndexFlatL2(self._d)
        idx = faiss.IndexIVFPQ(q, self._d, self._n_clusters,
                               self._pq_m, self._pq_nbits)
        idx.train(train_data)
        idx.nprobe = self._nprobe
        return idx

    def add(self, vecs):
        n = len(vecs)
        new_ids = list(range(self._next_id, self._next_id + n))
        if self._index is None:
            self._index = self._new_ivfpq(vecs)
        self._index.add(vecs)
        for i, gid in enumerate(new_ids):
            self._id_map[self._next_id + i] = gid
        self._next_id += n
        return new_ids

    def delete(self, ids):
        self._deleted.update(ids)

    def search(self, q, k=K):
        fetch = min(k * 10, self._index.ntotal)
        fetch = max(fetch, k)
        _, I  = self._index.search(q[None], fetch)
        live  = [self._id_map.get(int(i), -1) for i in I[0]
                 if i >= 0 and self._id_map.get(int(i), -1) not in self._deleted]
        return np.array(live[:k], dtype=np.int32)

    def n_live(self):
        return self._index.ntotal - len(self._deleted)


# ── simulation ────────────────────────────────────────────────────────────────

def run_simulation(rng, n_init, d, rounds, batch_insert, delete_frac, n_queries, k):
    print(f"\nn_init={n_init:,}  d={d}  rounds={rounds}"
          f"  batch_insert={batch_insert:,}  delete_frac={delete_frac:.0%}"
          f"  k={k}  CPH n_clusters={CPH_N_CLUSTERS} nprobe={CPH_NPROBE}\n")

    data_init = rng.standard_normal((n_init, d)).astype(np.float32)
    queries   = rng.standard_normal((n_queries, d)).astype(np.float32)

    cph    = CopenhagenRunner(d, CPH_N_CLUSTERS, CPH_NPROBE, CPH_SOFT_K, CPH_SPLIT_THRESHOLD)
    ivf_f  = FAISSIVFFilterRunner(d, CPH_N_CLUSTERS, CPH_NPROBE)
    ivf_r  = FAISSIVFRebuildRunner(d, CPH_N_CLUSTERS, CPH_NPROBE)
    ivfpq  = FAISSIVFPQFilterRunner(d, CPH_N_CLUSTERS, CPH_NPROBE, PQ_M, PQ_NBITS)

    print("Building initial indexes …")
    t0 = time.perf_counter()
    live_ids_cph   = cph.add(data_init)
    live_ids_ivf_f = ivf_f.add(data_init)
    live_ids_ivf_r = ivf_r.add(data_init)
    live_ids_ivfpq = ivfpq.add(data_init)
    ivf_r.rebuild()
    print(f"  build time: {time.perf_counter() - t0:.2f}s")

    live_ids_cph   = list(live_ids_cph)
    live_ids_ivf_f = list(live_ids_ivf_f)
    live_ids_ivf_r = list(live_ids_ivf_r)
    live_ids_ivfpq = list(live_ids_ivfpq)

    all_vecs = list(data_init)

    header = (f"\n{'Round':>6}  {'Live':>6}  {'Del%':>5}  "
              f"{'CPH R@10':>9}  {'IVF+filt R@10':>14}  {'IVF+rbld R@10':>14}  {'IVFPQ+filt R@10':>16}  "
              f"{'CPH ins/s':>10}  {'IVF-r ins/s':>12}  "
              f"{'CPH del/s':>10}  "
              f"{'CPH QPS':>8}  {'IVF-f QPS':>10}  {'IVFPQ QPS':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    history       = []
    total_deleted = 0

    for rnd in range(rounds + 1):
        # ground truth over current live corpus
        if live_ids_cph:
            live_mat   = np.stack([all_vecs[i] for i in live_ids_cph]).astype(np.float32)
            gt_local   = brute_knn(live_mat, queries, k)
            gt_global  = np.array([[live_ids_cph[j] for j in row] for row in gt_local])
        else:
            gt_global = np.zeros((n_queries, k), dtype=np.int32)

        def eval_recall(runner):
            found = [runner.search(q, k=k) for q in queries]
            return recall_at_k(gt_global, found, k)

        def eval_qps(runner):
            t = time.perf_counter()
            for q in queries:
                runner.search(q, k=k)
            return len(queries) / (time.perf_counter() - t)

        rec_cph    = eval_recall(cph)
        rec_ivf_f  = eval_recall(ivf_f)
        rec_ivf_r  = eval_recall(ivf_r)
        rec_ivfpq  = eval_recall(ivfpq)
        qps_cph    = eval_qps(cph)
        qps_ivf_f  = eval_qps(ivf_f)
        qps_ivf_r  = eval_qps(ivf_r)
        qps_ivfpq  = eval_qps(ivfpq)

        pct_del = total_deleted / max(1, total_deleted + len(live_ids_cph))
        history.append(dict(
            round=rnd, n_live=len(live_ids_cph), pct_del=pct_del,
            rec_cph=rec_cph, rec_ivf_f=rec_ivf_f, rec_ivf_r=rec_ivf_r, rec_ivfpq=rec_ivfpq,
            qps_cph=qps_cph, qps_ivf_f=qps_ivf_f, qps_ivf_r=qps_ivf_r, qps_ivfpq=qps_ivfpq,
        ))

        if rnd == rounds:
            break

        # insert
        new_vecs = rng.standard_normal((batch_insert, d)).astype(np.float32)
        all_vecs.extend(new_vecs)

        t0 = time.perf_counter()
        new_ids_cph = cph.add(new_vecs)
        t_ins_cph = time.perf_counter() - t0

        t0 = time.perf_counter()
        new_ids_ivf_f = ivf_f.add(new_vecs)
        new_ids_ivf_r = ivf_r.add(new_vecs)
        new_ids_ivfpq = ivfpq.add(new_vecs)
        t_ins_ivf_r = time.perf_counter() - t0

        live_ids_cph.extend(new_ids_cph)
        live_ids_ivf_f.extend(new_ids_ivf_f)
        live_ids_ivf_r.extend(new_ids_ivf_r)
        live_ids_ivfpq.extend(new_ids_ivfpq)

        ins_per_s_cph   = batch_insert / t_ins_cph
        ins_per_s_ivf_r = batch_insert / t_ins_ivf_r

        # delete oldest fraction
        n_delete  = max(1, int(len(live_ids_cph) * delete_frac))
        to_delete = live_ids_cph[:n_delete]
        total_deleted += n_delete

        t0 = time.perf_counter()
        cph.delete(to_delete)
        t_del_cph = time.perf_counter() - t0

        ivf_f.delete(to_delete)
        ivf_r.delete(to_delete)
        ivfpq.delete(to_delete)

        live_ids_cph   = live_ids_cph[n_delete:]
        live_ids_ivf_f = live_ids_ivf_f[n_delete:]
        live_ids_ivf_r = live_ids_ivf_r[n_delete:]
        live_ids_ivfpq = live_ids_ivfpq[n_delete:]

        del_per_s_cph = n_delete / t_del_cph

        ivf_r.rebuild()

        print(f"  {rnd+1:>5}  {len(live_ids_cph):>6,}  {pct_del:>4.0%}  "
              f"{rec_cph:>9.3f}  {rec_ivf_f:>14.3f}  {rec_ivf_r:>14.3f}  {rec_ivfpq:>16.3f}  "
              f"{ins_per_s_cph:>10,.0f}  {ins_per_s_ivf_r:>12,.0f}  "
              f"{del_per_s_cph:>10,.0f}  "
              f"{qps_cph:>8,.0f}  {qps_ivf_f:>10,.0f}  {qps_ivfpq:>10,.0f}")

    return history


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(history, suffix=""):
    rounds = [h["round"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(rounds, [h["rec_cph"]   for h in history],
            color="#1f77b4", marker="o", lw=2, label="Copenhagen")
    ax.plot(rounds, [h["rec_ivf_r"] for h in history],
            color="#2ca02c", marker="s", lw=2, ls="--", label="FAISS IVF + rebuild")
    ax.plot(rounds, [h["rec_ivf_f"] for h in history],
            color="#d62728", marker="^", lw=2, ls=":", label="FAISS IVF + filter")
    ax.plot(rounds, [h["rec_ivfpq"] for h in history],
            color="#ff7f0e", marker="D", lw=2, ls="-.", label=f"FAISS IVFPQ+filter (M={PQ_M})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Recall@10 under {DELETE_FRAC:.0%} churn per round")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rounds, [h["qps_cph"]   for h in history],
            color="#1f77b4", marker="o", lw=2, label="Copenhagen")
    ax.plot(rounds, [h["qps_ivf_r"] for h in history],
            color="#2ca02c", marker="s", lw=2, ls="--", label="FAISS IVF + rebuild")
    ax.plot(rounds, [h["qps_ivf_f"] for h in history],
            color="#d62728", marker="^", lw=2, ls=":", label="FAISS IVF + filter")
    ax.plot(rounds, [h["qps_ivfpq"] for h in history],
            color="#ff7f0e", marker="D", lw=2, ls="-.", label=f"FAISS IVFPQ+filter (M={PQ_M})")
    ax.set_xlabel("Round")
    ax.set_ylabel("QPS")
    ax.set_title("Search QPS over rounds")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Streaming churn vs FAISS IVF: n_init={N_INIT:,}, d={D}, "
        f"+{BATCH_INSERT:,}/round, delete {DELETE_FRAC:.0%} oldest  "
        f"(CPH nprobe={CPH_NPROBE}/{CPH_N_CLUSTERS})",
        fontsize=10,
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / f"ivf_churn{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int,   default=N_INIT,      help="Initial corpus size")
    ap.add_argument("--rounds", type=int,   default=ROUNDS,       help="Streaming rounds")
    ap.add_argument("--insert", type=int,   default=BATCH_INSERT, help="Vectors inserted per round")
    ap.add_argument("--delete", type=float, default=DELETE_FRAC,  help="Fraction deleted per round")
    ap.add_argument("--seed",   type=int,   default=42)
    args = ap.parse_args()

    N_INIT       = args.n
    ROUNDS       = args.rounds
    BATCH_INSERT = args.insert
    DELETE_FRAC  = args.delete

    rng = np.random.default_rng(args.seed)
    history = run_simulation(rng, N_INIT, D, ROUNDS, BATCH_INSERT, DELETE_FRAC,
                             N_QUERIES, K)
    plot(history)
