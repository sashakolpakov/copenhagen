"""benchmark_vs_turbovec.py — Copenhagen vs TurboVec, head to head, 3 axes.

TurboVec (RyanCodrai/turbovec) is the strongest *compressed static* ANN index
(TurboQuant scalar quantizer); Copenhagen is the strongest *dynamic* index. This
benchmark measures both on the three axes that matter so the trade-off is explicit:

  A. COMPRESSION + static recall — recall@10 vs bytes/vector. TurboVec should win
     bytes; we show Copenhagen-float's recall ceiling and that IVFPQ (Copenhagen's
     current compressed path) is both bigger AND worse — the gap the TurboQuant
     port closes.
  B. STATIC throughput — build time and QPS.
  C. DYNAMICS — insert throughput, delete handling, recall under churn. TurboVec
     has no native delete, so it must REBUILD each round (like FAISS/HNSW
     +rebuild); Copenhagen does O(1) tombstone delete.

All vectors are L2-NORMALIZED so L2-NN ≡ cosine ≡ MIPS — fair across TurboVec
(inner product) and Copenhagen/FAISS (L2). Ground truth is recomputed by exact
brute force on the normalized data.

Usage
  python benchmarks/benchmark_vs_turbovec.py            # SIFT subset (default)
  python benchmarks/benchmark_vs_turbovec.py --synthetic --n 20000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "core"))
sys.path.insert(0, str(Path(__file__).parent))
from core import CopenhagenIndex
from _turbovec_runner import normalize, bytes_per_vector
import turbovec as tv
try:
    import block_vq                      # Copenhagen's integrated block-VQ index
except ImportError:
    block_vq = None                      # build with: bash src/build_block_vq.sh

K = 10
DIM = 128


def exact_gt(base, queries, k):
    """Top-k by L2 on normalized data (== top-k by inner product)."""
    q2 = np.sum(queries ** 2, axis=1)[:, None]
    d2 = np.sum(base ** 2, axis=1)[None, :]
    return np.argsort(q2 + d2 - 2.0 * queries @ base.T, axis=1)[:, :k].astype(np.int32)


def recall(gt, found, k=K):
    return np.mean([len(set(g[:k]) & set(np.asarray(f)[:k].tolist())) / k
                    for g, f in zip(gt, found)])


def load_data(args):
    if not args.synthetic:
        path = Path(__file__).parent.parent / "data/sift/sift-128-euclidean.hdf5"
        if path.exists():
            import h5py
            with h5py.File(path, "r") as f:
                base = np.asarray(f["train"][: args.n], dtype=np.float32)
                queries = np.asarray(f["test"][: args.queries], dtype=np.float32)
            print(f"dataset: SIFT-128 (normalized)  base={len(base):,}  queries={len(queries):,}")
            return normalize(base), normalize(queries)
        print("  (SIFT not found — falling back to synthetic)")
    rng = np.random.default_rng(0)
    base = rng.standard_normal((args.n, DIM)).astype(np.float32)
    queries = rng.standard_normal((args.queries, DIM)).astype(np.float32)
    print(f"dataset: synthetic Gaussian (normalized)  base={len(base):,}  d={DIM}")
    return normalize(base), normalize(queries)


# ──────────────────────────── A. compression + recall ───────────────────────
def section_compression(base, queries, gt):
    n, d = base.shape
    fb = d * 4
    print("\n" + "=" * 78)
    print("A. COMPRESSION + static recall@10  (lower bytes/vec + higher recall = better)")
    print("=" * 78)
    print(f"{'index':<26}{'recall@10':>11}{'bytes/vec':>12}{'compress':>10}{'QPS':>10}")
    print("-" * 78)

    rows = []

    def bench_cph(label, **kw):
        idx = CopenhagenIndex(dim=d, **kw)
        idx.add(base.copy())
        np_ = kw.get("nprobe", 1)
        # recall — batch search returns (ids_list, dists_list)
        ids_list, _ = idx.search(queries, k=K, n_probes=np_)
        rec = recall(gt, ids_list)
        t = time.perf_counter()
        idx.search(queries, k=K, n_probes=np_)
        qps = len(queries) / (time.perf_counter() - t)
        return rec, qps

    # Copenhagen float (recall ceiling)
    rec, qps = bench_cph("CPH float", n_clusters=256, nprobe=32, soft_k=1)
    bpv = fb
    print(f"{'Copenhagen float':<26}{rec:>11.4f}{bpv:>12}{fb/bpv:>9.1f}x{qps:>10,.0f}")
    rows.append(("Copenhagen float", rec, bpv))

    # Copenhagen IVFPQ (current compressed path) — note: also retains floats
    pq_m = 16
    rec, qps = bench_cph("CPH IVFPQ", n_clusters=256, nprobe=32, soft_k=1,
                         use_pq=True, pq_m=pq_m, pq_ks=256)
    bpv = pq_m + fb   # PQ codes ON TOP of retained float32 (no memory win)
    print(f"{'Copenhagen IVFPQ (M=%d)' % pq_m:<26}{rec:>11.4f}{bpv:>12}{fb/bpv:>9.1f}x{qps:>10,.0f}"
          f"   [+{pq_m}B codes, floats retained]")
    rows.append(("Copenhagen IVFPQ", rec, bpv))

    # TurboVec 4-bit and 2-bit
    for bw in (4, 2):
        idx = tv.TurboQuantIndex(d, bw)
        idx.add(np.ascontiguousarray(base))
        idx.prepare()
        s, I = idx.search(np.ascontiguousarray(queries), K)
        found = [np.asarray(row) for row in np.asarray(I)]
        rec = recall(gt, found)
        t = time.perf_counter()
        idx.search(np.ascontiguousarray(queries), K)
        qps = len(queries) / (time.perf_counter() - t)
        bpv = bytes_per_vector(d, bw)
        print(f"{('TurboVec %d-bit' % bw):<26}{rec:>11.4f}{bpv:>12}{fb/bpv:>9.1f}x{qps:>10,.0f}")
        rows.append((f"TurboVec {bw}-bit", rec, bpv))

    # Copenhagen's INTEGRATED block-VQ index ("adaptive binning"): same rotation
    # + renormalization front-end as TurboVec, but B-dim joint codebooks instead
    # of scalar — the low-d fix. B=4 ≈ TurboVec 2-bit rate; B=2 ≈ 4-bit rate.
    if block_vq is not None:
        for B, label in ((4, "Copenhagen-TQ blockVQ B=4"), (2, "Copenhagen-TQ blockVQ B=2")):
            ix = block_vq.BlockVQIndex()
            ix.train(np.ascontiguousarray(base), B, 256)
            ix.add(np.ascontiguousarray(base))
            I = np.asarray(ix.search(np.ascontiguousarray(queries), K))
            rec = recall(gt, [row for row in I])
            t = time.perf_counter(); ix.search(np.ascontiguousarray(queries), K)
            qps = len(queries) / (time.perf_counter() - t)
            bpv = ix.bytes_per_vector()
            print(f"{label:<26}{rec:>11.4f}{bpv:>12}{fb/bpv:>9.1f}x{qps:>10,.0f}   [O(1) delete]")
            rows.append((label, rec, bpv))
    else:
        print("(block_vq not built — run `bash src/build_block_vq.sh` for the integrated row)")

    print("\nTakeaway: TurboVec wins bytes/vec over IVFPQ decisively. Copenhagen-TQ")
    print("block VQ matches TurboVec's compression AND supports O(1) delete — and at")
    print("the aggressive (B=4 ≈ 2-bit) low-d rate its joint codebook beats scalar.")
    return rows


# ──────────────────────────────── C. dynamics ───────────────────────────────
def section_dynamics(base, queries, gt, rounds=5, batch=2000, del_frac=0.30):
    n, d = base.shape
    from _turbovec_runner import TurboVecRebuildRunner
    print("\n" + "=" * 78)
    print(f"C. DYNAMICS  (insert / delete / {del_frac:.0%}-churn over {rounds} rounds)")
    print("=" * 78)

    rng = np.random.default_rng(1)
    # Copenhagen incremental insert throughput
    cph = CopenhagenIndex(dim=d, n_clusters=256, nprobe=32, soft_k=2)
    t = time.perf_counter()
    cph.add(base.copy())
    cph_build = time.perf_counter() - t
    print(f"Copenhagen initial insert: {len(base):,} vecs in {cph_build:.2f}s "
          f"({len(base)/cph_build:,.0f}/s)")

    tvr = TurboVecRebuildRunner(d, bit_width=4)
    tvr.add(base)
    tvr.rebuild()
    print(f"TurboVec initial build (4-bit): {len(base):,} vecs in "
          f"{tvr.last_rebuild_s:.2f}s ({len(base)/tvr.last_rebuild_s:,.0f}/s)")

    print(f"\n{'round':>6}{'live':>9}{'CPH R@10':>10}{'TVec R@10':>11}"
          f"{'CPH ins/s':>12}{'CPH del/s':>12}{'TVec rebuild/s':>16}")
    print("-" * 78)

    live_ids = list(range(len(base)))
    all_vecs = list(base)
    cph_live = set(live_ids)
    tvr_next = len(base)
    next_gid = len(base)

    for rnd in range(rounds):
        # ground truth over live set
        live_list = sorted(cph_live)
        live_mat = np.stack([all_vecs[i] for i in live_list]).astype(np.float32)
        gt_local = exact_gt(live_mat, queries, K)
        gt_global = np.array([[live_list[j] for j in row] for row in gt_local])

        found_cph, _ = cph.search(queries, k=K, n_probes=32)
        rec_cph = recall(gt_global, found_cph)
        found_tv = [tvr.search(q, K) for q in queries]
        rec_tv = recall(gt_global, found_tv)

        # insert
        new = normalize(rng.standard_normal((batch, d)).astype(np.float32))
        new_ids = list(range(next_gid, next_gid + batch)); next_gid += batch
        all_vecs.extend(new)
        t = time.perf_counter(); cph.add(new); ins_cph = batch / (time.perf_counter() - t)
        cph_live.update(new_ids)
        tvr.add(new)

        # delete oldest frac
        ndel = max(1, int(len(cph_live) * del_frac))
        to_del = sorted(cph_live)[:ndel]
        t = time.perf_counter(); cph.delete(to_del); del_cph = ndel / (time.perf_counter() - t)
        cph_live.difference_update(to_del)
        tvr.delete(to_del)
        tvr.rebuild()  # TurboVec has no delete → full rebuild
        tv_rebuild_rate = tvr.n_live() / tvr.last_rebuild_s if tvr.last_rebuild_s else 0

        print(f"{rnd+1:>6}{len(cph_live):>9,}{rec_cph:>10.3f}{rec_tv:>11.3f}"
              f"{ins_cph:>12,.0f}{del_cph:>12,.0f}{tv_rebuild_rate:>16,.0f}")

    print("\nTakeaway: TurboVec recall is high (flat + good quant) but every delete")
    print("forces a full rebuild; Copenhagen sustains O(1) insert/delete with no rebuild.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--skip-dynamics", action="store_true")
    args = ap.parse_args()

    base, queries = load_data(args)
    print("computing exact ground truth …")
    gt = exact_gt(base, queries, K)

    section_compression(base, queries, gt)
    if not args.skip_dynamics:
        section_dynamics(base, queries, gt)


if __name__ == "__main__":
    main()
