# Copenhagen — Benchmark Results

Latest published run: `benchmarks/results/REPORT_20260607_000220.md` on Linux
`5.15.0-143-generic x86_64`, Python `3.11.15`, full mode. Scripts are in
`benchmarks/`.

---

## 1. Streaming Churn vs HNSW (`benchmark_hnsw_churn.py`)

**Scenario**: n_init=50,000, +1,000 inserts/round, delete 30% oldest/round.
CPH: 64 clusters, nprobe=32 (50%), soft_k=2, split_threshold=3.0.
HNSW: M=32, efConstruction=64, efSearch=64.

| Round | Live   | Del%  | CPH R@10 | HNSW+filter R@10 | HNSW+rebuild R@10 | CPH ins/s | HNSW-r ins/s | CPH del/s |
|-------|--------|-------|----------|------------------|-------------------|-----------|--------------|-----------|
| 1     | 35,700 |  0%   | 0.960    | 0.542            | 0.526             | 711,983   | 6,239        | 1,445,350 |
| 2     | 25,690 | 30%   | 0.950    | 0.518            | 0.571             | 895,094   | 9,450        | 1,419,024 |
| 3     | 18,683 | 51%   | 0.949    | 0.477            | 0.641             | 906,650   | 10,750       | 1,544,021 |
| 4     | 13,779 | 65%   | 0.945    | 0.453            | 0.711             | 830,629   | 9,669        | 1,447,393 |
| 5     | 10,346 | 74%   | 0.940    | 0.429            | 0.759             | 802,065   | 11,033       | 1,409,322 |
| 6     | 7,943  | 81%   | 0.938    | 0.389            | 0.807             | 912,472   | 10,438       | 1,498,962 |
| 7     | 6,261  | 86%   | 0.935    | 0.335            | 0.853             | 832,728   | 9,367        | 1,463,241 |
| 8     | 5,083  | 89%   | 0.926    | 0.288            | 0.875             | 883,667   | 8,839        | 1,497,785 |
| 9     | 4,259  | 91%   | 0.927    | 0.268            | 0.911             | 969,431   | 9,033        | 1,523,704 |
| 10    | 3,682  | 93%   | 0.937    | 0.276            | 0.916             | 838,724   | 8,345        | 1,486,813 |

**Key numbers at 93% churn**:
- CPH: 0.937 recall, ~1.49M del/s, ~839k ins/s
- HNSW+filter: 0.276 recall (graph clogging), no native delete
- HNSW+rebuild: 0.916 recall, but requires full graph rebuild every round (~8.3k ins/s effective)

CPH beats HNSW+rebuild on both final recall (+2.1pp) and update throughput, while
avoiding the rebuild path entirely.

---

## 2. Streaming Churn vs FAISS IVF (`benchmark_ivf_churn.py`)

**Scenario**: same as above. IVF: same 64 clusters, nprobe=32 — direct apples-to-apples
with Copenhagen. For the removed FAISS product-quantized baseline and why it is
no longer part of Copenhagen's design discussion, see [IVFPQ.md](IVFPQ.md).

| Round | Live   | Del%  | CPH R@10 | IVF+filter R@10 | IVF+rebuild R@10 | Legacy PQ baseline R@10 |
|-------|--------|-------|----------|-----------------|------------------|-------------------|
| 1     | 35,700 |  0%   | 0.960    | 0.815           | 0.815            | 0.432             |
| 3     | 18,683 | 51%   | 0.949    | 0.789           | 0.802            | 0.426             |
| 5     | 10,346 | 74%   | 0.940    | 0.783           | 0.811            | 0.443             |
| 7     | 6,261  | 86%   | 0.935    | 0.759           | 0.815            | 0.443             |
| 10    | 3,682  | 93%   | 0.937    | 0.642           | 0.803            | 0.392             |

**Insert throughput** (round 10, 1000-vector batch):
- CPH: ~971,000 /s
- IVF+filter: add-only baseline, no retrain
- IVF+rebuild: ~339,000 /s (full retrain each round)

**Removed-path note**: the legacy product-quantized baseline sits around
0.39–0.45 recall here, far below Copenhagen's 0.94-0.96. Details and rationale:
[IVFPQ.md](IVFPQ.md).

---

## 3. Distribution Drift — MNIST → Fashion-MNIST (`benchmark_drift.py`)

**Scenario**: train on 20k MNIST (784d handwritten digits), insert 10k Fashion-MNIST
(784d clothing — completely different manifold). 32 IVF clusters, nprobe=4. Ground
truth: brute-force over all 30k vectors.

| Method                                          | Fashion R@10 | MNIST R@10 | Insert    |
|-------------------------------------------------|-------------|------------|-----------|
| FAISS IVF add-only (no retrain)                 | 0.9548      | 0.9736     | 31 ms     |
| FAISS IVF full rebuild                          | 0.9892      | 0.9738     | 226 ms    |
| Copenhagen baseline (fixed, soft_k=1)           | 0.9488      | 0.9766     | 57 ms     |
| Copenhagen adaptive (splits, soft_k=1)          | 0.9236      | 0.9602     | 100 ms    |
| Copenhagen soft_k=2 (fixed)                     | 0.9864      | 0.9936     | 101 ms    |
| Copenhagen best (splits + soft_k=2)             | 0.9864      | 0.9936     | 102 ms    |

**Key comparisons**:
- Copenhagen best vs FAISS full rebuild: -0.28pp fashion recall, +1.98pp MNIST recall, 2.2x faster insert (102 ms vs 226 ms)
- Copenhagen soft_k=2 (fixed) nearly matches FAISS full rebuild on fashion recall (0.9864 vs 0.9892) at 2.2x faster insert
- Copenhagen best vs FAISS add-only: +3.16pp MNIST recall at +71 ms insert cost
- AMPI was not installed on this host, so it was skipped in this run

FAISS add-only collapses on fashion queries because all fashion vectors land in the
few MNIST Voronoi cells that are "least wrong"; nprobe=4 scans only 12.5% of the
index and misses them.

---

## 4. Gradual Streaming Drift (`benchmark_drift_streaming.py`)

**Scenario**: train on 10k MNIST (784d), stream 5k Fashion-MNIST in 10 batches of 500.
32 clusters, nprobe=4. Recall@10 measured after each batch against brute-force ground truth.

| Method                                    | Start  | Mid    | Final  |
|-------------------------------------------|--------|--------|--------|
| FAISS add-only                            | 0.9455 | 0.9770 | 0.9795 |
| Copenhagen baseline (no splits, soft_k=1) | 0.9210 | 0.9580 | 0.9635 |
| Copenhagen best (splits + soft_k=2)       | 0.9720 | 0.9875 | 0.9900 |

Copenhagen best leads FAISS from the first batch (+2.65pp) and finishes +1.05pp ahead.
In this run the gain comes from `soft_k=2`; no adaptive splits fired.

---

## 5. Insert Scaling — O(1) vs O(log n) (`benchmark_insert_scaling.py`)

**Scenario**: SIFT-128 (real L2 vectors, d=128). Insert 1,000 vectors into an index
already holding n vectors. CPH and FAISS IVF use batch add; HNSW uses single-vector add
(its natural API). k=64 clusters, nprobe=8.

| n       | CPH µs/vec | CPH R@10 | IVF µs/vec | IVF R@10 | HNSW µs/vec | HNSW R@10 |
|---------|-----------|----------|-----------|----------|------------|----------|
| 5,000   | 1.18      | 0.991    | 0.50      | 0.951    | 75.96      | 1.000    |
| 10,000  | 1.41      | 0.996    | 0.59      | 0.975    | 113.09     | 1.000    |
| 25,000  | 1.39      | 0.996    | 0.59      | 0.970    | 214.41     | 1.000    |
| 50,000  | 1.35      | 0.999    | 0.76      | 0.975    | 308.25     | 0.998    |
| 100,000 | 1.82      | 0.999    | 0.54      | 0.985    | 512.70     | 0.995    |

**Insert cost change over 20× scale-up (5k → 100k):**
- CPH: 1.54x — flat enough to remain **O(1)**
- FAISS IVF: 1.08x — **O(1)**
- HNSW: 6.75x — **O(log n)** + cache effects

Both CPH and FAISS IVF remain effectively flat in insert cost; HNSW is still much
more expensive per inserted vector and lacks native tombstone delete.

---

## 7. Static Recall and Throughput (`benchmark_vs_faiss.py`)

**Scenario**: 10k Gaussian vectors, d=128, 200 queries, recall@10.

| Method                           | R@10  | QPS    |
|----------------------------------|-------|--------|
| FAISS Flat L2 (exact)            | 1.000 | 3,476  |
| FAISS IVF nprobe=1               | 0.595 | 8,886  |
| FAISS IVF nprobe=10              | 0.595 | 8,719  |
| Copenhagen n=16 nprobe=8         | 0.712 | 6,672  |
| Copenhagen n=32 nprobe=8         | 0.483 | 10,802 |
| Copenhagen n=64 nprobe=8         | 0.369 | 16,404 |

FAISS IVF recall is capped by nlist=100 (sqrt(10k)) — nprobe has no effect past the cluster count. Copenhagen R@10 peaks at n=16 clusters with nprobe=8 (0.712) and trades recall for QPS at higher cluster counts. On stable distributions CPH is competitive on QPS but not recall — the advantage is in dynamic workloads: ~900k ins/s and O(1) delete vs HNSW/IVF rebuild.

---

## 8. Static Recall vs HNSW (`benchmark_vs_hnsw.py`)

**Scenario**: SIFT-100k (d=128, 200 queries). HNSW tuned via (M, efSearch);
Copenhagen tuned via (n_clusters, nprobe). Selected points from the Pareto frontier:

| Method                        | R@10  | QPS    |
|-------------------------------|-------|--------|
| HNSW M=16 ef=32               | 0.952 | 13,425 |
| HNSW M=32 ef=32               | 0.966 | 11,633 |
| HNSW M=32 ef=64               | 0.996 | 8,167  |
| HNSW M=32 ef=128              | 0.999 | 5,217  |
| HNSW M=32 ef=256              | 1.000 | 3,070  |
| Copenhagen n=64 nprobe=4      | 0.923 | 2,714  |
| Copenhagen n=32 nprobe=4      | 0.969 | 2,975  |
| Copenhagen n=16 nprobe=4      | 0.996 | 1,691  |
| Copenhagen n=16 nprobe=8      | 1.000 | 906    |

On a static dataset HNSW dominates the recall/QPS frontier. At R@10 ≈ 0.97,
HNSW M=32 ef=32 (11,633 QPS) is 3.9x faster than CPH n=32 nprobe=4 (2,975 QPS).
CPH's advantage is elsewhere: effectively O(1) insert cost (see §5) and O(1)
tombstone delete (HNSW has no native delete).

---

## 9. Tombstone Delete Latency

**Scenario**: 10k vectors (MNIST 784d, 32 clusters), delete 3,333 (33%), 300 queries.

| Method          | Delete time | µs / delete | Recall@10 | Leaked results |
|-----------------|-------------|-------------|-----------|----------------|
| FAISS rebuild   | 11 ms       | —           | 0.609     | 0              |
| CPH tombstone   | 1 ms        | 0.3         | 0.565     | 0              |

CPH deletes are **12× faster** with zero leaked vectors. FAISS has no native delete —
it must rebuild the entire index. Recall gap (0.565 vs 0.609) reflects that tombstoned
vectors are excluded from search but still occupy cluster slots, slightly reducing
effective nprobe coverage until compaction runs.

---

## 10. GPU Performance (`tests/bench_gpu.py`)

**Hardware**: Apple M-series (MPS), d=64, k=32. Run: `python tests/bench_gpu.py`

**Transfer (host→device→host round-trip):**

| n       | MB   | ms   | GB/s |
|---------|------|------|------|
| 10,000  | 2.6  | 0.69 | 3.72 |
| 50,000  | 12.8 | 2.58 | 4.97 |
| 100,000 | 25.6 | 5.16 | 4.96 |

**Centroid assignment compute (torch.mm vs numpy):**

| n       | CPU (ms) | GPU/MPS (ms) | speedup |
|---------|----------|--------------|---------|
| 10,000  | 8.29     | 2.29         | 3.62×   |
| 50,000  | 91.36    | 22.42        | 4.08×   |
| 100,000 | 106.99   | 54.13        | 1.98×   |

**End-to-end insert throughput (vectors/s):**

| n       | CPU       | GPU/MPS   | speedup |
|---------|-----------|-----------|---------|
| 10,000  | 1,619,663 | 786,398   | 0.49×   |
| 50,000  | 1,630,291 | 1,242,005 | 0.76×   |
| 100,000 | 1,622,070 | 1,316,638 | 0.81×   |

GPU wins 3.5–4× on the gemm alone but loses end-to-end because C++ bookkeeping
(tombstones, `id_to_location`, soft-k dedup) dominates insert time and can't be
offloaded. Centroid pinning: 0.16 ms one-time cost.

---

## Running the Benchmarks

```bash
# Streaming churn: Copenhagen vs HNSW under 30% deletion/round
python benchmarks/benchmark_hnsw_churn.py

# Streaming churn: Copenhagen vs FAISS IVF and legacy PQ baseline
python benchmarks/benchmark_ivf_churn.py

# Distribution drift: MNIST → Fashion-MNIST
python benchmarks/benchmark_drift.py          # quick (~2 min)
python benchmarks/benchmark_drift.py --full   # full 60k+60k (~15 min)

# Insert scaling: O(1) CPH vs O(log n) HNSW on SIFT-128
python benchmarks/benchmark_insert_scaling.py

# Static recall/QPS tradeoff: Copenhagen vs HNSW
python benchmarks/benchmark_vs_hnsw.py          # SIFT-100k (default)
python benchmarks/benchmark_vs_hnsw.py gauss    # 10k Gaussian

# Static recall vs FAISS across variants
python benchmarks/benchmark_vs_faiss.py
python benchmarks/benchmark_vs_faiss.py all   # include MNIST, Fashion, SIFT
```

Figures are written to `figures/`. Results JSON to `results/`.

## Running the Test Suites

GPU (MPS/CUDA) tests and FAISS benchmarks **must be run in separate invocations**
to avoid an OpenMP runtime conflict on macOS (two incompatible `libomp` copies in
the same process). Use the pytest markers:

```bash
# GPU correctness + performance (initialises MPS/CUDA context)
pytest -m gpu

# FAISS throughput benchmarks (loads FAISS OpenMP runtime)
pytest -m faiss

# All other tests (Copenhagen-only, no GPU or FAISS)
pytest -m "not gpu and not faiss"
```
