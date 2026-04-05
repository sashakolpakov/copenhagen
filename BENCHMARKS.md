# Copenhagen — Benchmark Results

All benchmarks run on Apple M-series (Accelerate BLAS), d=128 synthetic Gaussian
unless noted. Scripts are in `benchmarks/`.

---

## 1. Streaming Churn vs HNSW (`benchmark_hnsw_churn.py`)

**Scenario**: n_init=50,000, +1,000 inserts/round, delete 30% oldest/round.
CPH: 64 clusters, nprobe=32 (50%), soft_k=2, split_threshold=3.0.
HNSW: M=32, efConstruction=64, efSearch=64.

| Round | Live   | Del%  | CPH R@10 | HNSW+filter R@10 | HNSW+rebuild R@10 | CPH ins/s | HNSW-r ins/s | CPH del/s |
|-------|--------|-------|----------|------------------|-------------------|-----------|--------------|-----------|
| 1     | 35,700 |  0%   | 0.951    | 0.698            | 0.692             | 870,922   | 11,469       | 1,191,353 |
| 2     | 25,690 | 30%   | 0.945    | 0.663            | 0.723             | 840,042   | 10,893       | 1,460,309 |
| 3     | 18,683 | 51%   | 0.945    | 0.644            | 0.800             | 932,220   | 9,905        | 1,457,043 |
| 4     | 13,779 | 65%   | 0.947    | 0.609            | 0.827             | 878,735   | 10,504       | 1,384,223 |
| 5     | 10,346 | 74%   | 0.941    | 0.588            | 0.853             | 916,695   | 9,358        | 1,083,179 |
| 6     | 7,943  | 81%   | 0.933    | 0.557            | 0.901             | 947,867   | 10,297       | 971,280   |
| 7     | 6,261  | 86%   | 0.930    | 0.511            | 0.906             | 940,955   | 10,035       | 1,010,804 |
| 8     | 5,083  | 89%   | 0.927    | 0.501            | 0.925             | 948,917   | 9,481        | 1,062,115 |
| 9     | 4,259  | 91%   | 0.925    | 0.492            | 0.952             | 933,998   | 6,856        | 969,289   |
| 10    | 3,682  | 93%   | 0.929    | 0.473            | 0.952             | 917,993   | 10,084       | 971,882   |

**Key numbers at 93% churn**:
- CPH: 0.929 recall, ~1M del/s, ~900k ins/s
- HNSW+filter: 0.473 recall (graph clogging), ~10k ins/s effective
- HNSW+rebuild: 0.952 recall, but requires full graph rebuild every round (~10k ins/s effective)

CPH maintains recall within 0.023 of an always-optimal HNSW+rebuild while inserting
and deleting **90–100× faster**.

---

## 2. Streaming Churn vs FAISS IVF (`benchmark_ivf_churn.py`)

**Scenario**: same as above. IVF: same 64 clusters, nprobe=32 — direct apples-to-apples
with Copenhagen. IVFPQ: M=32 subquantizers, 8 bits (32 bytes/vec vs 512 for float32).

| Round | Live   | Del%  | CPH R@10 | IVF+filter R@10 | IVF+rebuild R@10 | IVFPQ+filter R@10 |
|-------|--------|-------|----------|-----------------|------------------|-------------------|
| 1     | 35,700 |  0%   | 0.951    | 0.795           | 0.795            | 0.436             |
| 3     | 18,683 | 51%   | 0.945    | 0.782           | 0.786            | 0.447             |
| 5     | 10,346 | 74%   | 0.941    | 0.784           | 0.793            | 0.472             |
| 7     | 6,261  | 86%   | 0.930    | 0.757           | 0.808            | 0.454             |
| 10    | 3,682  | 93%   | 0.929    | 0.663           | 0.887            | 0.423             |

**Insert throughput** (round 1, 1000-vector batch):
- CPH: ~900,000 /s (tombstone + BLAS centroid assignment)
- IVF+filter: adds to frozen index, no retrain; ~same insert rate as CPH
- IVF+rebuild: ~90,000–120,000 /s (full retrain each round)

**IVFPQ note**: at M=32 (16× compression vs float32), IVFPQ achieves 0.42–0.47 recall —
less than half of CPH's 0.93–0.95. IVFPQ QPS is modestly higher due to PQ distance
table lookups, but the recall gap makes the comparison unfavorable. At M=8 (64×
compression) IVFPQ recall drops to 0.06–0.11.

---

## 3. Distribution Drift — MNIST → Fashion-MNIST (`benchmark_drift.py`)

**Scenario**: train on 20k MNIST (784d handwritten digits), insert 10k Fashion-MNIST
(784d clothing — completely different manifold). 32 IVF clusters, nprobe=4. Ground
truth: brute-force over all 30k vectors.

| Method                                          | Fashion R@10 | MNIST R@10 | Insert    |
|-------------------------------------------------|-------------|------------|-----------|
| FAISS IVF add-only (no retrain)                 | 0.953        | 0.973      | 21 ms     |
| FAISS IVF full rebuild                          | 0.989        | 0.974      | 119 ms    |
| AMPI (nlist=212, fans=16, probes=16)            | 0.997        | 0.974      | 7,063 ms  |
| Copenhagen baseline (fixed, soft_k=1)           | 0.962        | 0.980      | 29 ms     |
| Copenhagen adaptive (splits, soft_k=1)          | 0.945        | 0.977      | 93 ms     |
| Copenhagen soft_k=2 (fixed)                     | 0.989        | 0.994      | 65 ms     |
| Copenhagen best (splits + soft_k=2)             | 0.979        | 0.993      | 168 ms    |

**Key comparisons**:
- Copenhagen best vs FAISS full rebuild: −0.98pp fashion recall, +0.19pp MNIST recall, 0.7× insert speed (168 ms vs 119 ms)
- Copenhagen soft_k=2 (fixed) matches FAISS full rebuild on fashion recall (0.989 vs 0.989) at 1.8× faster insert (65 ms vs 119 ms) with +2.0pp MNIST recall
- Copenhagen best vs FAISS add-only: +2.6pp fashion recall, +2.3pp MNIST recall
- AMPI leads all on fashion recall (0.997) at 42× slower insert than Copenhagen best

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
| Copenhagen baseline (no splits, soft_k=1) | 0.9160 | 0.9590 | 0.9685 |
| Copenhagen best (splits + soft_k=2)       | 0.9715 | 0.9855 | 0.9865 |

Copenhagen best leads FAISS from the very first batch (+2.6pp) and finishes +0.7pp ahead.
Copenhagen baseline lags at the start but closes the gap as splits fire.

---

## 5. Insert Scaling — O(1) vs O(log n) (`benchmark_insert_scaling.py`)

**Scenario**: SIFT-128 (real L2 vectors, d=128). Insert 1,000 vectors into an index
already holding n vectors. CPH and FAISS IVF use batch add; HNSW uses single-vector add
(its natural API). k=64 clusters, nprobe=8.

| n       | CPH µs/vec | CPH R@10 | IVF µs/vec | IVF R@10 | HNSW µs/vec | HNSW R@10 |
|---------|-----------|----------|-----------|----------|------------|----------|
| 5,000   | 1.16      | 0.993    | 0.79      | 0.951    | 280        | 1.000    |
| 10,000  | 1.24      | 0.996    | 0.60      | 0.975    | 492        | 1.000    |
| 25,000  | 1.12      | 0.997    | 0.64      | 0.972    | 1,092      | 1.000    |
| 50,000  | 1.03      | 0.998    | 0.64      | 0.975    | 2,207      | 0.998    |
| 100,000 | 1.08      | 0.996    | 0.67      | 0.985    | 8,419      | 0.997    |

**Insert cost change over 20× scale-up (5k → 100k):**
- CPH: 0.93× — flat → **O(1)**
- FAISS IVF: 0.85× — flat → **O(1)**
- HNSW: 30× — growing → **O(log n)** (amplified by cache effects at large n)

HNSW starts at 240× the CPH insert cost and reaches 7,800× at n=100k. Both CPH and
FAISS IVF are O(1) in insert cost; the difference is recall under distribution shift
(see §3) and O(1) delete (FAISS IVF has no native delete).

---

## 7. Static Recall and Throughput (`benchmark_vs_faiss.py`)

**Scenario**: 10k Gaussian vectors, d=128, 200 queries, recall@10.

| Method                           | R@10  | QPS    |
|----------------------------------|-------|--------|
| FAISS Flat L2 (exact)            | 1.000 | 4,277  |
| FAISS IVF nprobe=1               | 0.595 | 10,121 |
| FAISS IVF nprobe=10              | 0.595 | 10,327 |
| Copenhagen n=16 nprobe=8         | 0.740 | 7,974  |
| Copenhagen n=32 nprobe=8         | 0.499 | 11,779 |
| Copenhagen n=64 nprobe=8         | 0.368 | 15,070 |

FAISS IVF recall is capped by nlist=100 (sqrt(10k)) — nprobe has no effect past the cluster count. Copenhagen R@10 peaks at n=16 clusters with nprobe=8 (0.740) and trades recall for QPS at higher cluster counts. On stable distributions CPH is competitive on QPS but not recall — the advantage is in dynamic workloads: ~900k ins/s and O(1) delete vs HNSW/IVF rebuild.

---

## 8. Static Recall vs HNSW (`benchmark_vs_hnsw.py`)

**Scenario**: SIFT-100k (d=128, 200 queries). HNSW tuned via (M, efSearch);
Copenhagen tuned via (n_clusters, nprobe). Selected points from the Pareto frontier:

| Method                        | R@10  | QPS    |
|-------------------------------|-------|--------|
| HNSW M=16 ef=32               | 0.961 | 6,209  |
| HNSW M=32 ef=32               | 0.972 | 5,803  |
| HNSW M=32 ef=64               | 0.994 | 4,454  |
| HNSW M=32 ef=128              | 0.998 | 3,179  |
| HNSW M=32 ef=256              | 1.000 | 1,698  |
| Copenhagen n=64 nprobe=4      | 0.920 | 5,031  |
| Copenhagen n=32 nprobe=4      | 0.971 | 3,661  |
| Copenhagen n=16 nprobe=4      | 0.991 | 1,947  |
| Copenhagen n=16 nprobe=8      | 1.000 | 1,178  |

On a static dataset HNSW dominates the recall/QPS frontier. At R@10 ≈ 0.97,
HNSW M=32 ef=32 (5,803 QPS) is 1.6× faster than CPH n=32 nprobe=4 (3,661 QPS).
CPH's advantage is elsewhere: O(1) insert cost (240–7,800× cheaper than HNSW per
vector, see §5) and O(1) tombstone delete (HNSW has no native delete).

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

# Streaming churn: Copenhagen vs FAISS IVF and IVFPQ
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
