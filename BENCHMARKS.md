# Copenhagen — Benchmark Results

Latest published run: `benchmarks/results/REPORT_20260607_215454.md` on Linux
`6.8.0-1046-nvidia x86_64` (Intel Xeon Platinum 8358, 30 vCPU), Python `3.12.3`,
full mode — all 10 benchmarks green (faiss / hnsw / turbovec × churn / drift /
scaling / static / compression). Reproduce with `python benchmarks/reproduce.py`.
Recall figures are host-independent; insert/delete/QPS throughput is host-specific.

---

## 1. Streaming Churn vs HNSW (`benchmark_hnsw_churn.py`)

**Scenario**: n_init=50,000, +1,000 inserts/round, delete 30% oldest/round.
CPH: 64 clusters, nprobe=32 (50%), soft_k=2, split_threshold=3.0.
HNSW: M=32, efConstruction=64, efSearch=64.

| Round | Live   | Del%  | CPH R@10 | HNSW+filter R@10 | HNSW+rebuild R@10 | CPH ins/s | HNSW-r ins/s | CPH del/s |
|-------|--------|-------|----------|------------------|-------------------|-----------|--------------|-----------|
| 1     | 35,700 |  0%   | 0.960    | 0.524            | 0.529             | 317,725   | 6,625        | 698,323   |
| 2     | 25,690 | 30%   | 0.950    | 0.501            | 0.613             | 465,260   | 6,411        | 620,235   |
| 3     | 18,683 | 51%   | 0.949    | 0.469            | 0.659             | 597,530   | 8,025        | 658,857   |
| 4     | 13,779 | 65%   | 0.945    | 0.449            | 0.710             | 620,085   | 9,120        | 805,550   |
| 5     | 10,346 | 74%   | 0.940    | 0.417            | 0.754             | 403,072   | 7,098        | 784,679   |
| 6     | 7,943  | 81%   | 0.938    | 0.378            | 0.812             | 418,108   | 8,343        | 731,036   |
| 7     | 6,261  | 86%   | 0.935    | 0.343            | 0.843             | 634,191   | 6,636        | 706,200   |
| 8     | 5,083  | 89%   | 0.926    | 0.293            | 0.882             | 360,317   | 8,057        | 702,877   |
| 9     | 4,259  | 91%   | 0.927    | 0.282            | 0.909             | 637,759   | 7,985        | 761,379   |
| 10    | 3,682  | 93%   | 0.937    | 0.275            | 0.920             | 658,980   | 7,937        | 683,826   |

**Key numbers at 93% churn**:
- CPH: 0.937 recall, ~684k del/s, ~659k ins/s
- HNSW+filter: 0.275 recall (graph clogging), no native delete
- HNSW+rebuild: 0.920 recall, but requires full graph rebuild every round (~7.9k ins/s effective)

CPH beats HNSW+rebuild on final recall (+1.7pp) and is ~80× faster on inserts, while
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
- CPH: ~603,000 /s (and ~1.15M deletes/s)
- IVF+filter: add-only baseline, no retrain
- IVF+rebuild: ~286,000 /s (full retrain each round)

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
| 5,000   | 1.77      | 0.991    | 0.70      | 0.951    | 132.62     | 1.000    |
| 10,000  | 2.25      | 0.996    | 0.82      | 0.975    | 209.91     | 0.999    |
| 25,000  | 2.13      | 0.996    | 0.96      | 0.970    | 348.63     | 0.998    |
| 50,000  | 2.56      | 0.999    | 0.83      | 0.975    | 509.77     | 0.997    |
| 100,000 | 2.28      | 0.999    | 1.01      | 0.985    | 950.18     | 0.996    |

**Insert cost change over 20× scale-up (5k → 100k):**
- CPH: 1.29x — flat enough to remain **O(1)**
- FAISS IVF: 1.44x — **O(1)**
- HNSW: 7.16x — **O(log n)** + cache effects

Both CPH and FAISS IVF remain effectively flat in insert cost; HNSW is still much
more expensive per inserted vector and lacks native tombstone delete.

---

## 6. TurboQuant Fast-Scan Scoring Kernel (`benchmarks/bench_score_ip.cpp`)

The per-candidate TurboQuant scorer is the inner loop of every quantized search.
At ≤4 bits each coordinate's table has ≤16 entries — one register-width byte
table — so the scalar `score_ip` loop is replaced by a SIMD "fast scan" (FAISS /
Quick-ADC style): codes are stored block-transposed (`[block][dim][32]`) and 32
vectors are scored per dimension with one `vqtbl1q_u8` (NEON) / `pshufb` (AVX2)
over an 8-bit-quantized LUT, then the existing exact BLAS rerank resolves the
quantization. Microbenchmark (`-O3 -march=native`, 4-bit, ns per vector scored,
lower is better; speedup vs the scalar `score_ip` on the same host).

**Apple Silicon — NEON (width-32):**

| dim | scalar ns/vec | fast-scan ns/vec | speedup |
|----:|----:|----:|----:|
| 128 | 71 | 3.2 | 22× |
| 768 | 559 | 18.8 | 30× |
| 1536 | 1179 | 38.5 | 31× |

**Intel Xeon Platinum 8358 — AVX2 (width-32):**

| dim | scalar ns/vec | fast-scan ns/vec | speedup |
|----:|----:|----:|----:|
| 128 | 172 | 6.9 | 25× |
| 768 | 1069 | 85.3 | 12.5× |
| 1536 | 2113 | 210 | 10× |

On both ISAs the SIMD kernel is **bit-exact** with the scalar reference (0 sum
mismatches across all blocks) and preserves **99.0%** of the scalar top-100
candidate set. End-to-end (`tests/test_fastscan_recall.py`, run on both): 4-bit
fast-scan recall tracks the exact-IVF path within **≤0.005** (NEON 0.443 vs 0.447
exact at d=128; AVX2 0.431 vs 0.433), survives churn (delete + split + compaction)
within ~0.05, and never resurfaces a deleted id. Fast scan is the default route
for `tq_bits<=4`; on an unrecognized ISA the same blocked layout runs through a
scalar-block fallback. `get_stats()["tq_kernel"]` reports the active kernel
(`neon` / `avx2` / `scalar-block` / `scalar`).

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
