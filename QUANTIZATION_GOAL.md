# Goal: TurboQuant-style quantization for Copenhagen

## The thesis

Copenhagen is the strongest **dynamic-workload** ANN index we have: O(1) amortized
insert, O(1) tombstone delete, soft multi-cluster assignment, adaptive splits.
Its one weak spot is **compression**. The old compressed path had two
problems:

1. **It saves no memory.** The rerank step reads `clusters[c].vectors` — the full
   float32 vectors are still stored. PQ codes are added *on top*, so the index gets
   *bigger*, not smaller.
2. **Recall collapses under real compression.** Our latest benchmark matrix shows
   the removed PQ baseline at 0.392–0.446 recall@10 at 16× compression under churn — less than
   half of Copenhagen's float path (0.926–0.960).

[TurboVec](https://github.com/RyanCodrai/turbovec) (Ryan Codrai) is the mirror
image: a **compressed static-ish** flat index built on Google Research's
**TurboQuant** scalar quantizer. It holds recall at 2–4 bits/coordinate where PQ
falls apart, and it's the quantizer — not the index structure — that does the work.

**The opportunity is to take the best of both worlds:** keep Copenhagen's dynamic
IVF structure, and replace the weak legacy compressed path with a TurboQuant scalar quantizer.
Result: an index that is *both* fully dynamic *and* genuinely compressed with good
recall-per-byte.

## What we are NOT doing

- **No Rust.** TurboVec's implementation is Rust; we only borrow the *algorithm*.
  Copenhagen stays C++17 + BLAS. We reimplement TurboQuant from scratch in C++.
- We are not adopting TurboVec's flat (brute-force) index. Copenhagen's IVF +
  dynamic machinery (tombstones, soft_k, splits) is the part we keep.

## The TurboQuant algorithm (what we reimplement)

Per the TurboVec source (`rotation.rs`, `codebook.rs`, `encode.rs`, `search.rs`):

1. **Normalize** each vector to a unit direction `u = v/||v||`; remember `||v||`.
2. **Random orthogonal rotation** `u_rot = R u`. `R` is a fixed `d×d` orthonormal
   matrix (seeded, deterministic). After rotation each coordinate of a unit vector
   follows a known Beta((d-1)/2, (d-1)/2) marginal on [-1, 1] — a predictable,
   well-behaved distribution to quantize.
3. **Lloyd-Max scalar quantization.** A single shared `2^bits`-level codebook
   (reconstruction `centroids` + `boundaries`) optimal for that coordinate
   distribution. Each rotated coordinate → one `bits`-wide code (bits ∈ {2,3,4}).
4. **TQ+ per-coordinate calibration** (optional). Real data is anisotropic, so each
   coord deviates from the canonical marginal. Fit `(shift[d], scale[d])` mapping
   the empirical 5/95% quantiles of coord `d` onto the canonical 5/95% quantiles;
   quantize `u_calib[d] = (u_rot[d] + shift[d]) * scale[d]`. The inverse is applied
   to the query at search time, so the same codebook fits better.
5. **Length-renormalized scoring (RaBitQ-style).** Store a per-vector
   `scale = ||v|| / <u_rot, x_hat>` where `x_hat` is the reconstruction. At search,
   `<q, v> ≈ scale · <q_rot, x_hat>` is an unbiased estimator of the true inner
   product. For Copenhagen's L2 ranking we use
   `||q - v||² ≈ ||q||² + ||v||² - 2·<q,v>` with a stored per-vector `||v||²`.

Scoring is a per-coordinate table lookup: `T[d][level] = (q_rot[d]/scale[d])·centroid[level]`,
score `= bias + Σ_d T[d][code_d]`, then `× vec_scale`. TurboVec packs this into
nibble-split SIMD LUT kernels (NEON/AVX-512); we start with a correct scalar
version and **leave a clean seam for SIMD kernels later** (NEON first, matching
the existing `-march=armv8-a` build).

## Design decisions for the C++ port

- **Global rotation + codebook**, shared across all clusters and fit once at
  `train()`. Lloyd-Max is fit on the **empirical pooled rotated coordinates of the
  training batch** (data-adaptive, avoids needing Beta special functions in C++,
  and is more robust to anisotropy — TQ+ then mops up per-coord residual).
- **Cluster storage becomes truly compressed.** A `TQCluster` stores `bits·d/8`
  bytes of codes + a 4-byte `scale` + a 4-byte `||v||²` per vector, replacing the
  full float32 block. This is the change that makes the index actually smaller.
- **Dynamics are preserved.** Codes are fixed-width per vector, so tombstone
  delete, lazy compaction, soft_k dedup, and cluster splitting all work unchanged
  (split re-encodes live codes; the rotation/codebook are global so no retrain).
- **Optional exact rerank.** Keep an opt-in float rerank for the top candidates
  *only if the user asks* — but the default path is compressed-only, so the memory
  win is real.

## Success criteria

1. **Recall-per-byte beats the removed baseline decisively.** Target: ≥ 0.85 recall@10 at 8× (4-bit)
   and ≥ 0.75 at 16× (2-bit) on SIFT/synthetic — vs 0.392–0.446 at 16×.
2. **Real memory reduction.** Compressed clusters store no float32; index size drops
   ~`32/bits ×` vs the float path (minus small per-vector overhead).
3. **Dynamics intact.** Insert / delete / soft_k / split all pass existing tests
   with the quantizer enabled.
4. **SIMD-ready.** Scalar scorer is structured so a NEON nibble-LUT kernel can drop
   in behind the same interface.

## Phase 2 results (src/tq_standalone_test.cpp, synthetic blobs, top-10)

| dim  | 2-bit raw | 2-bit +rerank@200 | 4-bit raw | 4-bit +rerank | compression (2b/4b) |
|------|-----------|-------------------|-----------|---------------|---------------------|
| 128  | 0.4581    | 0.9893            | 0.8183    | 1.0000        | 12.8× / 7.1×        |
| 768  | 0.5990    | 1.0000            | 0.8700    | 1.0000        | 15.4× / 7.8×        |
| 1536 | 0.6704    | 1.0000            | 0.9008    | 1.0000        | 15.7× / 7.9×        |

**Removed-path reference (latest churn matrix): 0.392–0.446 @ 16×.** See [IVFPQ.md](IVFPQ.md).

Conclusions: (1) raw recall **rises with dimension** — exactly as the marginal
theory predicts; d=128 (SIFT) is TurboQuant's worst case. (2) **4-bit
compressed-only already beats the removed baseline decisively** (0.82–0.90 vs 0.392–0.446) at 7–8×
with no float storage. (3) rerank@200 → ~0.99–1.00 at all bit widths, but rerank
needs floats (see split tension). (4) TQ+ calibration adds a consistent +1–2pp.

### Decision: cull the legacy compressed path

TurboQuant dominates the removed baseline at every operating point *and* actually
reduces memory. The quant path is now landed; `PQCodebook` / `PQCluster` /
`use_pq` were deleted. Removal rationale: [IVFPQ.md](IVFPQ.md).

---

## Why low-d breaks (precisely) — and where the real gains are

TurboQuant's justification is asymptotic. After a random rotation the coordinates
of a unit vector are *exchangeable* with marginal Beta((d−1)/2,(d−1)/2) on [−1,1],
which → N(0, 1/d) as d→∞. Three things degrade at small d, and TurboQuant
conflates them:

1. **Non-Gaussian marginal.** Beta(a,a) with small a is platykurtic/wide (U-shaped
   at d=2,3). Lloyd-Max on the exact Beta — or our empirical fit — already fixes
   this. *Not* the main loss; the part TurboQuant got right.
2. **Coordinate dependence (the real killer).** A unit vector lives on S^{d−1}:
   Σxᵢ²=1. That constraint induces **negative correlation** between coordinates of
   O(1/d) — negligible at d=1536, strong at d=128. Per-coordinate scalar
   quantization **assumes independence** and discards this structure. This is the
   gap.
3. **Less averaging.** ⟨q,v⟩ ≈ scale·Σ_d q_rot[d]·x̂[d] is a sum of d error terms;
   its variance shrinks ~1/d *only if errors are independent*. CLT concentration
   is what holds high-d recall at 2 bits. Fewer + dependent terms ⇒ the estimate
   doesn't concentrate.

Diagnosis: **they optimized the marginal and ignored the joint.** Low-d is where
the joint matters.

### Improvements, ranked by leverage

**1. Block / lattice VQ after rotation (biggest lever).** Replace per-coordinate
scalar Lloyd-Max with per-*block* VQ on 2-D/4-D/8-D sub-blocks of the rotated
vector — the one change that captures the sphere-induced correlation scalar QY
discards. Two flavors:
   - **Optimal lattices**: D4 (4-D), **E8 (8-D)** are provably optimal quantizing
     lattices for Gaussian-ish sources — E8 gains ~1 bit/coord over scalar at equal
     distortion, *for free*, closed-form decode, no trained codebook.
   - **Trained small VQ** per block (k-means, 16–256 codes / 4-D block) when data
     isn't isotropic after rotation.

   Crucially this **keeps the SIMD nibble-LUT architecture**: a block → one code →
   LUT entry = ⟨q_block, recon_block⟩. It is literally PQ's data layout with
   TurboQuant's rotation+renormalization front-end and a *good* codebook —
   genuine best-of-both-worlds (PQ's per-subspace k-means was always better than
   scalar; we bolt it onto the renormalized-inner-product estimator instead of raw
   L2).

**2. MIPS-aware (anisotropic) loss, not MSE.** Lloyd-Max minimizes reconstruction
MSE, but recall depends on inner-product error, and the residual component
*parallel* to the vector hurts ⟨q,v⟩ far more than the orthogonal one (ScaNN / Guo
et al. 2020). Reweight the codebook fit to penalize the parallel residual ~d×
more. A change to the same k-means *objective* — cheap, directly optimizes what we
measure.

**3. One residual-correction byte (extended-RaBitQ).** The estimator ignores
u_rot's component orthogonal to x̂. Store **one extra byte/vector** = residual
energy (or cos∠(u_rot, x̂)); ⟨q,v⟩ then gets an unbiased correction with a
*provable* error bound. We already store `scale` and `‖v‖²`; this is a third
scalar — cheap, and it tightens the estimate precisely where averaging fails at
low d.

**4. Learned rotation (OPQ/ITQ-style; already in TODO).** Random rotation only
guarantees identical marginals. A learned d×d rotation can *also* decorrelate for
the actual data — but pure PCA breaks the identical-marginal assumption the shared
codebook needs. The principled object is OPQ/ITQ alternating optimization trading
decorrelation vs marginal uniformity. Second-order win on top of block-VQ.

### The freedom we're not using

At low d we have **bits to spare**: d=128 at 4-bit is only 64 B. We can afford 6–8
effective bits/coord via block-VQ and still sit at 24–32× vs float32. The
rate–distortion budget is wide open exactly where TurboQuant gave up.

**Recommendation:** prototype **rotation + 4-D block VQ (trained k-means,
anisotropic loss) + one residual-correction byte**, scored through the existing
LUT seam. Attacks mechanisms 2 and 3 directly, keeps SIMD, and is a clean superset
of both TurboQuant (scalar = 1-D block) and PQ (raw L2 = no renormalization).
The **E8 lattice** variant is the elegant zero-training fallback to measure
against the trained codebook.

### Phase 3 results — block VQ vs scalar at MATCHED bytes (src/tq_block_test.cpp)

RAW recall@10 (compressed-only, no float rerank), synthetic blobs:

| d   | rate  | scalar (B=1) | block VQ B=… MSE | Δ        | bytes/vec |
|-----|-------|--------------|------------------|----------|-----------|
| 128 | 2-bit | 0.4581       | **0.5806** (B=4) | **+12.3pp** | 40     |
| 128 | 4-bit | 0.8183       | **0.8412** (B=2) | +2.3pp   | 72        |
| 768 | 2-bit | 0.5990       | **0.7288** (B=4) | **+13.0pp** | 200    |
| 768 | 4-bit | 0.8700       | **0.8956** (B=2) | +2.6pp   | 392       |

**Lever #1 confirmed.** Block VQ beats scalar at every operating point, and the
win is *largest at 2-bit* (+12–14pp) — exactly the aggressive-compression / low-d
regime where the unit-sphere coordinate correlation dominates and scalar QY
discards it. At 4-bit the gap narrows (fewer dims per block share less
correlation, and scalar 4-bit is already decent). Net: **block VQ is the path** —
it makes 2-bit (≈13–16×) genuinely usable where scalar and the removed baseline both fail.

**Lever #2 (anisotropic loss) — DONE, and the result is a finding, not a win.**
First the broken block-local proxy was rejected (hurt everywhere). Then the
*correct* ScaNN/Guo et al. formulation was implemented in full: full-vector
parallel-residual weighting `L = ‖r‖² + (η−1)(xᵀr)²` (x=u_rot is unit after
rotation, so ‖x‖²=1 and blocks couple only through the scalar s=xᵀr), with
block-coordinate-descent assignment and the closed-form B×B weighted-least-squares
codebook update (`refine_aniso`/`assign_aniso`/`solve_spd` in block_quant.hpp;
η=1 reproduces MSE — sanity check passes). The B×B solve is **Cholesky** (A is
SPD by construction for η≥1; identity-minus-PSD but still PD for η<1 since
λ_max(Σxxᵀ)≈N·B/d≪N) with a diagonal-jitter fallback — no pivot search, ~2× fewer
flops than LU; B is tiny by the SIMD-LUT design so BLAS/LAPACK isn't warranted.

η-sweep, RAW recall@10, B=4 (2-bit), src/tq_aniso_test.cpp:

| η      | 0.25       | 0.5    | 1 (MSE) | 2      | 10     | 30     | 100     |
|--------|------------|--------|---------|--------|--------|--------|---------|
| d=128  | **0.6007** | 0.6005 | 0.5987  | 0.5963 | 0.5912 | 0.5847 | 0.5682  |
| d=768  | **0.7248** | 0.7194 | 0.7164  | 0.7176 | 0.7130 | 0.7012 | 0.6144  |

**η>1 (classic ScaNN) monotonically HURTS.** Mechanism: our estimator already
applies RaBitQ length-renormalization `scale = ‖v‖/⟨u_rot,x̂⟩`, which *explicitly
divides out the reconstruction's parallel/norm error*. ScaNN's η>1 exists to
protect that same parallel component for indexes that DON'T renormalize — stacked
on renormalization it double-corrects and wastes codebook resolution. What
governs ranking here is the **orthogonal/angular** accuracy of x̂, pointing to η<1.

**η<1 — culled from main (5-seed verdict).** A single seed hinted at a gain; the
significance test `src/tq_aniso_signif.cpp` (5 seeds, varying data + init) shows
it is marginal at d=128 (+0.0028 ± 0.0017, ~1.6σ) and **net negative** at d=768
(−0.0035 ± 0.0026, 4/5 seeds). The anisotropic codebook does not earn its
training cost; it was **removed from main and preserved on the
`experiments/anisotropic` branch**. Bit-budget takeaway: spend bits on **joint
structure (block VQ, +12–14pp)**, not on parallel reweighting — once you
renormalize, the parallel direction is already handled.

Open next: (i) does block VQ + rerank reach ~1.0 like scalar did? (ii) residual
byte (lever #3); (iii) correct anisotropic; (iv) Kc=16 (4-bit codes) for the
NEON nibble-LUT path — measure recall cost of 16 vs 256 codes/block.

---

## Build plan / status

- [x] Phase 0 — Study TurboQuant + Copenhagen internals; formulate goal (this doc).
- [x] Phase 1 — `src/turbo_quant.hpp`: rotation, Lloyd-Max codebook, TQ+, encode,
      scalar scorer. Dependency-free, with dedicated microbenchmarks. (scalar = 1-D block VQ;
      the baseline the block-VQ prototype must beat.)
- [x] Phase 2 — `src/tq_standalone_test.cpp`: recall-per-byte vs exact, dim sweep
      128/768/1536. Thesis + low-d failure mode proven (tables above).
- [~] Phase 3 (**revised**) — **Block-VQ prototype** (`src/block_quant.hpp`,
      `src/tq_block_test.cpp`). Done: (a) scalar baseline, (b) trained k-means VQ
      B∈{2,4} Kc=256 MSE — **confirmed +12–14pp at 2-bit** (table above);
      (c) local-proxy anisotropic — **measured, it hurts, rejected**.
      Remaining: (c') correct ScaNN anisotropic; (d) E8 lattice reference;
      (e) residual-correction byte; (f) Kc=16 for the nibble-LUT path.
      Stretch target: d=128 2-bit raw → >0.75 (now 0.60; needs e/c'/rerank).
- [x] Phase 4 — Integrated a scalar-TQ `quant="tq"` mode in
      `dynamic_ivf.cpp` (`TQCluster`: codes + scale + ‖v‖²), wired
      `python/core/__init__.py`, rebuilt `.so`, and passed smoke/stress tests.
      Remaining integration work is to upgrade this live path toward block-VQ /
      decoded-split variants rather than to land TQ itself.
- [x] Phase 5 — SIMD nibble-LUT fast-scan kernel for the scalar-TQ path
      (`src/tq_fastscan.hpp`): NEON (`vqtbl1q_u8`) + AVX2 (`pshufb`) with a
      scalar-block fallback, runtime arch dispatch, blocked `[block][dim][32]`
      code layout in `TQCluster`. Bit-exact with the scalar scorer; ~10–25×
      faster (BENCHMARKS.md §6). Default for `tq_bits<=4`. The Kc=256 block-VQ
      scorer stays a gather (256 codes exceed a register table) — a future
      Kc=16 nibble variant could reuse this kernel.
- [x] Phase 6 — Benchmarked vs the removed baseline in `benchmarks/`, updated
      docs, and culled the old path after the TQ path passed the same suite.
