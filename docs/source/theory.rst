.. _theory:

==================================================
Quantization theory: from TurboQuant to block VQ
==================================================

This chapter develops, from first principles, the quantization mathematics
behind Copenhagen's compressed search path. It explains why the scalar
TurboQuant scheme [TurboQuant]_ (as shipped in TurboVec [TurboVec]_) is near-optimal in high
dimension and *degrades in low dimension*, isolates the precise mechanism of
that degradation, and derives the block / sub-vector vector-quantization (VQ)
scheme Copenhagen uses to repair it. All claims are accompanied by the
experiments that test them (see :ref:`benchmarks`).

.. contents::
   :local:
   :depth: 2


Problem setup and notation
==========================

We index a set of database vectors :math:`\{v_i\}_{i=1}^{n}\subset\mathbb
R^{d}` and answer queries :math:`q\in\mathbb R^{d}`. Write
:math:`u = v/\lVert v\rVert` for the unit direction of a vector and
:math:`\langle\cdot,\cdot\rangle` for the Euclidean inner product. For
normalized data the three common similarity rankings coincide,

.. math::
   :label: equiv

   \operatorname*{arg\,max}_i \langle q, v_i\rangle
   \;=\;
   \operatorname*{arg\,max}_i \cos(q,v_i)
   \;=\;
   \operatorname*{arg\,min}_i \lVert q - v_i\rVert^2 ,
   \qquad \lVert v_i\rVert = 1,

because :math:`\lVert q-v\rVert^2 = \lVert q\rVert^2 + \lVert v\rVert^2 -
2\langle q,v\rangle`. This identity is what lets a maximum-inner-product (MIPS)
index such as TurboVec and an :math:`L_2` index such as Copenhagen or FAISS be
compared on the *same* ground truth once the data is L2-normalized; it is the
basis of the fairness convention used throughout :ref:`benchmarks`.

A scalar/vector quantizer replaces each :math:`v_i` by a short code
:math:`c_i\in\{0,\dots,K-1\}^{m}` plus :math:`O(1)` scalars, and answers queries
by an *estimator* :math:`\widehat{\langle q,v_i\rangle}` computed from
:math:`q` and :math:`c_i` alone. Two quantizers are compared by **recall at a
fixed code length** (recall-per-byte): the quantity that matters is not raw
distortion but how often the estimator preserves the true top-:math:`k`.


The TurboQuant pipeline
=======================

TurboQuant [TurboQuant]_ is the composition of four classical ideas. We restate
each and give the exact estimator Copenhagen reimplements.

1. Normalize and rotate
-----------------------

Let :math:`R\in O(d)` be a fixed random orthogonal matrix (Copenhagen generates
it by modified Gram–Schmidt on a seeded Gaussian matrix; TurboVec uses a QR
factorization — both yield a Haar-distributed rotation). Set

.. math::

   u = v/\lVert v\rVert,\qquad u_{\mathrm{rot}} = R\,u .

Because :math:`R` is orthogonal, :math:`\lVert u_{\mathrm{rot}}\rVert = 1` and
inner products are preserved: :math:`\langle q,v\rangle =
\lVert v\rVert\,\langle Rq, u_{\mathrm{rot}}\rangle`. The point of the rotation
is purely statistical: it makes the coordinates of :math:`u_{\mathrm{rot}}`
*exchangeable* with a known marginal.

.. _beta-marginal:

2. The coordinate marginal is Beta
----------------------------------

For a vector drawn uniformly on the sphere :math:`S^{d-1}`, each coordinate
:math:`x_j` has the density of a (shifted, scaled) Beta law. Concretely
:math:`(x_j+1)/2 \sim \mathrm{Beta}\!\left(\tfrac{d-1}{2},\tfrac{d-1}{2}\right)`
on :math:`[0,1]`, so :math:`x_j\in[-1,1]` with

.. math::
   :label: betamarg

   \mathbb E[x_j] = 0,\qquad
   \operatorname{Var}(x_j) = \frac1d,\qquad
   x_j \xrightarrow[d\to\infty]{} \mathcal N\!\left(0,\tfrac1d\right).

A random rotation turns *arbitrary* unit data into data whose one-dimensional
marginals match :eq:`betamarg`. This is what makes a single shared scalar
codebook appropriate for every coordinate.

3. Lloyd–Max scalar quantization
--------------------------------

A scalar quantizer with :math:`K=2^{b}` levels is chosen to minimize expected
squared error against the marginal of :eq:`betamarg`. The optimal
reconstruction points :math:`\{t_\ell\}` and decision boundaries
:math:`\{\beta_\ell\}` satisfy the Lloyd–Max fixed point [Lloyd1982]_,
[Max1960]_,

.. math::
   :label: lloyd

   \beta_\ell = \tfrac12(t_\ell + t_{\ell+1}),\qquad
   t_\ell = \mathbb E\!\left[X \,\middle|\, X\in(\beta_{\ell-1},\beta_\ell]\right].

TurboVec solves :eq:`lloyd` against the analytic Beta density. Copenhagen
instead fits :eq:`lloyd` to the **empirical pooled rotated coordinates** of the
training batch (1-D :math:`k`-means). This is dependency-free (no incomplete-Beta
special function), and it is *data-adaptive*: when the rotated data deviates
from the ideal Beta — which is exactly the anisotropic, low-:math:`d` regime of
interest — the empirical fit tracks the true marginal.

.. _renorm:

4. Length-renormalized inner-product estimator
----------------------------------------------

Let :math:`\hat x = (t_{c_1},\dots,t_{c_d})` be the per-coordinate
reconstruction of :math:`u_{\mathrm{rot}}`. TurboQuant stores **one scalar per
vector**,

.. math::
   :label: scale

   s \;=\; \frac{\lVert v\rVert}{\langle u_{\mathrm{rot}},\,\hat x\rangle},

and estimates inner products by

.. math::
   :label: ipest

   \widehat{\langle q, v\rangle} \;=\; s\,\langle q_{\mathrm{rot}},\,\hat x\rangle,
   \qquad q_{\mathrm{rot}} = R q .

This is the RaBitQ construction [RaBitQ]_: dividing by
:math:`\langle u_{\mathrm{rot}},\hat x\rangle` makes :eq:`ipest` an *unbiased*
estimator of :math:`\langle q,v\rangle` under the random rotation, with a
variance that concentrates as :math:`d` grows. For Copenhagen's :math:`L_2`
ranking we additionally store :math:`\lVert v\rVert^2` and reconstruct

.. math::
   :label: l2est

   \widehat{\lVert q - v\rVert^2} = \lVert q\rVert^2 + \lVert v\rVert^2
   - 2\,\widehat{\langle q,v\rangle}.

The estimator :eq:`ipest` is a sum of :math:`d` terms; with a per-coordinate
table :math:`T[j,\ell] = (q_{\mathrm{rot}})_j\, t_\ell` it is a single gather-add
pass over the code — the structure that makes the SIMD nibble-LUT kernels of
TurboVec (and Copenhagen's planned NEON kernel) possible.


Per-coordinate calibration (TQ+)
================================

Real data is anisotropic: even after rotation, coordinate :math:`j` may not
match the canonical marginal :eq:`betamarg`. TurboVec's **TQ+** fits two scalars
per coordinate, a shift :math:`\sigma_j` and a scale :math:`\gamma_j`, mapping
the empirical 5/95 % quantiles of coordinate :math:`j` onto the canonical
quantiles, then quantizes :math:`(u_{\mathrm{rot}})_j' = (u_{\mathrm{rot},j} +
\sigma_j)\gamma_j`. The inverse is applied to the query and a per-query bias
:math:`-\langle q_{\mathrm{rot}}, \sigma\rangle` folded into :eq:`ipest`, so the
kernel is unchanged. Copenhagen implements the same correction
(:file:`src/turbo_quant.hpp`). It is a first-order fix to the *marginals*; the
next section shows the dominant low-:math:`d` error is in the *joint*
distribution, which TQ+ cannot touch.


.. _lowd:

Why low dimension breaks scalar quantization
=============================================

The asymptotic justification of TurboQuant rests on :eq:`betamarg`: the
coordinates become Gaussian *and* the estimator :eq:`ipest` concentrates. Three
effects degrade at small :math:`d`, and they are routinely conflated:

**(i) Non-Gaussian marginal.** :math:`\mathrm{Beta}(a,a)` with
:math:`a=(d-1)/2` small is platykurtic (U-shaped for :math:`d=2,3`). This is the
part TurboQuant *handles*: Lloyd–Max on the exact Beta — or our empirical fit —
already matches the marginal. It is not the main loss.

**(ii) Coordinate dependence — the dominant term.** A unit vector obeys
:math:`\sum_j x_j^2 = 1`. This single constraint forces the coordinates to be
*negatively correlated*. For the uniform distribution on :math:`S^{d-1}`,

.. math::
   :label: cov

   \operatorname{Var}(x_j) = \frac1d,\qquad
   \operatorname{Cov}(x_j,x_k) = -\frac{1}{d^2(d-1)}\;\text{ shaped }\;O\!\left(\tfrac1d\right)
   \ \ (j\neq k),

with the exact identity :math:`\sum_{k\ne j}\operatorname{Cov}(x_j,x_k) =
-\operatorname{Var}(x_j) = -1/d` obtained by differentiating the constraint
:math:`\sum_k x_k^2 = 1`. The pairwise correlation is :math:`O(1/d)`:
negligible at :math:`d=1536`, but a first-order effect at :math:`d=128`.
Per-coordinate scalar quantization **assumes independence** and discards exactly
this structure.

**(iii) Loss of averaging.** The estimator :eq:`ipest` is a sum of :math:`d`
error terms. Its variance contracts like :math:`O(1/d)` *only if those errors
are independent*; the central-limit concentration that keeps high-:math:`d`
recall high at 2 bits weakens when the terms are both fewer and dependent.

The diagnosis is therefore: **TurboQuant optimizes the marginal and ignores the
joint.** Low dimension is where the joint matters. The experiments in
:ref:`benchmarks` confirm the signature — scalar recall *rises monotonically with
dimension*, from 0.45 at :math:`d{=}128` to 0.66 at :math:`d{=}1536` at 2 bits.


.. _blockvq:

Block (sub-vector) vector quantization
======================================

The fix is to quantize the **joint** distribution of small groups of
coordinates. Partition the rotated vector into :math:`m=d/B` contiguous blocks
of :math:`B` coordinates, :math:`u_{\mathrm{rot}} = (u^{(1)},\dots,u^{(m)})`, and
learn a separate :math:`K`-entry codebook :math:`C_g\subset\mathbb R^{B}` per
block by :math:`k`-means. Each block is encoded by its nearest codeword
:math:`c_g = C_g[\,\kappa_g]`, the reconstruction is
:math:`\hat x = (c_1,\dots,c_m)`, and the **estimator is unchanged** —
:eq:`scale`–:eq:`l2est` apply verbatim with

.. math::

   \langle q_{\mathrm{rot}}, \hat x\rangle = \sum_{g=1}^{m}
     \big\langle q^{(g)}_{\mathrm{rot}},\, C_g[\kappa_g]\big\rangle
   \;=\; \sum_{g=1}^m \mathrm{LUT}_g[\kappa_g].

This is, deliberately, **Product Quantization's data layout** [PQ2011]_ — a
sub-vector maps to one code maps to one LUT entry, the structure SIMD scoring is
built around — combined with **TurboQuant's rotation + renormalization
front-end** :eq:`scale`–:eq:`ipest`. It is a strict superset of both endpoints:

* :math:`B=1` recovers scalar TurboQuant exactly;
* dropping the rotation and renormalization recovers raw-:math:`L_2` PQ.

At a matched bit budget (:math:`B=4`, :math:`K=256` is 8 bits per 4 coordinates =
2 bits/coordinate, identical bytes/vector to scalar 2-bit) the block codebook can
represent the :math:`O(1/d)` cross-coordinate covariance :eq:`cov` that scalar
quantization cannot. The measured gain is largest exactly where the theory says
it should be — at 2 bits and low :math:`d`:

.. list-table:: Raw recall@10, matched bytes/vector (synthetic, :ref:`benchmarks`)
   :header-rows: 1

   * - :math:`d`
     - rate
     - scalar (:math:`B{=}1`)
     - block VQ
     - :math:`\Delta`
   * - 128
     - 2-bit
     - 0.4581
     - **0.5806** (:math:`B{=}4`)
     - **+12.3 pp**
   * - 128
     - 4-bit
     - 0.8183
     - 0.8412 (:math:`B{=}2`)
     - +2.3 pp
   * - 768
     - 2-bit
     - 0.5990
     - **0.7288** (:math:`B{=}4`)
     - **+13.0 pp**

We refer to the block scheme as *adaptive binning*: the codebook learns the
shape of the joint cell where scalar quantization can only place axis-aligned
grids. The :math:`E_8` lattice [ConwaySloane]_ is the elegant zero-training
alternative for :math:`B=8` (optimal Gaussian quantizer, :math:`\sim 1`
bit/coordinate gain, closed-form decode) and is the natural next reference point.
A *learned* rotation that also decorrelates the data — Optimized Product
Quantization [OPQ2013]_ — is the natural second-order refinement on top of block
VQ (a random rotation only equalizes the marginals; OPQ additionally rotates to
align the data with the block structure).

.. admonition:: Implementation status
   :class: note

   Implemented and measured: the random rotation, scalar Lloyd–Max with TQ+
   calibration (:file:`src/turbo_quant.hpp`), and trained block VQ
   (:file:`src/block_quant.hpp`), exposed to Python as a flat compressed index
   with :math:`O(1)` tombstone delete (:file:`src/block_quant_py.cpp`,
   ``block_vq.BlockVQIndex``). The ScaNN anisotropic codebook was implemented,
   found not to help, and **culled to the** ``experiments/anisotropic`` **branch**.
   **Not implemented:** the :math:`E_8` lattice and OPQ (named as next reference
   points, not results), and wiring the quantizer into
   :file:`src/dynamic_ivf.cpp` as a first-class ``quant`` mode replacing IVFPQ
   (the IVF + block-VQ combination is currently exercised through
   ``BlockVQIndex``).

The bit budget at low dimension is *not* the constraint: :math:`d=128` at 4 bits
is only 64 bytes, so 6–8 effective bits/coordinate via block VQ still sits at
24–32× compression versus float32 — the rate–distortion room scalar TurboQuant
left on the table is precisely in the low-:math:`d` corner.


.. _anisotropy:

Anisotropic (MIPS-aware) loss — a negative result worth stating
===============================================================

ScaNN [ScaNN2020]_ observes that for MIPS the quantization residual
:math:`r = u_{\mathrm{rot}} - \hat x` decomposes into a part *parallel* to the
datapoint and a part orthogonal, and that the parallel part distorts
:math:`\langle q,\hat x\rangle` far more in expectation over :math:`q`. It
therefore minimizes the weighted loss

.. math::
   :label: aniso

   L(u_{\mathrm{rot}},\hat x) = \lVert r_\perp\rVert^2 + \eta\,\lVert r_\parallel\rVert^2
   = \lVert r\rVert^2 + (\eta-1)\,(u_{\mathrm{rot}}^\top r)^2,
   \qquad \eta > 1,

using :math:`\lVert u_{\mathrm{rot}}\rVert = 1`. Because the coupling is only
through the scalar :math:`s = u_{\mathrm{rot}}^\top r = \sum_g u^{(g)\top}r^{(g)}`,
:eq:`aniso` admits block-coordinate-descent assignment and a closed-form
:math:`B\times B` weighted-least-squares codebook update
(:math:`A = NI + (\eta-1)\sum x x^\top`, solved by Cholesky since :math:`A` is
SPD for :math:`\eta\ge 1`). Copenhagen implements this exactly
(:file:`src/block_quant.hpp`).

**The measured result is that** :math:`\eta>1` **does not help and large**
:math:`\eta` **hurts** (e.g. :math:`d{=}128`, 2-bit: :math:`0.5987` at
:math:`\eta{=}1` falls to :math:`0.5682` at :math:`\eta{=}100`). The reason is
specific to our pipeline: the length-renormalization :eq:`scale` *already divides
out the parallel/norm component of the reconstruction error*. ScaNN's
:math:`\eta>1` exists for indexes that do **not** renormalize; stacked on top of
RaBitQ-style renormalization it double-corrects and wastes codebook resolution.
What remains to govern ranking is the **orthogonal/angular** accuracy of
:math:`\hat x`, which points to :math:`\eta<1`.

A single seed suggested a small gain there; a **5-seed significance test**
(:file:`src/tq_aniso_signif.cpp`, branch ``experiments/anisotropic``) settles it:
at :math:`\eta{=}0.25` the gain is :math:`+0.0028\pm0.0017` at :math:`d{=}128`
(marginal, :math:`\sim1.6\sigma`) and :math:`-0.0035\pm0.0026` at :math:`d{=}768`
(net **negative**, 4/5 seeds). **The anisotropic codebook does not earn its
training cost and was culled from main**; the full closed-form implementation is
preserved on the ``experiments/anisotropic`` branch. The practical takeaway for
the bit budget stands: **spend bits on joint structure (block VQ), not on
parallel reweighting** — once you renormalize, the parallel direction is already
handled.


Integration with the dynamic index
===================================

The quantizer is global (one rotation, one codebook, fit once at ``train``) and
the per-vector code is fixed-width, so it composes with Copenhagen's dynamic IVF
core [Copenhagen]_ without special cases:

* **Insert / delete** store / tombstone fixed-width codes — :math:`O(1)`,
  unchanged.
* **Adaptive cluster splitting** runs :math:`k`-means on vectors *decoded* from
  their codes (``decode`` reconstructs :math:`v \approx \lVert v\rVert\,
  R^{\top}\hat x/\lVert\hat x\rVert`); split centroids are approximate, which is
  acceptable since cluster centroids are themselves approximate. No float
  side-store is required, so compression is real (unlike the current IVFPQ path,
  which retains float32 — see :ref:`benchmarks`).
* **Optional rerank** can use either the decoded vectors (modest lift) or an
  opt-in float store (near-exact, costs memory).

This is the construction by which Copenhagen aims to keep its dynamics advantage
over FAISS/HNSW *and* acquire TurboVec-class compression — with the low-:math:`d`
block-VQ improvement that vanilla TurboVec lacks.


References
==========

.. [TurboQuant] A. Zandieh et al., "TurboQuant: Online Vector Quantization with
   Near-optimal Distortion Rate," arXiv:2504.19874, Google & NYU, ICLR 2026.
   https://arxiv.org/abs/2504.19874 — random rotation → Beta-concentrated
   coordinates → per-coordinate optimal scalar quantizer (relying explicitly on
   the *near-independence of coordinates in high dimension* that
   :ref:`lowd` dissects), plus a 1-bit QJL residual stage. The algorithm
   reimplemented and extended here.

.. [TurboVec] R. Codrai, *TurboVec*: a vector-search index implementing
   TurboQuant. https://github.com/RyanCodrai/turbovec — the compression baseline
   in :ref:`benchmarks`.

.. [RaBitQ] J. Gao and C. Long, "RaBitQ: Quantizing High-Dimensional Vectors
   with a Theoretical Error Bound for Approximate Nearest Neighbor Search,"
   *Proc. ACM SIGMOD*, 2024. (Length-renormalized unbiased inner-product
   estimator, :eq:`scale`–:eq:`ipest`.)

.. [ScaNN2020] R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern,
   and S. Kumar, "Accelerating Large-Scale Inference with Anisotropic Vector
   Quantization," *Proc. ICML*, 2020. (Anisotropic / MIPS-aware loss
   :eq:`aniso`.)

.. [PQ2011] H. Jégou, M. Douze, C. Schmid, "Product Quantization for Nearest
   Neighbor Search," *IEEE TPAMI* 33(1), 2011. (Sub-vector codebooks + LUT
   scoring.)

.. [OPQ2013] T. Ge, K. He, Q. Ke, J. Sun, "Optimized Product Quantization,"
   *Proc. CVPR*, 2013. (Learned rotation balancing decorrelation vs. marginal
   uniformity — the second-order lever on top of block VQ.)

.. [Lloyd1982] S. P. Lloyd, "Least Squares Quantization in PCM," *IEEE Trans.
   Information Theory* 28(2), 1982.

.. [Max1960] J. Max, "Quantizing for Minimum Distortion," *IRE Trans.
   Information Theory* 6(1), 1960.

.. [ConwaySloane] J. H. Conway and N. J. A. Sloane, *Sphere Packings, Lattices
   and Groups*, 3rd ed., Springer, 1999. (:math:`D_4`, :math:`E_8` optimal
   quantizing lattices.)

.. [Copenhagen] T. van der Hoog, M. Reinstädtler, E. Rotenberg et al.,
   "Engineering Fully Dynamic Convex Hulls" / fully-dynamic logarithmic-method
   bucket structures, IT University of Copenhagen (arXiv:2604.00271). Source of
   Copenhagen's tombstone-correctness and bucket design.
