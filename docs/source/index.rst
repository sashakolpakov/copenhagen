==========================================
Copenhagen — dynamic ANN with compression
==========================================

**Copenhagen** is a fully dynamic approximate-nearest-neighbour (ANN) index built
on an Inverted File (IVF) structure. It sustains continuous inserts and deletes
without ever rebuilding — :math:`O(1)` amortized insert, :math:`O(1)` tombstone
delete, soft multi-cluster assignment, and adaptive cluster splitting — and it is
acquiring a compressed search path based on the TurboQuant scalar quantizer,
extended with **block / sub-vector vector quantization** that repairs
TurboQuant's low-dimension weakness.

The design goal, stated plainly:

   *Copenhagen + an integrated TurboQuant-style quantizer + adaptive binning at
   low dimension should beat FAISS and HNSW on dynamics, and beat vanilla
   TurboVec on compression — simultaneously.*

This documentation explains the mathematics behind that claim and how to
reproduce the evidence.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   theory
   benchmarks

Where the pieces live
======================

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Role
   * - ``src/dynamic_ivf.cpp``
     - The dynamic IVF index (insert/delete/split, BLAS search).
   * - ``src/turbo_quant.hpp``
     - Scalar TurboQuant: rotation, Lloyd–Max, TQ+, renormalized inner-product
       estimator (:math:`B{=}1` baseline).
   * - ``src/block_quant.hpp``
     - Block / sub-vector VQ (:ref:`blockvq`) + the ScaNN anisotropic codebook
       (:ref:`anisotropy`).
   * - ``benchmarks/reproduce.py``
     - One-shot: build, deps, download, run, write a report.
   * - ``benchmarks/benchmark_vs_turbovec.py``
     - Head-to-head vs TurboVec on all three axes.

The three-axis story
=====================

* **Dynamics (vs FAISS / HNSW).** Under 30 %/round churn, Copenhagen holds
  recall@10 ≈ 0.93–0.95 at ~1.7 M inserts/s and ~2.8 M deletes/s, while FAISS
  IVF+filter decays to 0.66 and HNSW+filter to 0.52; the rebuild variants match
  recall only at 10–150 k inserts/s. See :ref:`benchmarks`.
* **Compression (vs TurboVec / IVFPQ).** On normalized SIFT, TurboVec 4-bit
  reaches 0.85 recall@10 at 68 B/vec (7.5×) while Copenhagen's current IVFPQ path
  is *both larger (528 B) and worse (0.65)* — the gap the TurboQuant port closes.
* **Low dimension (the differentiator).** Scalar TurboQuant assumes coordinate
  independence, which the unit-sphere constraint violates at small :math:`d`
  (:ref:`lowd`). Block VQ recovers the joint structure: +12–14 pp recall at 2-bit
  on :math:`d{=}128` and :math:`d{=}768` at identical bytes/vector (:ref:`blockvq`).

Indices and tables
===================

* :ref:`genindex`
* :ref:`search`
