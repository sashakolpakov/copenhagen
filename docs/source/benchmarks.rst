.. _benchmarks:

==========================
Benchmarks & reproduction
==========================

Every number in :ref:`theory` is produced by a script in ``benchmarks/`` or
``src/``. This page describes the suite, the fairness conventions, and the
one-command reproduction.

.. contents::
   :local:


One command
===========

.. code-block:: bash

   python3 benchmarks/reproduce.py            # full suite, writes a report
   python3 benchmarks/reproduce.py --quick    # small sizes, fast smoke
   python3 benchmarks/reproduce.py --only compression

``reproduce.py`` builds the C++ extension and the standalone quantization
harnesses, installs the comparison libraries (``faiss-cpu``, ``hnswlib``,
``turbovec``, ``h5py``), downloads the ANN-benchmark datasets, runs the dynamics,
drift, and compression benchmarks, and writes a timestamped Markdown report with
all tables and figures to ``benchmarks/results/REPORT_*.md``.


Fairness conventions
=====================

* **Metric.** TurboVec ranks by inner product (MIPS); Copenhagen / FAISS / HNSW
  rank by :math:`L_2`. Whenever TurboVec is in the comparison the data is
  **L2-normalized**, so by :eq:`equiv` the rankings coincide and the ground truth
  is shared. (Normalizing barely moves the FAISS/HNSW/Copenhagen numbers.)
* **Deletes.** TurboVec is a flat, static index with no native delete; like
  FAISS-flat and HNSW it appears as a ``+rebuild`` baseline. Its dynamics story
  is insert/rebuild *throughput*, not delete.
* **Recall-per-byte.** Compression results report recall@10 *and* bytes/vector;
  one without the other is meaningless. Copenhagen's IVFPQ figure includes the
  retained float32 (its codes are stored *on top* of the full vectors), which is
  why its byte count exceeds float.


Dynamics — vs FAISS IVF and HNSW
================================

.. code-block:: bash

   python3 benchmarks/benchmark_ivf_churn.py     [--with-turbovec]
   python3 benchmarks/benchmark_hnsw_churn.py    [--with-turbovec]

50 000 initial vectors, 10 rounds, +1 000 inserts and 30 % oldest deleted per
round, :math:`d=128`. Representative final-round result (Apple M-series):

.. list-table:: Recall@10 and throughput at ~93 % cumulative churn
   :header-rows: 1

   * - Method
     - recall@10
     - inserts/s
     - deletes/s
   * - **Copenhagen**
     - **0.93–0.95**
     - **~1.7 M**
     - **~2.8 M**
   * - FAISS IVF + filter
     - 0.66
     - —
     - (tombstone)
   * - FAISS IVF + rebuild
     - 0.89
     - ~150 k
     - (rebuild)
   * - HNSW + filter
     - 0.52
     - —
     - (tombstone)
   * - HNSW + rebuild
     - 0.95
     - ~40 k
     - (rebuild)

Copenhagen matches the always-rebuild recall ceiling while inserting and deleting
60–100× faster, and never goes offline.


Compression — Copenhagen vs TurboVec vs IVFPQ
=============================================

.. code-block:: bash

   python3 benchmarks/benchmark_vs_turbovec.py            # normalized SIFT
   python3 benchmarks/benchmark_vs_turbovec.py --synthetic

.. list-table:: Static recall@10 vs bytes/vector, normalized SIFT-128 (50 k)
   :header-rows: 1

   * - Index
     - recall@10
     - bytes/vec
     - compression
   * - Copenhagen float
     - 0.995
     - 512
     - 1.0×
   * - Copenhagen IVFPQ (M=16)
     - 0.650
     - 528
     - *0.97× (larger!)*
   * - TurboVec 4-bit
     - 0.848
     - 68
     - 7.5×
   * - TurboVec 2-bit
     - 0.627
     - 36
     - 14.2×

IVFPQ is dominated on both axes; TurboVec wins bytes decisively. This is the
motivation for the TurboQuant port.


Quantizer micro-benchmarks (standalone C++)
===========================================

These isolate the quantizer from the index and measure recall-per-byte against
exact brute force.

.. code-block:: bash

   c++ -O3 -std=c++17 -march=native src/tq_standalone_test.cpp -o /tmp/tq && /tmp/tq
   c++ -O3 -std=c++17 -march=native src/tq_block_test.cpp      -o /tmp/tqb && /tmp/tqb
   c++ -O3 -std=c++17 -march=native src/tq_aniso_test.cpp      -o /tmp/tqa && /tmp/tqa

* ``tq_standalone_test`` — scalar TurboQuant recall-per-byte across
  :math:`d\in\{128,768,1536\}`; shows recall *rising with dimension*
  (:ref:`lowd`) and that a rerank@200 reaches ~0.99 at all bit widths.
* ``tq_block_test`` — block VQ vs scalar at **matched bytes/vector**; the
  +12–14 pp 2-bit gain of :ref:`blockvq`.
* ``tq_aniso_test`` — the :math:`\eta` sweep of :ref:`anisotropy`
  (:math:`\eta>1` hurts; :math:`\eta\approx0.25` helps slightly).


Datasets
========

Downloaded on demand from the `ann-benchmarks <http://ann-benchmarks.com>`_
project by ``benchmarks/download_data.py``:

.. list-table::
   :header-rows: 1

   * - Key
     - Dataset
     - Used by
   * - ``sift``
     - SIFT 1M × 128, Euclidean
     - insert scaling, vs-HNSW, vs-TurboVec
   * - ``mnist``
     - MNIST 60k × 784
     - drift
   * - ``fashion``
     - Fashion-MNIST 60k × 784
     - drift
