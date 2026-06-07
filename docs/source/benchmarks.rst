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

``reproduce.py`` builds the C++ extension and the quantization microbenchmark
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
  one without the other is meaningless. For the removed PQ baseline and why it
  was culled, see ``IVFPQ.md`` in the repository root.


Dynamics — vs FAISS IVF and HNSW
================================

.. code-block:: bash

   python3 benchmarks/benchmark_ivf_churn.py     [--with-turbovec]
   python3 benchmarks/benchmark_hnsw_churn.py    [--with-turbovec]

50 000 initial vectors, 10 rounds, +1 000 inserts and 30 % oldest deleted per
round, :math:`d=128`. Representative final-round result from
``REPORT_20260607_000220`` on Linux x86_64:

.. list-table:: Recall@10 and throughput at ~93 % cumulative churn
   :header-rows: 1

   * - Method
     - recall@10
     - inserts/s
     - deletes/s
   * - **Copenhagen**
     - **0.93–0.95**
     - **~0.84-0.97 M**
     - **~1.49 M**
   * - FAISS IVF + filter
     - 0.64
     - —
     - (tombstone)
   * - FAISS IVF + rebuild
     - 0.80
     - ~339 k
     - (rebuild)
   * - HNSW + filter
     - 0.28
     - —
     - (tombstone)
   * - HNSW + rebuild
     - 0.92
     - ~8.3 k
     - (rebuild)

Copenhagen stays near rebuild-level recall while updating much faster, and never
goes offline.


Compression — Copenhagen vs TurboVec vs legacy baseline
=======================================================

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
     - 0.9971
     - 512
     - 1.0×
   * - Legacy compressed path
     - 0.6463
     - 528
     - *0.97× (larger than float in absolute bytes)*
   * - TurboVec 4-bit
     - 0.8476
     - 68
     - 7.5×
   * - TurboVec 2-bit
     - 0.6266
     - 36
     - 14.2×
   * - Copenhagen-TQ block VQ ``B=2``
     - 0.9070
     - 72
     - 7.1×
   * - Copenhagen-TQ block VQ ``B=4``
     - 0.7002
     - 40
     - 12.8×

The removed legacy compressed path was dominated on both axes. Details and
removal rationale are in ``IVFPQ.md``. TurboVec still wins bytes decisively in
this comparison; that is the motivation for continuing to improve the
integrated TurboQuant path.


Quantizer micro-benchmarks
==========================

These isolate the quantizer logic from the full dynamic index and measure
recall-per-byte against exact brute force. They are diagnostic harnesses, not a
separate product mode.

.. code-block:: bash

   c++ -O3 -std=c++17 -march=native src/tq_standalone_test.cpp -o /tmp/tq && /tmp/tq
   c++ -O3 -std=c++17 -march=native src/tq_block_test.cpp      -o /tmp/tqb && /tmp/tqb
   c++ -O3 -std=c++17 -march=native src/tq_aniso_test.cpp      -o /tmp/tqa && /tmp/tqa

* ``tq_standalone_test`` — scalar TurboQuant recall-per-byte across
  :math:`d\in\{128,768,1536\}`; shows recall *rising with dimension*
  (:ref:`lowd`) and that a rerank@200 reaches 0.987-1.000 in the current run.
* ``tq_block_test`` — block VQ vs scalar at **matched bytes/vector**; the
  current run shows +12.3 pp at 128d / 2-bit and +13.0 pp at 768d / 2-bit.
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
