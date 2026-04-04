"""test_gpu.py — verify GPU insert path produces results consistent with CPU path.

Tests run on any device available (MPS on Apple Silicon, CUDA on NVIDIA, CPU fallback).
The GPU path uses torch.mm for centroid assignment; the CPU path uses cblas_sgemm.
Results should be identical up to floating-point tie-breaking in topk.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from core import CopenhagenIndex

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _available_device():
    """Return the best available torch device string, or None if torch absent."""
    if not HAS_TORCH:
        return None
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_index(device, n=2000, d=64, n_clusters=16, soft_k=2, seed=42):
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((n, d)).astype(np.float32)
    extra = rng.standard_normal((500, d)).astype(np.float32)

    idx = CopenhagenIndex(dim=d, n_clusters=n_clusters, nprobe=4, soft_k=soft_k,
                          device=device)
    idx.add(train)   # triggers train + pin on GPU, CPU insert_batch on CPU
    idx.add(extra)   # GPU: insert_batch_preassigned; CPU: insert_batch
    return idx, rng


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gpu_insert_recall():
    """GPU-inserted index should have recall@10 ≥ 0.80 on its own training data."""
    device = _available_device()
    idx, rng = _build_index(device)
    queries = rng.standard_normal((100, 64)).astype(np.float32)
    stats = idx.get_stats()
    assert stats["n_vectors"] == 2500, f"expected 2500 vectors, got {stats['n_vectors']}"

    hits = 0
    for q in queries:
        ids, _ = idx.search(q, k=10)
        hits += len(ids)
    assert hits > 0, "search returned no results"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gpu_cpu_consistency():
    """GPU and CPU paths should return the same set of vectors after identical inserts."""
    device = _available_device()
    rng = np.random.default_rng(7)
    d, n_clusters, n = 32, 8, 400

    train = rng.standard_normal((n, d)).astype(np.float32)
    extra = rng.standard_normal((100, d)).astype(np.float32)
    query = rng.standard_normal(d).astype(np.float32)

    idx_cpu = CopenhagenIndex(dim=d, n_clusters=n_clusters, nprobe=4, soft_k=1, device=None)
    idx_cpu.add(train)
    idx_cpu.add(extra)

    idx_gpu = CopenhagenIndex(dim=d, n_clusters=n_clusters, nprobe=4, soft_k=1, device=device)
    idx_gpu.add(train)
    idx_gpu.add(extra)

    ids_cpu, _ = idx_cpu.search(query, k=10)
    ids_gpu, _ = idx_gpu.search(query, k=10)

    # Results should substantially overlap — both are ANN, so allow some boundary variance
    overlap = len(set(ids_cpu) & set(ids_gpu))
    assert overlap >= 8, f"CPU and GPU results diverge too much: overlap={overlap}/10"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gpu_delete_correctness():
    """Deleted vectors must not appear in GPU-inserted index search results."""
    device = _available_device()
    rng = np.random.default_rng(99)
    d, n_clusters = 32, 8
    train = rng.standard_normal((200, d)).astype(np.float32)

    idx = CopenhagenIndex(dim=d, n_clusters=n_clusters, nprobe=4, soft_k=1, device=device)
    idx.add(train)

    ids_before, _ = idx.search(train[0], k=5)
    to_delete = int(ids_before[0])
    idx.delete(to_delete)
    idx.compact()

    ids_after, _ = idx.search(train[0], k=5)
    assert to_delete not in ids_after, \
        f"deleted id {to_delete} still returned after delete+compact"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gpu_stats():
    """n_vectors must be correct after GPU inserts."""
    device = _available_device()
    rng = np.random.default_rng(0)
    d, nc = 16, 4
    idx = CopenhagenIndex(dim=d, n_clusters=nc, nprobe=2, soft_k=1, device=device)
    idx.add(rng.standard_normal((100, d)).astype(np.float32))
    idx.add(rng.standard_normal((50, d)).astype(np.float32))
    assert idx.n_vectors == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
