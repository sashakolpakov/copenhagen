import os
import math
import tempfile

import numpy as np

from ..base.module import BaseANN


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class Copenhagen(BaseANN):
    def __init__(self, metric, dim, method_param):
        from python.core import CopenhagenIndex

        self._metric = metric
        self._dim = int(dim)
        self._method_param = dict(method_param)
        self._index_cls = CopenhagenIndex
        self._index = None

        self.cluster_factor = float(self._method_param.get("cluster_factor", 0.5))
        self.soft_k = int(self._method_param.get("soft_k", 1))
        self.split_threshold = float(self._method_param.get("split_threshold", 3.0))
        self.initial_nprobe = int(self._method_param.get("initial_nprobe", 8))
        self.max_clusters = int(self._method_param.get("max_clusters", 4096))
        self.min_clusters = int(self._method_param.get("min_clusters", 16))
        self.train_sample_size = int(self._method_param.get("train_sample_size", 250000))
        self.insert_batch_size = int(self._method_param.get("insert_batch_size", 50000))
        self.use_mmap = bool(self._method_param.get("use_mmap", False))
        self.force_mmap_threshold = int(self._method_param.get("force_mmap_threshold", 1_000_000))
        self._nprobe = self.initial_nprobe
        self._n_clusters = None
        self._name = "copenhagen(unfitted)"
        self._mmap_dir = None

    def _fit_n_clusters(self, n_rows: int) -> int:
        target = int(round(math.sqrt(max(n_rows, 1)) * self.cluster_factor))
        target = max(self.min_clusters, target)
        target = min(self.max_clusters, target, n_rows)
        target = max(target, self._nprobe)
        return target

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self._metric == "angular":
            X = _normalize_rows(X)

        n_rows = len(X)
        self._n_clusters = self._fit_n_clusters(len(X))
        use_mmap = self.use_mmap or n_rows >= self.force_mmap_threshold
        mmap_dir = ""
        if use_mmap:
            self._mmap_dir = tempfile.mkdtemp(prefix="annb-copenhagen-")
            mmap_dir = self._mmap_dir
        self._index = self._index_cls(
            dim=self._dim,
            n_clusters=self._n_clusters,
            nprobe=min(self._nprobe, self._n_clusters),
            soft_k=self.soft_k,
            use_mmap=use_mmap,
            mmap_dir=mmap_dir,
        )
        self._index._index.split_threshold = self.split_threshold
        train_rows = min(n_rows, self.train_sample_size)
        if train_rows < n_rows:
            sample_idx = np.linspace(0, n_rows - 1, num=train_rows, dtype=np.int64)
            train_data = np.ascontiguousarray(X[sample_idx])
        else:
            train_data = np.ascontiguousarray(X)
        self._index._index.train(train_data)

        for start in range(0, n_rows, self.insert_batch_size):
            batch = np.ascontiguousarray(X[start:start + self.insert_batch_size])
            self._index._index.insert_batch(batch)
        self._refresh_name()

    def _refresh_name(self):
        self.name = (
            f"copenhagen(n_clusters={self._n_clusters},nprobe={self._nprobe},"
            f"soft_k={self.soft_k},split_threshold={self.split_threshold})"
        )

    def set_query_arguments(self, nprobe):
        self._nprobe = int(nprobe)
        if self._n_clusters is not None:
            self._nprobe = min(self._nprobe, self._n_clusters)
        self._refresh_name()

    def query(self, v, n):
        q = np.asarray(v, dtype=np.float32).reshape(1, -1)
        if self._metric == "angular":
            q = _normalize_rows(q)
        ids, _ = self._index.search(q[0], n, n_probes=self._nprobe)
        return ids

    def get_additional(self):
        stats = self._index.get_stats()
        return {
            "soft_k": self.soft_k,
            "split_threshold": self.split_threshold,
            "n_clusters": self._n_clusters,
            "reported_n_clusters": stats.get("n_clusters"),
            "reported_n_vectors": stats.get("n_vectors"),
        }
