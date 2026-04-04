import numpy as np
import importlib.util
import json
import os

_dir = os.path.dirname(__file__)
_candidates = [f for f in os.listdir(_dir) if f.startswith('copenhagen') and f.endswith('.so')]
if not _candidates:
    raise ImportError(f"copenhagen extension not found in {_dir}. Run: pip install -e .")
# Prefer the most recently built .so (handles both generic and cpython-tagged names)
_candidates.sort(key=lambda f: os.path.getmtime(os.path.join(_dir, f)), reverse=True)
_module_path = os.path.join(_dir, _candidates[0])
_spec = importlib.util.spec_from_file_location("copenhagen", _module_path)
_copenhagen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_copenhagen)

DynamicIVF = _copenhagen.DynamicIVF


class CopenhagenIndex:
    """
    Copenhagen — Dynamic IVF index with O(1) delete, soft assignments, and adaptive splits.

    Optional GPU acceleration via PyTorch (install with `pip install copenhagen[gpu]`):
      idx = CopenhagenIndex(dim, n_clusters, device="cuda")   # NVIDIA
      idx = CopenhagenIndex(dim, n_clusters, device="mps")    # Apple M-series
      idx = CopenhagenIndex(dim, n_clusters, device="cpu")    # explicit CPU (default)

    When device is set, centroid distances for insert_batch are computed on the
    specified device via torch.mm, then pre-computed assignments are passed to the
    C++ layer via insert_batch_preassigned — skipping the CPU BLAS gemm entirely.
    Search remains on CPU (single-query centroid ranking is too small to benefit).
    """

    def __init__(self, dim, n_clusters, nprobe=1, use_pq=False, pq_m=8, pq_ks=256, soft_k=1,
                 use_mmap=False, mmap_dir="", device=None):
        """
        Args:
            dim: Dimension of the vectors
            n_clusters: Number of IVF clusters (Voronoi cells)
            nprobe: Number of clusters to search per query
            use_pq: Use Product Quantization for faster search
            pq_m: Number of PQ subspaces (higher = more accurate, slower)
            pq_ks: Number of centroids per subspace (256 = 8 bits)
            soft_k: Number of clusters each vector is indexed in (1 = standard IVF)
            use_mmap: Store cluster vectors in memory-mapped files (for large datasets)
            mmap_dir: Directory for mmap files (required when use_mmap=True)
        """
        if use_mmap and not mmap_dir:
            raise ValueError("mmap_dir must be set when use_mmap=True")
        self.dim = dim
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.use_pq = use_pq

        if use_mmap:
            os.makedirs(mmap_dir, exist_ok=True)

        self._index = DynamicIVF(dim, n_clusters, nprobe, 1 if use_pq else 0,
                                 pq_m, pq_ks, soft_k, use_mmap, mmap_dir)

        # GPU state — None means CPU-only path
        self._device       = device          # e.g. "cuda", "mps", "cpu", or None
        self._centroids_t  = None            # centroids pinned on device (set after train)
    
    def _pin_centroids(self):
        """Copy centroids to the target device after training."""
        import torch
        c = np.ascontiguousarray(self._index.get_centroids())   # (k, d)
        self._centroids_t = torch.from_numpy(c).to(self._device)

    def _gpu_assign(self, vectors):
        """Compute top-soft_k cluster assignments on device; return (n, soft_k) int32 array."""
        import torch
        v = torch.from_numpy(np.ascontiguousarray(vectors)).to(self._device)  # (n, d)
        c = self._centroids_t                                                  # (k, d)
        # L2²: ||v - c||² = ||v||² + ||c||² - 2·v·cᵀ
        v_sq = (v * v).sum(dim=1, keepdim=True)          # (n, 1)
        c_sq = (c * c).sum(dim=1, keepdim=True).T        # (1, k)
        dists = v_sq + c_sq - 2.0 * torch.mm(v, c.T)    # (n, k)
        k = min(self._index.soft_k, self._index.get_stats()["n_clusters"])
        _, idx = torch.topk(dists, k, dim=1, largest=False, sorted=True)
        return idx.cpu().numpy().astype(np.int32)         # (n, k)

    def add(self, vectors):
        """
        Add vectors to the index (incremental insert).

        Args:
            vectors: numpy array of shape (n, dim) or (dim,)
        """
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")

        if self._index.get_stats()["n_vectors"] == 0:
            self._index.train(vectors)
            if self._device is not None:
                self._pin_centroids()
        else:
            if self._device is not None and self._centroids_t is not None:
                assignments = self._gpu_assign(vectors)
                self._index.insert_batch_preassigned(vectors, assignments)
                # Re-pin if splits added new centroids
                if self._index.get_stats()["n_clusters"] != len(self._centroids_t):
                    self._pin_centroids()
            else:
                self._index.insert_batch(vectors)
    
    def delete(self, ids):
        """
        Delete vectors by ID.
        
        Args:
            ids: int or numpy array of int IDs to delete
        """
        if isinstance(ids, int):
            self._index.delete(ids)
        else:
            ids = np.asarray(ids, dtype=np.int32)
            self._index.delete_batch(ids)
    
    def search(self, query, k=10, n_probes=None, scan_limit=0):
        """
        Search for k nearest neighbors.
        
        Args:
            query: numpy array of shape (dim,) or (n_queries, dim)
            k: Number of results to return
            n_probes: Number of clusters to scan (default: self.nprobe)
            scan_limit: Max total vectors to scan (0 = scan all in selected clusters)
            
        Returns:
            If query is 1D: tuple of (ids, distances)
            If query is 2D: list of tuples of (ids, distances)
        """
        query = np.asarray(query, dtype=np.float32)
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        if query.shape[1] != self.dim:
            raise ValueError(f"Query dimension mismatch: expected {self.dim}, got {query.shape[1]}")
        
        if n_probes is None:
            n_probes = self.nprobe
        
        # Use batch search for better performance via BLAS gemm
        if len(query) > 1:
            return self.search_batch(query, k, n_probes)
        
        ids, dists = self._index.search(query[0], k, n_probes)
        return (ids, dists)
    
    def search_batch(self, queries, k=10, n_probes=None):
        """
        Batch search using BLAS gemm for centroid distances.
        
        Args:
            queries: numpy array of shape (n_queries, dim)
            k: Number of results per query
            n_probes: Number of clusters to scan
            
        Returns:
            List of (ids, distances) tuples
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        
        if n_probes is None:
            n_probes = self.nprobe
        
        return self._index.search_batch(queries, k, n_probes)
    
    def brute_force_search(self, query, k=10):
        """
        Brute force search for benchmark/comparison.
        
        Args:
            query: numpy array of shape (dim,)
            k: Number of results
            
        Returns:
            Tuple of (ids, distances)
        """
        query = np.asarray(query, dtype=np.float32)
        return self._index.brute_force_search(query, k)
    
    @property
    def n_vectors(self):
        """Number of vectors in the index."""
        return self._index.get_stats()["n_vectors"]
    
    def get_stats(self):
        """Return all stats."""
        return self._index.get_stats()

    def get_cluster_stats(self):
        """
        Return per-cluster breakdown as a list of dicts, one per cluster:
          cluster_id, live_size, physical_size, centroid (numpy array), last_split_round.

        last_split_round is -1 for clusters present at training time, and the
        insert-batch round number for clusters created by adaptive splits.
        Useful for monitoring imbalance and visualising split history.
        """
        return self._index.get_cluster_stats()

    @property
    def stats(self):
        """Return all stats."""
        return self._index.get_stats()

    def insert_stream(self, generator, chunk_size=1000):
        """
        Insert vectors from a generator without materialising the full batch.

        Args:
            generator: yields numpy arrays of shape (dim,) or (n, dim)
            chunk_size: number of vectors to accumulate before each insert_batch call

        Returns:
            (first_id, last_id) — inclusive range of auto-assigned IDs
        """
        first_id = None
        chunk = []
        for item in generator:
            vec = np.asarray(item, dtype=np.float32)
            if vec.ndim == 1:
                chunk.append(vec)
            else:
                chunk.extend(vec)
            if len(chunk) >= chunk_size:
                batch = np.stack(chunk)
                if first_id is None:
                    first_id = self._index.get_stats()["n_vectors"]
                self.add(batch)
                chunk.clear()
        if chunk:
            batch = np.stack(chunk)
            if first_id is None:
                first_id = self._index.get_stats()["n_vectors"]
            self.add(batch)
        if first_id is None:
            return None
        last_id = self._index.get_stats()["n_vectors"] - 1
        return (first_id, last_id)
    
    def compact(self):
        """
        Flush all tombstones: physically evict deleted vectors from every cluster
        and clear the deleted_ids set.

        Called automatically by insert_batch when deleted_ids exceeds 10% of
        n_vectors. Call manually after a bulk delete to reclaim memory immediately.
        """
        self._index.compact()

    def save(self, path):
        """
        Save index to a directory.

        Heap mode creates:
          <path>/clusters.npz   — centroids + all cluster vectors/ids
          <path>/metadata.json  — scalar state, deleted_ids, id_to_location

        mmap mode creates:
          <mmap_dir>/cluster_N.bin  — vector data (already persisted via MAP_SHARED)
          <path>/clusters_meta.npz  — centroids + ids only (no vectors)
          <path>/metadata.json      — scalar state, deleted_ids, id_to_location
        """
        os.makedirs(path, exist_ok=True)
        idx   = self._index
        stats = idx.get_stats()
        nc    = stats["n_clusters"]
        is_mmap = idx.use_mmap

        arrays = {"centroids": idx.get_centroids()}
        for c in range(nc):
            if not is_mmap:
                arrays[f"vecs_{c}"] = idx.get_cluster_vectors(c)
            arrays[f"ids_{c}"] = idx.get_cluster_ids(c)

        npz_name = "clusters_meta.npz" if is_mmap else "clusters.npz"
        np.savez(os.path.join(path, npz_name), **arrays)

        meta = {
            "dim":            self.dim,
            "n_clusters":     nc,
            "n_vectors":      stats["n_vectors"],
            "nprobe":         self.nprobe,
            "use_pq":         self.use_pq,
            "soft_k":         int(idx.soft_k),
            "split_threshold":float(idx.split_threshold),
            "max_split_iters":int(idx.max_split_iters),
            "use_mmap":       is_mmap,
            "mmap_dir":       idx.mmap_dir if is_mmap else "",
            "deleted_ids":    idx.get_deleted_ids_list(),
            "id_to_location": [list(map(list, locs)) for locs in idx.get_id_to_location()],
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path):
        """Load index from a directory created by save()."""
        with open(os.path.join(path, "metadata.json")) as f:
            meta = json.load(f)

        is_mmap  = meta.get("use_mmap", False)
        mmap_dir = meta.get("mmap_dir", "")

        obj = cls.__new__(cls)
        obj.dim        = meta["dim"]
        obj.n_clusters = meta["n_clusters"]
        obj.nprobe     = meta["nprobe"]
        obj.use_pq     = meta["use_pq"]

        obj._index = DynamicIVF(
            meta["dim"], meta["n_clusters"], meta["nprobe"],
            1 if meta["use_pq"] else 0,
            meta.get("pq_m", 8), meta.get("pq_ks", 256),
            meta["soft_k"],
            is_mmap, mmap_dir,
            False,  # truncate_mmap_files=False: existing .bin files hold valid data
        )
        obj._index.split_threshold = meta["split_threshold"]
        obj._index.max_split_iters = meta["max_split_iters"]

        if is_mmap:
            # Vectors are in cluster_N.bin files (mmap'd by constructor).
            # Load only centroids + ids from clusters_meta.npz.
            npz = np.load(os.path.join(path, "clusters_meta.npz"))
            obj._index.set_centroids(npz["centroids"])
            for c in range(meta["n_clusters"]):
                ids = npz[f"ids_{c}"]
                obj._index.restore_cluster_mmap(c, ids)
        else:
            npz = np.load(os.path.join(path, "clusters.npz"))
            obj._index.set_centroids(npz["centroids"])
            for c in range(meta["n_clusters"]):
                vecs = npz[f"vecs_{c}"]
                ids  = npz[f"ids_{c}"]
                if len(ids) > 0:
                    obj._index.restore_cluster(c, vecs, ids)

        id2loc = [[tuple(p) for p in locs] for locs in meta["id_to_location"]]
        obj._index.restore_state(meta["deleted_ids"], id2loc, meta["n_vectors"])

        return obj

    def __repr__(self):
        stats = self._index.get_stats()
        return f"CopenhagenIndex(dim={self.dim}, n_clusters={self.n_clusters}, n_vectors={stats['n_vectors']})"
