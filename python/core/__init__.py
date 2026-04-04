import numpy as np
import importlib.util
import json
import os

_module_path = os.path.join(os.path.dirname(__file__), 'copenhagen.so')
_spec = importlib.util.spec_from_file_location("copenhagen", _module_path)
_copenhagen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_copenhagen)

DynamicIVF = _copenhagen.DynamicIVF


class CopenhagenIndex:
    """
    Copenhagen - Dynamic IVF index with quantum-inspired updates.
    
    Implements the logarithmic method for incremental cluster updates
    based on paper 2604.00271v1.pdf from the University of Copenhagen.
    
    The name reflects our inspiration from the Copenhagen interpretation
    of quantum mechanics - where observation (measurement) collapses
    the wavefunction (state space), and quantum superposition allows
    entities to exist in multiple states until observed.
    
    In our algorithm, vectors maintain "soft assignments" to multiple
    clusters (superposition), which collapse to the nearest cluster
    only when necessary (measurement/search).
    """
    
    def __init__(self, dim, n_clusters, nprobe=1, use_pq=False, pq_m=8, pq_ks=256, soft_k=1):
        """
        Args:
            dim: Dimension of the vectors
            n_clusters: Number of IVF clusters (Voronoi cells)
            nprobe: Number of clusters to search per query
            use_pq: Use Product Quantization for faster search
            pq_m: Number of PQ subspaces (higher = more accurate, slower)
            pq_ks: Number of centroids per subspace (256 = 8 bits)
        """
        self.dim = dim
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.use_pq = use_pq
        
        self._index = DynamicIVF(dim, n_clusters, nprobe, 1 if use_pq else 0, pq_m, pq_ks, soft_k)
    
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
    
    @property
    def stats(self):
        """Return all stats."""
        return self._index.get_stats()
    
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

        Creates:
          <path>/clusters.npz   — centroids + all cluster vectors/ids in one file
          <path>/metadata.json  — scalar state, deleted_ids, id_to_location
        """
        os.makedirs(path, exist_ok=True)
        idx   = self._index
        stats = idx.get_stats()
        nc    = stats["n_clusters"]

        arrays = {"centroids": idx.get_centroids()}
        for c in range(nc):
            arrays[f"vecs_{c}"] = idx.get_cluster_vectors(c)
            arrays[f"ids_{c}"]  = idx.get_cluster_ids(c)
        np.savez(os.path.join(path, "clusters.npz"), **arrays)

        meta = {
            "dim":            self.dim,
            "n_clusters":     nc,
            "n_vectors":      stats["n_vectors"],
            "nprobe":         self.nprobe,
            "use_pq":         self.use_pq,
            "soft_k":         int(idx.soft_k),
            "split_threshold":float(idx.split_threshold),
            "max_split_iters":int(idx.max_split_iters),
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
        )
        obj._index.split_threshold  = meta["split_threshold"]
        obj._index.max_split_iters  = meta["max_split_iters"]

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
