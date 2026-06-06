"""_turbovec_runner.py — TurboVec baseline adapter, shared across benchmarks.

TurboVec (github.com/RyanCodrai/turbovec) is a FLAT, static, MIPS (inner-product)
index built on the TurboQuant scalar quantizer. Two facts shape a fair comparison:

  1. No native dynamic delete. Like FAISS-flat / HNSW it can only REBUILD (or
     filter) — so under churn it is a "+rebuild" baseline, exactly analogous to
     FAISSIVFRebuildRunner / the HNSW rebuild path. Its story on dynamics is
     insert/rebuild *throughput*, not delete.
  2. It ranks by inner product, while Copenhagen / FAISS / HNSW rank by L2. We
     therefore L2-NORMALIZE all vectors (see `normalize`) so that
     argmax⟨q,v⟩ == argmin‖q−v‖² and the comparison is apples-to-apples.

Exposes the same Runner interface as the FAISS/HNSW runners in the churn
benchmarks: add / delete / rebuild / search / n_live.
"""
import numpy as np
import turbovec as tv


def normalize(x):
    """L2-normalize rows; zero rows map to zero. Makes L2-NN ≡ cosine ≡ MIPS."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def bytes_per_vector(dim, bit_width):
    """TurboVec on-disk/in-RAM footprint per vector: packed codes + scale (f32)."""
    return bit_width * dim // 8 + 4


class TurboVecRebuildRunner:
    """Flat TurboQuant index; deletes handled by rebuilding from live vectors.

    Assumes vectors are already normalized (so MIPS top-k == L2 top-k). Matches
    the FAISS/HNSW +rebuild runner interface used by the churn benchmarks.
    """

    def __init__(self, d, bit_width=4):
        self._d = d
        self._bw = bit_width
        self._live = {}          # id -> vector
        self._next_id = 0
        self._index = None
        self._ids = []           # row position -> id
        self.last_rebuild_s = 0.0

    def add(self, vecs):
        ids = list(range(self._next_id, self._next_id + len(vecs)))
        for gid, v in zip(ids, vecs):
            self._live[gid] = v
        self._next_id += len(vecs)
        return ids

    def delete(self, ids):
        for gid in ids:
            self._live.pop(gid, None)

    def rebuild(self):
        import time
        if not self._live:
            self._index, self._ids = None, []
            return
        self._ids = list(self._live.keys())
        mat = np.ascontiguousarray(np.stack([self._live[i] for i in self._ids]),
                                   dtype=np.float32)
        t0 = time.perf_counter()
        idx = tv.TurboQuantIndex(self._d, self._bw)
        idx.add(mat)
        idx.prepare()
        self.last_rebuild_s = time.perf_counter() - t0
        self._index = idx

    def search(self, q, k=10):
        if self._index is None:
            return np.array([], dtype=np.int32)
        q2 = np.ascontiguousarray(np.asarray(q, dtype=np.float32).reshape(1, -1))
        _, I = self._index.search(q2, k)
        I = np.asarray(I).reshape(-1)
        return np.array([self._ids[i] for i in I if 0 <= i < len(self._ids)],
                        dtype=np.int32)

    def n_live(self):
        return len(self._live)


class TurboVecStaticRunner:
    """One-shot flat TurboQuant index for static recall / compression tests."""

    def __init__(self, d, bit_width=4):
        self._d = d
        self._bw = bit_width
        self._index = tv.TurboQuantIndex(d, bit_width)
        self._n = 0

    def add(self, vecs):
        self._index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        self._n += len(vecs)

    def prepare(self):
        self._index.prepare()

    def search(self, queries, k=10):
        s, I = self._index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return np.asarray(s), np.asarray(I)

    def bytes_per_vector(self):
        return bytes_per_vector(self._d, self._bw)
