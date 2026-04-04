/*
 * Copenhagen: A Quantum-Inspired Dynamic Index
 *
 * Based on the Copenhagen interpretation of quantum mechanics:
 * - Superposition: PQ encodes vectors as superpositions of subspace codes
 * - Entanglement: Vectors in the same cluster are quantum-correlated
 * - Wave Function Collapse: Search "collapses" to nearest clusters
 * - Heisenberg Uncertainty: Recall vs speed tradeoff (can't optimize both)
 * - Decoherence: Lazy deletion = vectors decohere from cluster state
 *
 * Inspired by arXiv:2604.00271 "Engineering Fully Dynamic Convex Hulls"
 * (van der Hoog, Reinstädtler, Rotenberg — IT University of Copenhagen).
 *
 * Key insight from that paper: tombstoning works for cumulative queries
 * (ANN distances) unlike Boolean convex hull queries. We exploit this for
 * O(1) lazy deletion. For distribution drift, cluster splitting (Feature 3)
 * adaptively adds centroids when Voronoi cells overflow with OOD data —
 * the IVF analogue of the paper's logarithmic-method bucket merging.
 *
 * Features:
 *   1. Tombstone deletion  — O(1) delete, lazy physical compaction at search
 *   2. Soft assignments    — each vector indexed in top-K clusters for better
 *                            recall at Voronoi boundaries
 *   3. Cluster rebalancing — split oversized clusters via mini k-means to
 *                            handle distribution drift without full rebuild
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unordered_set>
#include <string>
#include <stdexcept>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
#include <cblas.h>
#elif defined(USE_MKL)
#include <mkl_cblas.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__AVX2__)
static inline float avx256_horizontal_sum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

#if defined(__AVX2__)
static inline float simd_pq_distance(float* const* tables, const uint8_t* codes, int M) {
    __m256 acc = _mm256_setzero_ps();
    int m = 0;

    for (; m + 8 <= M; m += 8) {
        __m256i indices = _mm256_setr_epi32(
            codes[m], codes[m+1], codes[m+2], codes[m+3],
            codes[m+4], codes[m+5], codes[m+6], codes[m+7]
        );

        __m256 d0 = _mm256_i32gather_ps(tables[m], indices, 4);
        __m256 d1 = _mm256_i32gather_ps(tables[m+4], indices, 4);

        acc = _mm256_add_ps(acc, _mm256_add_ps(d0, d1));
    }

    float result = avx256_horizontal_sum(acc);

    for (; m < M; m++) {
        result += tables[m][codes[m]];
    }

    return result;
}

#elif defined(__ARM_NEON)
static inline float simd_pq_distance(float* const* tables, const uint8_t* codes, int M) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int m = 0;

    for (; m + 4 <= M; m += 4) {
        float vals[4] = {
            tables[m  ][codes[m  ]],
            tables[m+1][codes[m+1]],
            tables[m+2][codes[m+2]],
            tables[m+3][codes[m+3]],
        };
        acc = vaddq_f32(acc, vld1q_f32(vals));
    }

    float result = vaddvq_f32(acc);

    for (; m < M; m++) {
        result += tables[m][codes[m]];
    }

    return result;
}

#else
static inline float simd_pq_distance(const float** tables, const uint8_t* codes, int M) {
    float dist = 0;
    for (int m = 0; m < M; m++) {
        dist += tables[m][codes[m]];
    }
    return dist;
}
#endif

namespace py = pybind11;

struct Cluster {
    float*      vectors;
    float*      vec_sum;
    float*      norms;   // precomputed ||v||² per slot, parallel to ids
    int*        ids;
    int         size;
    int         capacity;

    // mmap state — empty path means heap mode
    std::string mmap_path;
    size_t      mmap_size;   // current mapping size in bytes

    Cluster() : vectors(nullptr), vec_sum(nullptr), norms(nullptr), ids(nullptr),
                size(0), capacity(0), mmap_size(0), mmap_path() {}

    Cluster(Cluster&& other) noexcept
        : vectors(other.vectors), vec_sum(other.vec_sum), norms(other.norms), ids(other.ids),
          size(other.size), capacity(other.capacity),
          mmap_path(std::move(other.mmap_path)), mmap_size(other.mmap_size) {
        other.vectors   = nullptr;
        other.vec_sum   = nullptr;
        other.norms     = nullptr;
        other.ids       = nullptr;
        other.size      = 0;
        other.capacity  = 0;
        other.mmap_size = 0;
    }

    Cluster& operator=(Cluster&& other) noexcept {
        if (this != &other) {
            _free_vectors();
            if (vec_sum) free(vec_sum);
            if (norms)   free(norms);
            if (ids)     free(ids);
            vectors    = other.vectors;  other.vectors   = nullptr;
            vec_sum    = other.vec_sum;  other.vec_sum   = nullptr;
            norms      = other.norms;    other.norms     = nullptr;
            ids        = other.ids;      other.ids       = nullptr;
            size       = other.size;     other.size      = 0;
            capacity   = other.capacity; other.capacity  = 0;
            mmap_path  = std::move(other.mmap_path);
            mmap_size  = other.mmap_size; other.mmap_size = 0;
        }
        return *this;
    }

    Cluster(const Cluster&) = delete;
    Cluster& operator=(const Cluster&) = delete;

    ~Cluster() {
        _free_vectors();
        if (vec_sum) free(vec_sum);
        if (norms)   free(norms);
        if (ids)     free(ids);
    }

    bool is_mmap() const { return !mmap_path.empty(); }

    // Free (or unmap) the vectors buffer.
    void _free_vectors() {
        if (!vectors) return;
#if !defined(_WIN32)
        if (is_mmap()) { munmap(vectors, mmap_size); }
        else
#endif
        { free(vectors); }
        vectors = nullptr;
    }

    // Open / create the cluster's .bin file, map `cap` vectors, close fd.
    // truncate_new=true: fresh construction — O_TRUNC prevents stale data reuse
    //                    when mmap_dir is reused across index runs.
    // truncate_new=false (default): load path — existing file content is valid.
    void _mmap_open(int cap, int dim, bool truncate_new = false) {
#if !defined(_WIN32)
        size_t want = (size_t)cap * dim * sizeof(float);
        int oflags = O_CREAT | O_RDWR | (truncate_new ? O_TRUNC : 0);
        int fd = open(mmap_path.c_str(), oflags, 0600);
        if (fd < 0)
            throw std::runtime_error("mmap open failed: " + mmap_path);
        struct stat st;
        fstat(fd, &st);
        size_t sz = ((size_t)st.st_size >= want) ? (size_t)st.st_size : want;
        if ((size_t)st.st_size < sz && ftruncate(fd, (off_t)sz) < 0) {
            close(fd);
            throw std::runtime_error("mmap ftruncate failed: " + mmap_path);
        }
        void* addr = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);   // fd-after-close: mapping survives on POSIX
        if (addr == MAP_FAILED)
            throw std::runtime_error("mmap failed: " + mmap_path);
        vectors   = static_cast<float*>(addr);
        mmap_size = sz;
        capacity  = (int)(sz / ((size_t)dim * sizeof(float)));
#else
        (void)cap; (void)dim;
        throw std::runtime_error("mmap not supported on Windows");
#endif
    }

    // Grow the mapping to new_cap vectors (munmap → ftruncate → mmap).
    void _mmap_grow(int new_cap, int dim) {
#if !defined(_WIN32)
        size_t new_sz = (size_t)new_cap * dim * sizeof(float);
        munmap(vectors, mmap_size);
        vectors = nullptr;
        int fd = open(mmap_path.c_str(), O_RDWR);
        if (fd < 0)
            throw std::runtime_error("mmap grow open failed: " + mmap_path);
        if (ftruncate(fd, (off_t)new_sz) < 0) {
            close(fd);
            throw std::runtime_error("mmap grow ftruncate failed: " + mmap_path);
        }
        void* addr = mmap(nullptr, new_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);   // fd-after-close: mapping survives on POSIX
        if (addr == MAP_FAILED)
            throw std::runtime_error("mmap grow remap failed: " + mmap_path);
        vectors   = static_cast<float*>(addr);
        mmap_size = new_sz;
        capacity  = new_cap;
#else
        (void)new_cap; (void)dim;
#endif
    }

    // Initialise a cluster. If path is non-empty, use mmap-backed storage.
    void init(int dim, const std::string& path = "", bool truncate_new = false) {
        vec_sum = (float*)aligned_alloc(32, dim * sizeof(float));
        std::memset(vec_sum, 0, dim * sizeof(float));
        if (!path.empty()) {
            mmap_path = path;
            _mmap_open(64, dim, truncate_new);   // initial capacity 64 vectors
            ids   = (int*)malloc((size_t)capacity * sizeof(int));
            norms = (float*)malloc((size_t)capacity * sizeof(float));
        }
        // Heap mode: vectors and ids allocated lazily on first add_vector
    }

    void add_vector(const float* vec, int dim, int id) {
        if (size >= capacity) {
            int new_cap = capacity == 0 ? 64 : capacity * 2;
            if (is_mmap()) {
                _mmap_grow(new_cap, dim);
                int*   new_ids   = (int*)realloc(ids,   (size_t)new_cap * sizeof(int));
                float* new_norms = (float*)realloc(norms, (size_t)new_cap * sizeof(float));
                if (!new_ids || !new_norms) throw std::bad_alloc();
                ids   = new_ids;
                norms = new_norms;
            } else {
                float* nv = (float*)aligned_alloc(32, (size_t)new_cap * dim * sizeof(float));
                int*   ni = (int*)malloc((size_t)new_cap * sizeof(int));
                float* nn = (float*)malloc((size_t)new_cap * sizeof(float));
                if (!nv || !ni || !nn) { free(nv); free(ni); free(nn); throw std::bad_alloc(); }
                if (vectors) { std::memcpy(nv, vectors, (size_t)size * dim * sizeof(float)); free(vectors); }
                if (ids)     { std::memcpy(ni, ids,     (size_t)size * sizeof(int));          free(ids); }
                if (norms)   { std::memcpy(nn, norms,   (size_t)size * sizeof(float));        free(norms); }
                vectors  = nv;
                ids      = ni;
                norms    = nn;
                capacity = new_cap;
            }
        }
        std::memcpy(vectors + size * dim, vec, dim * sizeof(float));
        ids[size] = id;
        float norm_sq = 0;
        for (int i = 0; i < dim; i++) norm_sq += vec[i] * vec[i];
        norms[size] = norm_sq;
        for (int i = 0; i < dim; i++) vec_sum[i] += vec[i];
        size++;
    }
};

struct PQCodebook {
    int M;
    int Ks;
    int d_sub;
    float** centroids;
    float** precomputed_tables;

    PQCodebook() : M(0), Ks(0), d_sub(0), centroids(nullptr), precomputed_tables(nullptr) {}

    ~PQCodebook() {
        if (centroids) {
            for (int m = 0; m < M; m++) {
                if (centroids[m]) free(centroids[m]);
            }
            free(centroids);
        }
        if (precomputed_tables) {
            for (int m = 0; m < M; m++) {
                if (precomputed_tables[m]) free(precomputed_tables[m]);
            }
            free(precomputed_tables);
        }
    }

    void init(int M_, int Ks_, int d_) {
        M = M_;
        Ks = Ks_;
        d_sub = d_ / M_;

        if (d_sub * M_ != d_) {
            fprintf(stderr, "ERROR: PQ dimension mismatch: d=%d, M=%d, d_sub=%d\n", d_, M_, d_sub);
            throw std::runtime_error("PQ dimension must be evenly divisible by M");
        }

        centroids = (float**)malloc(M * sizeof(float*));
        precomputed_tables = (float**)malloc(M * sizeof(float*));
        for (int m = 0; m < M; m++) {
            centroids[m] = (float*)aligned_alloc(32, Ks * d_sub * sizeof(float));
            precomputed_tables[m] = (float*)aligned_alloc(32, 256 * Ks * sizeof(float));
            if (!centroids[m] || !precomputed_tables[m]) {
                fprintf(stderr, "ERROR: PQ aligned_alloc failed for m=%d\n", m);
                throw std::bad_alloc();
            }
            std::memset(centroids[m], 0, Ks * d_sub * sizeof(float));
            std::memset(precomputed_tables[m], 0, 256 * Ks * sizeof(float));
        }
    }

    void train(const float* data, int n, int d, int iterations = 25) {
        fprintf(stderr, "Copenhagen PQ: Training with %d vectors, dim=%d\n", n, d);
        fprintf(stderr, "  M=%d subspaces, Ks=%d codes, d_sub=%d\n", M, Ks, d_sub);
        fflush(stderr);

        if (n == 0) {
            fprintf(stderr, "ERROR: No training data!\n");
            fflush(stderr);
            return;
        }

        std::mt19937 rng(42);

        std::vector<int> indices(n);
        for (int i = 0; i < n; i++) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), rng);

        fprintf(stderr, "  Initializing centroids...\n");
        fflush(stderr);

        for (int m = 0; m < M; m++) {
            for (int ks = 0; ks < Ks; ks++) {
                int idx = indices[(m * Ks + ks) % n];
                int offset = m * d_sub;
                for (int j = 0; j < d_sub; j++) {
                    centroids[m][ks * d_sub + j] = data[idx * d + offset + j];
                }
            }
        }

        std::vector<int> counts(M * Ks, 0);
        std::vector<float> sums(M * Ks * d_sub, 0.0f);

        fprintf(stderr, "  k-means iterations (%d)...\n", iterations);
        fflush(stderr);

        for (int iter = 0; iter < iterations; iter++) {
            std::fill(counts.begin(), counts.end(), 0);
            std::fill(sums.begin(), sums.end(), 0.0f);

            for (int m = 0; m < M; m++) {
                int offset = m * d_sub;

                std::vector<float> cent_mat(Ks * d_sub);
                for (int ks = 0; ks < Ks; ks++) {
                    for (int j = 0; j < d_sub; j++) {
                        cent_mat[ks * d_sub + j] = centroids[m][ks * d_sub + j];
                    }
                }

                std::vector<float> data_mat(n * d_sub);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < d_sub; j++) {
                        data_mat[i * d_sub + j] = data[i * d + offset + j];
                    }
                }

                std::vector<float> dists(n * Ks, 0.0f);

                std::vector<float> x_sq(n, 0.0f);
                for (int i = 0; i < n; i++) {
                    float s = 0;
                    for (int j = 0; j < d_sub; j++) {
                        float v = data_mat[i * d_sub + j];
                        s += v * v;
                    }
                    x_sq[i] = s;
                }

                std::vector<float> c_sq(Ks, 0.0f);
                for (int ks = 0; ks < Ks; ks++) {
                    float s = 0;
                    for (int j = 0; j < d_sub; j++) {
                        float v = cent_mat[ks * d_sub + j];
                        s += v * v;
                    }
                    c_sq[ks] = s;
                }

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) || defined(USE_MKL)
                float neg2 = -2.0f;
                float zero = 0.0f;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                           n, Ks, d_sub, neg2,
                           data_mat.data(), d_sub,
                           cent_mat.data(), d_sub,
                           zero, dists.data(), Ks);

                for (int i = 0; i < n; i++) {
                    for (int ks = 0; ks < Ks; ks++) {
                        dists[i * Ks + ks] += x_sq[i] + c_sq[ks];
                    }
                }
#else
                for (int i = 0; i < n; i++) {
                    for (int ks = 0; ks < Ks; ks++) {
                        float dot = 0.0f;
                        for (int j = 0; j < d_sub; j++)
                            dot += data_mat[i * d_sub + j] * cent_mat[ks * d_sub + j];
                        dists[i * Ks + ks] = x_sq[i] - 2.0f * dot + c_sq[ks];
                    }
                }
#endif

                for (int i = 0; i < n; i++) {
                    int best_ks = 0;
                    float best_dist = dists[i * Ks];
                    for (int ks = 1; ks < Ks; ks++) {
                        if (dists[i * Ks + ks] < best_dist) {
                            best_dist = dists[i * Ks + ks];
                            best_ks = ks;
                        }
                    }

                    counts[m * Ks + best_ks]++;

                    int sum_offset = (m * Ks + best_ks) * d_sub;
                    for (int j = 0; j < d_sub; j++) {
                        sums[sum_offset + j] += data_mat[i * d_sub + j];
                    }
                }
            }

            int empty = 0;
            for (int m = 0; m < M; m++) {
                for (int ks = 0; ks < Ks; ks++) {
                    int count = counts[m * Ks + ks];
                    if (count > 0) {
                        float inv = 1.0f / count;
                        int sum_offset = (m * Ks + ks) * d_sub;
                        for (int j = 0; j < d_sub; j++) {
                            centroids[m][ks * d_sub + j] = sums[sum_offset + j] * inv;
                        }
                    } else {
                        empty++;
                    }
                }
            }

            if (iter % 5 == 0 || iter == iterations - 1) {
                fprintf(stderr, "    iter %d: empty centroids=%d\n", iter + 1, empty);
                fflush(stderr);
            }
        }
        fprintf(stderr, "  Training complete\n");
        fflush(stderr);
    }

    void compute_precomputed_tables(const float* query, int d) {
        for (int m = 0; m < M; m++) {
            int offset = m * d_sub;
            for (int ks = 0; ks < Ks; ks++) {
                float dist = 0;
                for (int j = 0; j < d_sub; j++) {
                    float diff = query[offset + j] - centroids[m][ks * d_sub + j];
                    dist += diff * diff;
                }
                precomputed_tables[m][ks] = dist;
            }
        }
    }

    void encode(const float* vec, int d, uint8_t* out_codes) {
        for (int m = 0; m < M; m++) {
            int offset = m * d_sub;
            int best_ks = 0;
            float best_dist = 1e10;

            for (int ks = 0; ks < Ks; ks++) {
                float dist = 0;
                for (int j = 0; j < d_sub; j++) {
                    float diff = vec[offset + j] - centroids[m][ks * d_sub + j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_ks = ks;
                }
            }
            out_codes[m] = best_ks;
        }
    }

    float distance_to_code(uint8_t code, int m) {
        return precomputed_tables[m][code];
    }
};

struct PQCluster {
    uint8_t* codes;
    int* ids;
    int size;
    int capacity;
    int M;

    PQCluster() : codes(nullptr), ids(nullptr), size(0), capacity(0), M(0) {}

    PQCluster(PQCluster&& other) noexcept
        : codes(other.codes), ids(other.ids),
          size(other.size), capacity(other.capacity), M(other.M) {
        other.codes    = nullptr;
        other.ids      = nullptr;
        other.size     = 0;
        other.capacity = 0;
    }

    PQCluster& operator=(PQCluster&& other) noexcept {
        if (this != &other) {
            if (codes) free(codes);
            if (ids)   free(ids);
            codes     = other.codes;    other.codes    = nullptr;
            ids       = other.ids;      other.ids      = nullptr;
            size      = other.size;     other.size     = 0;
            capacity  = other.capacity; other.capacity = 0;
            M         = other.M;
        }
        return *this;
    }

    PQCluster(const PQCluster&) = delete;
    PQCluster& operator=(const PQCluster&) = delete;

    ~PQCluster() {
        if (codes) free(codes);
        if (ids) free(ids);
    }

    void set_m(int M_) { M = M_; }

    void reserve(int cap, int M_) {
        M = M_;
        if (cap > capacity) {
            if (codes) free(codes);
            if (ids) free(ids);
            codes = (uint8_t*)malloc(cap * M * sizeof(uint8_t));
            ids = (int*)malloc(cap * sizeof(int));
            capacity = cap;
        }
    }

    void add_codes(const uint8_t* new_codes, int M_, int id) {
        if (size >= capacity) {
            capacity = capacity == 0 ? 64 : capacity * 2;
            uint8_t* new_codes_arr = (uint8_t*)malloc(capacity * M * sizeof(uint8_t));
            int* new_ids_arr = (int*)malloc(capacity * sizeof(int));
            if (codes) {
                std::memcpy(new_codes_arr, codes, size * M * sizeof(uint8_t));
                std::memcpy(new_ids_arr, ids, size * sizeof(int));
                free(codes);
                free(ids);
            }
            codes = new_codes_arr;
            ids = new_ids_arr;
        }
        for (int m = 0; m < M; m++) {
            codes[size * M + m] = new_codes[m];
        }
        ids[size] = id;
        size++;
    }
};

inline void blas_l2_distances(const float* queries, int nq, int d,
                              const float* vecs, int nv,
                              float* dists) {
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) || defined(USE_MKL)
    float alpha = -2.0f, beta = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                nq, nv, d, alpha, queries, d, vecs, d, beta, dists, nv);

    std::vector<float> q_sq(nq);
    std::vector<float> v_sq(nv);

    for (int i = 0; i < nq; i++) {
        q_sq[i] = cblas_sdot(d, queries + i * d, 1, queries + i * d, 1);
    }
    for (int i = 0; i < nv; i++) {
        v_sq[i] = cblas_sdot(d, vecs + i * d, 1, vecs + i * d, 1);
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < nv; j++) {
            dists[i * nv + j] = dists[i * nv + j] + q_sq[i] + v_sq[j];
        }
    }
#else
    for (int q = 0; q < nq; q++) {
        for (int v = 0; v < nv; v++) {
            float dist = 0;
            for (int i = 0; i < d; i++) {
                float diff = queries[q * d + i] - vecs[v * d + i];
                dist += diff * diff;
            }
            dists[q * nv + v] = std::sqrt(dist);
        }
    }
#endif
}

// Like blas_l2_distances but avoids recomputing per-vector squared norms —
// caller supplies precomputed q_sq[nq] and v_sq[nv].
// Saves one full O(nq*d + nv*d) dot-product pass on every cluster scan.
inline void blas_l2_distances_precomp(
    const float* queries, const float* q_sq, int nq,
    int d,
    const float* vecs, const float* v_sq, int nv,
    float* dists)
{
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) || defined(USE_MKL)
    float alpha = -2.0f, beta = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                nq, nv, d, alpha, queries, d, vecs, d, beta, dists, nv);
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < nv; j++)
            dists[i * nv + j] += q_sq[i] + v_sq[j];
#else
    for (int q = 0; q < nq; q++) {
        for (int v = 0; v < nv; v++) {
            float dist = 0;
            for (int i = 0; i < d; i++) {
                float diff = queries[q * d + i] - vecs[v * d + i];
                dist += diff * diff;
            }
            dists[q * nv + v] = dist;
        }
    }
#endif
}

class DynamicIVF {
public:
    int dim;
    int n_clusters;
    int nprobe;
    int use_pq;
    int pq_m;
    int pq_ks;
    bool        use_mmap;
    std::string mmap_dir;
    int soft_k;           // number of clusters each vector is indexed in (1 = standard)
    float split_threshold; // split cluster when size > mean * split_threshold
    int max_split_iters;  // mini k-means iterations for cluster splitting

    float* centroids;
    int n_vectors;

    std::vector<Cluster> clusters;
    std::vector<PQCluster> pq_clusters;
    std::vector<int> search_ids;
    PQCodebook pq_codebook;

    // id -> list of (cluster_id, position_in_cluster) — one entry per soft assignment
    std::vector<std::vector<std::pair<int, int>>> id_to_location;

    // Tombstone set: O(1) delete, cleaned up lazily during search
    std::unordered_set<int> deleted_ids;

    // Live vector count per cluster: excludes tombstoned vectors.
    // Used by rebalance_if_needed() so that dead vectors don't skew split decisions.
    // Centroids are frozen — we deliberately do NOT use vec_sum to track a
    // "drifting mean". The centroid array is the source of truth; live_count
    // just lets us measure imbalance among actually-searchable vectors.
    std::vector<int> cluster_live_count;

    int max_vectors_per_probe;
    float* search_buffer;
    float* norms_buffer;  // precomputed v_sq values, parallel to search_buffer
    float* dists_buffer;

    std::mt19937 rng;

    DynamicIVF(int dim, int n_clusters, int nprobe = 1, int use_pq = 0,
               int pq_m = 8, int pq_ks = 256, int soft_k = 1,
               bool use_mmap = false, const std::string& mmap_dir = "")
        : dim(dim), n_clusters(n_clusters), nprobe(nprobe), use_pq(use_pq),
          pq_m(pq_m), pq_ks(pq_ks), soft_k(soft_k),
          use_mmap(use_mmap), mmap_dir(mmap_dir),
          split_threshold(3.0f), max_split_iters(10) {

        rng.seed(42);
        n_vectors = 0;

        centroids = (float*)aligned_alloc(32, n_clusters * dim * sizeof(float));
        std::memset(centroids, 0, n_clusters * dim * sizeof(float));

        max_vectors_per_probe = 100000;
        search_buffer = (float*)aligned_alloc(32, max_vectors_per_probe * dim * sizeof(float));
        norms_buffer  = (float*)malloc(max_vectors_per_probe * sizeof(float));
        dists_buffer  = (float*)aligned_alloc(32, max_vectors_per_probe * sizeof(float));
        search_ids.resize(max_vectors_per_probe);

        clusters.resize(n_clusters);
        pq_clusters.resize(n_clusters);
        cluster_live_count.resize(n_clusters, 0);
        id_to_location.resize(1000000);

        for (int c = 0; c < n_clusters; c++) {
            std::string path = use_mmap
                ? mmap_dir + "/cluster_" + std::to_string(c) + ".bin"
                : std::string();
            clusters[c].init(dim, path, /*truncate_new=*/use_mmap);
        }

        if (use_pq) {
            pq_codebook.init(pq_m, pq_ks, dim);
            for (auto& pc : pq_clusters) {
                pc.set_m(pq_m);
            }
        }
    }

    ~DynamicIVF() {
        free(centroids);
        free(search_buffer);
        free(norms_buffer);
        free(dists_buffer);
    }

    // -----------------------------------------------------------------------
    // Centroid helpers
    // -----------------------------------------------------------------------

    int find_nearest_centroid(const float* vec) {
        float best_dist = 1e10;
        int best = 0;
        for (int c = 0; c < n_clusters; c++) {
            float dist = 0;
            for (int i = 0; i < dim; i++) {
                float diff = vec[i] - centroids[c * dim + i];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best = c;
            }
        }
        return best;
    }

    // Returns indices of the k nearest centroids sorted by distance.
    std::vector<int> find_top_k_centroids(const float* vec, int k) {
        std::vector<float> dists(n_clusters);
        blas_l2_distances(vec, 1, dim, centroids, n_clusters, dists.data());
        std::vector<int> idx(n_clusters);
        std::iota(idx.begin(), idx.end(), 0);
        int actual_k = std::min(k, n_clusters);
        std::partial_sort(idx.begin(), idx.begin() + actual_k, idx.end(),
                          [&dists](int a, int b) { return dists[a] < dists[b]; });
        idx.resize(actual_k);
        return idx;
    }

    // -----------------------------------------------------------------------
    // Tombstone compaction helpers (lazy, triggered during search)
    // -----------------------------------------------------------------------

    // Physically removes deleted vectors from a cluster in-place (single pass).
    void compact_cluster(int cluster_id) {
        // Physically remove tombstoned vectors from the cluster array.
        // We do NOT touch vec_sum here: centroids are frozen after training and
        // vec_sum must not be used as a drifting centroid proxy. Subtracting
        // dead vectors from vec_sum would imply the centroid shifts with deletions,
        // which violates the design. cluster_live_count already tracks live mass.
        Cluster& cl = clusters[cluster_id];
        int write = 0;
        for (int read = 0; read < cl.size; read++) {
            int id = cl.ids[read];
            if (deleted_ids.count(id)) continue;
            if (write != read) {
                std::memcpy(cl.vectors + write * dim,
                            cl.vectors + read * dim, dim * sizeof(float));
                cl.ids[write]   = id;
                cl.norms[write] = cl.norms[read];
            }
            for (auto& loc : id_to_location[id])
                if (loc.first == cluster_id) { loc.second = write; break; }
            write++;
        }
        cl.size = write;
    }

    void compact_pq_cluster(int cluster_id) {
        if (!use_pq) return;
        PQCluster& pc = pq_clusters[cluster_id];
        int write = 0;
        for (int read = 0; read < pc.size; read++) {
            int id = pc.ids[read];
            if (deleted_ids.count(id)) continue;
            if (write != read) {
                std::memcpy(pc.codes + write * pq_m,
                            pc.codes + read * pq_m, pq_m);
                pc.ids[write] = id;
            }
            write++;
        }
        pc.size = write;
    }

    // Full compaction: evict all tombstones from every cluster, then clear
    // deleted_ids. Call this when deleted_ids grows large (e.g. > 10% of
    // n_vectors) to keep compact_cluster hash lookups at O(1).
    // cluster_live_count is already correct; no recompute needed here.
    void compact_all() {
        for (int c = 0; c < n_clusters; c++) {
            bool has_deleted = false;
            for (int i = 0; i < clusters[c].size && !has_deleted; i++)
                if (deleted_ids.count(clusters[c].ids[i])) has_deleted = true;
            if (has_deleted) {
                compact_cluster(c);
                compact_pq_cluster(c);
            }
        }
        deleted_ids.clear();
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    void train_centroids(const float* data, int n) {
        if (n < n_clusters) {
            for (int c = 0; c < n_clusters; c++) {
                for (int d = 0; d < dim; d++) {
                    centroids[c * dim + d] = ((float)rng() / rng.max()) * 2.0f - 1.0f;
                }
            }
            return;
        }

        std::vector<int> indices(n);
        for (int i = 0; i < n; i++) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int c = 0; c < n_clusters; c++) {
            int idx = indices[c];
            std::memcpy(centroids + c * dim, data + idx * dim, dim * sizeof(float));
        }

        std::vector<int> assignments(n);
        std::vector<float> dists(n * n_clusters);

        for (int iter = 0; iter < 25; iter++) {
            std::vector<int> counts(n_clusters, 0);

            blas_l2_distances(data, n, dim, centroids, n_clusters, dists.data());

            for (int i = 0; i < n; i++) {
                int best = 0;
                float min_d = dists[i * n_clusters];
                for (int c = 1; c < n_clusters; c++) {
                    float d = dists[i * n_clusters + c];
                    if (d < min_d) { min_d = d; best = c; }
                }
                assignments[i] = best;
                counts[best]++;
            }

            for (int c = 0; c < n_clusters; c++)
                std::memset(centroids + c * dim, 0, dim * sizeof(float));

            for (int i = 0; i < n; i++) {
                int c = assignments[i];
                const float* v = data + i * dim;
                float* cent = centroids + c * dim;
                for (int d = 0; d < dim; d++) cent[d] += v[d];
            }

            for (int c = 0; c < n_clusters; c++) {
                if (counts[c] > 0) {
                    float inv_count = 1.0f / counts[c];
                    float* cent = centroids + c * dim;
                    for (int d = 0; d < dim; d++) cent[d] *= inv_count;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Insert helpers (used by train and insert_vector)
    // -----------------------------------------------------------------------

    // Used only during train() — does NOT trigger rebalancing.
    void add_vector_no_centroid_update(const float* data) {
        int id = n_vectors;
        if (id >= (int)id_to_location.size())
            id_to_location.resize(id_to_location.size() * 2);
        id_to_location[id].clear();

        for (int cluster_id : find_top_k_centroids(data, soft_k)) {
            int pos = clusters[cluster_id].size;
            clusters[cluster_id].add_vector(data, dim, id);
            id_to_location[id].push_back({cluster_id, pos});
            cluster_live_count[cluster_id]++;
            if (use_pq) {
                std::vector<uint8_t> codes(pq_m);
                pq_codebook.encode(data, dim, codes.data());
                pq_clusters[cluster_id].add_codes(codes.data(), pq_m, id);
            }
        }
        n_vectors++;
    }

    void insert_vector(const float* data) {
        int id = n_vectors;
        if (id >= (int)id_to_location.size())
            id_to_location.resize(id_to_location.size() * 2);
        id_to_location[id].clear();

        for (int cluster_id : find_top_k_centroids(data, soft_k)) {
            int pos = clusters[cluster_id].size;
            clusters[cluster_id].add_vector(data, dim, id);
            id_to_location[id].push_back({cluster_id, pos});
            cluster_live_count[cluster_id]++;
            if (use_pq) {
                std::vector<uint8_t> codes(pq_m);
                pq_codebook.encode(data, dim, codes.data());
                pq_clusters[cluster_id].add_codes(codes.data(), pq_m, id);
            }
        }
        n_vectors++;
        rebalance_if_needed();
    }

    // -----------------------------------------------------------------------
    // Delete — O(1) tombstone
    // -----------------------------------------------------------------------

    void delete_vector(int id) {
        if (id < 0 || id >= n_vectors) return;
        if (deleted_ids.count(id)) return;   // already deleted, don't double-count
        deleted_ids.insert(id);
        // Decrement live count for every cluster this vector is assigned to
        if (id < (int)id_to_location.size()) {
            for (auto& [cid, pos] : id_to_location[id])
                if (cid < (int)cluster_live_count.size())
                    cluster_live_count[cid]--;
        }
    }

    // -----------------------------------------------------------------------
    // Cluster splitting — handles distribution drift by adaptively adding
    // centroids when a Voronoi cell overflows with out-of-distribution data.
    // This is the IVF analogue of the paper's logarithmic-method bucket merge.
    // -----------------------------------------------------------------------

    void split_cluster(int cluster_id) {
        // Save old data BEFORE any emplace_back (which may invalidate refs)
        int old_size = clusters[cluster_id].size;
        if (old_size < 2) return;

        std::vector<float> old_vecs(old_size * dim);
        std::vector<int>   old_ids(old_size);
        std::memcpy(old_vecs.data(), clusters[cluster_id].vectors,
                    old_size * dim * sizeof(float));
        std::memcpy(old_ids.data(), clusters[cluster_id].ids,
                    old_size * sizeof(int));

        // Collect live indices — tombstoned vectors must not influence centroid placement
        std::vector<int> live_idx;
        live_idx.reserve(old_size);
        for (int i = 0; i < old_size; i++)
            if (!deleted_ids.count(old_ids[i]))
                live_idx.push_back(i);
        if ((int)live_idx.size() < 2) return;

        // Mini k-means (k=2) seeded from live vectors only
        std::uniform_int_distribution<int> dist_rng(0, (int)live_idx.size() - 1);
        int seed_a = live_idx[dist_rng(rng)];
        int seed_b = live_idx[dist_rng(rng)];
        while (seed_b == seed_a && (int)live_idx.size() > 1)
            seed_b = live_idx[dist_rng(rng)];

        std::vector<float> c0(dim), c1(dim);
        std::memcpy(c0.data(), old_vecs.data() + seed_a * dim, dim * sizeof(float));
        std::memcpy(c1.data(), old_vecs.data() + seed_b * dim, dim * sizeof(float));

        std::vector<int> assign(old_size, 0);
        for (int iter = 0; iter < max_split_iters; iter++) {
            std::vector<float> sum0(dim, 0.f), sum1(dim, 0.f);
            int cnt0 = 0, cnt1 = 0;
            for (int i : live_idx) {
                float d0 = 0, d1 = 0;
                const float* v = old_vecs.data() + i * dim;
                for (int d = 0; d < dim; d++) {
                    float diff0 = v[d] - c0[d], diff1 = v[d] - c1[d];
                    d0 += diff0 * diff0;
                    d1 += diff1 * diff1;
                }
                assign[i] = (d1 < d0) ? 1 : 0;
                if (assign[i] == 0) { cnt0++; for (int d=0;d<dim;d++) sum0[d]+=old_vecs[i*dim+d]; }
                else                { cnt1++; for (int d=0;d<dim;d++) sum1[d]+=old_vecs[i*dim+d]; }
            }
            if (cnt0 > 0) { float inv=1.f/cnt0; for(int d=0;d<dim;d++) c0[d]=sum0[d]*inv; }
            if (cnt1 > 0) { float inv=1.f/cnt1; for(int d=0;d<dim;d++) c1[d]=sum1[d]*inv; }
        }

        // Expand centroids array (n_clusters+1)
        int new_cluster_id = n_clusters;
        float* new_centroids = (float*)aligned_alloc(32, (n_clusters + 1) * dim * sizeof(float));
        if (!new_centroids) throw std::bad_alloc();
        std::memcpy(new_centroids, centroids, n_clusters * dim * sizeof(float));
        free(centroids);
        centroids = new_centroids;
        n_clusters++;

        // Write updated centroids: c0 stays at cluster_id, c1 goes to new_cluster_id
        std::memcpy(centroids + cluster_id * dim, c0.data(), dim * sizeof(float));
        std::memcpy(centroids + new_cluster_id * dim, c1.data(), dim * sizeof(float));

        // Grow cluster structures (emplace_back may realloc — old refs are gone, saved above)
        clusters.emplace_back();
        {
            std::string path = use_mmap
                ? mmap_dir + "/cluster_" + std::to_string(new_cluster_id) + ".bin"
                : std::string();
            clusters.back().init(dim, path);
        }
        pq_clusters.emplace_back();
        if (use_pq) pq_clusters.back().set_m(pq_m);
        cluster_live_count.push_back(0);   // new cluster starts with 0 live vectors

        // Reset the original cluster (live count will be rebuilt during redistribution)
        clusters[cluster_id].size = 0;
        std::memset(clusters[cluster_id].vec_sum, 0, dim * sizeof(float));
        cluster_live_count[cluster_id] = 0;
        if (use_pq) pq_clusters[cluster_id].size = 0;

        // Redistribute saved vectors
        for (int i = 0; i < old_size; i++) {
            // Skip tombstoned vectors — don't reinsert them
            if (deleted_ids.count(old_ids[i])) continue;

            int target = (assign[i] == 0) ? cluster_id : new_cluster_id;
            int pos = clusters[target].size;
            clusters[target].add_vector(old_vecs.data() + i * dim, dim, old_ids[i]);

            if (use_pq) {
                std::vector<uint8_t> codes(pq_m);
                pq_codebook.encode(old_vecs.data() + i * dim, dim, codes.data());
                pq_clusters[target].add_codes(codes.data(), pq_m, old_ids[i]);
            }

            cluster_live_count[target]++;

            // Update id_to_location: find the entry for the old cluster_id and update it
            for (auto& loc : id_to_location[old_ids[i]]) {
                if (loc.first == cluster_id) {
                    loc.first  = target;
                    loc.second = pos;
                    break;
                }
            }
        }
    }

    // Trigger splits for any cluster exceeding mean_size * split_threshold.
    // Only processes clusters that existed at call time (no cascading).
    void rebalance_if_needed() {
        if (n_vectors == 0) return;
        // Use live counts (tombstones excluded) so deleted vectors don't
        // artificially inflate a cluster's apparent size and trigger spurious splits.
        int total_live = 0;
        for (int c = 0; c < n_clusters; c++) total_live += cluster_live_count[c];
        if (total_live == 0) return;
        float mean_size = (float)total_live / n_clusters;
        float threshold = mean_size * split_threshold;
        int current_count = n_clusters;  // don't process newly created clusters this pass
        for (int c = 0; c < current_count; c++) {
            if (cluster_live_count[c] > (int)threshold) {
                split_cluster(c);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Search — non-PQ path
    // -----------------------------------------------------------------------

    py::tuple search(
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        int n_results,
        int n_probes_in
    ) {
        auto buf = query.request();
        if (buf.ndim != 1) throw std::runtime_error("Query must be 1D array");
        float* ptr = (float*)buf.ptr;
        if (buf.shape[0] != dim) throw std::runtime_error("Query dimension mismatch");

        int probes = std::min(n_probes_in > 0 ? n_probes_in : nprobe, n_clusters);

        std::vector<float> centroid_dists_sq(n_clusters);
        blas_l2_distances(ptr, 1, dim, centroids, n_clusters, centroid_dists_sq.data());

        std::vector<std::pair<int, float>> centroid_dists;
        centroid_dists.reserve(n_clusters);
        for (int i = 0; i < n_clusters; i++)
            centroid_dists.push_back({i, centroid_dists_sq[i]});
        std::sort(centroid_dists.begin(), centroid_dists.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Compute total (pre-compaction estimate for buffer sizing)
        int total_vectors = 0;
        for (int p = 0; p < probes; p++)
            total_vectors += clusters[centroid_dists[p].first].size;

        if (total_vectors == 0) {
            return py::make_tuple(py::list(), py::list());
        }

        if (use_pq) {
            pq_codebook.compute_precomputed_tables(ptr, dim);
            const int rerank_k = 100;

            std::vector<std::pair<int, float>> pq_results;
            pq_results.reserve(total_vectors);

            for (int p = 0; p < probes; p++) {
                int cluster_id = centroid_dists[p].first;
                PQCluster& pq_cluster = pq_clusters[cluster_id];
                for (int i = 0; i < pq_cluster.size; i++) {
                    int id = pq_cluster.ids[i];
                    if (deleted_ids.count(id)) continue;
                    uint8_t* codes = pq_cluster.codes + i * pq_m;
                    float pq_dist = 0;
                    for (int m = 0; m < pq_m; m++)
                        pq_dist += pq_codebook.precomputed_tables[m][codes[m]];
                    pq_results.push_back({id, pq_dist});
                }
            }

            int n_pq = std::min(rerank_k, (int)pq_results.size());
            std::partial_sort(pq_results.begin(), pq_results.begin() + n_pq, pq_results.end(),
                              [](const auto& a, const auto& b) { return a.second < b.second; });

            // Dedup (soft_k > 1 means same id may appear in multiple clusters)
            if (soft_k > 1) {
                std::unordered_set<int> seen;
                int w = 0;
                for (int i = 0; i < n_pq; i++)
                    if (seen.insert(pq_results[i].first).second)
                        pq_results[w++] = pq_results[i];
                n_pq = w;
            }

            std::vector<float> rerank_vectors(n_pq * dim);
            for (int i = 0; i < n_pq; i++) {
                int id = pq_results[i].first;
                auto [c, pos] = id_to_location[id][0];
                std::memcpy(rerank_vectors.data() + i * dim,
                            clusters[c].vectors + pos * dim, dim * sizeof(float));
            }

            std::vector<float> exact_dists(n_pq);
            blas_l2_distances(ptr, 1, dim, rerank_vectors.data(), n_pq, exact_dists.data());

            std::vector<std::pair<int, float>> final_results;
            final_results.reserve(n_pq);
            for (int i = 0; i < n_pq; i++)
                final_results.push_back({pq_results[i].first, exact_dists[i]});

            int result_count = std::min(n_results, (int)final_results.size());
            std::partial_sort(final_results.begin(), final_results.begin() + result_count,
                              final_results.end(),
                              [](const auto& a, const auto& b) { return a.second < b.second; });

            py::list py_ids, py_distances;
            for (int i = 0; i < result_count; i++) {
                py_ids.append(final_results[i].first);
                py_distances.append(final_results[i].second);
            }
            return py::make_tuple(py_ids, py_distances);

        } else {
            // Ensure buffer is large enough for pre-compaction estimate
            if (total_vectors > max_vectors_per_probe) {
                max_vectors_per_probe = total_vectors;
                free(search_buffer);
                free(norms_buffer);
                free(dists_buffer);
                search_buffer = (float*)aligned_alloc(32, max_vectors_per_probe * dim * sizeof(float));
                norms_buffer  = (float*)malloc(max_vectors_per_probe * sizeof(float));
                dists_buffer  = (float*)aligned_alloc(32, max_vectors_per_probe * sizeof(float));
                search_ids.resize(max_vectors_per_probe);
            }

            // Fill loop: compact deleted vectors lazily, then copy live vectors + norms
            int idx = 0;
            for (int p = 0; p < probes; p++) {
                int cid = centroid_dists[p].first;
                Cluster& cluster = clusters[cid];
                if (cluster.size == 0) continue;

                // Check for tombstones; compact if any found
                bool has_deleted = false;
                for (int i = 0; i < cluster.size && !has_deleted; i++)
                    if (deleted_ids.count(cluster.ids[i])) has_deleted = true;
                if (has_deleted) {
                    compact_cluster(cid);
                    compact_pq_cluster(cid);
                }

                std::memcpy(search_buffer + idx * dim, cluster.vectors,
                            cluster.size * dim * sizeof(float));
                std::memcpy(norms_buffer + idx, cluster.norms,
                            cluster.size * sizeof(float));
                for (int i = 0; i < cluster.size; i++)
                    search_ids[idx + i] = cluster.ids[i];
                idx += cluster.size;
            }
            // idx is the true vector count after compaction
            total_vectors = idx;

            if (total_vectors == 0) return py::make_tuple(py::list(), py::list());

            // Use precomputed norms to skip the per-vector norm recomputation
            float q_sq = 0;
            for (int i = 0; i < dim; i++) q_sq += ptr[i] * ptr[i];
            blas_l2_distances_precomp(ptr, &q_sq, 1, dim,
                                      search_buffer, norms_buffer, total_vectors,
                                      dists_buffer);

            std::vector<std::pair<int, float>> results;
            results.reserve(total_vectors);
            for (int i = 0; i < total_vectors; i++)
                results.push_back({search_ids[i], dists_buffer[i]});

            // Dedup when soft_k > 1 (same vector appears in multiple clusters)
            if (soft_k > 1) {
                std::unordered_set<int> seen;
                int w = 0;
                for (int i = 0; i < (int)results.size(); i++)
                    if (seen.insert(results[i].first).second)
                        results[w++] = results[i];
                results.resize(w);
            }

            int result_count = std::min(n_results, (int)results.size());
            std::partial_sort(results.begin(), results.begin() + result_count, results.end(),
                              [](const auto& a, const auto& b) { return a.second < b.second; });

            py::list py_ids, py_distances;
            for (int i = 0; i < result_count; i++) {
                py_ids.append(results[i].first);
                py_distances.append(results[i].second);
            }
            return py::make_tuple(py_ids, py_distances);
        }
    }

    // -----------------------------------------------------------------------
    // Batch search
    // -----------------------------------------------------------------------

    py::tuple search_batch(
        py::array_t<float, py::array::c_style | py::array::forcecast> queries_in,
        int n_results,
        int n_probes_in
    ) {
        auto buf = queries_in.request();
        if (buf.ndim != 2) throw std::runtime_error("Queries must be 2D array");
        int nq = (int)buf.shape[0];
        if (buf.shape[1] != dim) throw std::runtime_error("Query dimension mismatch");

        float* queries = (float*)buf.ptr;
        int probes = std::min(n_probes_in > 0 ? n_probes_in : nprobe, n_clusters);

        std::vector<float> centroid_dists_sq(nq * n_clusters);
        blas_l2_distances(queries, nq, dim, centroids, n_clusters, centroid_dists_sq.data());

        std::vector<std::pair<int, float>> sorted_dists(n_clusters);

        py::list all_ids;
        py::list all_dists;

        if (nq == 0) return py::make_tuple(all_ids, all_dists);

        if (use_pq) {
            for (int q = 0; q < nq; q++) {
                pq_codebook.compute_precomputed_tables(queries + q * dim, dim);

                for (int c = 0; c < n_clusters; c++)
                    sorted_dists[c] = {c, centroid_dists_sq[q * n_clusters + c]};
                std::sort(sorted_dists.begin(), sorted_dists.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });

                std::vector<std::pair<int, float>> results;

                for (int p = 0; p < probes; p++) {
                    int cluster_id = sorted_dists[p].first;
                    PQCluster& pq_cluster = pq_clusters[cluster_id];
                    for (int i = 0; i < pq_cluster.size; i++) {
                        int id = pq_cluster.ids[i];
                        if (deleted_ids.count(id)) continue;
                        float dist = 0;
                        for (int m = 0; m < pq_m; m++)
                            dist += pq_codebook.precomputed_tables[m][pq_cluster.codes[i * pq_m + m]];
                        results.push_back({id, dist});
                    }
                }

                // Dedup for soft_k > 1
                if (soft_k > 1) {
                    std::unordered_set<int> seen;
                    int w = 0;
                    for (int i = 0; i < (int)results.size(); i++)
                        if (seen.insert(results[i].first).second)
                            results[w++] = results[i];
                    results.resize(w);
                }

                int result_count = std::min(n_results, (int)results.size());
                std::partial_sort(results.begin(), results.begin() + result_count, results.end(),
                                  [](const auto& a, const auto& b) { return a.second < b.second; });

                py::list py_ids, py_distances;
                for (int i = 0; i < result_count; i++) {
                    py_ids.append(results[i].first);
                    py_distances.append(results[i].second);
                }
                all_ids.append(py_ids);
                all_dists.append(py_distances);
            }
        } else {
            // Inverted cluster scan: group queries by probed cluster, then issue
            // one GEMM per cluster instead of one GEMV per (query × cluster) pair.
            // Turns nq*nprobe thin GEMVs into ≤n_clusters fat GEMMs.

            // Precompute per-query squared norms (reused across all cluster scans)
            std::vector<float> q_sq(nq);
            for (int q = 0; q < nq; q++) {
                float s = 0;
                const float* qv = queries + q * dim;
                for (int i = 0; i < dim; i++) s += qv[i] * qv[i];
                q_sq[q] = s;
            }

            // Build cluster→prober inverted lists
            std::vector<std::vector<int>> cluster_probers(n_clusters);
            for (int q = 0; q < nq; q++) {
                for (int c = 0; c < n_clusters; c++)
                    sorted_dists[c] = {c, centroid_dists_sq[q * n_clusters + c]};
                std::partial_sort(sorted_dists.begin(), sorted_dists.begin() + probes,
                                  sorted_dists.end(),
                                  [](const auto& a, const auto& b) { return a.second < b.second; });
                for (int p = 0; p < probes; p++)
                    cluster_probers[sorted_dists[p].first].push_back(q);
            }

            // Per-query candidate lists
            std::vector<std::vector<std::pair<int,float>>> candidates(nq);

            // Process each probed cluster once with a batched GEMM
            for (int c = 0; c < n_clusters; c++) {
                auto& probers = cluster_probers[c];
                if (probers.empty()) continue;

                Cluster& cluster = clusters[c];
                if (cluster.size == 0) continue;

                bool has_deleted = false;
                for (int i = 0; i < cluster.size && !has_deleted; i++)
                    if (deleted_ids.count(cluster.ids[i])) has_deleted = true;
                if (has_deleted) {
                    compact_cluster(c);
                    compact_pq_cluster(c);
                }

                int mc = cluster.size;
                if (mc == 0) continue;
                int np = (int)probers.size();

                // Collect contiguous query rows and their precomputed norms
                std::vector<float> qbatch(np * dim);
                std::vector<float> qbatch_sq(np);
                for (int qi = 0; qi < np; qi++) {
                    int q = probers[qi];
                    std::memcpy(qbatch.data() + qi * dim, queries + q * dim, dim * sizeof(float));
                    qbatch_sq[qi] = q_sq[q];
                }

                // Batched GEMM: (np × d) · (d × mc) with precomputed vector norms
                std::vector<float> dmat(np * mc);
                blas_l2_distances_precomp(qbatch.data(), qbatch_sq.data(), np,
                                          dim,
                                          cluster.vectors, cluster.norms, mc,
                                          dmat.data());

                // Distribute results into per-query candidate lists
                for (int qi = 0; qi < np; qi++) {
                    int q = probers[qi];
                    auto& cands = candidates[q];
                    cands.reserve(cands.size() + mc);
                    const float* row = dmat.data() + qi * mc;
                    for (int vi = 0; vi < mc; vi++)
                        cands.push_back({cluster.ids[vi], row[vi]});
                }
            }

            // Per-query dedup + top-k
            for (int q = 0; q < nq; q++) {
                auto& cands = candidates[q];
                if (soft_k > 1) {
                    std::unordered_set<int> seen;
                    int w = 0;
                    for (int i = 0; i < (int)cands.size(); i++)
                        if (seen.insert(cands[i].first).second)
                            cands[w++] = cands[i];
                    cands.resize(w);
                }
                int result_count = std::min(n_results, (int)cands.size());
                if (result_count > 0)
                    std::partial_sort(cands.begin(), cands.begin() + result_count, cands.end(),
                                      [](const auto& a, const auto& b) { return a.second < b.second; });
                py::list py_ids, py_distances;
                for (int i = 0; i < result_count; i++) {
                    py_ids.append(cands[i].first);
                    py_distances.append(cands[i].second);
                }
                all_ids.append(py_ids);
                all_dists.append(py_distances);
            }
        }

        return py::make_tuple(all_ids, all_dists);
    }

    // -----------------------------------------------------------------------
    // Train
    // -----------------------------------------------------------------------

    void train(py::array_t<float> data) {
        auto buf = data.request();
        float* ptr = (float*)buf.ptr;
        ssize_t n = buf.shape[0];

        train_centroids(ptr, (int)n);

        if (use_pq) {
            pq_codebook.train(ptr, (int)n, dim);
        }

        for (ssize_t i = 0; i < n; i++) {
            add_vector_no_centroid_update(ptr + i * dim);
        }
    }

    // -----------------------------------------------------------------------
    // Batch insert
    // -----------------------------------------------------------------------

    void insert_batch(py::array_t<float> data) {
        auto buf = data.request();
        float* ptr = (float*)buf.ptr;
        ssize_t n = buf.shape[0];

        // BLAS: compute all n vectors × all n_clusters centroid distances at once
        std::vector<float> all_dists(n * n_clusters);
        blas_l2_distances(ptr, n, dim, centroids, n_clusters, all_dists.data());

        for (ssize_t i = 0; i < n; i++) {
            int id = n_vectors + (int)i;
            if (id >= (int)id_to_location.size())
                id_to_location.resize(id_to_location.size() * 2);
            id_to_location[id].clear();

            // Find top-soft_k clusters using the pre-computed distance row
            std::vector<int> cidx(n_clusters);
            std::iota(cidx.begin(), cidx.end(), 0);
            int actual_k = std::min(soft_k, n_clusters);
            std::partial_sort(cidx.begin(), cidx.begin() + actual_k, cidx.end(),
                [&](int a, int b) {
                    return all_dists[i * n_clusters + a] < all_dists[i * n_clusters + b];
                });

            for (int ki = 0; ki < actual_k; ki++) {
                int cluster_id = cidx[ki];
                int pos = clusters[cluster_id].size;
                clusters[cluster_id].add_vector(ptr + i * dim, dim, id);
                id_to_location[id].push_back({cluster_id, pos});
                cluster_live_count[cluster_id]++;
                if (use_pq) {
                    std::vector<uint8_t> codes(pq_m);
                    pq_codebook.encode(ptr + i * dim, dim, codes.data());
                    pq_clusters[cluster_id].add_codes(codes.data(), pq_m, id);
                }
            }
        }
        n_vectors += (int)n;

        // Rebalance: split clusters that are too large due to distribution drift
        rebalance_if_needed();

        // Auto-compact: flush deleted_ids when tombstone load exceeds 10%.
        // Keeps deleted_ids.count() at O(1) in compact_cluster.
        if (n_vectors > 0 && (int)deleted_ids.size() > n_vectors / 10)
            compact_all();
    }

    // -----------------------------------------------------------------------
    // Insert with pre-computed cluster assignments (GPU path)
    // Accepts vectors (n × d) and assignments (n × soft_k) already computed
    // on the caller's device. Skips the centroid BLAS gemm entirely.
    // -----------------------------------------------------------------------

    void insert_batch_preassigned(py::array_t<float> data,
                                  py::array_t<int>   assignments) {
        auto buf  = data.request();
        auto abuf = assignments.request();
        float* ptr  = (float*)buf.ptr;
        int*   aptr = (int*)abuf.ptr;
        ssize_t n       = buf.shape[0];
        int     actual_k = (int)abuf.shape[1];

        for (ssize_t i = 0; i < n; i++) {
            int id = n_vectors + (int)i;
            if (id >= (int)id_to_location.size())
                id_to_location.resize(id_to_location.size() * 2);
            id_to_location[id].clear();

            for (int ki = 0; ki < actual_k; ki++) {
                int cluster_id = aptr[i * actual_k + ki];
                if (cluster_id < 0 || cluster_id >= n_clusters) continue;
                int pos = clusters[cluster_id].size;
                clusters[cluster_id].add_vector(ptr + i * dim, dim, id);
                id_to_location[id].push_back({cluster_id, pos});
                cluster_live_count[cluster_id]++;
                if (use_pq) {
                    std::vector<uint8_t> codes(pq_m);
                    pq_codebook.encode(ptr + i * dim, dim, codes.data());
                    pq_clusters[cluster_id].add_codes(codes.data(), pq_m, id);
                }
            }
        }
        n_vectors += (int)n;
        rebalance_if_needed();
        if (n_vectors > 0 && (int)deleted_ids.size() > n_vectors / 10)
            compact_all();
    }

    // -----------------------------------------------------------------------
    // Batch delete
    // -----------------------------------------------------------------------

    void delete_batch(py::array_t<int> ids) {
        auto buf = ids.request();
        int* ptr = (int*)buf.ptr;
        for (ssize_t i = 0; i < buf.shape[0]; i++) {
            delete_vector(ptr[i]);
        }
    }

    // -----------------------------------------------------------------------
    // Brute force search (exact; used as recall oracle in benchmarks)
    // -----------------------------------------------------------------------

    py::tuple brute_force_search(
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        int n_results
    ) {
        auto buf = query.request();
        float* ptr = (float*)buf.ptr;

        // Collect all live, unique vectors across all clusters
        int total_physical = 0;
        for (int c = 0; c < n_clusters; c++)
            total_physical += clusters[c].size;

        if (total_physical == 0) return py::make_tuple(py::list(), py::list());

        std::vector<float> all_vectors(total_physical * dim);
        std::vector<int>   all_ids(total_physical);
        std::unordered_set<int> seen_ids;
        int idx = 0;

        for (int c = 0; c < n_clusters; c++) {
            for (int i = 0; i < clusters[c].size; i++) {
                int id = clusters[c].ids[i];
                if (deleted_ids.count(id)) continue;
                if (!seen_ids.insert(id).second) continue;  // dedup for soft_k > 1
                std::memcpy(all_vectors.data() + idx * dim,
                            clusters[c].vectors + i * dim, dim * sizeof(float));
                all_ids[idx++] = id;
            }
        }

        int total = idx;
        if (total == 0) return py::make_tuple(py::list(), py::list());

        std::vector<float> dists(total);
        blas_l2_distances(ptr, 1, dim, all_vectors.data(), total, dists.data());

        std::vector<std::pair<int, float>> results;
        results.reserve(total);
        for (int i = 0; i < total; i++)
            results.push_back({all_ids[i], dists[i]});

        int result_count = std::min(n_results, total);
        std::partial_sort(results.begin(), results.begin() + result_count, results.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });

        py::list py_ids, py_distances;
        for (int i = 0; i < result_count; i++) {
            py_ids.append(results[i].first);
            py_distances.append(results[i].second);
        }
        return py::make_tuple(py_ids, py_distances);
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    // ── Persistence helpers ────────────────────────────────────────────────────

    py::array_t<float> get_centroids() {
        auto result = py::array_t<float>({n_clusters, dim});
        std::memcpy(result.mutable_data(), centroids, (size_t)n_clusters * dim * sizeof(float));
        return result;
    }

    py::array_t<float> get_cluster_vectors(int c) {
        const Cluster& cl = clusters[c];
        auto result = py::array_t<float>({cl.size, dim});
        if (cl.size > 0)
            std::memcpy(result.mutable_data(), cl.vectors, (size_t)cl.size * dim * sizeof(float));
        return result;
    }

    py::array_t<int> get_cluster_ids(int c) {
        const Cluster& cl = clusters[c];
        auto result = py::array_t<int>(std::vector<ssize_t>{cl.size});
        if (cl.size > 0)
            std::memcpy(result.mutable_data(), cl.ids, cl.size * sizeof(int));
        return result;
    }

    std::vector<int> get_deleted_ids_list() {
        return std::vector<int>(deleted_ids.begin(), deleted_ids.end());
    }

    std::vector<std::vector<std::pair<int,int>>> get_id_to_location_cpp() {
        return id_to_location;
    }

    void set_centroids(py::array_t<float> arr) {
        auto buf = arr.request();
        int nc = (int)buf.shape[0];
        if (centroids) free(centroids);
        centroids = (float*)aligned_alloc(32, (size_t)nc * dim * sizeof(float));
        std::memcpy(centroids, buf.ptr, (size_t)nc * dim * sizeof(float));
        int old_nc = n_clusters;
        n_clusters = nc;
        clusters.resize(nc);
        pq_clusters.resize(nc);
        cluster_live_count.resize(nc, 0);
        for (int c = old_nc; c < nc; c++) {
            std::string path = use_mmap
                ? mmap_dir + "/cluster_" + std::to_string(c) + ".bin"
                : std::string();
            clusters[c].init(dim, path, /*truncate_new=*/use_mmap);
        }
    }

    // Heap-mode restore: load vectors + ids from numpy arrays (non-mmap load path).
    void restore_cluster(int c, py::array_t<float> vecs, py::array_t<int> ids_arr) {
        auto vbuf = vecs.request();
        auto ibuf = ids_arr.request();
        int sz = (int)vbuf.shape[0];
        Cluster& cl = clusters[c];
        cl._free_vectors();
        if (cl.ids)   { free(cl.ids);   cl.ids   = nullptr; }
        if (cl.norms) { free(cl.norms); cl.norms = nullptr; }
        cl.size     = sz;
        cl.capacity = sz;
        if (sz > 0) {
            cl.vectors = (float*)aligned_alloc(32, (size_t)sz * dim * sizeof(float));
            cl.ids     = (int*)malloc((size_t)sz * sizeof(int));
            cl.norms   = (float*)malloc((size_t)sz * sizeof(float));
            std::memcpy(cl.vectors, vbuf.ptr, (size_t)sz * dim * sizeof(float));
            std::memcpy(cl.ids,     ibuf.ptr, (size_t)sz * sizeof(int));
            const float* vptr = (const float*)vbuf.ptr;
            for (int i = 0; i < sz; i++) {
                float s = 0;
                for (int j = 0; j < dim; j++) s += vptr[i*dim+j] * vptr[i*dim+j];
                cl.norms[i] = s;
            }
        }
        std::memset(cl.vec_sum, 0, dim * sizeof(float));
        if (c < (int)cluster_live_count.size())
            cluster_live_count[c] = sz;
    }

    // mmap-mode restore: vectors live in cluster_c.bin (already mmap'd from
    // constructor); just load the IDs from the numpy array and set size.
    void restore_cluster_mmap(int c, py::array_t<int> ids_arr) {
        auto ibuf = ids_arr.request();
        int sz = (int)ibuf.shape[0];
        Cluster& cl = clusters[c];
        if (cl.ids)   { free(cl.ids);   cl.ids   = nullptr; }
        if (cl.norms) { free(cl.norms); cl.norms = nullptr; }
        int alloc = std::max(sz, cl.capacity);  // at least as many as capacity
        cl.ids   = (int*)malloc((size_t)alloc * sizeof(int));
        cl.norms = (float*)malloc((size_t)alloc * sizeof(float));
        if (sz > 0) {
            std::memcpy(cl.ids, ibuf.ptr, (size_t)sz * sizeof(int));
            for (int i = 0; i < sz; i++) {
                float s = 0;
                for (int j = 0; j < dim; j++) s += cl.vectors[i*dim+j] * cl.vectors[i*dim+j];
                cl.norms[i] = s;
            }
        }
        cl.size = sz;
        std::memset(cl.vec_sum, 0, dim * sizeof(float));
        if (c < (int)cluster_live_count.size())
            cluster_live_count[c] = sz;
    }

    void restore_state(std::vector<int> del_ids,
                       std::vector<std::vector<std::pair<int,int>>> id2loc,
                       int nv) {
        deleted_ids    = std::unordered_set<int>(del_ids.begin(), del_ids.end());
        id_to_location = std::move(id2loc);
        n_vectors      = nv;
        // Recompute live counts: restored clusters have size = physical slots,
        // but some of those IDs may be in the tombstone set.
        cluster_live_count.assign(n_clusters, 0);
        for (int c = 0; c < n_clusters; c++) {
            int live = 0;
            for (int i = 0; i < clusters[c].size; i++)
                if (!deleted_ids.count(clusters[c].ids[i])) live++;
            cluster_live_count[c] = live;
        }
    }

    py::dict get_stats() {
        py::dict stats;
        int total_physical = 0;
        int max_cluster_size = 0;
        int max_live_size = 0;
        int total_live = 0;
        for (int c = 0; c < n_clusters; c++) {
            total_physical += clusters[c].size;
            max_cluster_size = std::max(max_cluster_size, clusters[c].size);
            total_live += cluster_live_count[c];
            max_live_size = std::max(max_live_size, cluster_live_count[c]);
        }
        stats["n_vectors"]          = n_vectors;
        stats["n_physical_slots"]   = total_physical;
        stats["n_clusters"]         = n_clusters;
        stats["dim"]                = dim;
        stats["use_pq"]             = use_pq;
        stats["soft_k"]             = soft_k;
        stats["deleted_count"]      = (int)deleted_ids.size();
        stats["max_cluster_size"]   = max_live_size;   // live vectors only
        stats["mean_cluster_size"]  = n_clusters > 0
            ? (float)total_live / n_clusters : 0.f;
        if (use_pq) {
            stats["pq_m"]  = pq_m;
            stats["pq_ks"] = pq_ks;
            size_t orig_size = (size_t)total_physical * dim * sizeof(float);
            size_t pq_size   = (size_t)total_physical * pq_m;
            stats["compression_ratio"] = pq_size > 0 ? (float)orig_size / pq_size : 0.f;
        }
        return stats;
    }
};

PYBIND11_MODULE(copenhagen, m) {
    m.doc() = "Copenhagen - A Quantum-Inspired Dynamic IVF Index";

    py::class_<DynamicIVF>(m, "DynamicIVF")
        .def(py::init<int, int, int, int, int, int, int, bool, const std::string&>(),
             py::arg("dim"),
             py::arg("n_clusters"),
             py::arg("nprobe")   = 1,
             py::arg("use_pq")   = 0,
             py::arg("pq_m")     = 8,
             py::arg("pq_ks")    = 256,
             py::arg("soft_k")   = 1,
             py::arg("use_mmap") = false,
             py::arg("mmap_dir") = "")
        .def("train",          &DynamicIVF::train,          py::arg("data"))
        .def("insert",         &DynamicIVF::insert_vector,  py::arg("vector"))
        .def("insert_batch",             &DynamicIVF::insert_batch,             py::arg("data"))
        .def("insert_batch_preassigned", &DynamicIVF::insert_batch_preassigned,
             py::arg("data"), py::arg("assignments"))
        .def("delete",         &DynamicIVF::delete_vector,  py::arg("id"))
        .def("delete_batch",   &DynamicIVF::delete_batch,   py::arg("ids"))
        .def("search",         &DynamicIVF::search,
             py::arg("query"), py::arg("n_results") = 10, py::arg("n_probes") = 1)
        .def("search_batch",   &DynamicIVF::search_batch,
             py::arg("queries"), py::arg("n_results") = 10, py::arg("n_probes") = 1)
        .def("brute_force_search", &DynamicIVF::brute_force_search,
             py::arg("query"), py::arg("n_results") = 10)
        .def("get_stats",             &DynamicIVF::get_stats)
        .def("compact",               &DynamicIVF::compact_all)
        // Persistence
        .def("get_centroids",         &DynamicIVF::get_centroids)
        .def("get_cluster_vectors",   &DynamicIVF::get_cluster_vectors,   py::arg("cluster_id"))
        .def("get_cluster_ids",       &DynamicIVF::get_cluster_ids,       py::arg("cluster_id"))
        .def("get_deleted_ids_list",  &DynamicIVF::get_deleted_ids_list)
        .def("get_id_to_location",    &DynamicIVF::get_id_to_location_cpp)
        .def("set_centroids",         &DynamicIVF::set_centroids,         py::arg("centroids"))
        .def("restore_cluster",       &DynamicIVF::restore_cluster,       py::arg("cluster_id"), py::arg("vectors"), py::arg("ids"))
        .def("restore_cluster_mmap",  &DynamicIVF::restore_cluster_mmap,  py::arg("cluster_id"), py::arg("ids"))
        .def("restore_state",         &DynamicIVF::restore_state,         py::arg("deleted_ids"), py::arg("id_to_location"), py::arg("n_vectors"))
        .def_readwrite("split_threshold",  &DynamicIVF::split_threshold)
        .def_readwrite("max_split_iters",  &DynamicIVF::max_split_iters)
        .def_readwrite("soft_k",           &DynamicIVF::soft_k)
        .def_readwrite("use_mmap",         &DynamicIVF::use_mmap)
        .def_readwrite("mmap_dir",         &DynamicIVF::mmap_dir);
}
