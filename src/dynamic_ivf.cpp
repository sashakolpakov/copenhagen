/*
 * Copenhagen: A Quantum-Inspired Dynamic Index
 * 
 * Based on the Copenhagen interpretation of quantum mechanics:
 * - Superposition: PQ encodes vectors as superpositions of subspace codes
 * - Entanglement: Vectors in the same cluster are quantum-correlated
 * - Wave Function Collapse: Search "collapses" to nearest clusters
 * - Heisenberg Uncertainty: Recall vs speed tradeoff (can't optimize both)
 * - Decoherence: Lazy deletion = vectors decohere from cluster state
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <cstring>
#include <cstdlib>
#include <ctime>

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
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
        uint32x4_t indices = vld1q_u32((const uint32_t*)(codes + m));
        
        float32x4_t d0 = vld1q_f32(tables[m] + codes[m]);
        float32x4_t d1 = vld1q_f32(tables[m+1] + codes[m+1]);
        float32x4_t d2 = vld1q_f32(tables[m+2] + codes[m+2]);
        float32x4_t d3 = vld1q_f32(tables[m+3] + codes[m+3]);
        
        acc = vaddq_f32(acc, vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3)));
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
    float* vectors;
    float* vec_sum;
    int* ids;
    int size;
    int capacity;
    
    Cluster() : vectors(nullptr), vec_sum(nullptr), ids(nullptr), size(0), capacity(0) {}
    
    ~Cluster() {
        if (vectors) free(vectors);
        if (vec_sum) free(vec_sum);
        if (ids) free(ids);
    }
    
    void init(int dim) {
        vec_sum = (float*)aligned_alloc(32, dim * sizeof(float));
        std::memset(vec_sum, 0, dim * sizeof(float));
    }
    
    void reserve(int cap, int dim) {
        if (cap > capacity) {
            if (vectors) free(vectors);
            if (ids) free(ids);
            vectors = (float*)aligned_alloc(32, cap * dim * sizeof(float));
            ids = (int*)malloc(cap * sizeof(int));
            capacity = cap;
        }
    }
    
    void add_vector(const float* vec, int dim, int id) {
        if (size >= capacity) {
            capacity = capacity == 0 ? 64 : capacity * 2;
            float* new_vecs = (float*)aligned_alloc(32, capacity * dim * sizeof(float));
            int* new_ids = (int*)malloc(capacity * sizeof(int));
            if (vectors) {
                std::memcpy(new_vecs, vectors, size * dim * sizeof(float));
                std::memcpy(new_ids, ids, size * sizeof(int));
                free(vectors);
                free(ids);
            }
            vectors = new_vecs;
            ids = new_ids;
        }
        std::memcpy(vectors + size * dim, vec, dim * sizeof(float));
        ids[size] = id;
        for (int i = 0; i < dim; i++) {
            vec_sum[i] += vec[i];
        }
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
        
        centroids = (float**)malloc(M * sizeof(float*));
        precomputed_tables = (float**)malloc(M * sizeof(float*));
        for (int m = 0; m < M; m++) {
            centroids[m] = (float*)aligned_alloc(32, Ks * d_sub * sizeof(float));
            precomputed_tables[m] = (float*)aligned_alloc(32, 256 * Ks * sizeof(float));
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
            
            // Assign each vector to nearest centroid using BLAS
            // For each subspace m, compute distances to all Ks centroids
            for (int m = 0; m < M; m++) {
                int offset = m * d_sub;
                
                // Build matrix of centroids: Ks x d_sub
                std::vector<float> cent_mat(Ks * d_sub);
                for (int ks = 0; ks < Ks; ks++) {
                    for (int j = 0; j < d_sub; j++) {
                        cent_mat[ks * d_sub + j] = centroids[m][ks * d_sub + j];
                    }
                }
                
                // Build matrix of data: n x d_sub (extracted subspace)
                std::vector<float> data_mat(n * d_sub);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < d_sub; j++) {
                        data_mat[i * d_sub + j] = data[i * d + offset + j];
                    }
                }
                
                // Compute squared distances using BLAS: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
                std::vector<float> dists(n * Ks, 0.0f);
                
                // ||x||^2 for each data point
                std::vector<float> x_sq(n, 0.0f);
                for (int i = 0; i < n; i++) {
                    float s = 0;
                    for (int j = 0; j < d_sub; j++) {
                        float v = data_mat[i * d_sub + j];
                        s += v * v;
                    }
                    x_sq[i] = s;
                }
                
                // ||c||^2 for each centroid
                std::vector<float> c_sq(Ks, 0.0f);
                for (int ks = 0; ks < Ks; ks++) {
                    float s = 0;
                    for (int j = 0; j < d_sub; j++) {
                        float v = cent_mat[ks * d_sub + j];
                        s += v * v;
                    }
                    c_sq[ks] = s;
                }
                
                // -2 * x . c^T using BLAS
                // C = A * B^T where A=n x d_sub, B=Ks x d_sub, result=n x Ks
                float neg2 = -2.0f;
                float zero = 0.0f;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                           n, Ks, d_sub, neg2, 
                           data_mat.data(), d_sub,
                           cent_mat.data(), d_sub,
                           zero, dists.data(), Ks);
                
                // Add ||x||^2 and ||c||^2
                for (int i = 0; i < n; i++) {
                    for (int ks = 0; ks < Ks; ks++) {
                        dists[i * Ks + ks] += x_sq[i] + c_sq[ks];
                    }
                }
                
                // Assign and accumulate
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
            
            // Update centroids
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
#if defined(USE_ACCELERATE)
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

class DynamicIVF {
public:
    int dim;
    int n_clusters;
    int nprobe;
    int use_pq;
    int pq_m;
    int pq_ks;
    
    float* centroids;
    int n_vectors;
    
    std::vector<Cluster> clusters;
    std::vector<PQCluster> pq_clusters;
    std::vector<int> search_ids;
    PQCodebook pq_codebook;
    
    std::vector<std::pair<int, int>> id_to_location;
    
    int max_vectors_per_probe;
    float* search_buffer;
    float* dists_buffer;
    
    std::mt19937 rng;
    
    DynamicIVF(int dim, int n_clusters, int nprobe = 1, int use_pq = 0, int pq_m = 8, int pq_ks = 256)
        : dim(dim), n_clusters(n_clusters), nprobe(nprobe), use_pq(use_pq), pq_m(pq_m), pq_ks(pq_ks) {
        
        rng.seed(42);
        n_vectors = 0;
        
        centroids = (float*)aligned_alloc(32, n_clusters * dim * sizeof(float));
        std::memset(centroids, 0, n_clusters * dim * sizeof(float));
        
        max_vectors_per_probe = 100000;
        search_buffer = (float*)aligned_alloc(32, max_vectors_per_probe * dim * sizeof(float));
        dists_buffer = (float*)aligned_alloc(32, max_vectors_per_probe * sizeof(float));
        search_ids.resize(max_vectors_per_probe);
        
        clusters.resize(n_clusters);
        pq_clusters.resize(n_clusters);
        id_to_location.resize(1000000);
        
        for (auto& c : clusters) {
            c.init(dim);
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
        free(dists_buffer);
    }
    
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
                    if (d < min_d) {
                        min_d = d;
                        best = c;
                    }
                }
                assignments[i] = best;
                counts[best]++;
            }
            
            for (int c = 0; c < n_clusters; c++) {
                std::memset(centroids + c * dim, 0, dim * sizeof(float));
            }
            
            for (int i = 0; i < n; i++) {
                int c = assignments[i];
                const float* v = data + i * dim;
                float* cent = centroids + c * dim;
                for (int d = 0; d < dim; d++) {
                    cent[d] += v[d];
                }
            }
            
            for (int c = 0; c < n_clusters; c++) {
                if (counts[c] > 0) {
                    float inv_count = 1.0f / counts[c];
                    float* cent = centroids + c * dim;
                    for (int d = 0; d < dim; d++) {
                        cent[d] *= inv_count;
                    }
                }
            }
        }
    }
    
    void insert_vector(const float* data) {
        int id = n_vectors;
        int cluster_id = find_nearest_centroid(data);
        int pos = clusters[cluster_id].size;
        clusters[cluster_id].add_vector(data, dim, id);
        
        if (use_pq) {
            std::vector<uint8_t> codes(pq_m);
            pq_codebook.encode(data, dim, codes.data());
            pq_clusters[cluster_id].add_codes(codes.data(), pq_m, id);
        }
        
        if (id >= (int)id_to_location.size()) {
            id_to_location.resize(id_to_location.size() * 2);
        }
        id_to_location[id] = {cluster_id, pos};
        
        n_vectors++;
    }
    
    void delete_vector(int id) {
    }
    
    py::tuple search(
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        int n_results,
        int n_probes_in
    ) {
        auto buf = query.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Query must be 1D array");
        }
        float* ptr = (float*)buf.ptr;
        ssize_t query_dim = buf.shape[0];
        
        if (query_dim != dim) {
            throw std::runtime_error("Query dimension mismatch");
        }
        
        int probes = std::min(n_probes_in > 0 ? n_probes_in : nprobe, n_clusters);
        
        std::vector<float> centroid_dists_sq(n_clusters);
        blas_l2_distances(ptr, 1, dim, centroids, n_clusters, centroid_dists_sq.data());
        
        std::vector<std::pair<int, float>> centroid_dists;
        centroid_dists.reserve(n_clusters);
        for (int i = 0; i < n_clusters; i++) {
            centroid_dists.push_back({i, centroid_dists_sq[i]});
        }
        std::sort(centroid_dists.begin(), centroid_dists.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        
        int total_vectors = 0;
        for (int p = 0; p < probes; p++) {
            total_vectors += clusters[centroid_dists[p].first].size;
        }
        
        if (total_vectors == 0) {
            py::list py_ids;
            py::list py_distances;
            return py::make_tuple(py_ids, py_distances);
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
                    uint8_t* codes = pq_cluster.codes + i * pq_m;
                    float pq_dist = 0;
                    for (int m = 0; m < pq_m; m++) {
                        pq_dist += pq_codebook.precomputed_tables[m][codes[m]];
                    }
                    pq_results.push_back({pq_cluster.ids[i], pq_dist});
                }
            }
            
            int n_pq = std::min(rerank_k, (int)pq_results.size());
            std::partial_sort(pq_results.begin(), pq_results.begin() + n_pq, pq_results.end(),
                            [](const auto& a, const auto& b) { return a.second < b.second; });
            
            std::vector<int> rerank_ids(n_pq);
            for (int i = 0; i < n_pq; i++) {
                rerank_ids[i] = pq_results[i].first;
            }
            
            std::vector<float> rerank_vectors(n_pq * dim);
            for (int i = 0; i < n_pq; i++) {
                int id = rerank_ids[i];
                auto [c, pos] = id_to_location[id];
                std::memcpy(rerank_vectors.data() + i * dim, clusters[c].vectors + pos * dim, dim * sizeof(float));
            }
            
            std::vector<float> exact_dists(n_pq);
            blas_l2_distances(ptr, 1, dim, rerank_vectors.data(), n_pq, exact_dists.data());
            
            std::vector<std::pair<int, float>> final_results;
            final_results.reserve(n_pq);
            for (int i = 0; i < n_pq; i++) {
                final_results.push_back({rerank_ids[i], exact_dists[i]});
            }
            
            int result_count = std::min(n_results, (int)final_results.size());
            std::partial_sort(final_results.begin(), final_results.begin() + result_count, final_results.end(),
                            [](const auto& a, const auto& b) { return a.second < b.second; });
            
            py::list py_ids;
            py::list py_distances;
            for (int i = 0; i < result_count; i++) {
                py_ids.append(final_results[i].first);
                py_distances.append(final_results[i].second);
            }
            
            return py::make_tuple(py_ids, py_distances);
        } else {
            if (total_vectors > max_vectors_per_probe) {
                max_vectors_per_probe = total_vectors;
                free(search_buffer);
                free(dists_buffer);
                search_buffer = (float*)aligned_alloc(32, max_vectors_per_probe * dim * sizeof(float));
                dists_buffer = (float*)aligned_alloc(32, max_vectors_per_probe * sizeof(float));
                search_ids.resize(max_vectors_per_probe);
            }
            
            int idx = 0;
            for (int p = 0; p < probes; p++) {
                Cluster& cluster = clusters[centroid_dists[p].first];
                if (cluster.size > 0) {
                    std::memcpy(search_buffer + idx * dim, cluster.vectors, cluster.size * dim * sizeof(float));
                    for (int i = 0; i < cluster.size; i++) {
                        search_ids[idx + i] = cluster.ids[i];
                    }
                    idx += cluster.size;
                }
            }
            
            blas_l2_distances(ptr, 1, dim, search_buffer, total_vectors, dists_buffer);
            
            std::vector<std::pair<int, float>> results;
            results.reserve(total_vectors);
            for (int i = 0; i < total_vectors; i++) {
                results.push_back({search_ids[i], dists_buffer[i]});
            }
            
            int result_count = std::min(n_results, (int)results.size());
            std::partial_sort(results.begin(), results.begin() + result_count, results.end(),
                              [](const auto& a, const auto& b) { return a.second < b.second; });
            
            py::list py_ids;
            py::list py_distances;
            for (int i = 0; i < result_count; i++) {
                py_ids.append(results[i].first);
                py_distances.append(results[i].second);
            }
            
            return py::make_tuple(py_ids, py_distances);
        }
    }
    
    py::tuple search_batch(
        py::array_t<float, py::array::c_style | py::array::forcecast> queries_in,
        int n_results,
        int n_probes_in
    ) {
        auto buf = queries_in.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Queries must be 2D array");
        }
        int nq = (int)buf.shape[0];
        ssize_t query_dim = buf.shape[1];
        
        if (query_dim != dim) {
            throw std::runtime_error("Query dimension mismatch");
        }
        
        float* queries = (float*)buf.ptr;
        int probes = std::min(n_probes_in > 0 ? n_probes_in : nprobe, n_clusters);
        
        std::vector<float> centroid_dists_sq(nq * n_clusters);
        blas_l2_distances(queries, nq, dim, centroids, n_clusters, centroid_dists_sq.data());
        
        std::vector<std::pair<int, float>> sorted_dists(n_clusters);
        
        py::list all_ids;
        py::list all_dists;
        
        int max_vectors = 0;
        for (int q = 0; q < nq; q++) {
            for (int c = 0; c < n_clusters; c++) {
                sorted_dists[c] = {c, centroid_dists_sq[q * n_clusters + c]};
            }
            std::sort(sorted_dists.begin(), sorted_dists.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });
            int total = 0;
            for (int p = 0; p < probes; p++) {
                total += clusters[sorted_dists[p].first].size;
            }
            max_vectors = std::max(max_vectors, total);
        }
        
        if (max_vectors == 0) {
            for (int qi = 0; qi < nq; qi++) {
                py::list py_ids;
                py::list py_distances;
                all_ids.append(py_ids);
                all_dists.append(py_distances);
            }
            return py::make_tuple(all_ids, all_dists);
        }
        
        if (use_pq) {
            for (int q = 0; q < nq; q++) {
                pq_codebook.compute_precomputed_tables(queries + q * dim, dim);
                
                for (int c = 0; c < n_clusters; c++) {
                    sorted_dists[c] = {c, centroid_dists_sq[q * n_clusters + c]};
                }
                std::sort(sorted_dists.begin(), sorted_dists.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
                
                std::vector<std::pair<int, float>> results;
                
                std::vector<int> vec_ids;
                std::vector<uint8_t> all_codes;
                
                for (int p = 0; p < probes; p++) {
                    int cluster_id = sorted_dists[p].first;
                    PQCluster& pq_cluster = pq_clusters[cluster_id];
                    
                    for (int i = 0; i < pq_cluster.size; i++) {
                        vec_ids.push_back(pq_cluster.ids[i]);
                        all_codes.insert(all_codes.end(), pq_cluster.codes + i * pq_m, pq_cluster.codes + (i + 1) * pq_m);
                    }
                }
                
                std::vector<float> vec_dists(vec_ids.size());
                for (size_t i = 0; i < vec_ids.size(); i++) {
                    float dist = 0;
                    for (int m = 0; m < pq_m; m++) {
                        dist += pq_codebook.precomputed_tables[m][all_codes[i * pq_m + m]];
                    }
                    vec_dists[i] = dist;
                }
                
                for (size_t i = 0; i < vec_ids.size(); i++) {
                    results.push_back({vec_ids[i], vec_dists[i]});
                }
                
                int result_count = std::min(n_results, (int)results.size());
                std::partial_sort(results.begin(), results.begin() + result_count, results.end(),
                                  [](const auto& a, const auto& b) { return a.second < b.second; });
                
                py::list py_ids;
                py::list py_distances;
                for (int i = 0; i < result_count; i++) {
                    py_ids.append(results[i].first);
                    py_distances.append(results[i].second);
                }
                
                all_ids.append(py_ids);
                all_dists.append(py_distances);
            }
        } else {
            if (max_vectors > max_vectors_per_probe) {
                max_vectors_per_probe = max_vectors;
                free(search_buffer);
                free(dists_buffer);
                search_buffer = (float*)aligned_alloc(32, max_vectors_per_probe * dim * sizeof(float));
                dists_buffer = (float*)aligned_alloc(32, max_vectors_per_probe * sizeof(float));
                search_ids.resize(max_vectors_per_probe);
            }
            
            std::vector<float> dists(nq * max_vectors);
            std::vector<int> batch_ids(max_vectors);
            
            for (int q = 0; q < nq; q++) {
                for (int c = 0; c < n_clusters; c++) {
                    sorted_dists[c] = {c, centroid_dists_sq[q * n_clusters + c]};
                }
                std::sort(sorted_dists.begin(), sorted_dists.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
                
                int idx = 0;
                for (int p = 0; p < probes; p++) {
                    Cluster& cluster = clusters[sorted_dists[p].first];
                    if (cluster.size > 0) {
                        std::memcpy(search_buffer + idx * dim, cluster.vectors, cluster.size * dim * sizeof(float));
                        for (int i = 0; i < cluster.size; i++) {
                            batch_ids[idx + i] = cluster.ids[i];
                        }
                        idx += cluster.size;
                    }
                }
                
                blas_l2_distances(queries + q * dim, 1, dim, search_buffer, idx, dists.data() + q * max_vectors);
            }
            
            for (int q = 0; q < nq; q++) {
                for (int c = 0; c < n_clusters; c++) {
                    sorted_dists[c] = {c, centroid_dists_sq[q * n_clusters + c]};
                }
                std::sort(sorted_dists.begin(), sorted_dists.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
                
                int total = 0;
                for (int p = 0; p < probes; p++) {
                    total += clusters[sorted_dists[p].first].size;
                }
                
                std::vector<std::pair<int, float>> results;
                results.reserve(total);
                for (int i = 0; i < total; i++) {
                    results.push_back({batch_ids[q * max_vectors + i], dists[q * max_vectors + i]});
                }
                
                int result_count = std::min(n_results, (int)results.size());
                std::partial_sort(results.begin(), results.begin() + result_count, results.end(),
                                  [](const auto& a, const auto& b) { return a.second < b.second; });
                
                py::list py_ids;
                py::list py_distances;
                for (int i = 0; i < result_count; i++) {
                    py_ids.append(results[i].first);
                    py_distances.append(results[i].second);
                }
                
                all_ids.append(py_ids);
                all_dists.append(py_distances);
            }
        }
        
        return py::make_tuple(all_ids, all_dists);
    }
    
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
    
    void add_vector_no_centroid_update(const float* data) {
        int id = n_vectors;
        int cluster_id = find_nearest_centroid(data);
        int pos = clusters[cluster_id].size;
        clusters[cluster_id].add_vector(data, dim, id);
        
        if (use_pq) {
            std::vector<uint8_t> codes(pq_m);
            pq_codebook.encode(data, dim, codes.data());
            pq_clusters[cluster_id].add_codes(codes.data(), pq_m, id);
        }
        
        if (id >= (int)id_to_location.size()) {
            id_to_location.resize(id_to_location.size() * 2);
        }
        id_to_location[id] = {cluster_id, pos};
        
        n_vectors++;
    }
    
    void insert_batch(py::array_t<float> data) {
        auto buf = data.request();
        float* ptr = (float*)buf.ptr;
        ssize_t n = buf.shape[0];
        
        for (ssize_t i = 0; i < n; i++) {
            insert_vector(ptr + i * dim);
        }
    }
    
    void delete_batch(py::array_t<int> ids) {
        auto buf = ids.request();
        int* ptr = (int*)buf.ptr;
        ssize_t n = buf.shape[0];
        
        for (ssize_t i = 0; i < n; i++) {
            delete_vector(ptr[i]);
        }
    }
    
    py::tuple brute_force_search(
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        int n_results
    ) {
        auto buf = query.request();
        float* ptr = (float*)buf.ptr;
        
        int total = 0;
        for (int c = 0; c < n_clusters; c++) {
            total += clusters[c].size;
        }
        
        if (total == 0) {
            py::list py_ids;
            py::list py_distances;
            return py::make_tuple(py_ids, py_distances);
        }
        
        std::vector<float> all_vectors(total * dim);
        std::vector<int> all_ids(total);
        int idx = 0;
        for (int c = 0; c < n_clusters; c++) {
            if (clusters[c].size > 0) {
                std::memcpy(all_vectors.data() + idx * dim, clusters[c].vectors, clusters[c].size * dim * sizeof(float));
                for (int i = 0; i < clusters[c].size; i++) {
                    all_ids[idx + i] = clusters[c].ids[i];
                }
                idx += clusters[c].size;
            }
        }
        
        std::vector<float> dists(total);
        blas_l2_distances(ptr, 1, dim, all_vectors.data(), total, dists.data());
        
        std::vector<std::pair<int, float>> results;
        results.reserve(total);
        for (int i = 0; i < total; i++) {
            results.push_back({all_ids[i], dists[i]});
        }
        
        int result_count = std::min(n_results, (int)results.size());
        std::partial_sort(results.begin(), results.begin() + result_count, results.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });
        
        py::list py_ids;
        py::list py_distances;
        for (int i = 0; i < result_count; i++) {
            py_ids.append(results[i].first);
            py_distances.append(results[i].second);
        }
        
        return py::make_tuple(py_ids, py_distances);
    }
    
    py::dict get_stats() {
        py::dict stats;
        int total = 0;
        for (int c = 0; c < n_clusters; c++) {
            total += clusters[c].size;
        }
        stats["n_vectors"] = total;
        stats["n_clusters"] = n_clusters;
        stats["dim"] = dim;
        stats["use_pq"] = use_pq;
        if (use_pq) {
            stats["pq_m"] = pq_m;
            stats["pq_ks"] = pq_ks;
            size_t orig_size = total * dim * sizeof(float);
            size_t pq_size = total * pq_m;
            stats["compression_ratio"] = (float)orig_size / pq_size;
        }
        return stats;
    }
};

PYBIND11_MODULE(copenhagen, m) {
    m.doc() = "Copenhagen - A Quantum-Inspired Dynamic IVF Index";
    
    py::class_<DynamicIVF>(m, "DynamicIVF")
        .def(py::init<int, int, int, int, int, int>(),
             py::arg("dim"),
             py::arg("n_clusters"),
             py::arg("nprobe") = 1,
             py::arg("use_pq") = 0,
             py::arg("pq_m") = 8,
             py::arg("pq_ks") = 256)
        .def("train", &DynamicIVF::train, py::arg("data"))
        .def("insert", &DynamicIVF::insert_vector, py::arg("vector"))
        .def("insert_batch", &DynamicIVF::insert_batch, py::arg("data"))
        .def("delete", &DynamicIVF::delete_vector, py::arg("id"))
        .def("delete_batch", &DynamicIVF::delete_batch, py::arg("ids"))
        .def("search", &DynamicIVF::search, 
             py::arg("query"),
             py::arg("n_results") = 10,
             py::arg("n_probes") = 1)
        .def("search_batch", &DynamicIVF::search_batch,
             py::arg("queries"),
             py::arg("n_results") = 10,
             py::arg("n_probes") = 1)
        .def("brute_force_search", &DynamicIVF::brute_force_search,
             py::arg("query"),
             py::arg("n_results") = 10)
        .def("get_stats", &DynamicIVF::get_stats);
}
