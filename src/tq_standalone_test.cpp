// tq_standalone_test.cpp — prove the TurboQuant thesis in isolation.
//
// Builds a flat TurboQuant index over synthetic data, runs top-10 search using
// the quantized inner-product estimate, and reports recall@10 and
// recall-per-byte at 2/3/4 bits, calibration on/off. No BLAS, no Copenhagen —
// just src/turbo_quant.hpp vs exact brute-force L2.
//
// Build:  c++ -O3 -std=c++17 -march=native src/tq_standalone_test.cpp -o /tmp/tqtest
// Run:    /tmp/tqtest

#include "turbo_quant.hpp"
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using namespace cph;

// Clustered Gaussian blobs with per-vector random norm scaling — exercises the
// length-renormalization path (varying ||v||), not just unit-norm data.
static std::vector<float> make_data(int n, int dim, int n_blobs, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::uniform_real_distribution<float> scale(0.3f, 3.0f);
    std::vector<std::vector<float>> centers(n_blobs, std::vector<float>(dim));
    for (auto& c : centers) for (auto& x : c) x = g(rng) * 3.0f;
    std::vector<float> data((size_t)n * dim);
    std::uniform_int_distribution<int> pick(0, n_blobs - 1);
    for (int i = 0; i < n; i++) {
        int b = pick(rng);
        float s = scale(rng);
        for (int d = 0; d < dim; d++)
            data[(size_t)i * dim + d] = (centers[b][d] + g(rng)) * s;
    }
    return data;
}

static void exact_topk(const float* q, const float* data, int n, int dim,
                       int k, std::vector<int>& out) {
    std::vector<std::pair<float,int>> d(n);
    for (int i = 0; i < n; i++) {
        const float* v = data + (size_t)i * dim;
        float s = 0; for (int j = 0; j < dim; j++) { float t = q[j]-v[j]; s += t*t; }
        d[i] = {s, i};
    }
    std::partial_sort(d.begin(), d.begin()+k, d.end());
    out.resize(k); for (int i = 0; i < k; i++) out[i] = d[i].second;
}

static double recall_at(const std::vector<int>& got, const std::vector<int>& gt, int K) {
    int h = 0;
    for (int a = 0; a < (int)got.size(); a++)
        for (int b = 0; b < K; b++) if (got[a] == gt[b]) { h++; break; }
    return (double)h;
}

static void run(int dim, int n, int nq, int n_blobs, int K, int rerank) {
    printf("\n=== Synthetic: n=%d dim=%d queries=%d blobs=%d  top-%d  rerank=%d ===\n",
           n, dim, nq, n_blobs, K, rerank);
    std::vector<float> data = make_data(n, dim, n_blobs, 1);
    std::vector<float> queries = make_data(nq, dim, n_blobs, 999);

    std::vector<std::vector<int>> gt(nq);
    for (int q = 0; q < nq; q++)
        exact_topk(&queries[(size_t)q*dim], data.data(), n, dim, K, gt[q]);

    float float_bytes = dim * 4.0f;
    printf("float32 baseline: %.0f bytes/vec\n", float_bytes);
    printf("%-6s %-6s %11s %12s %12s %10s\n",
           "bits", "calib", "raw R@10", "rerankR@10", "bytes/vec", "compress");
    printf("----------------------------------------------------------------------\n");

    for (int bits : {2, 4}) {
        for (bool calib : {false, true}) {
            TurboQuantizer tq; tq.seed = 42;
            tq.train(data.data(), n, dim, bits, calib);
            std::vector<uint8_t> codes((size_t)n * dim);
            std::vector<float> scales(n), sqnorms(n);
            for (int i = 0; i < n; i++)
                tq.encode(&data[(size_t)i*dim], &codes[(size_t)i*dim], &scales[i], &sqnorms[i]);

            long raw_hits = 0, rr_hits = 0;
            std::vector<float> table; float bias;
            for (int q = 0; q < nq; q++) {
                const float* qp = &queries[(size_t)q*dim];
                float q_sq = 0; for (int j = 0; j < dim; j++) q_sq += qp[j]*qp[j];
                tq.build_query_table(qp, table, bias);
                std::vector<std::pair<float,int>> est(n);
                for (int i = 0; i < n; i++) {
                    float ip = tq.score_ip(table.data(), bias, &codes[(size_t)i*dim], scales[i]);
                    est[i] = {tq.l2sq(q_sq, sqnorms[i], ip), i};
                }
                int cand = std::min(rerank, n);
                std::partial_sort(est.begin(), est.begin()+cand, est.end());
                std::vector<int> raw(K); for (int i=0;i<K;i++) raw[i]=est[i].second;
                raw_hits += (long)recall_at(raw, gt[q], K);
                // rerank top-`cand` by exact L2 (models optional float rerank store)
                std::vector<std::pair<float,int>> rr(cand);
                for (int i = 0; i < cand; i++) {
                    const float* v = data.data() + (size_t)est[i].second*dim;
                    float s=0; for (int j=0;j<dim;j++){float t=qp[j]-v[j];s+=t*t;}
                    rr[i] = {s, est[i].second};
                }
                std::partial_sort(rr.begin(), rr.begin()+K, rr.end());
                std::vector<int> rk(K); for (int i=0;i<K;i++) rk[i]=rr[i].second;
                rr_hits += (long)recall_at(rk, gt[q], K);
            }
            float bpv = tq.packed_bytes_per_vector() + 8;
            printf("%-6d %-6s %11.4f %12.4f %12.1f %9.1fx\n",
                   bits, calib?"on":"off",
                   (double)raw_hits/(nq*K), (double)rr_hits/(nq*K),
                   bpv, float_bytes/bpv);
        }
    }
}

int main() {
    run(128,  20000, 1000, 50, 10, 200);   // SIFT-like: TurboQuant worst case
    run(768,  20000,  500, 50, 10, 200);   // BERT-like
    run(1536, 15000,  500, 50, 10, 200);   // OpenAI-like: TurboQuant sweet spot
    printf("\n(bytes/vec = packed codes [bits*dim/8] + 4B scale + 4B ||v||^2)\n");
    printf("IVFPQ reference (Copenhagen BENCHMARKS.md): 0.42-0.47 R@10 at 16x, 0.06-0.11 at 64x.\n");
    return 0;
}
