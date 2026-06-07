// bench_score_ip.cpp — isolated throughput microbenchmark for the quantizer
// hot loops (TurboQuantizer::score_ip / BlockQuantizer::score_ip).
//
// These are the per-candidate scoring kernels hit millions of times per query.
// End-to-end QPS benches can't tell us how fast the kernel itself is, or how
// much headroom a SIMD rewrite would buy. This measures codes/sec and ns/vector
// for the scalar baseline, sized so the code buffer spills out of cache (the
// realistic cluster-scan regime).
//
// Build (native):
//   c++ -O3 -march=native -std=c++17 -I../src bench_score_ip.cpp -o bench_score_ip
// Run:
//   ./bench_score_ip

#include "turbo_quant.hpp"
#include "block_quant.hpp"
#include "tq_fastscan.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;

static std::vector<float> random_data(int n, int dim, uint64_t seed) {
    std::vector<float> v((size_t)n * dim);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (auto& x : v) x = g(rng);
    return v;
}

// Returns codes/sec. sink is accumulated to defeat dead-code elimination.
static double bench_turbo(int dim, int bits, int n_db, double& sink) {
    cph::TurboQuantizer q;
    auto train = random_data(20000, dim, 1);
    q.train(train.data(), 20000, dim, bits, /*calibrate=*/true);

    auto db = random_data(n_db, dim, 2);
    std::vector<uint8_t> codes((size_t)n_db * dim);
    std::vector<float>   scale(n_db), sqnorm(n_db);
    for (int i = 0; i < n_db; i++)
        q.encode(&db[(size_t)i * dim], &codes[(size_t)i * dim], &scale[i], &sqnorm[i]);

    auto query = random_data(1, dim, 3);
    std::vector<float> table; float bias;
    q.build_query_table(query.data(), table, bias);

    // Warm up + pick a repeat count that runs long enough to time stably.
    auto run_once = [&]() -> double {
        double acc = 0;
        for (int i = 0; i < n_db; i++)
            acc += q.score_ip(table.data(), bias, &codes[(size_t)i * dim], scale[i]);
        return acc;
    };
    sink += run_once();

    int reps = 50;
    auto t0 = clk::now();
    for (int r = 0; r < reps; r++) sink += run_once();
    auto t1 = clk::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    return (double)n_db * reps / secs;
}

static double bench_block(int dim, int B, int Kc, int n_db, double& sink) {
    cph::BlockQuantizer q;
    auto train = random_data(20000, dim, 1);
    q.train(train.data(), 20000, dim, B, Kc);

    auto db = random_data(n_db, dim, 2);
    int nb = dim / B;
    std::vector<uint16_t> codes((size_t)n_db * nb);
    std::vector<float>    scale(n_db), sqnorm(n_db);
    for (int i = 0; i < n_db; i++)
        q.encode(&db[(size_t)i * dim], &codes[(size_t)i * nb], &scale[i], &sqnorm[i]);

    auto query = random_data(1, dim, 3);
    std::vector<float> lut;
    q.build_query_luts(query.data(), lut);

    auto run_once = [&]() -> double {
        double acc = 0;
        for (int i = 0; i < n_db; i++)
            acc += q.score_ip(lut.data(), &codes[(size_t)i * nb], scale[i]);
        return acc;
    };
    sink += run_once();

    int reps = 50;
    auto t0 = clk::now();
    for (int r = 0; r < reps; r++) sink += run_once();
    auto t1 = clk::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    return (double)n_db * reps / secs;
}

// Top-k indices by ascending value.
static std::vector<int> topk(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, (int)v.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] < v[b]; });
    idx.resize(k);
    return idx;
}

// Benchmarks the TurboQuant fast-scan kernel against the scalar score_ip, and
// checks accuracy (top-k candidate overlap — what the rerank stage consumes).
static void bench_turbo_fastscan(int dim, int bits, int n_db, double& sink) {
    namespace fs = cph::fastscan;
    cph::TurboQuantizer q;
    auto train = random_data(20000, dim, 1);
    q.train(train.data(), 20000, dim, bits, /*calibrate=*/true);
    int L = q.n_levels;

    auto db = random_data(n_db, dim, 2);
    std::vector<uint8_t> codes((size_t)n_db * dim);
    std::vector<float>   scale(n_db), sqnorm(n_db);
    for (int i = 0; i < n_db; i++)
        q.encode(&db[(size_t)i * dim], &codes[(size_t)i * dim], &scale[i], &sqnorm[i]);

    auto query = random_data(1, dim, 3);
    float q_sq = 0; for (int i = 0; i < dim; i++) q_sq += query[i] * query[i];
    std::vector<float> table; float bias;
    q.build_query_table(query.data(), table, bias);

    // --- scalar reference: approx l2sq for all vectors ---
    std::vector<float> ref(n_db);
    for (int i = 0; i < n_db; i++) {
        float ip = q.score_ip(table.data(), bias, &codes[(size_t)i * dim], scale[i]);
        ref[i] = q.l2sq(q_sq, sqnorm[i], ip);
    }

    // --- fast-scan setup ---
    std::vector<uint8_t> qlut((size_t)dim * fs::kLutStride);
    float step, lo;
    fs::quantize_lut(table.data(), dim, L, qlut.data(), step, lo);
    std::vector<uint8_t> packed;
    fs::pack_codes(codes.data(), n_db, dim, packed);
    auto fn = fs::select_block_fn();

    // Cross-check: the SIMD kernel must match the scalar kernel bit-for-bit
    // (same uint8 LUT, same integer adds). Validates the scalar fallback and
    // gives the (un-runnable-here) AVX2 path a trusted reference.
    {
        int nb = fs::n_blocks(n_db);
        uint32_t a[fs::kWidth], b[fs::kWidth];
        long mism = 0;
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t* p = packed.data() + (size_t)blk * dim * fs::kWidth;
            fs::block_scalar(p, qlut.data(), dim, a);
            fn(p, qlut.data(), dim, b);
            for (int v = 0; v < fs::kWidth; v++) if (a[v] != b[v]) mism++;
        }
        if (mism) printf("  !! WARN d=%d: %ld scalar/SIMD sum mismatches\n", dim, mism);
    }

    std::vector<float> got(n_db);
    auto scan = [&]() {
        fs::scan_l2sq(packed.data(), n_db, dim, qlut.data(),
                      step, lo, bias, scale.data(), sqnorm.data(), q_sq, got.data(), fn);
    };
    scan();
    double s = 0; for (float x : got) s += x; sink += s;

    int reps = 50;
    auto t0 = clk::now();
    for (int r = 0; r < reps; r++) { scan(); sink += got[r % n_db]; }
    auto t1 = clk::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double cps = (double)n_db * reps / secs;

    // --- accuracy: top-100 candidate overlap + estimate error ---
    const int K = 100;
    auto a = topk(ref, K), b = topk(got, K);
    std::vector<int> sa = a, sb = b;
    std::sort(sa.begin(), sa.end()); std::sort(sb.begin(), sb.end());
    std::vector<int> inter;
    std::set_intersection(sa.begin(), sa.end(), sb.begin(), sb.end(),
                          std::back_inserter(inter));
    double overlap = 100.0 * inter.size() / K;

    double max_abs = 0, sum_abs = 0;
    for (int i = 0; i < n_db; i++) {
        double e = std::fabs((double)got[i] - ref[i]);
        max_abs = std::max(max_abs, e); sum_abs += e;
    }

    printf("%-24s %12.1f %10.1f   top%d overlap %5.1f%%  |err| mean %.4g max %.4g\n",
           (std::string("TQ-fastscan d=") + std::to_string(dim)).c_str(),
           cps / 1e6, 1e9 / cps, K, overlap, sum_abs / n_db, max_abs);
}

int main() {
    // 20k vectors already spills L2 (e.g. 30 MB of TQ codes at d=1536), which is
    // all we need for a memory-bound measurement; reps drive the timing window.
    const int n_db = 20000;
    double sink = 0;

    printf("score_ip throughput — scalar baseline (n_db=%d per pass)\n", n_db);
    printf("%-28s %14s %12s %14s\n", "kernel", "Mcodes/s", "ns/vector", "Glookups/s");
    printf("--------------------------------------------------------------------------\n");

    struct { const char* name; int dim; int bits; } tq[] = {
        {"TurboQuant d=128 b=4", 128, 4},
        {"TurboQuant d=768 b=4", 768, 4},
        {"TurboQuant d=1536 b=4", 1536, 4},
    };
    for (auto& c : tq) {
        double cps = bench_turbo(c.dim, c.bits, n_db, sink);
        printf("%-28s %14.1f %12.1f %14.2f\n",
               c.name, cps / 1e6, 1e9 / cps, cps * c.dim / 1e9);
    }

    struct { const char* name; int dim; int B; int Kc; } bq[] = {
        {"Block d=128 B=4 Kc=256", 128, 4, 256},
        {"Block d=768 B=4 Kc=256", 768, 4, 256},
        {"Block d=1536 B=4 Kc=256", 1536, 4, 256},
    };
    for (auto& c : bq) {
        double cps = bench_block(c.dim, c.B, c.Kc, n_db, sink);
        int nb = c.dim / c.B;
        printf("%-28s %14.1f %12.1f %14.2f\n",
               c.name, cps / 1e6, 1e9 / cps, cps * nb / 1e9);
    }

    printf("\nTurboQuant fast-scan (SIMD %s, width=%d) vs scalar above\n",
           cph::fastscan::simd_available() ? "ON" : "OFF (scalar fallback)",
           cph::fastscan::kWidth);
    printf("%-24s %12s %10s\n", "kernel", "Mcodes/s", "ns/vector");
    printf("--------------------------------------------------------------------------\n");
    for (int dim : {128, 768, 1536}) bench_turbo_fastscan(dim, 4, n_db, sink);

    printf("--------------------------------------------------------------------------\n");
    printf("(checksum %.3e — ignore)\n", sink);
    return 0;
}
