// turbo_quant.hpp — TurboQuant-style scalar quantizer for Copenhagen.
//
// A from-scratch C++ reimplementation of the TurboQuant algorithm (Google
// Research; reference impl github.com/RyanCodrai/turbovec, Rust — algorithm
// only, no code reused). Replaces Copenhagen's weak IVFPQ path with a scalar
// quantizer that holds recall at 2-4 bits/coordinate.
//
// Pipeline:  normalize -> random orthogonal rotation -> per-coord TQ+
// calibration -> Lloyd-Max scalar quantize -> length-renormalized scoring.
//
// Dependency-free (no BLAS, no special functions): the Lloyd-Max codebook is
// fit on the empirical pooled rotated coordinates of the training batch, which
// is data-adaptive and sidesteps needing the Beta CDF in C++.
//
// Scoring is a per-coordinate table lookup, structured to leave a clean seam
// for a NEON nibble-LUT SIMD kernel later (see score_ip / build_query_table).

#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace cph {

struct TurboQuantizer {
    int   dim       = 0;
    int   bits      = 4;
    int   n_levels  = 16;       // 1 << bits
    bool  calibrate = true;     // TQ+ per-coordinate calibration
    uint64_t seed   = 42;

    // Global, fit once at train(); shared across all clusters.
    std::vector<float> rotation;    // dim*dim, row-major; u_rot = R * u
    std::vector<float> centroids;   // n_levels reconstruction values (ascending)
    std::vector<float> boundaries;  // n_levels-1 thresholds (ascending)
    std::vector<float> shift;       // dim, TQ+ (0 when !calibrate)
    std::vector<float> scale_tq;    // dim, TQ+ (1 when !calibrate)
    std::vector<float> inv_scale_tq;// dim, precomputed 1/scale_tq

    bool trained = false;

    // ---- packed code size for honest recall-per-byte reporting ----
    // v1 stores one byte per code internally (clean scalar score loop); the
    // packed-equivalent on-disk/in-RAM size is bits*dim/8 bytes/vector.
    int packed_bytes_per_vector() const { return (bits * dim + 7) / 8; }
    // What a TQCluster actually pays per vector: codes + scale + ||v||^2.
    int stored_bytes_per_vector() const { return packed_bytes_per_vector() + 8; }

    // ---------------------------------------------------------------- rotation
    // Seeded Gaussian d*d matrix, orthonormalized by modified Gram-Schmidt on
    // rows. Deterministic given (dim, seed).
    void make_rotation() {
        rotation.assign((size_t)dim * dim, 0.0f);
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::vector<double> R((size_t)dim * dim);
        for (auto& x : R) x = gauss(rng);

        // Modified Gram-Schmidt over rows.
        for (int i = 0; i < dim; i++) {
            double* ri = &R[(size_t)i * dim];
            for (int j = 0; j < i; j++) {
                const double* rj = &R[(size_t)j * dim];
                double dot = 0.0;
                for (int k = 0; k < dim; k++) dot += ri[k] * rj[k];
                for (int k = 0; k < dim; k++) ri[k] -= dot * rj[k];
            }
            double nrm = 0.0;
            for (int k = 0; k < dim; k++) nrm += ri[k] * ri[k];
            nrm = std::sqrt(nrm);
            double inv = nrm > 1e-12 ? 1.0 / nrm : 0.0;
            for (int k = 0; k < dim; k++) ri[k] *= inv;
        }
        for (size_t i = 0; i < R.size(); i++) rotation[i] = (float)R[i];
    }

    // out[i] = sum_k R[i][k] * u[k]
    void rotate(const float* u, float* out) const {
        for (int i = 0; i < dim; i++) {
            const float* ri = &rotation[(size_t)i * dim];
            float acc = 0.0f;
            for (int k = 0; k < dim; k++) acc += ri[k] * u[k];
            out[i] = acc;
        }
    }

    // ------------------------------------------------------------------- train
    void train(const float* data, int n, int d, int bits_, bool calibrate_) {
        dim = d; bits = bits_; n_levels = 1 << bits; calibrate = calibrate_;
        make_rotation();

        // Rotated unit vectors of the (sub)sampled training batch.
        const int max_train = 50000;             // cap cost for huge batches
        int step = (n > max_train) ? (n / max_train) : 1;
        std::vector<float> rotated;              // (n_used * dim)
        std::vector<float> u(dim), ur(dim);
        int n_used = 0;
        for (int i = 0; i < n; i += step) {
            const float* v = data + (size_t)i * dim;
            double nrm = 0.0;
            for (int k = 0; k < dim; k++) nrm += (double)v[k] * v[k];
            nrm = std::sqrt(nrm);
            float inv = nrm > 1e-10 ? (float)(1.0 / nrm) : 0.0f;
            for (int k = 0; k < dim; k++) u[k] = v[k] * inv;
            rotate(u.data(), ur.data());
            rotated.insert(rotated.end(), ur.begin(), ur.end());
            n_used++;
        }

        // Lloyd-Max on the pooled rotated coordinates.
        fit_codebook(rotated.data(), n_used);

        // TQ+ per-coordinate calibration.
        shift.assign(dim, 0.0f);
        scale_tq.assign(dim, 1.0f);
        if (calibrate && n_used >= 1000) fit_calibration(rotated.data(), n_used);
        inv_scale_tq.resize(dim);
        for (int d2 = 0; d2 < dim; d2++) inv_scale_tq[d2] = 1.0f / scale_tq[d2];

        trained = true;
    }

    // 1-D Lloyd-Max (k-means) over all pooled coordinate samples.
    void fit_codebook(const float* rotated, int n_used) {
        size_t total = (size_t)n_used * dim;
        // Subsample the pool for the fit if very large.
        const size_t max_pool = 2'000'000;
        size_t pstep = total > max_pool ? total / max_pool : 1;
        std::vector<float> pool;
        pool.reserve(total / pstep + 1);
        for (size_t i = 0; i < total; i += pstep) pool.push_back(rotated[i]);
        std::sort(pool.begin(), pool.end());
        int P = (int)pool.size();

        // Init centroids at evenly spaced quantiles.
        centroids.assign(n_levels, 0.0f);
        for (int l = 0; l < n_levels; l++) {
            double q = (l + 0.5) / n_levels;
            centroids[l] = pool[std::min(P - 1, (int)(q * P))];
        }

        boundaries.assign(n_levels - 1, 0.0f);
        for (int iter = 0; iter < 50; iter++) {
            for (int l = 0; l < n_levels - 1; l++)
                boundaries[l] = 0.5f * (centroids[l] + centroids[l + 1]);

            // Assign sorted pool to bins via boundary walk; accumulate means.
            std::vector<double> sum(n_levels, 0.0);
            std::vector<long>   cnt(n_levels, 0);
            int bin = 0;
            for (int i = 0; i < P; i++) {
                float x = pool[i];
                while (bin < n_levels - 1 && x > boundaries[bin]) bin++;
                sum[bin] += x; cnt[bin]++;
            }
            float max_change = 0.0f;
            for (int l = 0; l < n_levels; l++) {
                if (cnt[l] > 0) {
                    float nc = (float)(sum[l] / cnt[l]);
                    max_change = std::max(max_change, std::fabs(nc - centroids[l]));
                    centroids[l] = nc;
                }
            }
            if (max_change < 1e-7f) break;
        }
        for (int l = 0; l < n_levels - 1; l++)
            boundaries[l] = 0.5f * (centroids[l] + centroids[l + 1]);
    }

    // Per-coord (shift, scale) mapping empirical 5/95 quantiles onto the
    // global pooled 5/95 quantiles (canonical marginal proxy).
    void fit_calibration(const float* rotated, int n_used) {
        const double P_LO = 0.05, P_HI = 0.95;
        // Global canonical quantiles from the codebook pool: reuse centroids'
        // span as a cheap proxy is too coarse; recompute from a coord-agnostic
        // pooled sample.
        std::vector<float> gpool;
        gpool.reserve((size_t)n_used);
        for (int i = 0; i < n_used; i++) gpool.push_back(rotated[(size_t)i * dim + (i % dim)]);
        std::sort(gpool.begin(), gpool.end());
        int G = (int)gpool.size();
        float qc_lo = gpool[(int)(P_LO * G)];
        float qc_hi = gpool[std::min(G - 1, (int)(P_HI * G))];
        float qc_span = qc_hi - qc_lo;
        if (qc_span < 1e-6f) return;

        std::vector<float> coord(n_used);
        int lo_idx = (int)(P_LO * n_used);
        int hi_idx = std::min(n_used - 1, (int)(P_HI * n_used));
        for (int d2 = 0; d2 < dim; d2++) {
            for (int i = 0; i < n_used; i++) coord[i] = rotated[(size_t)i * dim + d2];
            std::nth_element(coord.begin(), coord.begin() + lo_idx, coord.end());
            float qe_lo = coord[lo_idx];
            std::nth_element(coord.begin(), coord.begin() + hi_idx, coord.end());
            float qe_hi = coord[hi_idx];
            float qe_span = qe_hi - qe_lo;
            if (qe_span > 1e-6f) {
                float sc = qc_span / qe_span;
                scale_tq[d2] = sc;
                shift[d2]    = qc_lo / sc - qe_lo;
            }
        }
    }

    // ------------------------------------------------------------------ encode
    int quantize(float calibrated) const {
        // boundaries ascending; return index of bin.
        int lo = 0, hi = n_levels - 1;
        // linear walk is fine for <=16 levels and is branch-predictable.
        int code = 0;
        for (int b = 0; b < n_levels - 1; b++) if (calibrated > boundaries[b]) code = b + 1;
        (void)lo; (void)hi;
        return code;
    }

    // Encode one vector. codes_out: dim bytes (one code each). Also returns the
    // length-renormalization scale and ||v||^2 for L2 reconstruction.
    void encode(const float* v, uint8_t* codes_out,
                float* scale_out, float* sqnorm_out) const {
        std::vector<float> u(dim), ur(dim);
        double nrm2 = 0.0;
        for (int k = 0; k < dim; k++) nrm2 += (double)v[k] * v[k];
        float norm = (float)std::sqrt(nrm2);
        float inv = norm > 1e-10f ? 1.0f / norm : 0.0f;
        for (int k = 0; k < dim; k++) u[k] = v[k] * inv;
        rotate(u.data(), ur.data());

        double inner = 0.0;   // <u_rot, x_hat_orig>
        for (int d2 = 0; d2 < dim; d2++) {
            float cal = (ur[d2] + shift[d2]) * scale_tq[d2];
            int code = quantize(cal);
            codes_out[d2] = (uint8_t)code;
            // reconstruction in original (uncalibrated) space
            float x_hat = centroids[code] * inv_scale_tq[d2] - shift[d2];
            inner += (double)ur[d2] * x_hat;
        }
        float innerf = (float)(inner > 1e-10 ? inner : 1e-10);
        *scale_out  = norm / innerf;
        *sqnorm_out = (float)nrm2;
    }

    // -------------------------------------------------------------- query prep
    // Build the per-coordinate scoring table for one query.
    //   table[d*n_levels + l] = (q_rot[d] / scale_tq[d]) * centroids[l]
    //   bias = -sum_d q_rot[d] * shift[d]
    // Then  <q,v> ~= vec_scale * ( bias + sum_d table[d*n_levels + code_d] ).
    // (q is NOT normalized, so the estimate is the true inner product <q,v>.)
    void build_query_table(const float* q, std::vector<float>& table, float& bias) const {
        std::vector<float> qr(dim);
        rotate(q, qr.data());
        table.assign((size_t)dim * n_levels, 0.0f);
        double b = 0.0;
        for (int d2 = 0; d2 < dim; d2++) {
            float qcal = qr[d2] * inv_scale_tq[d2];
            float* row = &table[(size_t)d2 * n_levels];
            for (int l = 0; l < n_levels; l++) row[l] = qcal * centroids[l];
            b -= (double)qr[d2] * shift[d2];
        }
        bias = (float)b;
    }

    // Inner-product estimate for one code vector given a prepared query table.
    // This is the hot loop; a NEON nibble-LUT kernel replaces it later.
    inline float score_ip(const float* table, float bias,
                          const uint8_t* codes, float vec_scale) const {
        float acc = bias;
        for (int d2 = 0; d2 < dim; d2++) acc += table[(size_t)d2 * n_levels + codes[d2]];
        return acc * vec_scale;
    }

    // L2^2 estimate from the inner-product estimate.
    inline float l2sq(float q_sq, float v_sq, float ip) const {
        return q_sq + v_sq - 2.0f * ip;
    }
};

} // namespace cph
