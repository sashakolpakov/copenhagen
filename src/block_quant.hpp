// block_quant.hpp — Block / sub-vector VQ on top of the TurboQuant front-end.
//
// Generalizes the scalar TurboQuant quantizer (turbo_quant.hpp, = 1-D blocks)
// to B-dimensional sub-blocks with a trained k-means codebook per block. This
// is the "biggest lever" from QUANTIZATION_GOAL.md: scalar per-coordinate
// quantization assumes coordinate independence and throws away the negative
// correlation the unit-sphere constraint (Σxᵢ²=1) induces between coordinates —
// an O(1/d) effect that is negligible at d=1536 but the dominant error at d=128.
// Block VQ quantizes the joint distribution of B coordinates at once and
// recovers that structure.
//
// It is, deliberately, PQ's data layout (sub-vector → 1 code → LUT entry) with
// TurboQuant's rotation + length-renormalization front-end and a good codebook:
// best of both worlds. Scoring stays a per-block table lookup (SIMD-LUT-ready).
//
// Front-end identical to turbo_quant.hpp:
//   normalize -> rotate -> per-block nearest codeword -> store codes + scale + ‖v‖²
// Estimator (renormalized inner product, RaBitQ-style):
//   scale = ‖v‖ / <u_rot, x_hat>,  <q,v> ≈ scale · Σ_g <q_rot^(g), C_g[code_g]>
//   ‖q-v‖² ≈ ‖q‖² + ‖v‖² - 2<q,v>
//
// Dependency-free. MSE k-means codebook. (The ScaNN anisotropic variant was
// tried and culled — it does not beat MSE once the estimator renormalizes; see
// the experiments/anisotropic branch. The E8 lattice is a future alternative.)

#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace cph {

struct BlockQuantizer {
    int dim = 0;
    int B   = 4;          // block dimension (B=1 reproduces scalar TurboQuant)
    int nb  = 0;          // number of blocks = dim / B
    int Kc  = 256;        // codewords per block (256 = 1 byte/block)
    uint64_t seed = 42;

    std::vector<float> rotation;   // dim*dim, row-major; u_rot = R u
    std::vector<float> codebook;   // nb * Kc * B   (block g, code c, dim j)
    bool trained = false;

    int codes_per_vector() const { return nb; }
    // 1 byte/block at Kc<=256; scale + ‖v‖² add 8 bytes.
    int stored_bytes_per_vector() const {
        int code_bits = (int)std::ceil(std::log2((double)Kc));
        return (code_bits * nb + 7) / 8 + 8;
    }

    // ---------------------------------------------------------------- rotation
    void make_rotation() {
        rotation.assign((size_t)dim * dim, 0.0f);
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::vector<double> R((size_t)dim * dim);
        for (auto& x : R) x = gauss(rng);
        for (int i = 0; i < dim; i++) {
            double* ri = &R[(size_t)i * dim];
            for (int j = 0; j < i; j++) {
                const double* rj = &R[(size_t)j * dim];
                double dot = 0.0;
                for (int k = 0; k < dim; k++) dot += ri[k] * rj[k];
                for (int k = 0; k < dim; k++) ri[k] -= dot * rj[k];
            }
            double nrm = 0.0; for (int k = 0; k < dim; k++) nrm += ri[k]*ri[k];
            double inv = nrm > 1e-12 ? 1.0/std::sqrt(nrm) : 0.0;
            for (int k = 0; k < dim; k++) ri[k] *= inv;
        }
        for (size_t i = 0; i < R.size(); i++) rotation[i] = (float)R[i];
    }
    void rotate(const float* u, float* out) const {
        for (int i = 0; i < dim; i++) {
            const float* ri = &rotation[(size_t)i * dim];
            float acc = 0.0f;
            for (int k = 0; k < dim; k++) acc += ri[k] * u[k];
            out[i] = acc;
        }
    }

    // ------------------------------------------------------------------- train
    void train(const float* data, int n, int d, int B_, int Kc_) {
        dim = d; B = B_; Kc = Kc_; nb = d / B_;
        make_rotation();

        // Rotated unit vectors of a (sub)sampled batch.
        const int max_train = 50000;
        int step = (n > max_train) ? (n / max_train) : 1;
        std::vector<float> rot; std::vector<float> u(dim), ur(dim);
        int n_used = 0;
        for (int i = 0; i < n; i += step) {
            const float* v = data + (size_t)i * dim;
            double nrm2 = 0.0; for (int k=0;k<dim;k++) nrm2 += (double)v[k]*v[k];
            float inv = nrm2 > 1e-20 ? (float)(1.0/std::sqrt(nrm2)) : 0.0f;
            for (int k=0;k<dim;k++) u[k] = v[k]*inv;
            rotate(u.data(), ur.data());
            rot.insert(rot.end(), ur.begin(), ur.end());
            n_used++;
        }

        codebook.assign((size_t)nb * Kc * B, 0.0f);
        std::vector<float> block(n_used * B);
        for (int g = 0; g < nb; g++) {
            // gather block g across all training vectors (contiguous B-dim rows)
            for (int i = 0; i < n_used; i++)
                for (int j = 0; j < B; j++)
                    block[i*B + j] = rot[(size_t)i*dim + g*B + j];
            kmeans_block(block.data(), n_used, &codebook[(size_t)g*Kc*B]);
        }
        trained = true;
    }

    // k-means (Lloyd) on one block's B-dim samples → Kc codewords.
    void kmeans_block(const float* X, int n, float* C) {
        std::mt19937_64 rng(seed + 12345);
        std::uniform_int_distribution<int> pick(0, n - 1);
        for (int c = 0; c < Kc; c++) {
            int idx = pick(rng);
            for (int j = 0; j < B; j++) C[c*B + j] = X[(size_t)idx*B + j];
        }
        std::vector<int> assign(n, 0);
        std::vector<double> sum((size_t)Kc*B);
        std::vector<long>   cnt(Kc);
        for (int iter = 0; iter < 25; iter++) {
            std::fill(sum.begin(), sum.end(), 0.0);
            std::fill(cnt.begin(), cnt.end(), 0L);
            long changed = 0;
            for (int i = 0; i < n; i++) {
                const float* x = X + (size_t)i*B;
                int best = 0; float bd = 1e30f;
                for (int c = 0; c < Kc; c++) {
                    const float* cc = C + (size_t)c*B;
                    float dd = block_dist(x, cc);
                    if (dd < bd) { bd = dd; best = c; }
                }
                if (best != assign[i]) { assign[i] = best; changed++; }
                cnt[best]++;
                double* s = &sum[(size_t)best*B];
                for (int j = 0; j < B; j++) s[j] += x[j];
            }
            for (int c = 0; c < Kc; c++) {
                if (cnt[c] > 0) {
                    float invc = 1.0f / cnt[c];
                    for (int j = 0; j < B; j++) C[c*B+j] = (float)(sum[(size_t)c*B+j]*invc);
                } else {
                    int idx = pick(rng);                 // re-seed empty cell
                    for (int j = 0; j < B; j++) C[c*B+j] = X[(size_t)idx*B+j];
                }
            }
            if (changed == 0 && iter > 0) break;
        }
    }

    // Plain MSE block distance (used for the MSE codebook and the warm start).
    inline float block_dist(const float* x, const float* c) const {
        float s = 0; for (int j=0;j<B;j++){float t=x[j]-c[j]; s+=t*t;} return s;
    }

    // NOTE: the ScaNN anisotropic (MIPS-aware) codebook lived here. A 5-seed
    // significance test (src/tq_aniso_signif.cpp on branch experiments/anisotropic)
    // showed η<1 is marginal at best (d=128: +0.28pp, ~1.6σ) and net negative at
    // d=768 — because the RaBitQ length-renormalization already corrects the
    // parallel residual ScaNN targets. It was culled from main; the full
    // implementation is preserved on the experiments/anisotropic branch.

    // ------------------------------------------------------------------ encode
    void encode(const float* v, uint16_t* codes_out,
                float* scale_out, float* sqnorm_out) const {
        std::vector<float> u(dim), ur(dim);
        double nrm2 = 0.0; for (int k=0;k<dim;k++) nrm2 += (double)v[k]*v[k];
        float norm = (float)std::sqrt(nrm2);
        float inv = norm > 1e-10f ? 1.0f/norm : 0.0f;
        for (int k=0;k<dim;k++) u[k] = v[k]*inv;
        rotate(u.data(), ur.data());

        for (int g = 0; g < nb; g++) {                       // per-block nearest codeword
            const float* x = &ur[(size_t)g*B];
            const float* Cg = &codebook[(size_t)g*Kc*B];
            int best = 0; float bd = 1e30f;
            for (int c = 0; c < Kc; c++) {
                const float* cc = Cg + (size_t)c*B;
                float dd = 0; for (int j=0;j<B;j++){float t=x[j]-cc[j]; dd+=t*t;}
                if (dd < bd) { bd = dd; best = c; }
            }
            codes_out[g] = (uint16_t)best;
        }
        double inner = 0.0;                                  // <u_rot, x_hat>
        for (int g = 0; g < nb; g++) {
            const float* x = &ur[(size_t)g*B];
            const float* cc = &codebook[((size_t)g*Kc + codes_out[g])*B];
            for (int j=0;j<B;j++) inner += (double)x[j]*cc[j];
        }
        float innerf = (float)(inner > 1e-10 ? inner : 1e-10);
        *scale_out  = norm / innerf;
        *sqnorm_out = (float)nrm2;
    }

    // Approximate decode (for split k-means in the dynamic index): reconstruct
    // v ≈ ‖v‖ · R^T x_hat / ‖x_hat‖.  Good enough for split centroids.
    void decode(const uint16_t* codes, float scale, float sqnorm, float* out) const {
        (void)scale;
        std::vector<float> xhat(dim);
        float h2 = 0;
        for (int g = 0; g < nb; g++) {
            const float* cc = &codebook[((size_t)g*Kc + codes[g])*B];
            for (int j=0;j<B;j++){ xhat[g*B+j]=cc[j]; h2 += cc[j]*cc[j]; }
        }
        float norm = std::sqrt(sqnorm);
        float s = h2 > 1e-12f ? norm/std::sqrt(h2) : 0.0f;
        // out = s * R^T xhat  (R orthonormal rows → R^T is columns)
        for (int k = 0; k < dim; k++) {
            float acc = 0;
            for (int i = 0; i < dim; i++) acc += rotation[(size_t)i*dim + k]*xhat[i];
            out[k] = acc * s;
        }
    }

    // -------------------------------------------------------------- query prep
    // Per-block LUT: lut[g*Kc + c] = <q_rot^(g), C_g[c]>.
    void build_query_luts(const float* q, std::vector<float>& lut) const {
        std::vector<float> qr(dim);
        rotate(q, qr.data());
        lut.assign((size_t)nb * Kc, 0.0f);
        for (int g = 0; g < nb; g++) {
            const float* qg = &qr[(size_t)g*B];
            const float* Cg = &codebook[(size_t)g*Kc*B];
            float* lg = &lut[(size_t)g*Kc];
            for (int c = 0; c < Kc; c++) {
                const float* cc = Cg + (size_t)c*B;
                float s = 0; for (int j=0;j<B;j++) s += qg[j]*cc[j];
                lg[c] = s;
            }
        }
    }

    inline float score_ip(const float* lut, const uint16_t* codes, float vec_scale) const {
        float acc = 0;
        for (int g = 0; g < nb; g++) acc += lut[(size_t)g*Kc + codes[g]];
        return acc * vec_scale;
    }
    inline float l2sq(float q_sq, float v_sq, float ip) const { return q_sq + v_sq - 2.0f*ip; }
};

} // namespace cph
