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
// Dependency-free. MSE codebook now; anisotropic (parallel-residual-weighted)
// loss and E8 lattice are flagged TODOs (see QUANTIZATION_GOAL.md levers 2 & 4).

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
    bool anisotropic = false;  // lever #2: parallel-residual-weighted k-means
    float eta = 4.0f;          // anisotropic parallel-residual weight (η>1)
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
    void train(const float* data, int n, int d, int B_, int Kc_,
               bool anisotropic_ = false, float eta_ = 4.0f) {
        dim = d; B = B_; Kc = Kc_; nb = d / B_;
        anisotropic = anisotropic_; eta = eta_;
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
        // Lever #2: refine the MSE codebook under the anisotropic (MIPS) loss.
        if (anisotropic) refine_aniso(rot.data(), n_used);
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

    // ------------------------------------------------ ScaNN anisotropic (lever #2)
    //
    // Guo et al. 2020, "Accelerating Large-Scale Inference with Anisotropic
    // Vector Quantization." For MIPS the residual r = x - x̂ splits into a part
    // PARALLEL to the datapoint direction and a part orthogonal; the parallel
    // part distorts ⟨q,x̂⟩ far more (in expectation over q). Weight it by η>1:
    //
    //     L(x,x̂) = ‖r⊥‖² + η‖r∥‖²  =  ‖r‖² + (η−1)(x̂ᵀr)²/‖x‖²
    //
    // Here x = u_rot is unit (orthonormal rotation), so ‖x‖²=1 and the parallel
    // direction is x itself. The (xᵀr)² term couples the blocks ONLY through the
    // scalar s = xᵀr = Σ_g x_gᵀr_g — so assignment is block coordinate descent
    // and the codebook update is a tiny B×B weighted least-squares solve. This
    // is the full-vector formulation; the earlier block-local proxy was wrong
    // (it hurt recall — see QUANTIZATION_GOAL.md Phase 3 results).

    // Solve A x = b for the B×B system, A symmetric, row-major, overwritten.
    //
    // A = N·I + (η−1)·Σ x_g x_gᵀ is SPD by construction for η≥1, so Cholesky
    // (LLᵀ) is the right solver: no pivot search, ~2× fewer flops than LU. Our
    // tuned regime uses η<1, where A = N·I − (1−η)·Σ x_g x_gᵀ — identity minus a
    // PSD term, still PD in practice because λ_max(Σ x_g x_gᵀ) ≤ trace ≈ N·B/d
    // and B/d ≪ 1 by the SIMD-LUT block design. The diagonal clamp below is the
    // cheap insurance for the η<1 and very-large-η ill-conditioned edges (it
    // degrades gracefully to a damped solve rather than producing NaNs).
    // B is tiny by construction (block → one LUT code), so BLAS/LAPACK or
    // Sherman–Morrison would not pay here.
    static void solve_spd(int B, double* A, double* b, float* out) {
        double L[64];                              // B ≤ 8 ⇒ B*B ≤ 64
        for (int j = 0; j < B; j++) {
            double d = A[j*B+j];
            for (int k = 0; k < j; k++) d -= L[j*B+k]*L[j*B+k];
            if (d < 1e-10) d = 1e-10;              // jitter: keep factor real & bounded
            double Ljj = std::sqrt(d);
            L[j*B+j] = Ljj;
            for (int i = j+1; i < B; i++) {
                double s = A[i*B+j];
                for (int k = 0; k < j; k++) s -= L[i*B+k]*L[j*B+k];
                L[i*B+j] = s / Ljj;
            }
        }
        double y[8];
        for (int i = 0; i < B; i++) {              // forward solve L y = b
            double s = b[i];
            for (int k = 0; k < i; k++) s -= L[i*B+k]*y[k];
            y[i] = s / L[i*B+i];
        }
        for (int i = B-1; i >= 0; i--) {           // back solve Lᵀ x = y
            double s = y[i];
            for (int k = i+1; k < B; k++) s -= L[k*B+i]*out[k];
            out[i] = (float)(s / L[i*B+i]);
        }
    }

    // Assign one rotated unit vector to per-block codes minimizing the
    // anisotropic loss, via block coordinate descent. Returns nothing; fills
    // codes[nb]. xn2[g] = ‖x_g‖² may be passed (else recomputed).
    void assign_aniso(const float* x, uint16_t* codes) const {
        std::vector<float> xn2(nb);
        for (int g=0;g<nb;g++){ float s=0; const float* xg=x+(size_t)g*B; for(int j=0;j<B;j++) s+=xg[j]*xg[j]; xn2[g]=s; }
        // warm start: per-block MSE argmin
        float S = 0.0f;
        for (int g=0;g<nb;g++){
            const float* xg=x+(size_t)g*B; const float* Cg=&codebook[(size_t)g*Kc*B];
            int best=0; float bd=1e30f;
            for(int c=0;c<Kc;c++){const float* cc=Cg+(size_t)c*B; float dd=0; for(int j=0;j<B;j++){float t=xg[j]-cc[j];dd+=t*t;} if(dd<bd){bd=dd;best=c;}}
            codes[g]=(uint16_t)best;
            float xc=0; const float* cc=Cg+(size_t)best*B; for(int j=0;j<B;j++) xc+=xg[j]*cc[j];
            S += xn2[g] - xc;                          // s_g = x_gᵀ(x_g - c)
        }
        // coordinate descent
        for (int round=0; round<4; round++) {
            bool changed=false;
            for (int g=0;g<nb;g++){
                const float* xg=x+(size_t)g*B; const float* Cg=&codebook[(size_t)g*Kc*B];
                // remove current block contribution
                const float* ccur=Cg+(size_t)codes[g]*B; float xccur=0; for(int j=0;j<B;j++) xccur+=xg[j]*ccur[j];
                float s_cur = xn2[g]-xccur;
                float S_minus = S - s_cur;
                int best=codes[g]; float bestcost=1e30f, best_s=s_cur;
                for(int c=0;c<Kc;c++){
                    const float* cc=Cg+(size_t)c*B; float r2=0,xc=0;
                    for(int j=0;j<B;j++){float t=xg[j]-cc[j]; r2+=t*t; xc+=xg[j]*cc[j];}
                    float s_k = xn2[g]-xc;
                    float tot = S_minus + s_k;
                    float cost = r2 + (eta-1.0f)*tot*tot;
                    if(cost<bestcost){bestcost=cost;best=c;best_s=s_k;}
                }
                if(best!=codes[g]) changed=true;
                codes[g]=(uint16_t)best; S = S_minus + best_s;
            }
            if(!changed) break;
        }
    }

    // Alternating refinement of the codebook under the anisotropic loss, warm
    // started from the MSE codebook. `rot` = n_used rotated unit vectors.
    void refine_aniso(const float* rot, int n_used) {
        std::vector<uint16_t> codes((size_t)n_used*nb);
        for (int iter=0; iter<5; iter++) {
            // E-step: assign all points; track S_total per point.
            std::vector<float> Stot(n_used);
            for (int i=0;i<n_used;i++){
                const float* x = rot+(size_t)i*dim;
                assign_aniso(x, &codes[(size_t)i*nb]);
                float S=0;
                for(int g=0;g<nb;g++){ const float* xg=x+(size_t)g*B; const float* cc=&codebook[((size_t)g*Kc+codes[(size_t)i*nb+g])*B];
                    float xc=0,xx=0; for(int j=0;j<B;j++){xc+=xg[j]*cc[j];xx+=xg[j]*xg[j];} S+=xx-xc; }
                Stot[i]=S;
            }
            // M-step: per (block, code) weighted least squares  A c = b
            //   A = N I + (η−1) Σ x_g x_gᵀ
            //   b = Σ x_g + (η−1) Σ (‖x_g‖² + S_minus_i) x_g
            std::vector<double> A((size_t)Kc*B*B), b((size_t)Kc*B);
            std::vector<long> cnt(Kc);
            for (int g=0; g<nb; g++) {
                std::fill(A.begin(),A.end(),0.0); std::fill(b.begin(),b.end(),0.0); std::fill(cnt.begin(),cnt.end(),0L);
                for (int i=0;i<n_used;i++){
                    int k = codes[(size_t)i*nb+g];
                    const float* xg = rot+(size_t)i*dim+(size_t)g*B;
                    const float* ccur = &codebook[((size_t)g*Kc+k)*B];
                    float xc=0,xx=0; for(int j=0;j<B;j++){xc+=xg[j]*ccur[j];xx+=xg[j]*xg[j];}
                    float s_cur = xx-xc;
                    float S_minus = Stot[i]-s_cur;
                    double* Ak=&A[(size_t)k*B*B]; double* bk=&b[(size_t)k*B];
                    double w = (double)eta-1.0;
                    double coef = w*(xx + S_minus);
                    for(int a=0;a<B;a++){
                        bk[a] += xg[a] + coef*xg[a];
                        for(int c2=0;c2<B;c2++) Ak[a*B+c2] += w*(double)xg[a]*xg[c2];
                    }
                    cnt[k]++;
                }
                for (int k=0;k<Kc;k++){
                    if(cnt[k]==0) continue;
                    double* Ak=&A[(size_t)k*B*B]; double* bk=&b[(size_t)k*B];
                    for(int a=0;a<B;a++) Ak[a*B+a] += (double)cnt[k];
                    solve_spd(B, Ak, bk, &codebook[((size_t)g*Kc+k)*B]);
                }
            }
        }
    }

    // ------------------------------------------------------------------ encode
    void encode(const float* v, uint16_t* codes_out,
                float* scale_out, float* sqnorm_out) const {
        std::vector<float> u(dim), ur(dim);
        double nrm2 = 0.0; for (int k=0;k<dim;k++) nrm2 += (double)v[k]*v[k];
        float norm = (float)std::sqrt(nrm2);
        float inv = norm > 1e-10f ? 1.0f/norm : 0.0f;
        for (int k=0;k<dim;k++) u[k] = v[k]*inv;
        rotate(u.data(), ur.data());

        if (anisotropic) {
            // assign to minimize the same anisotropic loss the codebook was fit for
            assign_aniso(ur.data(), codes_out);
        } else {
            for (int g = 0; g < nb; g++) {
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
