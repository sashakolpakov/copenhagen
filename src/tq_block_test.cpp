// tq_block_test.cpp — scalar TurboQuant vs block VQ at MATCHED byte counts.
//
// The decisive experiment for QUANTIZATION_GOAL.md lever #1: at d=128 (the
// regime where the unit-sphere coordinate correlation dominates), does
// quantizing B coordinates jointly beat per-coordinate scalar quantization at
// the SAME bitrate? Matched pairs (identical bytes/vector):
//   scalar 2-bit  ↔  block B=4 Kc=256   (2 bits/coord, 40 B/vec @ d=128)
//   scalar 4-bit  ↔  block B=2 Kc=256   (4 bits/coord, 72 B/vec @ d=128)
// Reports RAW recall@10 (compressed-only, no float rerank) — the honest number.
//
// Build: c++ -O3 -std=c++17 -march=native src/tq_block_test.cpp -o /tmp/tqblock
// Run:   /tmp/tqblock

#include "turbo_quant.hpp"
#include "block_quant.hpp"
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

using namespace cph;

static std::vector<float> make_data(int n, int dim, int n_blobs, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.f,1.f);
    std::uniform_real_distribution<float> scale(0.3f,3.0f);
    std::vector<std::vector<float>> ctr(n_blobs, std::vector<float>(dim));
    for (auto& c : ctr) for (auto& x : c) x = g(rng)*3.0f;
    std::vector<float> data((size_t)n*dim);
    std::uniform_int_distribution<int> pick(0,n_blobs-1);
    for (int i=0;i<n;i++){int b=pick(rng);float s=scale(rng);
        for(int d=0;d<dim;d++) data[(size_t)i*dim+d]=(ctr[b][d]+g(rng))*s;}
    return data;
}
static void exact_topk(const float* q,const float* data,int n,int dim,int k,std::vector<int>& out){
    std::vector<std::pair<float,int>> d(n);
    for(int i=0;i<n;i++){const float* v=data+(size_t)i*dim;float s=0;
        for(int j=0;j<dim;j++){float t=q[j]-v[j];s+=t*t;} d[i]={s,i};}
    std::partial_sort(d.begin(),d.begin()+k,d.end());
    out.resize(k); for(int i=0;i<k;i++) out[i]=d[i].second;
}
static double recall_of(const std::vector<int>& got,const std::vector<int>& gt,int K){
    int h=0; for(int a=0;a<(int)got.size();a++) for(int b=0;b<K;b++) if(got[a]==gt[b]){h++;break;}
    return (double)h/K;
}

int main() {
    const int K=10, n_blobs=50;
    for (int dim : {128, 768}) {
        int n  = 20000, nq = (dim==128?1000:500);
        printf("\n================ d=%d  n=%d  q=%d  top-%d ================\n", dim, n, nq, K);
        std::vector<float> data = make_data(n,dim,n_blobs,1);
        std::vector<float> queries = make_data(nq,dim,n_blobs,999);
        std::vector<std::vector<int>> gt(nq);
        for (int q=0;q<nq;q++) exact_topk(&queries[(size_t)q*dim],data.data(),n,dim,K,gt[q]);

        printf("%-26s %5s %10s %12s\n","method","B","raw R@10","bytes/vec");
        printf("------------------------------------------------------------\n");

        // ---- scalar baselines (turbo_quant.hpp) ----
        for (int bits : {2,4}) {
            TurboQuantizer tq; tq.seed=42; tq.train(data.data(),n,dim,bits,true);
            std::vector<uint8_t> codes((size_t)n*dim);
            std::vector<float> sc(n), sq(n);
            for(int i=0;i<n;i++) tq.encode(&data[(size_t)i*dim],&codes[(size_t)i*dim],&sc[i],&sq[i]);
            double hits=0; std::vector<float> tbl; float bias;
            for(int q=0;q<nq;q++){const float* qp=&queries[(size_t)q*dim];
                float qsq=0; for(int j=0;j<dim;j++) qsq+=qp[j]*qp[j];
                tq.build_query_table(qp,tbl,bias);
                std::vector<std::pair<float,int>> e(n);
                for(int i=0;i<n;i++){float ip=tq.score_ip(tbl.data(),bias,&codes[(size_t)i*dim],sc[i]);
                    e[i]={tq.l2sq(qsq,sq[i],ip),i};}
                std::partial_sort(e.begin(),e.begin()+K,e.end());
                std::vector<int> r(K); for(int i=0;i<K;i++) r[i]=e[i].second;
                hits += recall_of(r,gt[q],K);}
            char nm[64]; snprintf(nm,64,"scalar %d-bit (TurboQuant)",bits);
            printf("%-26s %5d %10.4f %12d\n", nm, 1, hits/nq, tq.packed_bytes_per_vector()+8);
        }

        // ---- block VQ (block_quant.hpp), matched bitrate ----
        struct Cfg{int B;const char* nm;};
        Cfg cfgs[] = {
            {4,"block VQ Kc=256 MSE"},     // matches scalar 2-bit
            {2,"block VQ Kc=256 MSE"},     // matches scalar 4-bit
        };
        for (auto& cf : cfgs) {
            BlockQuantizer bq; bq.seed=42;
            bq.train(data.data(),n,dim,cf.B,256);
            std::vector<uint16_t> codes((size_t)n*bq.nb);
            std::vector<float> sc(n), sq(n);
            for(int i=0;i<n;i++) bq.encode(&data[(size_t)i*dim],&codes[(size_t)i*bq.nb],&sc[i],&sq[i]);
            double hits=0; std::vector<float> lut;
            for(int q=0;q<nq;q++){const float* qp=&queries[(size_t)q*dim];
                float qsq=0; for(int j=0;j<dim;j++) qsq+=qp[j]*qp[j];
                bq.build_query_luts(qp,lut);
                std::vector<std::pair<float,int>> e(n);
                for(int i=0;i<n;i++){float ip=bq.score_ip(lut.data(),&codes[(size_t)i*bq.nb],sc[i]);
                    e[i]={bq.l2sq(qsq,sq[i],ip),i};}
                std::partial_sort(e.begin(),e.begin()+K,e.end());
                std::vector<int> r(K); for(int i=0;i<K;i++) r[i]=e[i].second;
                hits += recall_of(r,gt[q],K);}
            printf("%-26s %5d %10.4f %12d\n", cf.nm, cf.B, hits/nq, bq.stored_bytes_per_vector());
        }
    }
    printf("\nMatched-rate pairs: scalar 2-bit ↔ block B=4 (40B@128); scalar 4-bit ↔ block B=2 (72B@128).\n");
    return 0;
}
