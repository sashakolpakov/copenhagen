// tq_aniso_test.cpp — lever #2: ScaNN anisotropic block VQ vs MSE block VQ.
//
// Sweeps the anisotropic weight η and reports RAW recall@10 (compressed-only)
// against the MSE block-VQ baseline, at the 2-bit (B=4) operating point where
// the parallel-residual error dominates. η=1 must reproduce MSE (sanity check).
//
// Build: c++ -O3 -std=c++17 -march=native src/tq_aniso_test.cpp -o /tmp/tqaniso
// Run:   /tmp/tqaniso

#include "block_quant.hpp"
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

using namespace cph;

static std::vector<float> make_data(int n,int dim,int nb,uint64_t seed){
    std::mt19937_64 rng(seed); std::normal_distribution<float> g(0,1);
    std::uniform_real_distribution<float> sc(0.3f,3.0f);
    std::vector<std::vector<float>> ctr(nb,std::vector<float>(dim));
    for(auto&c:ctr)for(auto&x:c)x=g(rng)*3.0f;
    std::vector<float> d((size_t)n*dim); std::uniform_int_distribution<int> pk(0,nb-1);
    for(int i=0;i<n;i++){int b=pk(rng);float s=sc(rng);for(int j=0;j<dim;j++)d[(size_t)i*dim+j]=(ctr[b][j]+g(rng))*s;}
    return d;
}
static void exact_topk(const float* q,const float* D,int n,int dim,int k,std::vector<int>& o){
    std::vector<std::pair<float,int>> d(n);
    for(int i=0;i<n;i++){const float* v=D+(size_t)i*dim;float s=0;for(int j=0;j<dim;j++){float t=q[j]-v[j];s+=t*t;}d[i]={s,i};}
    std::partial_sort(d.begin(),d.begin()+k,d.end()); o.resize(k);for(int i=0;i<k;i++)o[i]=d[i].second;
}
static double rec(const std::vector<int>& g,const std::vector<int>& gt,int K){int h=0;for(int a=0;a<(int)g.size();a++)for(int b=0;b<K;b++)if(g[a]==gt[b]){h++;break;}return (double)h/K;}

static double eval_block(int dim,int n,int B,bool aniso,float eta,
                         const std::vector<float>& data,const std::vector<float>& Q,int nq,int K,
                         const std::vector<std::vector<int>>& gt){
    BlockQuantizer bq; bq.seed=42; bq.train(data.data(),n,dim,B,256,aniso,eta);
    std::vector<uint16_t> codes((size_t)n*bq.nb); std::vector<float> sc(n),sq(n);
    for(int i=0;i<n;i++) bq.encode(&data[(size_t)i*dim],&codes[(size_t)i*bq.nb],&sc[i],&sq[i]);
    double hits=0; std::vector<float> lut;
    for(int q=0;q<nq;q++){const float* qp=&Q[(size_t)q*dim];float qsq=0;for(int j=0;j<dim;j++)qsq+=qp[j]*qp[j];
        bq.build_query_luts(qp,lut);
        std::vector<std::pair<float,int>> e(n);
        for(int i=0;i<n;i++){float ip=bq.score_ip(lut.data(),&codes[(size_t)i*bq.nb],sc[i]);e[i]={bq.l2sq(qsq,sq[i],ip),i};}
        std::partial_sort(e.begin(),e.begin()+K,e.end());
        std::vector<int> r(K);for(int i=0;i<K;i++)r[i]=e[i].second;hits+=rec(r,gt[q],K);}
    return hits/nq;
}

int main(){
    const int K=10,nblob=50;
    for(int dim : {128,768}){
        int n=20000, nq=(dim==128?1000:500);
        printf("\n=========== d=%d  n=%d  q=%d  (B=4, 2-bit-equiv, 1B/block) ===========\n",dim,n,nq);
        std::vector<float> data=make_data(n,dim,nblob,1), Q=make_data(nq,dim,nblob,999);
        std::vector<std::vector<int>> gt(nq);
        for(int q=0;q<nq;q++) exact_topk(&Q[(size_t)q*dim],data.data(),n,dim,K,gt[q]);

        double mse = eval_block(dim,n,4,false,1.0f,data,Q,nq,K,gt);
        printf("  MSE  (B=4)            raw R@10 = %.4f\n", mse);
        for(float eta : {0.1f, 0.25f, 0.5f, 0.75f, 1.0f}){
            double a = eval_block(dim,n,4,true,eta,data,Q,nq,K,gt);
            printf("  aniso eta=%-6.0f       raw R@10 = %.4f   (%+.4f vs MSE)\n", eta, a, a-mse);
        }
    }
    printf("\n(eta=1 should ~reproduce MSE; >0 deltas at eta>1 = lever #2 working.)\n");
    return 0;
}
