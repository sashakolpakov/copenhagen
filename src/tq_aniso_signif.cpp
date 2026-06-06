// tq_aniso_signif.cpp — is the eta<1 anisotropic gain real, or within noise?
//
// Decides whether the ScaNN anisotropic codebook earns its place in main. For
// several independent seeds (varying BOTH data and quantizer init), measures
// raw recall@10 of MSE (eta=1) vs anisotropic eta=0.25 block VQ, and reports the
// per-seed delta plus mean ± std. If mean(delta) is within ~1 std of 0, the
// "gain" is noise → move anisotropic to an experiments branch, keep main MSE.
//
// Build: c++ -O3 -std=c++17 -march=native src/tq_aniso_signif.cpp -o /tmp/sig && /tmp/sig

#include "block_quant.hpp"
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

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

static double eval(int dim,int n,int nq,int K,uint64_t seed,bool aniso,float eta,
                   const std::vector<float>& data,const std::vector<float>& Q,
                   const std::vector<std::vector<int>>& gt){
    BlockQuantizer bq; bq.seed=seed; bq.train(data.data(),n,dim,4,256,aniso,eta);
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
    const int K=10,nblob=50,NSEED=5;
    for(int dim : {128,768}){
        int n=20000, nq=(dim==128?1000:500);
        printf("\n=== d=%d  (B=4, 2-bit)  %d seeds:  MSE(eta=1) vs aniso(eta=0.25) ===\n",dim,NSEED);
        printf("%6s %12s %14s %12s\n","seed","MSE R@10","aniso R@10","delta");
        std::vector<double> deltas;
        for(int s=1;s<=NSEED;s++){
            std::vector<float> data=make_data(n,dim,nblob,1000+s), Q=make_data(nq,dim,nblob,9000+s);
            std::vector<std::vector<int>> gt(nq);
            for(int q=0;q<nq;q++) exact_topk(&Q[(size_t)q*dim],data.data(),n,dim,K,gt[q]);
            double m=eval(dim,n,nq,K,42+s,false,1.0f,data,Q,gt);
            double a=eval(dim,n,nq,K,42+s,true,0.25f,data,Q,gt);
            deltas.push_back(a-m);
            printf("%6d %12.4f %14.4f %+12.4f\n",s,m,a,a-m);
        }
        double mean=0; for(double x:deltas) mean+=x; mean/=NSEED;
        double var=0; for(double x:deltas) var+=(x-mean)*(x-mean); var/=NSEED;
        double sd=std::sqrt(var);
        printf("  delta: mean=%+.4f  std=%.4f  -> %s\n", mean, sd,
               (mean > 2*sd && mean > 0.001) ? "SIGNIFICANT (keep)" :
               (mean > sd ? "weak/marginal" : "NOISE (cull to branch)"));
    }
    return 0;
}
