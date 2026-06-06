// block_quant_py.cpp — pybind11 bindings for the block-VQ quantizer.
//
// Exposes a flat, compressed, *dynamically deletable* index built on
// BlockQuantizer (block_quant.hpp): TurboQuant's rotation + renormalization
// front-end, block/sub-vector VQ ("adaptive binning"), and O(1) tombstone
// delete. This is the "Copenhagen-style dynamics + TurboVec-style compression"
// combination, callable from Python so it can be benchmarked head-to-head
// against FAISS / HNSW / vanilla TurboVec on the same harness.
//
// Build (see src/build_block_vq.sh):
//   c++ -O3 -march=native -shared -std=c++17 -fPIC $(python -m pybind11 --includes) \
//       block_quant_py.cpp -o ../python/core/block_vq.so

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include "block_quant.hpp"

namespace py = pybind11;
using namespace cph;

struct BlockVQIndex {
    BlockQuantizer q;
    std::vector<uint16_t> codes;
    std::vector<float> scales, sqnorms;
    std::vector<uint8_t> deleted;     // tombstone bitmap, indexed by id
    int n = 0;
    int dim = 0;

    void train(py::array_t<float, py::array::c_style | py::array::forcecast> data,
               int B, int Kc, bool aniso, float eta) {
        auto b = data.request();
        int N = (int)b.shape[0]; dim = (int)b.shape[1];
        q.train((const float*)b.ptr, N, dim, B, Kc, aniso, eta);
    }

    // Encode and append. Returns the [first, last) id range assigned.
    py::tuple add(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
        auto b = data.request();
        int N = (int)b.shape[0], d = (int)b.shape[1];
        const float* p = (const float*)b.ptr;
        int base = n;
        codes.resize((size_t)(n + N) * q.nb);
        scales.resize(n + N); sqnorms.resize(n + N); deleted.resize(n + N, 0);
        for (int i = 0; i < N; i++)
            q.encode(p + (size_t)i * d, &codes[(size_t)(base + i) * q.nb],
                     &scales[base + i], &sqnorms[base + i]);
        n += N;
        return py::make_tuple(base, n);
    }

    void remove(int id) { if (id >= 0 && id < n) deleted[id] = 1; }   // O(1) tombstone

    // Flat top-k search by the renormalized L2 estimate, skipping tombstones.
    py::array_t<int> search(
            py::array_t<float, py::array::c_style | py::array::forcecast> queries, int k) {
        auto qb = queries.request();
        int nq = (int)qb.shape[0], d = (int)qb.shape[1];
        const float* qp = (const float*)qb.ptr;
        py::array_t<int> out({nq, k});
        int* oi = (int*)out.request().ptr;
        std::vector<float> lut;
        std::vector<std::pair<float,int>> est;
        for (int Q = 0; Q < nq; Q++) {
            float qsq = 0;
            for (int j = 0; j < d; j++) { float v = qp[(size_t)Q*d+j]; qsq += v*v; }
            q.build_query_luts(qp + (size_t)Q*d, lut);
            est.clear(); est.reserve(n);
            for (int i = 0; i < n; i++) {
                if (deleted[i]) continue;
                float ip = q.score_ip(lut.data(), &codes[(size_t)i * q.nb], scales[i]);
                est.push_back({q.l2sq(qsq, sqnorms[i], ip), i});
            }
            int kk = std::min((int)est.size(), k);
            std::partial_sort(est.begin(), est.begin() + kk, est.end());
            for (int j = 0; j < k; j++) oi[Q*k + j] = (j < kk) ? est[j].second : -1;
        }
        return out;
    }

    int bytes_per_vector() const { return q.stored_bytes_per_vector(); }
    int n_live() const { int c = 0; for (auto x : deleted) c += !x; return c; }
};

PYBIND11_MODULE(block_vq, m) {
    m.doc() = "Flat compressed index: TurboQuant rotation+renorm + block VQ + O(1) tombstone delete";
    py::class_<BlockVQIndex>(m, "BlockVQIndex")
        .def(py::init<>())
        .def("train", &BlockVQIndex::train,
             py::arg("data"), py::arg("B") = 4, py::arg("Kc") = 256,
             py::arg("aniso") = false, py::arg("eta") = 1.0f)
        .def("add", &BlockVQIndex::add)
        .def("remove", &BlockVQIndex::remove)
        .def("search", &BlockVQIndex::search, py::arg("queries"), py::arg("k") = 10)
        .def("bytes_per_vector", &BlockVQIndex::bytes_per_vector)
        .def("n_live", &BlockVQIndex::n_live);
}
