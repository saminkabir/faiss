// ============================================================
// demo_flatnav_shared_storage.cpp
//
// Build a FlatNav graph directly on a faiss::IndexFlatL2 (shared storage).
//
//   faiss::IndexFlatL2 shared_storage(d);  shared_storage.add(n, xb);
//   auto flatnav = flatnav_faiss::make_flatnav_l2(shared_storage, M);
//   flatnav_faiss::build(*flatnav, ef_construction, threads);
//
// FlatNav allocates only its graph; every distance reads the vector out of
// shared_storage.codes. Node id i == faiss id i, so the same storage can also
// back IndexHNSW(&shared_storage) / IndexNSG(&shared_storage) (see
// demo_paired_hnsw_nsg_same_storage_parallal.cpp) and results are directly
// comparable.
//
// Usage:
//   demo_flatnav_shared_storage base.fvecs query.fvecs [gt.ivecs|-] [M] [efC] [efS] [threads]
//   demo_flatnav_shared_storage --synthetic [n] [d] [M] [efC] [efS] [threads]
//
// If no ground truth is given, exact kNN is computed with shared_storage.search()
// (the same IndexFlatL2 FlatNav is reading from).
// ============================================================

#include <faiss/IndexFlat.h>

#include "flatnav_faiss_storage.hpp"

#include <omp.h>
#include <sys/stat.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

double now_s() {
    return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

// .fvecs / .ivecs readers (same format as the faiss demos). Caller owns memory.
float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        std::abort();
    }
    int d;
    if (fread(&d, 1, sizeof(int), f) != sizeof(int)) {
        fprintf(stderr, "could not read %s\n", fname);
        std::abort();
    }
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);
    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");
    for (size_t i = 0; i < n; i++) {
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));
    }
    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return reinterpret_cast<int*>(fvecs_read(fname, d_out, n_out));
}

float* synthetic(size_t n, size_t d, uint32_t seed) {
    // Low intrinsic dimension (like real embeddings): x = A z + noise, z in R^16.
    // The projection A is fixed (seed 1234) so base and queries share it.
    const size_t r = std::min<size_t>(16, d);
    std::mt19937 arng(1234);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> A(d * r);
    for (auto& v : A) v = g(arng);
    std::mt19937 rng(seed);
    float* x = new float[n * d];
    std::vector<float> z(r);
    for (size_t i = 0; i < n; i++) {
        for (auto& v : z) v = g(rng);
        for (size_t j = 0; j < d; j++) {
            float s = 0.1f * g(rng);
            for (size_t k = 0; k < r; k++) s += A[j * r + k] * z[k];
            x[i * d + j] = s;
        }
    }
    return x;
}

double recall_at_k(const faiss::idx_t* I, const faiss::idx_t* gt, size_t nq, int k, int gt_k) {
    size_t hits = 0;
    for (size_t q = 0; q < nq; q++) {
        std::unordered_set<faiss::idx_t> truth(gt + q * gt_k, gt + q * gt_k + k);
        for (int j = 0; j < k; j++) hits += truth.count(I[q * k + j]);
    }
    return static_cast<double>(hits) / (nq * k);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " base.fvecs query.fvecs [gt.ivecs|-] [M] [efC] [efS] [threads]\n  " << argv[0]
                  << " --synthetic [n] [d] [M] [efC] [efS] [threads]\n";
        return 1;
    }

    const bool synth = std::string(argv[1]) == "--synthetic";
    size_t d = 0, n = 0, nq = 0;
    float* xb = nullptr;
    float* xq = nullptr;
    int* gt_raw = nullptr;
    size_t gt_k = 0;
    int a = 0;  // index of the first numeric parameter

    if (synth) {
        n = argc > 2 ? std::stoul(argv[2]) : 100000;
        d = argc > 3 ? std::stoul(argv[3]) : 128;
        nq = 1000;
        xb = synthetic(n, d, 1);
        xq = synthetic(nq, d, 2);
        a = 4;
    } else {
        if (argc < 3) {
            std::cerr << "need base.fvecs and query.fvecs\n";
            return 1;
        }
        size_t dq;
        xb = fvecs_read(argv[1], &d, &n);
        xq = fvecs_read(argv[2], &dq, &nq);
        assert(d == dq);
        if (argc > 3 && std::string(argv[3]) != "-") {
            size_t ngt;
            gt_raw = ivecs_read(argv[3], &gt_k, &ngt);
            assert(ngt == nq);
        }
        a = 4;
    }
    const int M = argc > a ? std::stoi(argv[a]) : 32;
    const int EF_C = argc > a + 1 ? std::stoi(argv[a + 1]) : 100;
    const int EF_S = argc > a + 2 ? std::stoi(argv[a + 2]) : 64;
    const int THREADS = argc > a + 3 ? std::stoi(argv[a + 3])
                                     : static_cast<int>(std::thread::hardware_concurrency());
    const int K = 10;

    printf("n=%zu d=%zu nq=%zu | M=%d efC=%d efS=%d threads=%d\n", n, d, nq, M, EF_C, EF_S, THREADS);

    // --------------------------------------------------------
    // 1. Populate ONE faiss storage. This is the only copy of the vectors.
    // --------------------------------------------------------
    faiss::IndexFlatL2 shared_storage(static_cast<faiss::idx_t>(d));
    shared_storage.add(static_cast<faiss::idx_t>(n), xb);
    delete[] xb;  // FlatNav never sees xb: it reads shared_storage directly
    xb = nullptr;

    // --------------------------------------------------------
    // 2. FlatNav graph on the shared storage (no vector copy).
    // --------------------------------------------------------
    auto flatnav = flatnav_faiss::make_flatnav_l2(shared_storage, M);

    double t0 = now_s();
    flatnav_faiss::build(*flatnav, EF_C, THREADS);
    double build_s = now_s() - t0;
    printf("FlatNav build on shared storage: %.2f s (%zu nodes)\n", build_s, flatnav->currentNumNodes());

    // --------------------------------------------------------
    // 3. Verify the storage really is shared.
    // --------------------------------------------------------
    assert(flatnav->usesExternalStorage());
    assert(flatnav->externalStorageData() == static_cast<const void*>(shared_storage.get_xb()));
    const uint64_t vec_bytes = shared_storage.codes.size();
    const uint64_t graph_bytes = flatnav_faiss::graph_bytes(*flatnav);
    const uint64_t owned_equiv = graph_bytes + static_cast<uint64_t>(n) * d * sizeof(float);
    printf("shared_storage.get_xb()      = %p\n", static_cast<const void*>(shared_storage.get_xb()));
    printf("flatnav external data ptr    = %p\n", flatnav->externalStorageData());
    printf("faiss vectors (shared)       = %.1f MB\n", vec_bytes / 1e6);
    printf("flatnav graph (links+labels) = %.1f MB  (node = %zu B)\n", graph_bytes / 1e6,
           flatnav->nodeSizeBytes());
    printf("flatnav with its own copy    = %.1f MB  -> saved %.1f MB\n", owned_equiv / 1e6,
           (owned_equiv - graph_bytes) / 1e6);

    // --------------------------------------------------------
    // 4. Ground truth: given, or exact search on the SAME storage.
    // --------------------------------------------------------
    std::vector<faiss::idx_t> gt;
    if (gt_raw) {
        gt.assign(gt_raw, gt_raw + nq * gt_k);
        delete[] gt_raw;
    } else {
        gt_k = K;
        gt.resize(nq * K);
        std::vector<float> gd(nq * K);
        shared_storage.search(static_cast<faiss::idx_t>(nq), xq, K, gd.data(), gt.data());
    }

    // --------------------------------------------------------
    // 5. Search FlatNav; ids are faiss ids.
    // --------------------------------------------------------
    std::vector<float> D(nq * K);
    std::vector<faiss::idx_t> I(nq * K);
    for (int ef : {EF_S / 2, EF_S, EF_S * 2, EF_S * 4}) {
        if (ef < K) continue;
        double s0 = now_s();
        flatnav_faiss::search(*flatnav, static_cast<faiss::idx_t>(nq), xq, K, ef, D.data(), I.data(), 1);
        double dt = now_s() - s0;
        printf("FlatNav efS=%4d  recall@%d=%.4f  QPS(1 thread)=%.0f\n", ef, K,
               recall_at_k(I.data(), gt.data(), nq, K, static_cast<int>(gt_k)), nq / dt);
    }

    // --------------------------------------------------------
    // 6. Save only the graph; reload and re-attach to the faiss storage.
    //    (Persist the vectors with faiss::write_index(&shared_storage, ...).)
    // --------------------------------------------------------
    const std::string graph_file = "flatnav_shared_storage.graph";
    flatnav->saveIndex(graph_file);
    auto reloaded = flatnav_faiss::FlatNavL2::loadIndex(graph_file);
    flatnav_faiss::reattach(*reloaded, shared_storage);
    std::vector<faiss::idx_t> I2(nq * K);
    std::vector<float> D2(nq * K);
    flatnav_faiss::search(*reloaded, static_cast<faiss::idx_t>(nq), xq, K, EF_S, D2.data(), I2.data(), 1);
    flatnav_faiss::search(*flatnav, static_cast<faiss::idx_t>(nq), xq, K, EF_S, D.data(), I.data(), 1);
    struct stat st;
    stat(graph_file.c_str(), &st);
    printf("graph file: %.1f MB, reloaded results identical: %s\n", st.st_size / 1e6, I == I2 ? "yes" : "NO");
    std::remove(graph_file.c_str());

    delete[] xq;
    printf("OK\n");
    return I == I2 ? 0 : 2;
}
