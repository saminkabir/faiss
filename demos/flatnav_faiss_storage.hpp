#pragma once
// ============================================================
// flatnav_faiss_storage.hpp
//
// Glue that lets a FlatNav graph use a faiss::IndexFlat as its vector
// storage (shared storage). FlatNav allocates only its graph
// ([M links][label] per node); every distance reads the vector
// straight out of faiss' `codes` buffer (IndexFlat::get_xb()).
//
// Node id i in FlatNav == row i in the faiss storage == faiss id i,
// so FlatNav, IndexHNSW(&storage) and IndexNSG(&storage) all agree on
// ids and can be compared / paired directly.
//
// Requirements
//   * FlatNav with ExternalVectorStorage support (flatnav branch
//     `faiss-shared-storage`, include/flatnav/index/Index.h).
//   * SIMD: the demos build FlatNav with AVX2 (no AVX-512) by default;
//     see FLATNAV_SIMD in demos/CMakeLists.txt.
//   * storage is float32 IndexFlatL2 (-> FlatNav L2) or IndexFlatIP
//     (-> FlatNav inner product). Scalar-quantized FlatNav cannot share
//     because its internal vector format differs from faiss'.
//   * Populate the storage before building, and don't call
//     storage.add() while FlatNav is reading it (std::vector growth
//     can move the buffer). If you do grow it, call
//     reattach(index, storage) before the next build/search.
// ============================================================

#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/impl/FaissAssert.h>

#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <flatnav/util/Multithreading.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace flatnav_faiss {

using flatnav::util::DataType;
using L2Distance = flatnav::distances::SquaredL2Distance<DataType::float32>;
using IPDistance = flatnav::distances::InnerProductDistance<DataType::float32>;

// int labels: label i == faiss id i (fine up to 2^31 - 1 vectors).
using FlatNavL2 = flatnav::Index<L2Distance, int>;
using FlatNavIP = flatnav::Index<IPDistance, int>;

// View of a faiss IndexFlat's vectors as FlatNav external storage.
inline flatnav::ExternalVectorStorage storage_view(const faiss::IndexFlat& storage) {
    return flatnav::ExternalVectorStorage{
            /* data = */ storage.get_xb(),
            /* num_vectors = */ static_cast<size_t>(storage.ntotal),
            /* row_stride_bytes = */ static_cast<size_t>(storage.d) * sizeof(float)};
}

// Empty FlatNav L2 graph whose vectors are `storage` (an IndexFlatL2).
// max_nodes = 0 -> storage.ntotal.
inline std::unique_ptr<FlatNavL2> make_flatnav_l2(
        const faiss::IndexFlat& storage,
        int M,
        size_t max_nodes = 0,
        bool collect_stats = false) {
    FAISS_THROW_IF_NOT_MSG(
            storage.metric_type == faiss::METRIC_L2,
            "make_flatnav_l2: storage must be an IndexFlatL2");
    FAISS_THROW_IF_NOT_MSG(storage.ntotal > 0, "storage is empty: add vectors first");
    return std::make_unique<FlatNavL2>(
            L2Distance::create(storage.d),
            static_cast<int>(max_nodes ? max_nodes : storage.ntotal),
            M,
            storage_view(storage),
            collect_stats,
            DataType::float32);
}

// Empty FlatNav inner-product graph whose vectors are `storage` (an IndexFlatIP).
inline std::unique_ptr<FlatNavIP> make_flatnav_ip(
        const faiss::IndexFlat& storage,
        int M,
        size_t max_nodes = 0,
        bool collect_stats = false) {
    FAISS_THROW_IF_NOT_MSG(
            storage.metric_type == faiss::METRIC_INNER_PRODUCT,
            "make_flatnav_ip: storage must be an IndexFlatIP");
    FAISS_THROW_IF_NOT_MSG(storage.ntotal > 0, "storage is empty: add vectors first");
    return std::make_unique<FlatNavIP>(
            IPDistance::create(storage.d),
            static_cast<int>(max_nodes ? max_nodes : storage.ntotal),
            M,
            storage_view(storage),
            collect_stats,
            DataType::float32);
}

// Call after storage.add() (buffer may have moved) or after
// FlatNavL2::loadIndex() of a graph saved in shared-storage mode.
template <class FlatNavIndex>
inline void reattach(FlatNavIndex& index, const faiss::IndexFlat& storage) {
    index.attachExternalStorage(storage_view(storage));
}

// Build the graph over storage rows not yet in the graph.
// num_threads = FlatNav's own std::thread workers (independent of OpenMP).
template <class FlatNavIndex>
inline void build(
        FlatNavIndex& index,
        int ef_construction,
        int num_threads = 1,
        int num_initializations = 100) {
    index.setNumThreads(static_cast<uint32_t>(std::max(1, num_threads)));
    size_t todo = std::min(index.externalStorageNumVectors(), index.maxNodeCount()) -
            index.currentNumNodes();
    index.addFromStorage(todo, ef_construction, num_initializations);
}

// Faiss-style batched search: fills D/I as (nq x k), -1 padded.
// Uses FlatNav's thread pool (index.getNumThreads()) unless num_threads > 0.
template <class FlatNavIndex>
inline void search(
        FlatNavIndex& index,
        faiss::idx_t nq,
        const float* xq,
        int k,
        int ef_search,
        float* D,
        faiss::idx_t* I,
        int num_threads = 0,
        int num_initializations = 100) {
    const size_t d = index.dataDimension();
    auto one = [&](uint32_t q) {
        auto res = index.search(xq + static_cast<size_t>(q) * d, k, ef_search, num_initializations);
        const size_t row = static_cast<size_t>(q) * static_cast<size_t>(k);
        for (int j = 0; j < k; j++) {
            bool have = j < static_cast<int>(res.size());
            D[row + j] = have ? res[j].first : -1.0f;
            I[row + j] = have ? static_cast<faiss::idx_t>(res[j].second) : -1;
        }
    };
    uint32_t threads = num_threads > 0 ? static_cast<uint32_t>(num_threads) : index.getNumThreads();
    if (threads <= 1) {
        for (faiss::idx_t q = 0; q < nq; q++) {
            one(static_cast<uint32_t>(q));
        }
        return;
    }
    flatnav::executeInParallel(0, static_cast<uint32_t>(nq), threads, one);
}

// Bytes FlatNav itself allocated for node blocks (graph only in shared mode,
// vectors + graph in owned mode).
template <class FlatNavIndex>
inline uint64_t graph_bytes(const FlatNavIndex& index) {
    return index.getTotalIndexMemory();
}

// Vector bytes held inside FlatNav's own memory (0 when shared).
template <class FlatNavIndex>
inline uint64_t vector_bytes(const FlatNavIndex& index) {
    return index.usesExternalStorage()
            ? 0
            : static_cast<uint64_t>(index.maxNodeCount()) * index.dataSizeBytes();
}

// Graph-only bytes (M links + label per node), comparable across the
// shared and owned layouts and with hnsw/nsg_graph_bytes_direct().
template <class FlatNavIndex>
inline uint64_t graph_overhead_bytes(const FlatNavIndex& index) {
    return index.getTotalIndexMemory() - vector_bytes(index);
}

// ------------------------------------------------------------
// Separate (owned) storage: the regular FlatNav index that keeps its
// own copy of the vectors in its interleaved [vector][links][label]
// node layout. Used by the *_separate_storage_* demos.
// ------------------------------------------------------------
inline std::unique_ptr<FlatNavL2> make_flatnav_l2_owned(
        int d,
        size_t max_nodes,
        int M,
        bool collect_stats = false) {
    return std::make_unique<FlatNavL2>(
            L2Distance::create(d),
            static_cast<int>(max_nodes),
            M,
            collect_stats,
            DataType::float32);
}

inline std::unique_ptr<FlatNavIP> make_flatnav_ip_owned(
        int d,
        size_t max_nodes,
        int M,
        bool collect_stats = false) {
    return std::make_unique<FlatNavIP>(
            IPDistance::create(d),
            static_cast<int>(max_nodes),
            M,
            collect_stats,
            DataType::float32);
}

// Copies x (n x d, row-major) into the owned index and builds the graph.
// Labels are 0..n-1, i.e. the same ids faiss assigns to the same rows.
template <class FlatNavIndex>
inline void build_owned(
        FlatNavIndex& index,
        const float* x,
        size_t n,
        int ef_construction,
        int num_threads = 1,
        int num_initializations = 100) {
    if (index.usesExternalStorage()) {
        throw std::logic_error("build_owned(): index uses shared storage; use build()");
    }
    std::vector<int> labels(n);
    std::iota(labels.begin(), labels.end(), 0);
    index.setNumThreads(static_cast<uint32_t>(std::max(1, num_threads)));
    index.template addBatch<float>(
            const_cast<float*>(x), labels, ef_construction, num_initializations);
}

} // namespace flatnav_faiss
