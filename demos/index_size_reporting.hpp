// ============================================================
// index_size_reporting.hpp
//
// Drop-in helpers for measuring the on-heap size of the paired
// HNSW + NSG demo (shared_storage / hnsw / nsg from
// demo_paired_hnsw_nsg.cpp), broken into:
//
//   1. Vector storage (the shared IndexFlatL2)      — counted once
//   2. HNSW graph overhead (levels/offsets/neighbors)
//   3. NSG  graph overhead (final_graph adjacency)
//
// IMPORTANT — why this is direct-field-based, not serialization-based:
//
// An earlier version of this header used faiss::write_index() into
// an in-memory buffer to measure sizes. That works fine for
// shared_storage and hnsw, but throws for nsg:
//
//   terminate called after throwing an instance of 'faiss::FaissException'
//   what():  ... don't know how to serialize this IndexNSG subtype
//
// This is a real faiss limitation, confirmed against
// faiss/impl/index_write.cpp: write_index() for NSG only recognizes
// the three concrete subclasses IndexNSGFlat / IndexNSGPQ /
// IndexNSGSQ (each dispatched via dynamic_cast to pick a fourcc
// tag). IndexHNSW has an explicit fallback for the plain base class
// (typeid(*idx) == typeid(IndexHNSW) -> fourcc("IH00")); IndexNSG
// has no such fallback, so a plain `faiss::IndexNSG(&storage, R)`
// like the one built in demo_paired_hnsw_nsg.cpp has no known tag
// and the FAISS_THROW_IF_MSG(h == 0, ...) fires.
//
// So every size below is computed directly from public struct
// fields instead, cross-checked against faiss's own write_HNSW() /
// write_NSG() (faiss/impl/index_write.cpp) field-for-field, and
// against process RSS as an independent sanity check. Nothing here
// calls faiss::write_index().
//
// ------------------------------------------------------------
// Field provenance (faiss/impl/HNSW.h, faiss/impl/NSG.h,
// faiss/impl/index_write.cpp — verified against the faiss source
// as of this writing):
//
//   write_HNSW() writes exactly: assign_probas, cum_nneighbor_per_level,
//   levels, offsets, neighbors (all std::vector, via WRITEVECTOR) plus
//   a handful of scalars. hnsw_graph_bytes_direct() below sums the
//   byte size of those same five vectors — same fields, same order.
//
//   NSG's graph lives in nsg.final_graph, a
//   std::shared_ptr<nsg::Graph<int32_t>> with public fields
//   `data` (int32_t*), `N`, `K`, `own_fields`. After NSG::build(),
//   own_fields is true and data is one dense heap allocation of
//   N*K int32_t (see nsg::Graph's `Graph(int N_in, int K_in)`
//   constructor, which does `data = new node_t[N_in * K_in]`).
//   So its exact RAM footprint is N * K * sizeof(int32_t) —
//   nsg_graph_bytes_direct() below.
//
// ------------------------------------------------------------
//
// Usage (see report_index_sizes() below for the one-call version):
//
//     report_index_sizes(shared_storage, hnsw, nsg);
//
// ============================================================

#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexNSG.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/NSG.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

// ------------------------------------------------------------
// Shared vector storage: IndexFlat's `codes` is a std::vector<uint8_t>
// holding the raw vectors back to back (code_size = d * sizeof(float)
// for IndexFlatL2/IP). codes.size() is therefore already the exact
// byte count — no need to recompute from ntotal * d * sizeof(float)
// yourself, and it can't drift from whatever code_size the index
// actually has.
// ------------------------------------------------------------

inline size_t flat_storage_bytes_direct(const faiss::IndexFlat& flat) {
    return flat.codes.size();
}

// ------------------------------------------------------------
// HNSW graph structure only (no storage). Fields per
// faiss/impl/HNSW.h, matched against write_HNSW() in
// faiss/impl/index_write.cpp:
//
//   levels                 : vector<int>           — one level per point
//   offsets                : vector<size_t>         — CSR row pointers, ntotal+1
//   neighbors               : vector<storage_idx_t> (int32) — CSR adjacency
//   cum_nneighbor_per_level : vector<int>            — tiny, one per level
//   assign_probas           : vector<double>         — tiny, one per level
// ------------------------------------------------------------

inline size_t hnsw_graph_bytes_direct(const faiss::HNSW& h) {
    return h.neighbors.size() * sizeof(faiss::HNSW::storage_idx_t) +
            h.offsets.size() * sizeof(size_t) +
            h.levels.size() * sizeof(int) +
            h.cum_nneighbor_per_level.size() * sizeof(int) +
            h.assign_probas.size() * sizeof(double);
}

// ------------------------------------------------------------
// NSG graph structure only (no storage). final_graph is a dense
// N-by-K int32 adjacency matrix (K = R = max out-degree, padded
// with -1 for nodes with fewer than R real neighbors), allocated
// as one heap block by nsg::Graph's owning constructor. See the
// file header comment above for how this was confirmed against
// faiss's own NSG.h / write_NSG().
// ------------------------------------------------------------

inline size_t nsg_graph_bytes_direct(const faiss::NSG& nsg) {
    if (!nsg.is_built || !nsg.final_graph) {
        // Not built yet (or build_type path that didn't populate
        // final_graph) — nothing allocated for the graph itself.
        return 0;
    }
    size_t N = static_cast<size_t>(nsg.final_graph->N);
    size_t K = static_cast<size_t>(nsg.final_graph->K);
    return N * K * sizeof(int32_t);
}

// ------------------------------------------------------------
// Process RSS (resident set size) in bytes, read from
// /proc/self/status. Linux-only. Use as a sanity check on the
// numbers above, not as the primary source — it reflects the
// whole process (allocator overhead, fragmentation, anything
// else you've allocated), not just these three objects.
// ------------------------------------------------------------

inline size_t get_current_rss_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            size_t kb = std::stoul(line.substr(6));
            return kb * 1024;
        }
    }
    return 0;
}

// ------------------------------------------------------------
// One-call report. All three components are counted directly
// from live heap structures (no serialization, no faiss version
// dependence beyond the field names cited above), printed, and
// returned so you can also feed the numbers into log_metric() if
// you want them in benchmark_log.csv alongside build/query timings.
// ------------------------------------------------------------

struct IndexSizeReport {
    size_t storage_bytes = 0;
    size_t hnsw_graph_overhead_bytes = 0;
    size_t nsg_graph_overhead_bytes = 0;
    size_t combined_bytes = 0; // storage counted once + both graphs
    size_t unshared_total_bytes = 0; // what it would cost with two
                                      // independent copies of storage
};

inline IndexSizeReport report_index_sizes(
        const faiss::IndexFlatL2& shared_storage,
        const faiss::IndexHNSW& hnsw,
        const faiss::IndexNSG& nsg) {

    IndexSizeReport r;

    r.storage_bytes = flat_storage_bytes_direct(shared_storage);
    r.hnsw_graph_overhead_bytes = hnsw_graph_bytes_direct(hnsw.hnsw);
    r.nsg_graph_overhead_bytes = nsg_graph_bytes_direct(nsg.nsg);
    r.combined_bytes = r.storage_bytes + r.hnsw_graph_overhead_bytes +
            r.nsg_graph_overhead_bytes;
    r.unshared_total_bytes = 2 * r.storage_bytes +
            r.hnsw_graph_overhead_bytes + r.nsg_graph_overhead_bytes;

    auto MB = [](size_t bytes) { return bytes / (1024.0 * 1024.0); };

    std::cout << "\n========================================\n"
              << "Index size breakdown (direct field measurement)\n"
              << "========================================\n";

    std::cout << "  Shared vector storage  : " << r.storage_bytes
              << " bytes (" << MB(r.storage_bytes) << " MB)\n";
    std::cout << "  HNSW graph overhead    : " << r.hnsw_graph_overhead_bytes
              << " bytes (" << MB(r.hnsw_graph_overhead_bytes) << " MB)\n";
    std::cout << "  NSG  graph overhead    : " << r.nsg_graph_overhead_bytes
              << " bytes (" << MB(r.nsg_graph_overhead_bytes) << " MB)\n";
    std::cout << "  Combined (storage x1)  : " << r.combined_bytes
              << " bytes (" << MB(r.combined_bytes) << " MB)\n";
    std::cout << "  If storage were NOT shared, total would be "
              << MB(r.unshared_total_bytes) << " MB (saved "
              << MB(r.unshared_total_bytes - r.combined_bytes)
              << " MB by sharing)\n";

    std::cout << "\n[process RSS, informational cross-check only — "
              << "includes allocator/fragmentation overhead beyond "
              << "just these three objects]\n";
    std::cout << "  Current RSS            : "
              << MB(get_current_rss_bytes()) << " MB\n";

    return r;
}
