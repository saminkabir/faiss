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
// Three independent ways to get the numbers are provided so you
// can cross-check them against each other:
//
//   A) serialized_size()      — exact, version-independent, uses
//                                faiss::write_index() into memory.
//                                Doubles peak memory briefly (it
//                                re-serializes the shared vectors
//                                once per index), so avoid it on
//                                huge datasets if RAM is tight.
//   B) hnsw_graph_bytes_direct() — reads HNSW's public containers
//                                directly, zero extra memory, but
//                                depends on faiss/impl/HNSW.h's
//                                field layout (stable across recent
//                                faiss releases, but verify against
//                                your checkout if in doubt).
//   C) get_current_rss_bytes() — OS-level ground truth (VmRSS
//                                delta). Includes allocator
//                                overhead/fragmentation so it will
//                                run a bit higher than A/B, but
//                                needs no internal knowledge at all
//                                and costs nothing extra to call.
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
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

// ------------------------------------------------------------
// A) Exact, version-independent size via in-memory serialization.
//
// faiss::write_index() always writes the FULL index, including
// whatever storage it points to — even when that storage is
// shared with another index. So serializing hnsw and nsg
// separately each embeds a full redundant copy of the base
// vectors. Serializing shared_storage alone gives that redundant
// amount, which lets you subtract it back out:
//
//     hnsw_graph_overhead = serialized_size(&hnsw) - serialized_size(&shared_storage)
//     nsg_graph_overhead  = serialized_size(&nsg)  - serialized_size(&shared_storage)
//     combined            = storage_bytes + hnsw_graph_overhead + nsg_graph_overhead
// ------------------------------------------------------------

inline size_t serialized_size(const faiss::Index* index) {
    faiss::VectorIOWriter writer;
    faiss::write_index(index, &writer);
    return writer.data.size();
}

// ------------------------------------------------------------
// B) Direct field-sum for HNSW's graph structure only (no
// storage, no serialization, no extra memory). Fields per
// faiss/impl/HNSW.h:
//
//   levels                 : vector<int>          — one level per point
//   offsets                : vector<size_t>        — CSR row pointers, ntotal+1
//   neighbors               : vector<storage_idx_t> (int32) — CSR adjacency
//   cum_nneighbor_per_level : vector<int>           — tiny, one per level
//   assign_probas           : vector<double>        — tiny, one per level
// ------------------------------------------------------------

inline size_t hnsw_graph_bytes_direct(const faiss::HNSW& h) {
    return h.neighbors.size() * sizeof(faiss::HNSW::storage_idx_t) +
            h.offsets.size() * sizeof(size_t) +
            h.levels.size() * sizeof(int) +
            h.cum_nneighbor_per_level.size() * sizeof(int) +
            h.assign_probas.size() * sizeof(double);
}

// Exact bytes actually held by IndexFlat's raw code buffer
// (codes.size() is already in bytes — no need to multiply by d
// or sizeof(float) yourself, and it can't drift from whatever
// code_size the index was actually built with).
inline size_t flat_storage_bytes_direct(const faiss::IndexFlat& flat) {
    return flat.codes.size();
}

// ------------------------------------------------------------
// C) Process RSS (resident set size) in bytes, read from
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
// One-call report combining A) and B), printed and returned so
// you can also feed the numbers into log_metric() if you want
// them in benchmark_log.csv alongside build/query timings.
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

    // Method A (serialization-based) — the numbers you should trust.
    size_t storage_serialized = serialized_size(&shared_storage);
    size_t hnsw_serialized = serialized_size(&hnsw);
    size_t nsg_serialized = serialized_size(&nsg);

    r.storage_bytes = storage_serialized;
    r.hnsw_graph_overhead_bytes = hnsw_serialized - storage_serialized;
    r.nsg_graph_overhead_bytes = nsg_serialized - storage_serialized;
    r.combined_bytes = r.storage_bytes + r.hnsw_graph_overhead_bytes +
            r.nsg_graph_overhead_bytes;
    r.unshared_total_bytes = hnsw_serialized + nsg_serialized;

    // Method B (direct field sum) — cheap cross-check, no reserialization.
    size_t storage_direct = flat_storage_bytes_direct(shared_storage);
    size_t hnsw_direct = hnsw_graph_bytes_direct(hnsw.hnsw);

    auto MB = [](size_t bytes) { return bytes / (1024.0 * 1024.0); };

    std::cout << "\n========================================\n"
              << "Index size breakdown\n"
              << "========================================\n";

    std::cout << "[serialization method]\n";
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

    std::cout << "\n[direct field-sum cross-check, HNSW + storage only]\n";
    std::cout << "  Shared vector storage  : " << storage_direct
              << " bytes (" << MB(storage_direct) << " MB) "
              << (storage_direct == r.storage_bytes - /* header slack */ 0
                          ? "(matches serialized size minus a small header)\n"
                          : "\n");
    std::cout << "  HNSW graph overhead    : " << hnsw_direct << " bytes ("
              << MB(hnsw_direct) << " MB)\n";

    std::cout << "\n[process RSS, informational only]\n";
    std::cout << "  Current RSS            : "
              << MB(get_current_rss_bytes()) << " MB\n";

    return r;
}
