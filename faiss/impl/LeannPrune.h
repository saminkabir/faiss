/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace faiss {

/** High-degree preserving graph pruning, following the LEANN paper
 *
 *   "LEANN: A Low-Storage Overhead Vector Index", Wang et al., 2025
 *   (Section 5, Algorithm 3)
 *
 * Idea: a small fraction `hub_fraction` (beta) of the nodes are "hubs", the
 * nodes that a graph search visits most often. Hubs keep a large out-degree
 * budget (`hub_max_degree`, "M" in the paper); all other nodes get a small
 * one (`low_max_degree`, "m" in the paper, m = M / 5), and every node keeps a
 * route into the hubs.
 *
 * The same parameters are used for HNSW (level 0) and NSG:
 *
 *  - IndexHNSW::prune_leann is the paper's algorithm: the level-0 lists are
 *    rebuilt by re-inserting all nodes, hubs choosing up to M neighbors and
 *    the others m; every edge is also added in reverse and may grow any node
 *    up to M.
 *  - IndexNSG::prune_leann is an adaptation: NSG's edges are pruned in
 *    place, hubs keep their edges, the others keep their edges into hubs and
 *    then their nearest ones, up to m.
 *
 * Neither changes the in-memory layout of the graph (fixed-size, -1 padded
 * neighbor tables): the savings show up in LeannPruneStats::edge_bytes_after,
 * the size of the neighbor ids in a compact (CSR) encoding.
 */

/// How a node's importance ("hub-ness") is estimated on the *original* graph.
enum LeannHubScore : int {
    /// out-degree in the original graph (what the paper does)
    LEANN_HUB_OUT_DEGREE = 0,
    /// in-degree in the original graph (how often a node is pointed to)
    LEANN_HUB_IN_DEGREE = 1,
    /// out-degree + in-degree
    LEANN_HUB_TOTAL_DEGREE = 2,
};

struct LeannPruneParams {
    /// fraction of nodes that are kept as high-degree hubs (beta, paper: 2%).
    /// 0 => no hub; 1 => every node is a hub.
    float hub_fraction = 0.02f;

    /// max out-degree of hubs (M in the paper); for HNSW also the cap that
    /// reverse edges may grow any node to.
    /// 0 => the widest the graph can hold (2*M for HNSW level 0, R for NSG).
    int hub_max_degree = 0;

    /// max out-degree of the edges a non-hub node picks for itself
    /// (m in the paper). For HNSW, reverse edges can add to it.
    /// 0 => hub_max_degree / 5.
    int low_max_degree = 0;

    /// how the hubs are selected on the original graph
    LeannHubScore hub_score = LEANN_HUB_OUT_DEGREE;

    /// HNSW only: size of the candidate list used to select the neighbors of
    /// every re-inserted node. 0 => HNSW::efConstruction.
    int candidate_pool_size = 0;

    /// HNSW only: when a reverse edge overflows a node's list, prune it to
    /// hub_max_degree * (1 - reverse_headroom). 0 reproduces the paper
    /// ("shrink to M"). A positive value trades a slightly lower degree for
    /// fewer re-prunes.
    float reverse_headroom = 0.0f;

    /// print progress and degree statistics
    bool verbose = false;
};

/// Result of a pruning pass. Degrees are out-degrees on the pruned layer
/// (HNSW: level 0, NSG: the whole graph).
struct LeannPruneStats {
    size_t n_nodes = 0;
    size_t n_hubs = 0;

    // effective parameters that were used
    int hub_max_degree = 0;
    int low_max_degree = 0;

    size_t edges_before = 0;
    size_t edges_after = 0;
    double avg_degree_before = 0;
    double avg_degree_after = 0;
    int max_degree_before = 0;
    int max_degree_after = 0;

    /// out-degree of hubs / non-hubs after pruning
    double avg_degree_hubs_after = 0;
    double avg_degree_non_hubs_after = 0;

    /// nodes without any incoming edge after pruning
    size_t n_zero_indegree_after = 0;

    /// nodes that a graph traversal cannot reach after pruning. HNSW: not
    /// reachable on level 0 from the entry point or from any node of the
    /// upper levels (where a search lands after its descent). NSG: not
    /// reachable from the navigating node (the pass re-attaches such nodes,
    /// so this is 0).
    size_t n_unreachable_after = 0;

    /// bytes of the neighbor ids if the graph were stored compactly (CSR,
    /// 4 bytes per id, no padding). The in-memory HNSW/NSG arrays keep their
    /// fixed-size, -1 padded layout, so their footprint does not change.
    size_t edge_bytes_before = 0;
    size_t edge_bytes_after = 0;
};

/** Mark the top `hub_fraction` nodes by hub score in `is_hub` (resized to n).
 * Ties are broken by node id, so the selection is deterministic.
 * @return the number of hubs selected
 */
inline size_t leann_select_hubs(
        const std::vector<int>& out_degree,
        const std::vector<int>& in_degree,
        LeannHubScore score_kind,
        float hub_fraction,
        std::vector<uint8_t>& is_hub) {
    const size_t n = out_degree.size();
    is_hub.assign(n, 0);
    size_t n_hubs = static_cast<size_t>(
            std::ceil(static_cast<double>(hub_fraction) * double(n)));
    n_hubs = std::min(n_hubs, n);
    if (n_hubs == n) {
        std::fill(is_hub.begin(), is_hub.end(), 1);
        return n_hubs;
    }
    if (n_hubs == 0) {
        return 0;
    }
    auto score = [&](size_t i) -> int64_t {
        switch (score_kind) {
            case LEANN_HUB_IN_DEGREE:
                return in_degree[i];
            case LEANN_HUB_TOTAL_DEGREE:
                return int64_t(in_degree[i]) + out_degree[i];
            default:
                return out_degree[i];
        }
    };
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t(0));
    std::nth_element(
            order.begin(),
            order.begin() + n_hubs,
            order.end(),
            [&](size_t a, size_t b) {
                int64_t sa = score(a), sb = score(b);
                return sa != sb ? sa > sb : a < b;
            });
    for (size_t k = 0; k < n_hubs; k++) {
        is_hub[order[k]] = 1;
    }
    return n_hubs;
}

} // namespace faiss
