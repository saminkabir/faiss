/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests for the LEANN-style high-degree preserving graph pruning
// (IndexHNSW::prune_leann and IndexNSG::prune_leann).

#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stack>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexNSG.h>
#include <faiss/clone_index.h>
#include <faiss/impl/FaissException.h>

using namespace faiss;

namespace {

// Gaussian mixture, so that the data has neighborhood structure
std::vector<float> make_clustered_data(size_t n, size_t d, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.f, 1.f);
    const size_t ncenters = 40;
    std::vector<float> centers(ncenters * d);
    for (auto& c : centers) {
        c = 4.f * gauss(rng);
    }
    std::vector<float> x(n * d);
    for (size_t i = 0; i < n; i++) {
        const float* c = &centers[(rng() % ncenters) * d];
        for (size_t j = 0; j < d; j++) {
            x[i * d + j] = c[j] + gauss(rng);
        }
    }
    return x;
}

std::vector<idx_t> exact_neighbors(
        const IndexFlat& flat,
        const float* q,
        size_t nq,
        size_t k) {
    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    flat.search(nq, q, k, D.data(), I.data());
    return I;
}

double recall_at_k(
        const std::vector<idx_t>& gt,
        const std::vector<idx_t>& res,
        size_t nq,
        size_t k) {
    size_t hit = 0;
    for (size_t i = 0; i < nq; i++) {
        std::set<idx_t> g(gt.begin() + i * k, gt.begin() + (i + 1) * k);
        for (size_t j = 0; j < k; j++) {
            hit += g.count(res[i * k + j]);
        }
    }
    return double(hit) / (nq * k);
}

// checks the structure of one -1 padded adjacency row
void check_row(
        const int32_t* row,
        int slots,
        int64_t self,
        int64_t n,
        int max_degree) {
    std::set<int32_t> seen;
    int deg = 0;
    bool padding = false;
    for (int j = 0; j < slots; j++) {
        if (row[j] < 0) {
            padding = true;
            continue;
        }
        ASSERT_FALSE(padding) << "gap in the adjacency row of node " << self;
        ASSERT_LT(row[j], n);
        ASSERT_NE(row[j], self) << "self loop on node " << self;
        ASSERT_TRUE(seen.insert(row[j]).second)
                << "duplicate edge " << self << " -> " << row[j];
        deg++;
    }
    ASSERT_LE(deg, max_degree) << "node " << self;
}

} // namespace

TEST(LeannPruneHNSW, invariants_and_recall) {
    const size_t d = 32, n = 6000, nq = 200, k = 10;
    const int M = 16;
    auto xb = make_clustered_data(n, d, 1);
    auto xq = make_clustered_data(nq, d, 2);

    IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = 100;
    index.add(n, xb.data());

    IndexFlat flat(d);
    flat.add(n, xb.data());
    auto gt = exact_neighbors(flat, xq.data(), nq, k);

    const HNSW& h = index.hnsw;
    const std::vector<int32_t> neighbors_before(
            h.neighbors.data(), h.neighbors.data() + h.neighbors.size());
    const int slots = h.nb_neighbors(0);
    ASSERT_EQ(slots, 2 * M);

    LeannPruneParams params;
    params.hub_fraction = 0.02f;
    params.hub_max_degree = 0; // -> 2 * M
    params.low_max_degree = 4;
    LeannPruneStats st = index.prune_leann(params);

    EXPECT_EQ(st.n_nodes, n);
    EXPECT_GE(st.n_hubs, size_t(std::ceil(0.02 * n)));
    EXPECT_LE(st.n_hubs, size_t(std::ceil(0.02 * n)) + 1);
    EXPECT_EQ(st.hub_max_degree, slots);
    EXPECT_EQ(st.low_max_degree, 4);

    // the pass really removed edges, and kept more on the hubs
    EXPECT_LT(st.edges_after, st.edges_before);
    EXPECT_LT(st.avg_degree_after, st.avg_degree_before);
    EXPECT_GT(st.avg_degree_hubs_after, st.avg_degree_non_hubs_after);
    EXPECT_EQ(st.edge_bytes_after, st.edges_after * sizeof(int32_t));
    EXPECT_LE(st.n_unreachable_after, n / 100);

    // structure of every level-0 row, and edge count matches the stats
    size_t edges = 0;
    for (size_t i = 0; i < n; i++) {
        size_t b, e;
        h.neighbor_range(i, 0, &b, &e);
        ASSERT_EQ(int(e - b), slots);
        check_row(&h.neighbors[b], slots, i, n, slots);
        for (size_t j = b; j < e && h.neighbors[j] >= 0; j++) {
            edges++;
        }
    }
    EXPECT_EQ(edges, st.edges_after);

    // upper levels are not touched
    for (size_t i = 0; i < n; i++) {
        size_t b, e;
        h.neighbor_range(i, 0, &b, &e);
        for (size_t j = e; j < h.offsets[i + 1]; j++) {
            ASSERT_EQ(h.neighbors[j], neighbors_before[j])
                    << "upper level slot changed for node " << i;
        }
    }

    // the pruned graph must still be a usable index
    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    index.hnsw.efSearch = 128;
    index.search(nq, xq.data(), k, D.data(), I.data());
    EXPECT_GE(recall_at_k(gt, I, nq, k), 0.85);
}

TEST(LeannPruneHNSW, hubs_keep_more_edges_than_the_others) {
    const size_t d = 16, n = 3000;
    auto xb = make_clustered_data(n, d, 3);
    IndexHNSWFlat index(d, 12);
    index.add(n, xb.data());

    LeannPruneParams params;
    params.hub_fraction = 0.05f;
    params.low_max_degree = 3;
    LeannPruneStats st = index.prune_leann(params);

    EXPECT_GE(st.n_hubs, size_t(std::ceil(0.05 * n)));
    // hubs keep a large budget, the others only a small one (plus reverse
    // edges)
    EXPECT_GT(st.avg_degree_hubs_after, 1.3 * st.avg_degree_non_hubs_after);
    EXPECT_LE(st.avg_degree_non_hubs_after, 2.0 * params.low_max_degree + 1);
}

TEST(LeannPruneHNSW, deterministic_across_thread_counts) {
    const size_t d = 24, n = 4000;
    auto xb = make_clustered_data(n, d, 4);
    IndexHNSWFlat index(d, 12);
    index.add(n, xb.data());

    std::unique_ptr<Index> copy(clone_index(&index));
    IndexHNSW* index2 = dynamic_cast<IndexHNSW*>(copy.get());
    ASSERT_NE(index2, nullptr);

    LeannPruneParams params;
    params.low_max_degree = 4;

    int nt = omp_get_max_threads();
    omp_set_num_threads(1);
    LeannPruneStats s1 = index.prune_leann(params);
    omp_set_num_threads(std::max(4, nt));
    LeannPruneStats s2 = index2->prune_leann(params);
    omp_set_num_threads(nt);

    EXPECT_EQ(s1.edges_after, s2.edges_after);
    ASSERT_EQ(index.hnsw.neighbors.size(), index2->hnsw.neighbors.size());
    for (size_t i = 0; i < index.hnsw.neighbors.size(); i++) {
        ASSERT_EQ(index.hnsw.neighbors[i], index2->hnsw.neighbors[i]) << i;
    }
}

TEST(LeannPruneHNSW, inner_product) {
    const size_t d = 24, n = 3000, nq = 100, k = 5;
    auto xb = make_clustered_data(n, d, 5);
    auto xq = make_clustered_data(nq, d, 6);
    IndexHNSWFlat index(d, 12, METRIC_INNER_PRODUCT);
    index.add(n, xb.data());

    IndexFlatIP flat(d);
    flat.add(n, xb.data());
    auto gt = exact_neighbors(flat, xq.data(), nq, k);

    LeannPruneParams params;
    params.low_max_degree = 5;
    LeannPruneStats st = index.prune_leann(params);
    EXPECT_LT(st.edges_after, st.edges_before);

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    index.hnsw.efSearch = 128;
    index.search(nq, xq.data(), k, D.data(), I.data());
    EXPECT_GE(recall_at_k(gt, I, nq, k), 0.8);
}

TEST(LeannPruneHNSW, rejects_bad_parameters) {
    const size_t d = 8, n = 500;
    auto xb = make_clustered_data(n, d, 7);
    IndexHNSWFlat index(d, 8);
    index.add(n, xb.data());

    LeannPruneParams too_wide;
    too_wide.hub_max_degree = 2 * 8 + 1; // more than the level-0 slots
    EXPECT_THROW(index.prune_leann(too_wide), FaissException);

    LeannPruneParams inverted;
    inverted.hub_max_degree = 8;
    inverted.low_max_degree = 9;
    EXPECT_THROW(index.prune_leann(inverted), FaissException);

    LeannPruneParams bad_fraction;
    bad_fraction.hub_fraction = 1.5f;
    EXPECT_THROW(index.prune_leann(bad_fraction), FaissException);

    IndexHNSWFlat empty(d, 8);
    EXPECT_THROW(empty.prune_leann(), FaissException);
}

TEST(LeannPruneHNSW, all_hubs_is_a_plain_rebuild_of_level_0) {
    // hub_fraction = 1 => every node has the full budget, i.e. the pass is
    // just a rebuild of the level-0 links, and must not hurt the recall.
    const size_t d = 16, n = 2000, nq = 100, k = 10;
    auto xb = make_clustered_data(n, d, 8);
    auto xq = make_clustered_data(nq, d, 9);
    IndexHNSWFlat index(d, 12);
    index.hnsw.efConstruction = 80;
    index.add(n, xb.data());

    IndexFlat flat(d);
    flat.add(n, xb.data());
    auto gt = exact_neighbors(flat, xq.data(), nq, k);

    LeannPruneParams params;
    params.hub_fraction = 1.0f;
    LeannPruneStats st = index.prune_leann(params);
    EXPECT_EQ(st.n_hubs, n);

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    index.hnsw.efSearch = 100;
    index.search(nq, xq.data(), k, D.data(), I.data());
    EXPECT_GE(recall_at_k(gt, I, nq, k), 0.9);
}

TEST(LeannPruneHNSW, quantized_storage) {
    // the vectors are only available as codes: the pass works on them
    const size_t d = 32, n = 3000, nq = 100, k = 10;
    auto xb = make_clustered_data(n, d, 15);
    auto xq = make_clustered_data(nq, d, 16);

    IndexHNSWSQ index(d, ScalarQuantizer::QT_8bit, 12);
    index.train(n, xb.data());
    index.add(n, xb.data());

    IndexFlat flat(d);
    flat.add(n, xb.data());
    auto gt = exact_neighbors(flat, xq.data(), nq, k);

    LeannPruneParams params;
    params.low_max_degree = 5;
    LeannPruneStats st = index.prune_leann(params);
    EXPECT_LT(st.edges_after, st.edges_before);

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    index.hnsw.efSearch = 128;
    index.search(nq, xq.data(), k, D.data(), I.data());
    EXPECT_GE(recall_at_k(gt, I, nq, k), 0.8);
}

TEST(LeannPruneNSG, invariants_connectivity_and_recall) {
    const size_t d = 32, n = 4000, nq = 100, k = 10;
    const int R = 24;
    auto xb = make_clustered_data(n, d, 10);
    auto xq = make_clustered_data(nq, d, 11);

    IndexNSGFlat index(d, R);
    index.build_type = 1; // NNDescent
    index.add(n, xb.data());
    ASSERT_TRUE(index.is_built);

    IndexFlat flat(d);
    flat.add(n, xb.data());
    auto gt = exact_neighbors(flat, xq.data(), nq, k);

    LeannPruneParams params;
    params.hub_fraction = 0.05f;
    params.low_max_degree = 5;
    LeannPruneStats st = index.prune_leann(params);

    EXPECT_EQ(st.n_nodes, n);
    EXPECT_EQ(st.hub_max_degree, R);
    EXPECT_LT(st.edges_after, st.edges_before);
    EXPECT_GT(st.avg_degree_hubs_after, st.avg_degree_non_hubs_after);
    EXPECT_EQ(st.n_unreachable_after, 0u);
    EXPECT_EQ(st.edge_bytes_after, st.edges_after * sizeof(int32_t));

    // adjacency structure
    const auto& g = *index.nsg.final_graph;
    ASSERT_EQ(g.K, R);
    size_t edges = 0;
    for (size_t i = 0; i < n; i++) {
        check_row(&g.data[i * R], R, i, n, R);
        for (int j = 0; j < R && g.data[i * R + j] >= 0; j++) {
            edges++;
        }
    }
    EXPECT_EQ(edges, st.edges_after);

    // every node is reachable from the navigating node (tree_grow)
    std::vector<char> seen(n, 0);
    std::stack<int> todo;
    todo.push(index.nsg.enterpoint);
    seen[index.nsg.enterpoint] = 1;
    size_t reached = 1;
    while (!todo.empty()) {
        int u = todo.top();
        todo.pop();
        for (int j = 0; j < R && g.data[size_t(u) * R + j] >= 0; j++) {
            int v = g.data[size_t(u) * R + j];
            if (!seen[v]) {
                seen[v] = 1;
                reached++;
                todo.push(v);
            }
        }
    }
    EXPECT_EQ(reached, n);

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);
    index.nsg.search_L = 100;
    index.search(nq, xq.data(), k, D.data(), I.data());
    EXPECT_GE(recall_at_k(gt, I, nq, k), 0.85);
}

TEST(LeannPruneNSG, keeps_edges_into_hubs_first) {
    const size_t d = 24, n = 3000;
    const int R = 24, m = 4;
    auto xb = make_clustered_data(n, d, 14);
    IndexNSGFlat index(d, R);
    index.build_type = 1;
    index.add(n, xb.data());

    const auto& g = *index.nsg.final_graph;
    std::vector<int32_t> orig(g.data, g.data + n * R);

    LeannPruneParams params;
    params.hub_fraction = 0.05f;
    params.low_max_degree = m;
    index.prune_leann(params);

    // recompute the hubs the way the pass does
    std::vector<int> out_deg(n, 0), in_deg(n, 0);
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < R && orig[i * R + j] >= 0; j++) {
            out_deg[i]++;
            in_deg[orig[i * R + j]]++;
        }
    }
    std::vector<uint8_t> is_hub;
    leann_select_hubs(
            out_deg, in_deg, params.hub_score, params.hub_fraction, is_hub);
    is_hub[index.nsg.enterpoint] = 1;

    size_t n_truncated = 0;
    for (size_t q = 0; q < n; q++) {
        const int budget = is_hub[q] ? R : m;
        // a built NSG can hold a few duplicate entries, which the pass drops
        std::vector<int32_t> uniq;
        for (int j = 0; j < out_deg[q]; j++) {
            if (std::find(uniq.begin(), uniq.end(), orig[q * R + j]) ==
                uniq.end()) {
                uniq.push_back(orig[q * R + j]);
            }
        }
        const std::set<int32_t> old_row(uniq.begin(), uniq.end());
        const int kept = std::min<int>(uniq.size(), budget);
        // the first `kept` edges are edges of the original graph (tree_grow
        // may only append after them)
        for (int j = 0; j < kept; j++) {
            ASSERT_TRUE(old_row.count(g.data[q * R + j]))
                    << "node " << q << " got a new edge";
        }
        if (static_cast<int>(uniq.size()) <= budget) {
            // nothing to prune: the row is left as it was
            for (size_t j = 0; j < uniq.size(); j++) {
                ASSERT_EQ(g.data[q * R + j], uniq[j]);
            }
            continue;
        }
        n_truncated++;
        // edges into hubs come first
        int n_hub_targets = 0;
        for (int32_t v : old_row) {
            n_hub_targets += is_hub[v];
        }
        for (int j = 0; j < std::min(n_hub_targets, kept); j++) {
            ASSERT_TRUE(is_hub[g.data[q * R + j]])
                    << "node " << q << " lost a hub edge for a non-hub one";
        }
    }
    EXPECT_GT(n_truncated, n / 2);
}

TEST(LeannPruneNSG, deterministic_across_thread_counts) {
    const size_t d = 16, n = 2000;
    auto xb = make_clustered_data(n, d, 12);
    IndexNSGFlat index(d, 16);
    index.build_type = 1;
    index.add(n, xb.data());

    std::vector<int32_t> orig(
            index.nsg.final_graph->data, index.nsg.final_graph->data + n * 16);

    LeannPruneParams params;
    params.low_max_degree = 3;

    int nt = omp_get_max_threads();
    omp_set_num_threads(1);
    index.prune_leann(params);
    std::vector<int32_t> a(
            index.nsg.final_graph->data, index.nsg.final_graph->data + n * 16);

    // put the original graph back and prune again with more threads
    std::copy(orig.begin(), orig.end(), index.nsg.final_graph->data);
    omp_set_num_threads(std::max(4, nt));
    index.prune_leann(params);
    std::vector<int32_t> b(
            index.nsg.final_graph->data, index.nsg.final_graph->data + n * 16);
    omp_set_num_threads(nt);

    EXPECT_EQ(a, b);
}

TEST(LeannPruneNSG, rejects_bad_parameters) {
    const size_t d = 8, n = 300;
    auto xb = make_clustered_data(n, d, 13);
    IndexNSGFlat index(d, 16);
    index.build_type = 1;

    EXPECT_THROW(index.prune_leann(), FaissException); // not built

    index.add(n, xb.data());
    LeannPruneParams too_wide;
    too_wide.hub_max_degree = 17;
    EXPECT_THROW(index.prune_leann(too_wide), FaissException);

    LeannPruneParams inverted;
    inverted.hub_max_degree = 8;
    inverted.low_max_degree = 9;
    EXPECT_THROW(index.prune_leann(inverted), FaissException);
}
