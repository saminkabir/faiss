/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexNSG.h>

namespace faiss {

/**
 * Paired HNSW + NSG index with one shared IndexFlat storage.
 *
 * Architecture:
 *
 *                    IndexNSGHNSWFlat
 *                           |
 *                    shared IndexFlat
 *                     /           \
 *                    /             \
 *             IndexHNSW         IndexNSG
 *
 * Raw vectors are stored exactly once.
 *
 * Search:
 *
 *   1. Search HNSW for hnsw_k candidates.
 *   2. Search NSG for nsg_k candidates.
 *   3. Union the candidate IDs.
 *   4. Recompute exact distance using shared raw vectors.
 *   5. Return final top-k.
 *
 * NOTE:
 *
 * IndexHNSW and IndexNSG must provide:
 *
 *     void add_links_only(idx_t n, const float* x);
 *
 * Those functions must build graph structure WITHOUT calling:
 *
 *     storage->add(...)
 */
struct IndexNSGHNSWFlat : Index {
    /// Shared raw-vector storage.
    /// This object is owned by IndexNSGHNSWFlat.
    IndexFlat* storage = nullptr;

    /// HNSW graph using the shared storage.
    IndexHNSW* hnsw_index = nullptr;

    /// NSG graph using the shared storage.
    IndexNSG* nsg_index = nullptr;

    /**
     * Candidate count retrieved from HNSW.
     *
     * If <= 0, k from search() is used.
     */
    idx_t hnsw_k = 0;

    /**
     * Candidate count retrieved from NSG.
     *
     * If <= 0, k from search() is used.
     */
    idx_t nsg_k = 0;

    /**
     * Default constructor.
     *
     * Mainly useful for SWIG / Faiss infrastructure.
     */
    IndexNSGHNSWFlat();

    /**
     * Construct paired HNSW + NSG index.
     *
     * @param d          vector dimension
     * @param hnsw_M     HNSW M
     * @param nsg_R      NSG R
     * @param metric     METRIC_L2 or METRIC_INNER_PRODUCT
     */
    IndexNSGHNSWFlat(
            int d,
            int hnsw_M,
            int nsg_R,
            MetricType metric = METRIC_L2);

    ~IndexNSGHNSWFlat() override;

    /**
     * Train storage.
     *
     * IndexFlat itself does not require training, but this maintains
     * the standard Faiss interface.
     */
    void train(idx_t n, const float* x) override;

    /**
     * Add vectors.
     *
     * Raw vectors are added exactly once to shared storage.
     *
     * Then:
     *
     *     HNSW graph is constructed over existing storage.
     *     NSG graph is constructed over existing storage.
     *
     * Incremental add is intentionally disabled because NSG does
     * not support incremental addition.
     */
    void add(idx_t n, const float* x) override;

    /**
     * Paired search.
     */
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /**
     * Reconstruct using shared IndexFlat storage.
     */
    void reconstruct(idx_t key, float* recons) const override;

    /**
     * Reset graphs and shared storage.
     */
    void reset() override;

   private:
    /**
     * Search one query.
     */
    void search_one(
            const float* query,
            idx_t k,
            float* distances,
            idx_t* labels) const;
};

} // namespace faiss