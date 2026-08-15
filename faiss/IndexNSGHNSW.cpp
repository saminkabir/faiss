/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNSGHNSW.h>

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

namespace faiss {

/***************************************************************
 * Helper candidate structure
 ***************************************************************/

namespace {

struct PairedCandidate {
    idx_t id;
    float distance;
};

} // namespace

/***************************************************************
 * Default constructor
 ***************************************************************/

IndexNSGHNSWFlat::IndexNSGHNSWFlat()
        : Index(0, METRIC_L2) {
    storage = nullptr;
    hnsw_index = nullptr;
    nsg_index = nullptr;

    hnsw_k = 0;
    nsg_k = 0;

    is_trained = true;
}

/***************************************************************
 * Main constructor
 ***************************************************************/

IndexNSGHNSWFlat::IndexNSGHNSWFlat(
        int d_in,
        int hnsw_M,
        int nsg_R,
        MetricType metric)
        : Index(d_in, metric) {
    FAISS_THROW_IF_NOT_MSG(
            d_in > 0,
            "IndexNSGHNSWFlat: dimension must be > 0");

    FAISS_THROW_IF_NOT_MSG(
            hnsw_M > 0,
            "IndexNSGHNSWFlat: HNSW M must be > 0");

    FAISS_THROW_IF_NOT_MSG(
            nsg_R > 0,
            "IndexNSGHNSWFlat: NSG R must be > 0");

    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 ||
                    metric == METRIC_INNER_PRODUCT,
            "IndexNSGHNSWFlat supports only L2 and inner product");

    /*
     * =========================================================
     * Shared storage
     * =========================================================
     *
     * There is exactly ONE IndexFlat containing the vectors.
     */

    storage = new IndexFlat(
            d_in,
            metric);

    /*
     * =========================================================
     * HNSW
     * =========================================================
     *
     * IndexHNSW accepts an externally supplied Index* storage.
     */

    hnsw_index = new IndexHNSW(
            storage,
            hnsw_M);

    /*
     * =========================================================
     * NSG
     * =========================================================
     *
     * IndexNSG also accepts an externally supplied Index* storage.
     */

    nsg_index = new IndexNSG(
            storage,
            nsg_R);

    /*
     * =========================================================
     * Ownership
     * =========================================================
     *
     * Both children point to the same storage.
     *
     * Therefore neither child may delete storage.
     *
     * IndexNSGHNSWFlat is the ONLY owner of storage.
     */

    hnsw_index->own_fields = false;
    nsg_index->own_fields = false;

    /*
     * IndexFlat does not require training.
     */

    storage->is_trained = true;

    hnsw_index->is_trained = true;
    nsg_index->is_trained = true;

    is_trained = true;

    ntotal = 0;

    /*
     * Candidate counts.
     *
     * Zero means:
     *
     *     use k passed to search().
     */

    hnsw_k = 0;
    nsg_k = 0;
}

/***************************************************************
 * Destructor
 ***************************************************************/

IndexNSGHNSWFlat::~IndexNSGHNSWFlat() {
    /*
     * Delete graph indexes first.
     *
     * Since own_fields == false, neither graph deletes storage.
     */

    if (hnsw_index != nullptr) {
        delete hnsw_index;
        hnsw_index = nullptr;
    }

    if (nsg_index != nullptr) {
        delete nsg_index;
        nsg_index = nullptr;
    }

    /*
     * Delete shared storage exactly once.
     */

    if (storage != nullptr) {
        delete storage;
        storage = nullptr;
    }
}

/***************************************************************
 * Train
 ***************************************************************/

void IndexNSGHNSWFlat::train(
        idx_t n,
        const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage != nullptr,
            "IndexNSGHNSWFlat: storage is null");

    FAISS_THROW_IF_NOT_MSG(
            x != nullptr,
            "IndexNSGHNSWFlat::train: x is null");

    /*
     * IndexFlat train() is effectively a no-op,
     * but calling through the interface is correct.
     */

    storage->train(
            n,
            x);

    is_trained =
            storage->is_trained;

    if (hnsw_index != nullptr) {
        hnsw_index->is_trained =
                is_trained;
    }

    if (nsg_index != nullptr) {
        nsg_index->is_trained =
                is_trained;
    }
}

/***************************************************************
 * Add
 ***************************************************************/

void IndexNSGHNSWFlat::add(
        idx_t n,
        const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage != nullptr,
            "IndexNSGHNSWFlat: storage is null");

    FAISS_THROW_IF_NOT_MSG(
            hnsw_index != nullptr,
            "IndexNSGHNSWFlat: HNSW index is null");

    FAISS_THROW_IF_NOT_MSG(
            nsg_index != nullptr,
            "IndexNSGHNSWFlat: NSG index is null");

    FAISS_THROW_IF_NOT_MSG(
            is_trained,
            "IndexNSGHNSWFlat is not trained");

    FAISS_THROW_IF_NOT_MSG(
            x != nullptr,
            "IndexNSGHNSWFlat::add: x is null");

    FAISS_THROW_IF_NOT_MSG(
            n > 0,
            "IndexNSGHNSWFlat::add requires n > 0");

    /*
     * =========================================================
     * Build-once restriction
     * =========================================================
     *
     * NSG does not support incremental addition.
     */

    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0,
            "IndexNSGHNSWFlat does not support incremental add");

    FAISS_THROW_IF_NOT_MSG(
            storage->ntotal == 0,
            "IndexNSGHNSWFlat: shared storage is not empty");

    FAISS_THROW_IF_NOT_MSG(
            hnsw_index->ntotal == 0,
            "IndexNSGHNSWFlat: HNSW is not empty");

    FAISS_THROW_IF_NOT_MSG(
            nsg_index->ntotal == 0,
            "IndexNSGHNSWFlat: NSG is not empty");

    /*
     * =========================================================
     * STEP 1:
     *
     * Store vectors exactly once.
     * =========================================================
     */

    storage->add(
            n,
            x);

    FAISS_THROW_IF_NOT_MSG(
            storage->ntotal == n,
            "IndexNSGHNSWFlat: storage add failed");

    /*
     * Main index now contains n vectors.
     */

    ntotal =
            storage->ntotal;

    /*
     * =========================================================
     * STEP 2:
     *
     * Build HNSW links only.
     * =========================================================
     *
     * IMPORTANT:
     *
     * add_links_only() must NOT call:
     *
     *     storage->add(...)
     */

    hnsw_index->add_links_only(
            n,
            x);

    /*
     * Ensure metadata is synchronized.
     */

    hnsw_index->ntotal =
            ntotal;

    /*
     * =========================================================
     * STEP 3:
     *
     * Build NSG links only.
     * =========================================================
     */

    nsg_index->add_links_only(
            n,
            x);

    nsg_index->ntotal =
            ntotal;

    /*
     * =========================================================
     * Final consistency checks
     * =========================================================
     */

    FAISS_THROW_IF_NOT_MSG(
            storage->ntotal == ntotal,
            "IndexNSGHNSWFlat: storage ntotal mismatch");

    FAISS_THROW_IF_NOT_MSG(
            hnsw_index->ntotal == ntotal,
            "IndexNSGHNSWFlat: HNSW ntotal mismatch");

    FAISS_THROW_IF_NOT_MSG(
            nsg_index->ntotal == ntotal,
            "IndexNSGHNSWFlat: NSG ntotal mismatch");

    FAISS_THROW_IF_NOT_MSG(
            nsg_index->is_built,
            "IndexNSGHNSWFlat: NSG graph was not built");
}

/***************************************************************
 * Search one query
 ***************************************************************/

void IndexNSGHNSWFlat::search_one(
        const float* query,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    /*
     * Number of candidates requested from each graph.
     *
     * Never request fewer than final k.
     */

    const idx_t kh =
            hnsw_k > 0
            ? std::max(
                      hnsw_k,
                      k)
            : k;

    const idx_t kn =
            nsg_k > 0
            ? std::max(
                      nsg_k,
                      k)
            : k;

    /*
     * =========================================================
     * HNSW search
     * =========================================================
     */

    std::vector<float> hnsw_D(
            static_cast<size_t>(kh));

    std::vector<idx_t> hnsw_I(
            static_cast<size_t>(kh));

    hnsw_index->search(
            1,
            query,
            kh,
            hnsw_D.data(),
            hnsw_I.data());

    /*
     * =========================================================
     * NSG search
     * =========================================================
     */

    std::vector<float> nsg_D(
            static_cast<size_t>(kn));

    std::vector<idx_t> nsg_I(
            static_cast<size_t>(kn));

    nsg_index->search(
            1,
            query,
            kn,
            nsg_D.data(),
            nsg_I.data());

    /*
     * =========================================================
     * Union candidate IDs
     * =========================================================
     */

    std::unordered_set<idx_t> seen;

    seen.reserve(
            static_cast<size_t>(
                    kh + kn));

    std::vector<idx_t> candidate_ids;

    candidate_ids.reserve(
            static_cast<size_t>(
                    kh + kn));

    /*
     * HNSW candidates.
     */

    for (idx_t j = 0;
         j < kh;
         ++j) {
        const idx_t id =
                hnsw_I[
                        static_cast<size_t>(j)];

        /*
         * Faiss may return -1 if not enough
         * results are available.
         */

        if (id < 0 ||
            id >= ntotal) {
            continue;
        }

        if (seen.insert(id).second) {
            candidate_ids.push_back(id);
        }
    }

    /*
     * NSG candidates.
     */

    for (idx_t j = 0;
         j < kn;
         ++j) {
        const idx_t id =
                nsg_I[
                        static_cast<size_t>(j)];

        if (id < 0 ||
            id >= ntotal) {
            continue;
        }

        if (seen.insert(id).second) {
            candidate_ids.push_back(id);
        }
    }

    /*
     * =========================================================
     * Exact reranking
     * =========================================================
     *
     * Shared IndexFlat stores raw vectors contiguously.
     *
     * get_xb() returns pointer to the first vector.
     */

    const float* xb =
            storage->get_xb();

    FAISS_THROW_IF_NOT_MSG(
            xb != nullptr,
            "IndexNSGHNSWFlat: shared storage is empty");

    std::vector<PairedCandidate> candidates;

    candidates.reserve(
            candidate_ids.size());

    for (const idx_t id : candidate_ids) {
        const float* base_vector =
                xb +
                static_cast<size_t>(id) *
                        static_cast<size_t>(d);

        float distance;

        /*
         * -----------------------------------------------------
         * L2
         * -----------------------------------------------------
         */

        if (metric_type == METRIC_L2) {
            distance =
                    fvec_L2sqr(
                            query,
                            base_vector,
                            d);
        }

        /*
         * -----------------------------------------------------
         * Inner product
         * -----------------------------------------------------
         */

        else if (
                metric_type ==
                METRIC_INNER_PRODUCT) {
            distance =
                    fvec_inner_product(
                            query,
                            base_vector,
                            d);
        }

        else {
            FAISS_THROW_MSG(
                    "IndexNSGHNSWFlat: unsupported metric");
        }

        candidates.push_back(
                {
                        id,
                        distance
                });
    }

    /*
     * =========================================================
     * Sort exact candidates
     * =========================================================
     */

    if (metric_type == METRIC_L2) {
        /*
         * Smaller L2 is better.
         */

        std::sort(
                candidates.begin(),
                candidates.end(),
                [](
                        const PairedCandidate& a,
                        const PairedCandidate& b) {
                    if (a.distance !=
                        b.distance) {
                        return a.distance <
                                b.distance;
                    }

                    return a.id <
                            b.id;
                });

    } else {
        /*
         * Larger similarity is better.
         */

        std::sort(
                candidates.begin(),
                candidates.end(),
                [](
                        const PairedCandidate& a,
                        const PairedCandidate& b) {
                    if (a.distance !=
                        b.distance) {
                        return a.distance >
                                b.distance;
                    }

                    return a.id <
                            b.id;
                });
    }

    /*
     * =========================================================
     * Return top-k
     * =========================================================
     */

    const idx_t available =
            static_cast<idx_t>(
                    candidates.size());

    const idx_t result_count =
            std::min(
                    k,
                    available);

    for (idx_t j = 0;
         j < result_count;
         ++j) {
        const PairedCandidate& candidate =
                candidates[
                        static_cast<size_t>(j)];

        labels[j] =
                candidate.id;

        distances[j] =
                candidate.distance;
    }

    /*
     * Fill unused output positions.
     */

    for (idx_t j = result_count;
         j < k;
         ++j) {
        labels[j] = -1;

        if (metric_type == METRIC_L2) {
            distances[j] =
                    std::numeric_limits<
                            float>::infinity();

        } else {
            distances[j] =
                    -std::numeric_limits<
                            float>::infinity();
        }
    }
}

/***************************************************************
 * Search
 ***************************************************************/

void IndexNSGHNSWFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            params == nullptr,
            "IndexNSGHNSWFlat currently does not support "
            "custom SearchParameters");

    FAISS_THROW_IF_NOT_MSG(
            storage != nullptr,
            "IndexNSGHNSWFlat: storage is null");

    FAISS_THROW_IF_NOT_MSG(
            hnsw_index != nullptr,
            "IndexNSGHNSWFlat: HNSW index is null");

    FAISS_THROW_IF_NOT_MSG(
            nsg_index != nullptr,
            "IndexNSGHNSWFlat: NSG index is null");

    FAISS_THROW_IF_NOT_MSG(
            x != nullptr,
            "IndexNSGHNSWFlat::search: x is null");

    FAISS_THROW_IF_NOT_MSG(
            distances != nullptr,
            "IndexNSGHNSWFlat::search: distances is null");

    FAISS_THROW_IF_NOT_MSG(
            labels != nullptr,
            "IndexNSGHNSWFlat::search: labels is null");

    FAISS_THROW_IF_NOT_MSG(
            ntotal > 0,
            "IndexNSGHNSWFlat is empty");

    FAISS_THROW_IF_NOT_MSG(
            n > 0,
            "IndexNSGHNSWFlat::search requires n > 0");

    FAISS_THROW_IF_NOT_MSG(
            k > 0,
            "IndexNSGHNSWFlat::search requires k > 0");

    /*
     * =========================================================
     * Parallelize across queries.
     *
     * For each query, HNSW and NSG searches are currently
     * performed sequentially.
     *
     * This avoids nested OpenMP complications.
     * =========================================================
     */

#pragma omp parallel for
    for (idx_t i = 0;
         i < n;
         ++i) {
        const float* query =
                x +
                static_cast<size_t>(i) *
                        static_cast<size_t>(d);

        float* query_D =
                distances +
                static_cast<size_t>(i) *
                        static_cast<size_t>(k);

        idx_t* query_I =
                labels +
                static_cast<size_t>(i) *
                        static_cast<size_t>(k);

        search_one(
                query,
                k,
                query_D,
                query_I);
    }
}

/***************************************************************
 * Reconstruct
 ***************************************************************/

void IndexNSGHNSWFlat::reconstruct(
        idx_t key,
        float* recons) const {
    FAISS_THROW_IF_NOT_MSG(
            storage != nullptr,
            "IndexNSGHNSWFlat: storage is null");

    FAISS_THROW_IF_NOT_MSG(
            key >= 0 &&
                    key < ntotal,
            "IndexNSGHNSWFlat::reconstruct: invalid key");

    storage->reconstruct(
            key,
            recons);
}

/***************************************************************
 * Reset
 ***************************************************************/

void IndexNSGHNSWFlat::reset() {
    /*
     * IMPORTANT:
     *
     * Do NOT call:
     *
     *     hnsw_index->reset();
     *
     * because current IndexHNSW::reset() also executes:
     *
     *     storage->reset();
     *
     * Likewise, IndexNSG::reset() also resets its storage.
     *
     * Since both graphs share the same storage, reset the
     * graph structures manually and reset storage once.
     */

    if (hnsw_index != nullptr) {
        /*
         * Clear HNSW graph.
         */

        hnsw_index->hnsw.reset();

        /*
         * Current HNSW also maintains graph-construction locks.
         */

        hnsw_index->locks.clear();

        hnsw_index->ntotal = 0;
    }

    if (nsg_index != nullptr) {
        /*
         * Clear NSG graph.
         */

        nsg_index->nsg.reset();

        nsg_index->ntotal = 0;

        nsg_index->is_built = false;
    }

    /*
     * Reset shared raw-vector storage exactly once.
     */

    if (storage != nullptr) {
        storage->reset();
    }

    ntotal = 0;
}

} // namespace faiss