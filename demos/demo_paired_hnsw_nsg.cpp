#include <faiss/IndexHNSW.h>
#include <faiss/IndexNSG.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <sys/stat.h>


// ============================================================
// Simple timing log
//
// Appends one CSV row per metric to a log file, in addition
// to whatever gets printed to stdout. Flushed after every
// write so the log is intact even if the run is interrupted
// partway through (e.g. during a long NSG build on a big
// dataset).
//
// Columns: run_timestamp, phase, metric, value
// ============================================================

std::ofstream g_log_file;
std::string g_run_timestamp;

void log_init(const std::string& path) {

    g_log_file.open(
            path,
            std::ios::out |
                    std::ios::app);

    if (!g_log_file.is_open()) {

        fprintf(
                stderr,
                "Warning: could not open log file '%s' — "
                "continuing without file logging.\n",
                path.c_str());

        return;
    }

    std::time_t now =
            std::time(nullptr);

    char buf[32];

    std::strftime(
            buf,
            sizeof(buf),
            "%Y-%m-%dT%H:%M:%S",
            std::localtime(&now));

    g_run_timestamp =
            std::string(buf);

    // Write header only if the file is new/empty.
    g_log_file.seekp(
            0,
            std::ios::end);

    if (g_log_file.tellp() == 0) {
        g_log_file
                << "run_timestamp,phase,metric,value\n";
    }

    g_log_file.flush();

    std::cout
            << "Logging timings to: "
            << path
            << "\n";
}

void log_metric(
        const std::string& phase,
        const std::string& metric,
        double value) {

    // Always print to console.
    std::cout
            << "[LOG] "
            << phase
            << " | "
            << metric
            << " = "
            << value
            << "\n";

    if (!g_log_file.is_open()) {
        return;
    }

    g_log_file
            << g_run_timestamp
            << ","
            << phase
            << ","
            << metric
            << ","
            << value
            << "\n";

    g_log_file.flush();
}


// ============================================================
// Read .fvecs file
//
// Format:
//
// int32 dimension
// float[d] vector
// int32 dimension
// float[d] vector
// ...
//
// Returns a contiguous float array:
//
//     [n][d]
//
// Caller owns returned memory.
// ============================================================

float* fvecs_read(
        const char* fname,
        size_t* d_out,
        size_t* n_out) {

    FILE* f = fopen(fname, "rb");

    if (!f) {
        fprintf(
                stderr,
                "Could not open file: %s\n",
                fname);

        perror("");

        std::abort();
    }

    int d = 0;

    size_t read_count =
            fread(
                    &d,
                    sizeof(int),
                    1,
                    f);

    if (read_count != 1) {
        fprintf(
                stderr,
                "Could not read dimension from %s\n",
                fname);

        fclose(f);
        std::abort();
    }

    assert(
            d > 0 &&
            d < 1000000);

    // --------------------------------------------------------
    // Determine file size
    // --------------------------------------------------------

    fseek(
            f,
            0,
            SEEK_END);

    long file_size =
            ftell(f);

    fseek(
            f,
            0,
            SEEK_SET);

    const size_t vector_size_bytes =
            sizeof(int) +
            static_cast<size_t>(d) *
                    sizeof(float);

    if (
            file_size <= 0 ||
            static_cast<size_t>(file_size) %
                            vector_size_bytes !=
                    0) {

        fprintf(
                stderr,
                "Invalid fvecs file size for %s\n",
                fname);

        fclose(f);
        std::abort();
    }

    size_t n =
            static_cast<size_t>(file_size) /
            vector_size_bytes;

    printf(
            "Reading %s\n",
            fname);

    printf(
            "  vectors   = %zu\n"
            "  dimension = %d\n",
            n,
            d);

    // --------------------------------------------------------
    // Allocate output
    // --------------------------------------------------------

    float* data =
            new float[
                    n *
                    static_cast<size_t>(d)];

    // --------------------------------------------------------
    // Read vectors
    // --------------------------------------------------------

    for (size_t i = 0; i < n; ++i) {

        int current_d = 0;

        if (
                fread(
                        &current_d,
                        sizeof(int),
                        1,
                        f) !=
                1) {

            fprintf(
                    stderr,
                    "Failed reading dimension at vector %zu\n",
                    i);

            delete[] data;
            fclose(f);
            std::abort();
        }

        if (current_d != d) {

            fprintf(
                    stderr,
                    "Dimension mismatch at vector %zu: "
                    "expected %d but found %d\n",
                    i,
                    d,
                    current_d);

            delete[] data;
            fclose(f);
            std::abort();
        }

        if (
                fread(
                        data +
                                i *
                                        static_cast<size_t>(d),
                        sizeof(float),
                        d,
                        f) !=
                static_cast<size_t>(d)) {

            fprintf(
                    stderr,
                    "Failed reading vector %zu\n",
                    i);

            delete[] data;
            fclose(f);
            std::abort();
        }
    }

    fclose(f);

    *d_out =
            static_cast<size_t>(d);

    *n_out =
            n;

    return data;
}


// ============================================================
// Read .ivecs file (ground truth neighbor ids)
//
// Same on-disk layout as .fvecs, but the payload per row is
// int32 instead of float32:
//
// int32 k
// int32[k] neighbor ids
// int32 k
// int32[k] neighbor ids
// ...
//
// Returns a contiguous int64 (idx_t) array:
//
//     [nq][k]
//
// Caller owns returned memory.
// ============================================================

faiss::idx_t* ivecs_read(
        const char* fname,
        size_t* k_out,
        size_t* nq_out) {

    FILE* f = fopen(fname, "rb");

    if (!f) {
        fprintf(
                stderr,
                "Could not open file: %s\n",
                fname);

        perror("");

        std::abort();
    }

    int k = 0;

    if (
            fread(
                    &k,
                    sizeof(int),
                    1,
                    f) !=
            1) {

        fprintf(
                stderr,
                "Could not read k from %s\n",
                fname);

        fclose(f);
        std::abort();
    }

    assert(
            k > 0 &&
            k < 1000000);

    fseek(
            f,
            0,
            SEEK_END);

    long file_size =
            ftell(f);

    fseek(
            f,
            0,
            SEEK_SET);

    const size_t row_size_bytes =
            sizeof(int) +
            static_cast<size_t>(k) *
                    sizeof(int);

    if (
            file_size <= 0 ||
            static_cast<size_t>(file_size) %
                            row_size_bytes !=
                    0) {

        fprintf(
                stderr,
                "Invalid ivecs file size for %s\n",
                fname);

        fclose(f);
        std::abort();
    }

    size_t nq =
            static_cast<size_t>(file_size) /
            row_size_bytes;

    printf(
            "Reading %s\n",
            fname);

    printf(
            "  queries with GT = %zu\n"
            "  k (GT depth)    = %d\n",
            nq,
            k);

    faiss::idx_t* data =
            new faiss::idx_t[
                    nq *
                    static_cast<size_t>(k)];

    std::vector<int> row_buf(
            static_cast<size_t>(k));

    for (size_t i = 0; i < nq; ++i) {

        int current_k = 0;

        if (
                fread(
                        &current_k,
                        sizeof(int),
                        1,
                        f) !=
                1) {

            fprintf(
                    stderr,
                    "Failed reading k at row %zu\n",
                    i);

            delete[] data;
            fclose(f);
            std::abort();
        }

        if (current_k != k) {

            fprintf(
                    stderr,
                    "k mismatch at row %zu: "
                    "expected %d but found %d\n",
                    i,
                    k,
                    current_k);

            delete[] data;
            fclose(f);
            std::abort();
        }

        if (
                fread(
                        row_buf.data(),
                        sizeof(int),
                        k,
                        f) !=
                static_cast<size_t>(k)) {

            fprintf(
                    stderr,
                    "Failed reading row %zu\n",
                    i);

            delete[] data;
            fclose(f);
            std::abort();
        }

        for (int j = 0; j < k; ++j) {
            data[i * static_cast<size_t>(k) + j] =
                    static_cast<faiss::idx_t>(
                            row_buf[j]);
        }
    }

    fclose(f);

    *k_out =
            static_cast<size_t>(k);

    *nq_out =
            nq;

    return data;
}


// ============================================================
// Recall@k helper
//
// For each query, count how many of the top-`eval_k` returned
// ids appear in the ground truth's top-`eval_k` ids.
//
// Returns fraction in [0, 1].
// ============================================================

double compute_recall_at_k(
        const std::vector<faiss::idx_t>& result_labels,
        faiss::idx_t search_k,
        const faiss::idx_t* gt,
        size_t gt_k,
        size_t nq,
        size_t eval_k) {

    assert(eval_k <= static_cast<size_t>(search_k));
    assert(eval_k <= gt_k);

    size_t total_correct = 0;
    size_t total_expected = 0;

    for (size_t q = 0; q < nq; ++q) {

        std::unordered_set<faiss::idx_t> truth(
                gt + q * gt_k,
                gt + q * gt_k + eval_k);

        for (size_t j = 0; j < eval_k; ++j) {

            faiss::idx_t got =
                    result_labels[
                            q * static_cast<size_t>(search_k) +
                            j];

            if (truth.count(got)) {
                ++total_correct;
            }
        }

        total_expected += eval_k;
    }

    return static_cast<double>(total_correct) /
            static_cast<double>(total_expected);
}


// ============================================================
// Main
//
// Usage:
//
// ./demo_paired_hnsw_nsg \
//     /path/to/base.fvecs \
//     /path/to/query.fvecs \
//     [/path/to/groundtruth.ivecs] \
//     [/path/to/log.csv]
//
// Example (SIFT1M):
//
// ./build/demos/demo_paired_hnsw_nsg \
//     /home/cc/datasets/sift/base.fvecs \
//     /home/cc/datasets/sift/query.fvecs \
//     /home/cc/datasets/sift/groundtruth.ivecs \
//     /home/cc/results/sift_paired_run.csv
// ============================================================

int main(
        int argc,
        char** argv) {

    if (argc < 3) {

        std::cerr
                << "Usage:\n"
                << "  "
                << argv[0]
                << " /path/to/base.fvecs"
                << " /path/to/query.fvecs"
                << " [/path/to/groundtruth.ivecs]"
                << " [/path/to/log.csv]\n";

        return 1;
    }

    const char* base_file =
            argv[1];

    const char* query_file =
            argv[2];

    const char* gt_file =
            (argc >= 4) ? argv[3] : nullptr;

    const std::string log_path =
            (argc >= 5) ? argv[4] : "benchmark_log.csv";

    log_init(log_path);


    // ========================================================
    // Parameters
    // ========================================================

    const int HNSW_M =
            32;

    const int HNSW_EF_CONSTRUCTION =
            100;

    const int HNSW_EF_SEARCH =
            64;

    const int NSG_R =
            32;

    const int NSG_GK =
            64;

    const int NSG_SEARCH_L =
            64;

    const faiss::idx_t K =
            10;

    const size_t RECALL_EVAL_K =
            10;


    std::cout
            << "\n========================================\n"
            << "Paired HNSW + NSG Demo\n"
            << "========================================\n";

    std::cout
            << "HNSW M              = "
            << HNSW_M
            << "\n";

    std::cout
            << "HNSW efConstruction = "
            << HNSW_EF_CONSTRUCTION
            << "\n";

    std::cout
            << "HNSW efSearch       = "
            << HNSW_EF_SEARCH
            << "\n";

    std::cout
            << "NSG R                = "
            << NSG_R
            << "\n";

    std::cout
            << "NSG GK               = "
            << NSG_GK
            << "\n";

    std::cout
            << "NSG search_L         = "
            << NSG_SEARCH_L
            << "\n\n";


    // ========================================================
    // Load base dataset
    // ========================================================

    size_t d_size =
            0;

    size_t nb =
            0;

    float* xb =
            fvecs_read(
                    base_file,
                    &d_size,
                    &nb);

    const int d =
            static_cast<int>(
                    d_size);

    const faiss::idx_t n =
            static_cast<faiss::idx_t>(
                    nb);


    std::cout
            << "\nBase dataset loaded\n";

    std::cout
            << "d      = "
            << d
            << "\n";

    std::cout
            << "nb     = "
            << nb
            << "\n";


    // ========================================================
    // Load query dataset
    // ========================================================

    size_t qd_size =
            0;

    size_t nq_size =
            0;

    float* xq =
            fvecs_read(
                    query_file,
                    &qd_size,
                    &nq_size);

    if (
            static_cast<int>(qd_size) !=
            d) {

        fprintf(
                stderr,
                "Query dimension (%zu) does not match "
                "base dimension (%d)\n",
                qd_size,
                d);

        delete[] xb;
        delete[] xq;

        return 1;
    }

    const faiss::idx_t nq =
            static_cast<faiss::idx_t>(
                    nq_size);

    std::cout
            << "\nQuery dataset loaded\n";

    std::cout
            << "nq     = "
            << nq
            << "\n";


    // ========================================================
    // Load ground truth (optional)
    // ========================================================

    faiss::idx_t* gt =
            nullptr;

    size_t gt_k =
            0;

    size_t gt_nq =
            0;

    if (gt_file != nullptr) {

        gt =
                ivecs_read(
                        gt_file,
                        &gt_k,
                        &gt_nq);

        if (gt_nq != nq_size) {

            fprintf(
                    stderr,
                    "Ground truth query count (%zu) does not "
                    "match query file count (%zu)\n",
                    gt_nq,
                    nq_size);

            delete[] xb;
            delete[] xq;
            delete[] gt;

            return 1;
        }

        if (gt_k < RECALL_EVAL_K) {

            fprintf(
                    stderr,
                    "Ground truth depth (%zu) is smaller than "
                    "RECALL_EVAL_K (%zu)\n",
                    gt_k,
                    RECALL_EVAL_K);

            delete[] xb;
            delete[] xq;
            delete[] gt;

            return 1;
        }
    } else {

        std::cout
                << "\nNo ground truth file provided — "
                << "recall will not be computed.\n";
    }


    // ========================================================
    // Build HNSW
    // ========================================================

    std::cout
            << "\n========================================\n"
            << "Building HNSW\n"
            << "========================================\n";


    faiss::IndexHNSWFlat hnsw(
            d,
            HNSW_M,
            faiss::METRIC_L2);


    hnsw.hnsw.efConstruction =
            HNSW_EF_CONSTRUCTION;

    hnsw.hnsw.efSearch =
            HNSW_EF_SEARCH;


    auto hnsw_start =
            std::chrono::
                    high_resolution_clock::
                            now();


    hnsw.add(
            n,
            xb);


    auto hnsw_end =
            std::chrono::
                    high_resolution_clock::
                            now();


    double hnsw_build_seconds =
            std::chrono::duration<double>(
                    hnsw_end -
                    hnsw_start)
                    .count();


    std::cout
            << "HNSW construction complete\n";

    std::cout
            << "HNSW ntotal = "
            << hnsw.ntotal
            << "\n";

    log_metric(
            "construction",
            "hnsw_build_seconds",
            hnsw_build_seconds);


    // ========================================================
    // Build NSG
    // ========================================================

    std::cout
            << "\n========================================\n"
            << "Building NSG\n"
            << "========================================\n";


    faiss::IndexNSGFlat nsg(
            d,
            NSG_R,
            faiss::METRIC_L2);


    // Candidate kNN graph size used while constructing NSG.
    nsg.GK =
            NSG_GK;

    // Faiss IndexNSG:
    //
    // build_type = 0:
    //     brute-force kNN graph
    //
    // build_type = 1:
    //     NNDescent-based construction
    //
    // For very large datasets you probably want 1.
    nsg.build_type =
            0;


    auto nsg_start =
            std::chrono::
                    high_resolution_clock::
                            now();


    nsg.add(
            n,
            xb);


    auto nsg_end =
            std::chrono::
                    high_resolution_clock::
                            now();


    double nsg_build_seconds =
            std::chrono::duration<double>(
                    nsg_end -
                    nsg_start)
                    .count();


    // search_L is NSG's search-time candidate-list depth —
    // the direct analog of HNSW's efSearch. Set it up front
    // so both indexes are configured before the query phase.
    nsg.nsg.search_L =
            NSG_SEARCH_L;


    std::cout
            << "NSG construction complete\n";

    std::cout
            << "NSG ntotal = "
            << nsg.ntotal
            << "\n";

    log_metric(
            "construction",
            "nsg_build_seconds",
            nsg_build_seconds);


    // ========================================================
    // Share vector storage: NSG borrows HNSW's IndexFlat
    // ========================================================

    std::cout
            << "\n========================================\n"
            << "Sharing vector storage\n"
            << "========================================\n";


    std::cout
            << "HNSW storage pointer (before) = "
            << static_cast<void*>(
                       hnsw.storage)
            << "\n";

    std::cout
            << "NSG storage pointer  (before) = "
            << static_cast<void*>(
                       nsg.storage)
            << "\n";


    assert(
            hnsw.storage !=
            nullptr);

    assert(
            nsg.storage !=
            nullptr);

    assert(
            hnsw.storage !=
            nsg.storage);


    if (
            nsg.own_fields &&
            nsg.storage !=
                    nullptr) {

        delete nsg.storage;
    }


    nsg.storage =
            hnsw.storage;

    // IMPORTANT: HNSW owns this object now; NSG must not
    // delete it in its own destructor.
    nsg.own_fields =
            false;


    std::cout
            << "HNSW storage pointer (after)  = "
            << static_cast<void*>(
                       hnsw.storage)
            << "\n";

    std::cout
            << "NSG storage pointer  (after)  = "
            << static_cast<void*>(
                       nsg.storage)
            << "\n";


    assert(
            hnsw.storage ==
            nsg.storage);

    assert(
            hnsw.storage->ntotal ==
            n);

    assert(
            nsg.storage->ntotal ==
            n);


    std::cout
            << "\nSUCCESS: HNSW and NSG "
            << "share the same vector storage.\n";


    // ========================================================
    // Query phase
    //
    // Batch-search both indexes over the full query set,
    // measure wall-clock time / QPS, and (if ground truth was
    // provided) compute recall@RECALL_EVAL_K.
    // ========================================================

    std::cout
            << "\n========================================\n"
            << "Query phase\n"
            << "========================================\n";

    std::cout
            << "nq = "
            << nq
            << ", k = "
            << K
            << "\n";


    std::vector<float>
            hnsw_distances(
                    static_cast<size_t>(nq) *
                    static_cast<size_t>(K));

    std::vector<faiss::idx_t>
            hnsw_labels(
                    static_cast<size_t>(nq) *
                    static_cast<size_t>(K));

    std::vector<float>
            nsg_distances(
                    static_cast<size_t>(nq) *
                    static_cast<size_t>(K));

    std::vector<faiss::idx_t>
            nsg_labels(
                    static_cast<size_t>(nq) *
                    static_cast<size_t>(K));


    // --------------------------------------------------------
    // HNSW batch search
    // --------------------------------------------------------

    std::cout
            << "\nRunning HNSW batch search "
            << "(efSearch = "
            << hnsw.hnsw.efSearch
            << ")...\n";

    auto hnsw_search_start =
            std::chrono::
                    high_resolution_clock::
                            now();

    hnsw.search(
            nq,
            xq,
            K,
            hnsw_distances.data(),
            hnsw_labels.data());

    auto hnsw_search_end =
            std::chrono::
                    high_resolution_clock::
                            now();

    double hnsw_search_seconds =
            std::chrono::duration<double>(
                    hnsw_search_end -
                    hnsw_search_start)
                    .count();

    double hnsw_qps =
            static_cast<double>(nq) /
            hnsw_search_seconds;

    double hnsw_ms_per_query =
            1000.0 *
            hnsw_search_seconds /
            static_cast<double>(nq);


    std::cout
            << "HNSW search time  = "
            << hnsw_search_seconds
            << " sec total, "
            << hnsw_ms_per_query
            << " ms/query, "
            << hnsw_qps
            << " QPS\n";

    log_metric(
            "query",
            "hnsw_search_seconds_total",
            hnsw_search_seconds);

    log_metric(
            "query",
            "hnsw_ms_per_query",
            hnsw_ms_per_query);

    log_metric(
            "query",
            "hnsw_qps",
            hnsw_qps);


    // --------------------------------------------------------
    // NSG batch search
    // --------------------------------------------------------

    std::cout
            << "\nRunning NSG batch search "
            << "(search_L = "
            << nsg.nsg.search_L
            << ")...\n";

    auto nsg_search_start =
            std::chrono::
                    high_resolution_clock::
                            now();

    nsg.search(
            nq,
            xq,
            K,
            nsg_distances.data(),
            nsg_labels.data());

    auto nsg_search_end =
            std::chrono::
                    high_resolution_clock::
                            now();

    double nsg_search_seconds =
            std::chrono::duration<double>(
                    nsg_search_end -
                    nsg_search_start)
                    .count();

    double nsg_qps =
            static_cast<double>(nq) /
            nsg_search_seconds;

    double nsg_ms_per_query =
            1000.0 *
            nsg_search_seconds /
            static_cast<double>(nq);


    std::cout
            << "NSG search time   = "
            << nsg_search_seconds
            << " sec total, "
            << nsg_ms_per_query
            << " ms/query, "
            << nsg_qps
            << " QPS\n";

    log_metric(
            "query",
            "nsg_search_seconds_total",
            nsg_search_seconds);

    log_metric(
            "query",
            "nsg_ms_per_query",
            nsg_ms_per_query);

    log_metric(
            "query",
            "nsg_qps",
            nsg_qps);


    // --------------------------------------------------------
    // Recall (if ground truth was provided)
    // --------------------------------------------------------

    if (gt != nullptr) {

        double hnsw_recall =
                compute_recall_at_k(
                        hnsw_labels,
                        K,
                        gt,
                        gt_k,
                        static_cast<size_t>(nq),
                        RECALL_EVAL_K);

        double nsg_recall =
                compute_recall_at_k(
                        nsg_labels,
                        K,
                        gt,
                        gt_k,
                        static_cast<size_t>(nq),
                        RECALL_EVAL_K);

        std::cout
                << "\n----------------------------------------\n"
                << "Recall@"
                << RECALL_EVAL_K
                << "\n"
                << "----------------------------------------\n";

        std::cout
                << "HNSW recall@"
                << RECALL_EVAL_K
                << " = "
                << hnsw_recall
                << "\n";

        std::cout
                << "NSG  recall@"
                << RECALL_EVAL_K
                << " = "
                << nsg_recall
                << "\n";

        log_metric(
                "query",
                "hnsw_recall_at_" +
                        std::to_string(RECALL_EVAL_K),
                hnsw_recall);

        log_metric(
                "query",
                "nsg_recall_at_" +
                        std::to_string(RECALL_EVAL_K),
                nsg_recall);
    }


    // --------------------------------------------------------
    // Show first query's neighbors for a quick sanity look
    // --------------------------------------------------------

    std::cout
            << "\nHNSW neighbors for query 0:\n";

    for (
            faiss::idx_t i = 0;
            i < K;
            ++i) {

        std::cout
                << "  "
                << i
                << ": id="
                << hnsw_labels[i]
                << " distance="
                << hnsw_distances[i]
                << "\n";
    }

    std::cout
            << "\nNSG neighbors for query 0:\n";

    for (
            faiss::idx_t i = 0;
            i < K;
            ++i) {

        std::cout
                << "  "
                << i
                << ": id="
                << nsg_labels[i]
                << " distance="
                << nsg_distances[i]
                << "\n";
    }


    // ========================================================
    // Final summary
    // ========================================================

    std::cout
            << "\n========================================\n"
            << "Summary\n"
            << "========================================\n";

    std::cout
            << "Vectors               = "
            << n
            << "\n";

    std::cout
            << "Queries                = "
            << nq
            << "\n";

    std::cout
            << "Dimension              = "
            << d
            << "\n";

    std::cout
            << "Shared storage total   = "
            << hnsw.storage->ntotal
            << "\n";

    std::cout
            << "HNSW build time        = "
            << hnsw_build_seconds
            << " sec\n";

    std::cout
            << "NSG  build time        = "
            << nsg_build_seconds
            << " sec\n";

    std::cout
            << "HNSW QPS                = "
            << hnsw_qps
            << "\n";

    std::cout
            << "NSG  QPS                = "
            << nsg_qps
            << "\n";


    // ========================================================
    // Cleanup
    //
    // xb / xq are separate host buffers; Faiss keeps its own
    // copy inside the shared IndexFlat, so these can be freed
    // freely once both build and query phases are done.
    // ========================================================

    delete[] xb;
    delete[] xq;

    if (gt != nullptr) {
        delete[] gt;
    }

    if (g_log_file.is_open()) {
        g_log_file.close();
    }


    std::cout
            << "\n========================================\n"
            << "Done\n"
            << "========================================\n";


    return 0;
}