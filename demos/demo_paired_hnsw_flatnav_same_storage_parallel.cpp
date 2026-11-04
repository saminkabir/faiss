// ============================================================
// demo_paired_hnsw_flatnav_same_storage_parallel.cpp
//
// HNSW (faiss) + FlatNav built in parallel on ONE shared
// faiss::IndexFlatL2. The vectors exist exactly once:
//
//   shared_storage (IndexFlatL2)  <- the only copy of the vectors
//     ^                    ^
//     |                    |
//   faiss::IndexHNSW     FlatNav (graph only: M links + label per node,
//   (&shared_storage)     distances read from shared_storage.codes)
//
// Node / vector id i is the same in both indexes, so results can be
// compared or paired directly.
//
// Each build runs on its own pinned CPU with exactly one thread.
// FlatNav and faiss are both compiled for AVX2 (no AVX-512); the
// SIMD configuration is printed and logged at startup.
//
// Usage:
//   demo_paired_hnsw_flatnav_same_storage_parallel
//       base.fvecs query.fvecs [gt.ivecs|-] [log.csv]
//       [HNSW_M=32] [HNSW_efConstruction=100] [HNSW_efSearch=64]
//       [FLATNAV_M=32] [FLATNAV_efConstruction=100] [FLATNAV_efSearch=64]
// ============================================================

#include "paired_flatnav_common.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " base.fvecs query.fvecs [gt.ivecs|-] [log.csv]"
                  << " [HNSW_M] [HNSW_efC] [HNSW_efS]"
                  << " [FLATNAV_M] [FLATNAV_efC] [FLATNAV_efS]\n";
        return 1;
    }
    const char* base_file = argv[1];
    const char* query_file = argv[2];
    const char* gt_file = (argc >= 4) ? argv[3] : nullptr;
    const std::string log_path = (argc >= 5) ? argv[4] : "benchmark_log.csv";
    log_init(log_path);

    // ========================================================
    // Parameters
    // ========================================================
    const int HNSW_M = (argc >= 6) ? stoi(argv[5]) : 32;
    const int HNSW_EF_CONSTRUCTION = (argc >= 7) ? stoi(argv[6]) : 100;
    const int HNSW_EF_SEARCH = (argc >= 8) ? stoi(argv[7]) : 64;
    const int FLATNAV_M = (argc >= 9) ? stoi(argv[8]) : 32;
    const int FLATNAV_EF_CONSTRUCTION = (argc >= 10) ? stoi(argv[9]) : 100;
    const int FLATNAV_EF_SEARCH = (argc >= 11) ? stoi(argv[10]) : 64;
    const faiss::idx_t K = 20;
    const size_t RECALL_EVAL_K = 20;

    std::cout << "\n========================================\n"
              << "Paired HNSW + FlatNav Demo (same storage)\n"
              << "========================================\n"
              << "HNSW M                 = " << HNSW_M << "\n"
              << "HNSW efConstruction    = " << HNSW_EF_CONSTRUCTION << "\n"
              << "HNSW efSearch          = " << HNSW_EF_SEARCH << "\n"
              << "FlatNav M              = " << FLATNAV_M << "\n"
              << "FlatNav efConstruction = " << FLATNAV_EF_CONSTRUCTION << "\n"
              << "FlatNav efSearch       = " << FLATNAV_EF_SEARCH << "\n";
    log_metric("params", "hnsw_M", HNSW_M);
    log_metric("params", "hnsw_ef_construction", HNSW_EF_CONSTRUCTION);
    log_metric("params", "hnsw_ef_search", HNSW_EF_SEARCH);
    log_metric("params", "flatnav_M", FLATNAV_M);
    log_metric("params", "flatnav_ef_construction", FLATNAV_EF_CONSTRUCTION);
    log_metric("params", "flatnav_ef_search", FLATNAV_EF_SEARCH);

    report_simd_config();

    PairDataset ds;
    if (!load_pair_dataset(base_file, query_file, gt_file, RECALL_EVAL_K, ds)) {
        return 1;
    }
    const int d = ds.d;
    const faiss::idx_t n = ds.n;

    // ========================================================
    // Populate ONE shared storage, once
    // ========================================================
    std::cout << "\n========================================\n"
              << "Populating shared storage\n"
              << "========================================\n";
    faiss::IndexFlatL2 shared_storage(d);
    shared_storage.add(n, ds.xb);
    std::cout << "shared_storage.ntotal = " << shared_storage.ntotal << "\n";

    // ========================================================
    // Construct both indexes directly on the shared storage
    // ========================================================
    // Index*-taking ctor: own_fields = false, HNSW never adds to or
    // frees shared_storage.
    faiss::IndexHNSW hnsw(&shared_storage, HNSW_M);
    hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION;
    hnsw.hnsw.efSearch = HNSW_EF_SEARCH;
    PAIR_CHECK(!hnsw.own_fields);

    // FlatNav reads shared_storage.codes directly (no copy).
    auto flatnav_index = flatnav_faiss::make_flatnav_l2(shared_storage, FLATNAV_M);
    PAIR_CHECK(flatnav_index->usesExternalStorage());

    // ========================================================
    // Build both graphs in parallel, one pinned thread each
    // ========================================================
    std::cout << "\n========================================\n"
              << "Building HNSW + FlatNav in parallel on shared storage\n"
              << "========================================\n";
    const int total_threads = available_cpu_count();
    const std::pair<int, int> build_cpus = pick_two_cpus();
    const int cpu_for_hnsw = build_cpus.first;
    const int cpu_for_flatnav = build_cpus.second;
    std::cout << "HNSW thread -> CPU " << cpu_for_hnsw << ", FlatNav thread -> CPU "
              << cpu_for_flatnav << "\n";

    PinnedBuild hnsw_build, flatnav_build;
    double parallel_build_wall_seconds = time_seconds([&]() {
        std::thread hnsw_thread = start_pinned_build(cpu_for_hnsw, hnsw_build, [&]() {
            build_hnsw_graph_on_shared_storage(
                    hnsw, static_cast<size_t>(n), ds.xb, /*verbose=*/false);
        });
        std::thread flatnav_thread = start_pinned_build(cpu_for_flatnav, flatnav_build, [&]() {
            flatnav_faiss::build(*flatnav_index, FLATNAV_EF_CONSTRUCTION, /*num_threads=*/1);
        });
        hnsw_thread.join();
        flatnav_thread.join();
    });

    std::cout << "HNSW    build: " << hnsw_build.seconds << " sec on CPU " << hnsw_build.ran_on_cpu
              << " (pinned to " << hnsw_build.requested_cpu << "), ntotal = " << hnsw.ntotal << "\n"
              << "FlatNav build: " << flatnav_build.seconds << " sec on CPU "
              << flatnav_build.ran_on_cpu << " (pinned to " << flatnav_build.requested_cpu
              << "), nodes = " << flatnav_index->currentNumNodes() << "\n"
              << "Combined wall-clock time = " << parallel_build_wall_seconds << " sec (vs. "
              << (hnsw_build.seconds + flatnav_build.seconds) << " sec if run sequentially)\n";
    log_metric("construction", "hnsw_build_seconds", hnsw_build.seconds);
    log_metric("construction", "flatnav_build_seconds", flatnav_build.seconds);
    log_metric("construction", "parallel_build_wall_seconds", parallel_build_wall_seconds);

    // ========================================================
    // Verify the storage really was shared
    // ========================================================
    std::cout << "\n========================================\n"
              << "Verifying shared storage\n"
              << "========================================\n"
              << "shared_storage address      = " << static_cast<void*>(&shared_storage) << "\n"
              << "hnsw.storage                = " << static_cast<void*>(hnsw.storage) << "\n"
              << "shared_storage.get_xb()     = "
              << static_cast<const void*>(shared_storage.get_xb()) << "\n"
              << "FlatNav vector pointer      = " << flatnav_index->externalStorageData() << "\n";
    PAIR_CHECK(hnsw.storage == &shared_storage);
    PAIR_CHECK(!hnsw.own_fields);
    PAIR_CHECK(flatnav_index->externalStorageData() ==
               static_cast<const void*>(shared_storage.get_xb()));
    PAIR_CHECK(shared_storage.ntotal == n);
    PAIR_CHECK(hnsw.ntotal == n);
    PAIR_CHECK(flatnav_index->currentNumNodes() == static_cast<size_t>(n));
    std::cout << "\nSUCCESS: HNSW and FlatNav were built in parallel directly on one "
              << "shared storage instance — no vectors were ever duplicated.\n";

    // ========================================================
    // Index sizes
    // ========================================================
    const size_t storage_bytes = flat_storage_bytes_direct(shared_storage);
    const size_t hnsw_graph = hnsw_graph_bytes_direct(hnsw.hnsw);
    const size_t flatnav_graph = flatnav_faiss::graph_overhead_bytes(*flatnav_index);
    const size_t combined = storage_bytes + hnsw_graph + flatnav_graph;
    const size_t unshared = 2 * storage_bytes + hnsw_graph + flatnav_graph;
    std::cout << "\n========================================\n"
              << "Index size breakdown\n"
              << "========================================\n"
              << "  Shared vector storage   : " << to_mb(storage_bytes) << " MB\n"
              << "  HNSW graph overhead     : " << to_mb(hnsw_graph) << " MB\n"
              << "  FlatNav graph overhead  : " << to_mb(flatnav_graph) << " MB\n"
              << "  Combined (storage x1)   : " << to_mb(combined) << " MB\n"
              << "  If not shared           : " << to_mb(unshared) << " MB (saved "
              << to_mb(unshared - combined) << " MB)\n"
              << "  Current RSS             : " << to_mb(get_current_rss_bytes()) << " MB\n";
    log_metric("index_size", "storage_bytes", static_cast<double>(storage_bytes));
    log_metric("index_size", "hnsw_graph_overhead_bytes", static_cast<double>(hnsw_graph));
    log_metric("index_size", "flatnav_graph_overhead_bytes", static_cast<double>(flatnav_graph));
    log_metric("index_size", "combined_bytes", static_cast<double>(combined));

    // ========================================================
    // Query phase (full parallelism, same thread count for both)
    // ========================================================
    std::cout << "\n========================================\n"
              << "Query phase (nq = " << ds.nq << ", k = " << K << ", threads = "
              << total_threads << ")\n"
              << "========================================\n";
    omp_set_num_threads(total_threads);
    const size_t nk = static_cast<size_t>(ds.nq) * static_cast<size_t>(K);

    std::vector<float> hnsw_distances(nk);
    std::vector<faiss::idx_t> hnsw_labels(nk);
    double hnsw_seconds = time_seconds([&]() {
        hnsw.search(ds.nq, ds.xq, K, hnsw_distances.data(), hnsw_labels.data());
    });
    QueryReport hnsw_q = report_query("hnsw", hnsw_seconds, hnsw_labels, K, ds, RECALL_EVAL_K);

    std::vector<float> flatnav_distances(nk);
    std::vector<faiss::idx_t> flatnav_labels(nk);
    double flatnav_seconds = time_seconds([&]() {
        flatnav_faiss::search(
                *flatnav_index,
                ds.nq,
                ds.xq,
                static_cast<int>(K),
                FLATNAV_EF_SEARCH,
                flatnav_distances.data(),
                flatnav_labels.data(),
                /*num_threads=*/total_threads);
    });
    QueryReport flatnav_q =
            report_query("flatnav", flatnav_seconds, flatnav_labels, K, ds, RECALL_EVAL_K);

    // ========================================================
    // Summary
    // ========================================================
    std::cout << "\n========================================\n"
              << "Summary (same storage)\n"
              << "========================================\n"
              << "HNSW    build = " << hnsw_build.seconds << " s, QPS = " << hnsw_q.qps
              << ", recall@" << RECALL_EVAL_K << " = " << hnsw_q.recall << "\n"
              << "FlatNav build = " << flatnav_build.seconds << " s, QPS = " << flatnav_q.qps
              << ", recall@" << RECALL_EVAL_K << " = " << flatnav_q.recall << "\n"
              << "Parallel build wall time = " << parallel_build_wall_seconds << " s\n"
              << "Total memory (storage x1) = " << to_mb(combined) << " MB\n";

    if (g_log_file.is_open()) {
        g_log_file.close();
    }
    std::cout << "\nDone\n";
    return 0;
}
