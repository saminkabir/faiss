#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexNSG.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/NSG.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#include <omp.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <pthread.h>
#include <sched.h>
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
// available_cpu_count
//
// std::thread::hardware_concurrency() reports the machine's
// TOTAL core count and does NOT shrink under a taskset/cgroup
// CPU affinity restriction (verified empirically: running under
// `taskset -c 0` on a 2-core box still reports 2, not 1). Using
// it to size internal OpenMP thread pools means that under
// `taskset -c 1-2` (2 CPUs) on, say, a 32-core machine, each of
// the 2 build threads below would try to spawn ~16 OpenMP
// workers — 32 threads fighting over 2 pinned cores instead of
// 2 threads cleanly occupying 2 cores.
//
// sched_getaffinity() reports the process's actual current CPU
// affinity mask, which DOES reflect taskset/cgroup restrictions
// (also verified: `taskset -c 0` correctly yields a mask of 1).
// This is what should drive how many threads we ask for.
// ============================================================

int available_cpu_count() {
    cpu_set_t set;
    CPU_ZERO(&set);

    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int count = CPU_COUNT(&set);
        if (count > 0) {
            return count;
        }
    }

    // Fallback (e.g. non-Linux, or the syscall failed): total
    // core count is still better than hardcoding 1.
    return std::max(
            1,
            static_cast<int>(
                    std::thread::hardware_concurrency()));
}


// ============================================================
// affinity_cpu_list / pin_thread_to_cpu
//
// affinity_cpu_list() returns the actual CPU ids this process is
// currently allowed to run on (respects taskset/cgroups — same
// mask available_cpu_count() reads, just returned as a list of
// ids instead of a count).
//
// pin_thread_to_cpu() pins a std::thread to exactly one CPU id
// via pthread_setaffinity_np (Linux). This is what gives a hard
// guarantee — independent of the OS scheduler's whims — that
// the HNSW build thread and the NSG build thread each run on
// their own dedicated core and never get scheduled onto the
// same one (when at least 2 CPUs are available).
// ============================================================

std::vector<int> affinity_cpu_list() {
    std::vector<int> cpus;

    cpu_set_t set;
    CPU_ZERO(&set);

    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
            if (CPU_ISSET(cpu, &set)) {
                cpus.push_back(cpu);
            }
        }
    }

    if (cpus.empty()) {
        // Fallback: assume CPUs 0..N-1 are usable.
        int n = std::max(
                1,
                static_cast<int>(
                        std::thread::hardware_concurrency()));
        for (int cpu = 0; cpu < n; cpu++) {
            cpus.push_back(cpu);
        }
    }

    return cpus;
}

void pin_thread_to_cpu(std::thread& t, int cpu_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);

    int rc = pthread_setaffinity_np(
            t.native_handle(),
            sizeof(set),
            &set);

    if (rc != 0) {
        fprintf(
                stderr,
                "Warning: pthread_setaffinity_np(cpu=%d) failed "
                "(rc=%d) — thread will run unpinned.\n",
                cpu_id,
                rc);
    }
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
    // Construct HNSW and NSG with SEPARATE, independent storage
    //
    // Each index gets its own IndexFlat, built and owned
    // internally by the *Flat constructors (own_fields = true
    // on both, the default). Nothing is shared here — this is
    // deliberately the opposite of the shared-storage version:
    // two independent copies of the base vectors will exist,
    // one inside hnsw.storage and one inside nsg.storage, each
    // populated by that index's own standard add() call below.
    //
    // Because neither index's add() path touches the other
    // index's storage or any global state (verified earlier
    // against the faiss source — HNSW's build-time RNGs/locks
    // and NSG's RNG are all per-instance, not shared/global),
    // it's safe for the two add() calls to run concurrently on
    // separate threads even though each is doing its own
    // storage->add() internally — they're touching different
    // storage objects.
    // ========================================================

    faiss::IndexHNSWFlat hnsw(
            d,
            HNSW_M,
            faiss::METRIC_L2);

    hnsw.hnsw.efConstruction =
            HNSW_EF_CONSTRUCTION;

    hnsw.hnsw.efSearch =
            HNSW_EF_SEARCH;

    assert(hnsw.own_fields);


    faiss::IndexNSGFlat nsg(
            d,
            NSG_R,
            faiss::METRIC_L2);

    nsg.GK =
            NSG_GK;

    nsg.build_type =
            0;

    nsg.nsg.search_L =
            NSG_SEARCH_L;

    assert(nsg.own_fields);


    // ========================================================
    // Build both graphs in parallel, each on its own storage
    // ========================================================

    std::cout
            << "\n========================================\n"
            << "Building HNSW + NSG in parallel "
            << "(separate storage)\n"
            << "========================================\n";

    double hnsw_build_seconds =
            0.0;

    double nsg_build_seconds =
            0.0;

    // Uses the process's actual CPU affinity mask (respects
    // taskset/cgroups), not std::thread::hardware_concurrency()
    // — see available_cpu_count() for why that distinction
    // matters here. Used below to restore full parallelism for
    // the (single-threaded caller) query phase.
    const int total_threads =
            available_cpu_count();

    // ------------------------------------------------------
    // Hard guarantee: exactly 2 threads do the building, one
    // dedicated to HNSW and one dedicated to NSG.
    //
    //  1. Each build thread forces omp_set_num_threads(1) for
    //     itself before calling into faiss. Faiss's internal
    //     #pragma omp parallel regions then run entirely on the
    //     calling thread instead of spawning extra OpenMP
    //     worker threads — so no matter how many CPUs the
    //     process can see, each build stays exactly 1 thread.
    //     Total thread count for the build phase is therefore
    //     always 2 (main spawns hnsw_thread + nsg_thread), never
    //     more — this is what "half the CPUs" from the previous
    //     version could NOT promise if it happened to compute
    //     more than 1 thread per build.
    //
    //  2. Each thread is additionally pinned with
    //     pthread_setaffinity_np to its own CPU id (drawn from
    //     the process's actual affinity list, so it honors
    //     taskset if present). This removes the OS scheduler
    //     from the picture: HNSW's thread and NSG's thread are
    //     each locked to a specific core for the whole build,
    //     confirmed below via sched_getcpu().
    // ------------------------------------------------------

    std::vector<int> cpus =
            affinity_cpu_list();

    int cpu_for_hnsw =
            cpus[0];

    int cpu_for_nsg =
            (cpus.size() >= 2) ? cpus[1] : cpus[0];

    std::cout
            << "Process CPU affinity list:";

    for (int c : cpus) {
        std::cout << " " << c;
    }

    std::cout << "\n";

    if (cpus.size() < 2) {
        std::cout
                << "WARNING: only 1 CPU available to this "
                << "process — HNSW and NSG threads will both "
                << "be pinned to CPU "
                << cpu_for_hnsw
                << " and will time-share it. For true 2-core "
                << "parallelism, run under e.g. "
                << "`taskset -c 0-1 ./this_binary ...`.\n";
    } else {
        std::cout
                << "HNSW thread -> CPU "
                << cpu_for_hnsw
                << ", NSG thread -> CPU "
                << cpu_for_nsg
                << "\n";
    }

    int hnsw_ran_on_cpu =
            -1;

    int nsg_ran_on_cpu =
            -1;

    auto overall_start =
            std::chrono::
                    high_resolution_clock::
                            now();

    std::thread hnsw_thread(
            [&]() {
                // Exactly 1 thread for this build, always —
                // not derived from CPU count.
                omp_set_num_threads(1);

                auto t0 =
                        std::chrono::
                                high_resolution_clock::
                                        now();

                // Standard public API: this populates hnsw's own
                // (independent) storage AND builds the graph, in
                // one call, exactly as stock faiss intends.
                hnsw.add(
                        n,
                        xb);

                auto t1 =
                        std::chrono::
                                high_resolution_clock::
                                        now();

                hnsw_build_seconds =
                        std::chrono::duration<double>(
                                t1 -
                                t0)
                                .count();

                hnsw_ran_on_cpu =
                        sched_getcpu();
            });

    pin_thread_to_cpu(
            hnsw_thread,
            cpu_for_hnsw);

    std::thread nsg_thread(
            [&]() {
                // Exactly 1 thread for this build, always —
                // not derived from CPU count.
                omp_set_num_threads(1);

                auto t0 =
                        std::chrono::
                                high_resolution_clock::
                                        now();

                // Standard public API: this populates nsg's own
                // (independent) storage AND builds the graph, in
                // one call, exactly as stock faiss intends.
                nsg.add(
                        n,
                        xb);

                auto t1 =
                        std::chrono::
                                high_resolution_clock::
                                        now();

                nsg_build_seconds =
                        std::chrono::duration<double>(
                                t1 -
                                t0)
                                .count();

                nsg_ran_on_cpu =
                        sched_getcpu();
            });

    pin_thread_to_cpu(
            nsg_thread,
            cpu_for_nsg);

    hnsw_thread.join();
    nsg_thread.join();

    std::cout
            << "HNSW build thread actually ran on CPU "
            << hnsw_ran_on_cpu
            << " (pinned to "
            << cpu_for_hnsw
            << ")\n";

    std::cout
            << "NSG  build thread actually ran on CPU "
            << nsg_ran_on_cpu
            << " (pinned to "
            << cpu_for_nsg
            << ")\n";

    auto overall_end =
            std::chrono::
                    high_resolution_clock::
                            now();

    double parallel_build_wall_seconds =
            std::chrono::duration<double>(
                    overall_end -
                    overall_start)
                    .count();

    std::cout
            << "HNSW graph build complete "
            << "(ntotal = "
            << hnsw.ntotal
            << ", "
            << hnsw_build_seconds
            << " sec on its own thread)\n";

    std::cout
            << "NSG  graph build complete "
            << "(ntotal = "
            << nsg.ntotal
            << ", "
            << nsg_build_seconds
            << " sec on its own thread)\n";

    std::cout
            << "Combined wall-clock time  = "
            << parallel_build_wall_seconds
            << " sec (vs. "
            << (hnsw_build_seconds + nsg_build_seconds)
            << " sec if run sequentially)\n";

    log_metric(
            "construction",
            "hnsw_build_seconds",
            hnsw_build_seconds);

    log_metric(
            "construction",
            "nsg_build_seconds",
            nsg_build_seconds);

    log_metric(
            "construction",
            "parallel_build_wall_seconds",
            parallel_build_wall_seconds);


    // ========================================================
    // Verify the storage really is separate (not shared)
    // ========================================================

    std::cout
            << "\n========================================\n"
            << "Verifying independent vector storage\n"
            << "========================================\n";

    std::cout
            << "hnsw.storage = "
            << static_cast<void*>(hnsw.storage)
            << "\n";

    std::cout
            << "nsg.storage  = "
            << static_cast<void*>(nsg.storage)
            << "\n";

    assert(hnsw.storage != nullptr);
    assert(nsg.storage != nullptr);
    // The whole point of this version: two distinct IndexFlat
    // objects, each owned by its respective index.
    assert(hnsw.storage != nsg.storage);
    assert(hnsw.own_fields);
    assert(nsg.own_fields);
    assert(hnsw.storage->ntotal == n);
    assert(nsg.storage->ntotal == n);
    assert(hnsw.ntotal == n);
    assert(nsg.ntotal == n);

    std::cout
            << "\nSUCCESS: HNSW and NSG were built in parallel, "
            << "each on its own separate, independent vector "
            << "storage.\n";


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

    // Both builds ran with a capped OMP thread count; restore
    // full parallelism for the (single, sequential) query phase.
    omp_set_num_threads(
            total_threads);


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
            << "HNSW storage total     = "
            << hnsw.storage->ntotal
            << "\n";

    std::cout
            << "NSG  storage total     = "
            << nsg.storage->ntotal
            << "\n";

    std::cout
            << "HNSW build time (own thread) = "
            << hnsw_build_seconds
            << " sec\n";

    std::cout
            << "NSG  build time (own thread) = "
            << nsg_build_seconds
            << " sec\n";

    std::cout
            << "Parallel build wall time     = "
            << parallel_build_wall_seconds
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
    // xb / xq are separate host buffers; hnsw and nsg each own
    // their own storage (own_fields = true on both, the default
    // for the *Flat constructors), so their destructors clean
    // up their own vector copies independently when hnsw/nsg go
    // out of scope right after this — no manual storage cleanup
    // needed here.
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
