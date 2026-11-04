// ============================================================
// paired_flatnav_common.hpp
//
// Shared code for the four HNSW/NSG + FlatNav pair demos:
//
//   demo_paired_hnsw_flatnav_same_storage_parallel.cpp
//   demo_paired_hnsw_flatnav_separate_storage_parallel.cpp
//   demo_paired_nsg_flatnav_same_storage_parallel.cpp
//   demo_paired_nsg_flatnav_separate_storage_parallel.cpp
//
// Part 1 is copied verbatim from demo_paired_hnsw_nsg_same_storage_parallal.cpp
// (CSV logging, fvecs/ivecs readers, recall, CPU affinity helpers, and
// build_hnsw_graph_on_shared_storage / build_nsg_graph_on_shared_storage),
// so the pair demos log and measure exactly like the HNSW+NSG demos.
// Part 2 adds the FlatNav / AVX2 helpers.
//
// Header-only and meant for single-translation-unit demos (it defines
// non-inline functions and globals, like the original demo file).
// ============================================================

#pragma once

// ---------------- Part 1: copied from the HNSW+NSG demo ----------------
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexNSG.h>
#include "index_size_reporting.hpp"

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
 using namespace std;

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
// build_hnsw_graph_on_shared_storage
//
// A faithful, from-scratch port of faiss's internal
// `hnsw_add_vertices` (IndexHNSW.cpp), which is normally only
// reachable via IndexHNSW::add() and cannot be called directly
// from outside faiss because it lives in an anonymous
// namespace. It is reconstructed here using ONLY public HNSW /
// IndexHNSW members (hnsw.prepare_level_tab, hnsw.levels,
// hnsw.add_with_locks, index_hnsw.locks, index_hnsw.storage,
// etc. — all verified public in faiss/impl/HNSW.h and
// faiss/IndexHNSW.h).
//
// Why this exists at all: IndexHNSW::add() always calls
// storage->add(n, x) itself as its first step, so it cannot be
// pointed at storage that has already been populated by someone
// else (here, by us, once, up front) without adding the vectors
// a second time. This function skips that step and only builds
// the graph structure against storage that the CALLER has
// already populated — which is exactly what lets HNSW and NSG
// build their graphs concurrently on one shared IndexFlat with
// zero vector duplication, ever.
//
// Precondition: index_hnsw.storage already contains exactly the
// n vectors in x (ids 0..n-1), and this is a from-scratch build
// (no pre-existing graph — n0 is fixed at 0 below).
// ============================================================

void build_hnsw_graph_on_shared_storage(
        faiss::IndexHNSW& index_hnsw,
        size_t n,
        const float* x,
        bool verbose) {

    using storage_idx_t = faiss::HNSW::storage_idx_t;

    if (n == 0) {
        return;
    }

    const size_t n0 = 0; // from-scratch build only
    const size_t d = static_cast<size_t>(index_hnsw.d);
    faiss::HNSW& hnsw = index_hnsw.hnsw;
    const size_t ntotal = n0 + n;

    int max_level =
            hnsw.prepare_level_tab(
                    n,
                    /*preset_levels=*/false);

    if (verbose) {
        printf("build_hnsw_graph_on_shared_storage: max_level = %d\n",
               max_level);
    }

    auto& locks = index_hnsw.locks;
    locks.prepare(ntotal);

    // Bucket points by level, highest first — identical logic to
    // faiss's internal hnsw_add_vertices.
    std::vector<int> hist;
    std::vector<int> order(n);

    {
        for (size_t i = 0; i < n; i++) {
            storage_idx_t pt_id = static_cast<storage_idx_t>(i + n0);
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= static_cast<int>(hist.size())) {
                hist.push_back(0);
            }
            hist[pt_level]++;
        }

        std::vector<int> offsets(hist.size() + 1, 0);
        for (size_t i = 0; i + 1 < hist.size(); i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        for (size_t i = 0; i < n; i++) {
            storage_idx_t pt_id = static_cast<storage_idx_t>(i + n0);
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    size_t check_period =
            faiss::InterruptCallback::get_period_hint(
                    static_cast<size_t>(max_level) *
                    d *
                    static_cast<size_t>(hnsw.efConstruction));

    faiss::RandomGenerator rng2(789);

    size_t i1 = n;

    for (int pt_level = static_cast<int>(hist.size()) - 1;
         pt_level >= static_cast<int>(!index_hnsw.init_level0);
         pt_level--) {

        size_t i0 = i1 - hist[pt_level];

        // Random permutation within this level's bucket, to remove
        // dataset-order bias — same as the internal implementation.
        for (size_t j = i0; j < i1; j++) {
            std::swap(
                    order[j],
                    order[j + rng2.rand_int(static_cast<int>(i1 - j))]);
        }

        bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
        {
            std::unique_ptr<faiss::VisitedTable> vt =
                    faiss::VisitedTable::create(
                            ntotal,
                            hnsw.use_visited_hashset);

            // storage_distance_computer() in faiss is just this one-liner,
            // but it's a private helper (anonymous namespace), so it's
            // inlined here directly — same call, same effect.
            std::unique_ptr<faiss::DistanceComputer> dis(
                    index_hnsw.storage->get_distance_computer());

            size_t counter = 0;

#pragma omp for schedule(static)
            for (int64_t i = static_cast<int64_t>(i0);
                 i < static_cast<int64_t>(i1);
                 i++) {

                storage_idx_t pt_id = order[i];
                dis->set_query(x + (pt_id - n0) * d);

                if (interrupt) {
                    continue;
                }

                hnsw.add_with_locks(
                        *dis,
                        pt_level,
                        pt_id,
                        locks,
                        *vt,
                        index_hnsw.keep_max_size_level0 &&
                                (pt_level == 0));

                if (counter % check_period == 0) {
                    if (faiss::InterruptCallback::is_interrupted()) {
                        interrupt = true;
                    }
                }
                counter++;
            }
        }

        if (interrupt) {
            fprintf(stderr, "HNSW build interrupted\n");
            std::abort();
        }

        i1 = i0;
    }

    if (!index_hnsw.retain_locks) {
        locks.clear();
    }

    // IndexHNSW::add() normally sets this from storage->ntotal after
    // calling storage->add(); we set it directly since storage was
    // already populated by the caller before this function ran.
    index_hnsw.ntotal = static_cast<faiss::idx_t>(ntotal);
}


// ============================================================
// build_nsg_graph_on_shared_storage
//
// A faithful port of the build_type == 0 (brute-force kNN
// graph) branch of IndexNSG::add() (IndexNSG.cpp), skipping the
// storage->add(n, x) call that branch normally makes first —
// again, so this can run against storage the caller already
// populated, shared with another index being built concurrently.
//
// Only the brute-force path (build_type == 0) is ported; NSG's
// build_type == 1 (NNDescent) path is not needed for this demo.
//
// Precondition: same as build_hnsw_graph_on_shared_storage — the
// n vectors in x are already present in index_nsg.storage as ids
// 0..n-1, and this is a from-scratch build.
// ============================================================

void build_nsg_graph_on_shared_storage(
        faiss::IndexNSG& index_nsg,
        size_t n,
        const float* x,
        bool verbose) {

    faiss::Index* storage = index_nsg.storage;
    const int GK = index_nsg.GK;
    const faiss::idx_t n_idx = static_cast<faiss::idx_t>(n);

    if (verbose) {
        printf("build_nsg_graph_on_shared_storage: brute-force kNN, GK=%d\n",
               GK);
    }

    // storage->assign() is a plain read-only nearest-neighbor query — safe
    // to run concurrently against a fully-populated, read-only IndexFlat
    // while another thread is doing the same (or reading via
    // get_distance_computer()) for HNSW.
    std::vector<faiss::idx_t> knng(
            static_cast<size_t>(n) * (GK + 1));

    storage->assign(n_idx, x, knng.data(), GK + 1);

    std::vector<faiss::idx_t> knng_trimmed(
            static_cast<size_t>(n) * GK);

    // Remove each point from its own neighbor list — identical logic to
    // IndexNSG::add().
    if (storage->metric_type == faiss::METRIC_INNER_PRODUCT) {
        for (faiss::idx_t i = 0; i < n_idx; i++) {
            int count = 0;
            for (int j = 0; j < GK + 1; j++) {
                faiss::idx_t id = knng[i * (GK + 1) + j];
                if (id != i) {
                    knng_trimmed[i * GK + count] = id;
                    count += 1;
                }
                if (count == GK) {
                    break;
                }
            }
        }
    } else {
        for (faiss::idx_t i = 0; i < n_idx; i++) {
            memmove(
                    knng_trimmed.data() + i * GK,
                    knng.data() + i * (GK + 1) + 1,
                    static_cast<size_t>(GK) * sizeof(faiss::idx_t));
        }
    }

    index_nsg.check_knn_graph(knng_trimmed.data(), n_idx, GK);

    const faiss::nsg::Graph<faiss::idx_t> knn_graph(
            knng_trimmed.data(),
            static_cast<int>(n),
            GK);

    // The actual graph-construction step — public API, read-only against
    // storage (only reads distances/vectors, never mutates it).
    index_nsg.nsg.build(storage, n_idx, knn_graph, verbose);

    index_nsg.ntotal = n_idx;
    index_nsg.is_built = true;
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

// ---------------- Part 2: FlatNav pair helpers ----------------

// ============================================================
// ------------- FlatNav pair helpers (new) -------------------
// ============================================================

#include "flatnav_faiss_storage.hpp"

#include <faiss/utils/utils.h>

#include <functional>
#include <numeric>

// ------------------------------------------------------------
// AVX2-only guard.
//
// CMake compiles these demos with -mavx2 -mfma -mno-avx512f and
// -DFLATNAV_FORBID_AVX512 (FLATNAV_SIMD=avx2, the default). If
// something re-enables AVX-512 (e.g. -march=native in
// CMAKE_CXX_FLAGS placed after our flags), fail the build instead
// of silently benchmarking FlatNav with AVX-512 kernels.
// ------------------------------------------------------------
#if defined(FLATNAV_FORBID_AVX512) && defined(__AVX512F__)
#error "AVX-512 is enabled for a FLATNAV_SIMD=avx2 build. Remove -march=native / -mavx512* from CMAKE_CXX_FLAGS."
#endif

// Always-on check (assert() disappears in Release builds with -DNDEBUG).
#define PAIR_CHECK(cond)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__,   \
                    __LINE__);                                               \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

// SIMD level FlatNav's distance kernels were compiled with.
inline std::string flatnav_simd_level() {
#if defined(USE_AVX512)
    return "AVX-512";
#elif defined(__AVX2__) && defined(USE_AVX)
    return "AVX2";
#elif defined(USE_AVX)
    return "AVX";
#elif defined(USE_SSE)
    return "SSE";
#else
    return "scalar";
#endif
}

// Prints which SIMD level faiss and FlatNav use; aborts if either is
// AVX-512 when AVX2 was requested.
inline void report_simd_config() {
    const std::string faiss_opts = faiss::get_compile_options();
    const std::string fn_level = flatnav_simd_level();

    std::cout << "\n========================================\n"
              << "SIMD configuration\n"
              << "========================================\n"
              << "faiss compile options  = " << faiss_opts << "\n"
              << "FlatNav SIMD kernels   = " << fn_level << "\n";

    log_metric("config", "faiss_avx2", faiss_opts.find("AVX2") != std::string::npos ? 1.0 : 0.0);
    log_metric("config", "faiss_avx512", faiss_opts.find("AVX512") != std::string::npos ? 1.0 : 0.0);
    log_metric("config", "flatnav_avx2", fn_level == "AVX2" ? 1.0 : 0.0);
    log_metric("config", "flatnav_avx512", fn_level == "AVX-512" ? 1.0 : 0.0);

#if defined(FLATNAV_FORBID_AVX512)
    if (faiss_opts.find("AVX512") != std::string::npos) {
        fprintf(stderr,
                "faiss library is an AVX-512 build (%s). Link faiss_avx2 "
                "(-DFLATNAV_FAISS_TARGET=faiss_avx2).\n",
                faiss_opts.c_str());
        std::abort();
    }
    if (faiss_opts.find("AVX2") == std::string::npos) {
        std::cout << "WARNING: faiss library is not an AVX2 build (" << faiss_opts
                  << "). Configure faiss with -DFAISS_OPT_LEVEL=avx2 and link faiss_avx2.\n";
    }
#endif
}

// ------------------------------------------------------------
// Dataset loading (same files / checks as the HNSW+NSG demos).
// ------------------------------------------------------------
struct PairDataset {
    float* xb = nullptr;
    float* xq = nullptr;
    faiss::idx_t* gt = nullptr;
    int d = 0;
    faiss::idx_t n = 0;
    faiss::idx_t nq = 0;
    size_t gt_k = 0;

    ~PairDataset() {
        delete[] xb;
        delete[] xq;
        delete[] gt;
    }
};

inline bool load_pair_dataset(
        const char* base_file,
        const char* query_file,
        const char* gt_file,
        size_t recall_eval_k,
        PairDataset& ds) {
    size_t d_size = 0, nb = 0;
    ds.xb = fvecs_read(base_file, &d_size, &nb);
    ds.d = static_cast<int>(d_size);
    ds.n = static_cast<faiss::idx_t>(nb);
    std::cout << "\nBase dataset loaded\n"
              << "d      = " << ds.d << "\n"
              << "nb     = " << nb << "\n";

    size_t qd_size = 0, nq_size = 0;
    ds.xq = fvecs_read(query_file, &qd_size, &nq_size);
    if (static_cast<int>(qd_size) != ds.d) {
        fprintf(stderr,
                "Query dimension (%zu) does not match base dimension (%d)\n",
                qd_size,
                ds.d);
        return false;
    }
    ds.nq = static_cast<faiss::idx_t>(nq_size);
    std::cout << "\nQuery dataset loaded\n"
              << "nq     = " << ds.nq << "\n";

    if (gt_file != nullptr && std::string(gt_file) != "-") {
        size_t gt_nq = 0;
        ds.gt = ivecs_read(gt_file, &ds.gt_k, &gt_nq);
        if (gt_nq != nq_size) {
            fprintf(stderr,
                    "Ground truth query count (%zu) does not match query "
                    "file count (%zu)\n",
                    gt_nq,
                    nq_size);
            return false;
        }
        if (ds.gt_k < recall_eval_k) {
            fprintf(stderr,
                    "Ground truth depth (%zu) is smaller than "
                    "RECALL_EVAL_K (%zu)\n",
                    ds.gt_k,
                    recall_eval_k);
            return false;
        }
    } else {
        std::cout << "\nNo ground truth file provided — "
                  << "recall will not be computed.\n";
    }
    return true;
}

// ------------------------------------------------------------
// Pinned, single-threaded build thread.
//
// The thread pins ITSELF (pthread_self) before doing any work, so
// the whole build runs on `cpu_id` (pinning from the parent after
// std::thread starts leaves a short unpinned window). OpenMP is
// set to 1 thread inside it so faiss' #pragma omp regions stay on
// this thread; FlatNav builds with its own num_threads=1.
// ------------------------------------------------------------
struct PinnedBuild {
    double seconds = 0.0;
    int requested_cpu = -1;
    int ran_on_cpu = -1;
};

inline void pin_current_thread_to_cpu(int cpu_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        fprintf(stderr,
                "Warning: pthread_setaffinity_np(cpu=%d) failed (rc=%d) — "
                "thread will run unpinned.\n",
                cpu_id,
                rc);
    }
}

inline std::thread start_pinned_build(
        int cpu_id,
        PinnedBuild& out,
        std::function<void()> work) {
    out.requested_cpu = cpu_id;
    return std::thread([cpu_id, &out, work]() {
        pin_current_thread_to_cpu(cpu_id);
        omp_set_num_threads(1);
        auto t0 = std::chrono::high_resolution_clock::now();
        work();
        auto t1 = std::chrono::high_resolution_clock::now();
        out.seconds = std::chrono::duration<double>(t1 - t0).count();
        out.ran_on_cpu = sched_getcpu();
    });
}

// Picks two distinct CPUs from the process affinity mask.
inline std::pair<int, int> pick_two_cpus() {
    std::vector<int> cpus = affinity_cpu_list();
    std::cout << "Process CPU affinity list:";
    for (int c : cpus) {
        std::cout << " " << c;
    }
    std::cout << "\n";
    if (cpus.size() < 2) {
        std::cout << "WARNING: only 1 CPU available — both build threads "
                  << "will share CPU " << cpus[0]
                  << ". Use e.g. `taskset -c 0-1` for true 2-core parallelism.\n";
        return {cpus[0], cpus[0]};
    }
    return {cpus[0], cpus[1]};
}

// ------------------------------------------------------------
// Query reporting: QPS / ms per query / recall, printed + logged
// with the same metric names the HNSW+NSG demos use
// ("<name>_qps", "<name>_recall_at_<k>", ...).
// ------------------------------------------------------------
struct QueryReport {
    double seconds = 0.0;
    double qps = 0.0;
    double ms_per_query = 0.0;
    double recall = -1.0;
};

inline QueryReport report_query(
        const std::string& name,
        double seconds,
        const std::vector<faiss::idx_t>& labels,
        faiss::idx_t search_k,
        const PairDataset& ds,
        size_t recall_eval_k) {
    QueryReport r;
    r.seconds = seconds;
    r.qps = static_cast<double>(ds.nq) / seconds;
    r.ms_per_query = 1000.0 * seconds / static_cast<double>(ds.nq);
    std::cout << name << " search time = " << seconds << " sec total, "
              << r.ms_per_query << " ms/query, " << r.qps << " QPS\n";
    log_metric("query", name + "_search_seconds_total", seconds);
    log_metric("query", name + "_ms_per_query", r.ms_per_query);
    log_metric("query", name + "_qps", r.qps);
    if (ds.gt != nullptr) {
        r.recall = compute_recall_at_k(
                labels,
                search_k,
                ds.gt,
                ds.gt_k,
                static_cast<size_t>(ds.nq),
                recall_eval_k);
        std::cout << name << " recall@" << recall_eval_k << " = " << r.recall << "\n";
        log_metric("query", name + "_recall_at_" + std::to_string(recall_eval_k), r.recall);
    }
    return r;
}

template <class F>
inline double time_seconds(F&& f) {
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

inline double to_mb(size_t bytes) {
    return bytes / (1024.0 * 1024.0);
}
