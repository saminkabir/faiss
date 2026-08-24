#include <faiss/IndexHNSW.h>
#include <faiss/IndexNSG.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <sys/stat.h>


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
// Main
//
// Usage:
//
// ./demo_paired_hnsw_nsg /path/to/base.fvecs
//
// Example:
//
// ./build/demos/demo_paired_hnsw_nsg \
//     /home/cc/datasets/datasets/sift/base.fvecs
// ============================================================

int main(
        int argc,
        char** argv) {

    if (argc < 2) {

        std::cerr
                << "Usage:\n"
                << "  "
                << argv[0]
                << " /path/to/base.fvecs\n";

        return 1;
    }

    const char* base_file =
            argv[1];


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
            << "\n\n";


    // ========================================================
    // Load dataset
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
            << "\nDataset loaded\n";

    std::cout
            << "d      = "
            << d
            << "\n";

    std::cout
            << "nb     = "
            << nb
            << "\n";


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


    double hnsw_seconds =
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

    std::cout
            << "HNSW storage ntotal = "
            << hnsw.storage->ntotal
            << "\n";

    std::cout
            << "HNSW construction time = "
            << hnsw_seconds
            << " sec\n";


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


    double nsg_seconds =
            std::chrono::duration<double>(
                    nsg_end -
                    nsg_start)
                    .count();


    std::cout
            << "NSG construction complete\n";

    std::cout
            << "NSG ntotal = "
            << nsg.ntotal
            << "\n";

    std::cout
            << "NSG storage ntotal = "
            << nsg.storage->ntotal
            << "\n";

    std::cout
            << "NSG construction time = "
            << nsg_seconds
            << " sec\n";


    // ========================================================
    // At this point:
    //
    // HNSW:
    //
    //     hnsw.storage
    //           |
    //           v
    //       IndexFlat
    //
    //
    // NSG:
    //
    //     nsg.storage
    //           |
    //           v
    //       IndexFlat
    //
    //
    // The two IndexFlat objects contain duplicate copies of
    // the same base vectors.
    //
    // We want:
    //
    //
    //       HNSW --------+
    //                    |
    //                    v
    //                IndexFlat
    //                    ^
    //                    |
    //       NSG ----------+
    //
    //
    // HNSW owns the shared IndexFlat.
    //
    // NSG only borrows it.
    // ========================================================


    std::cout
            << "\n========================================\n"
            << "Before sharing storage\n"
            << "========================================\n";


    std::cout
            << "HNSW storage pointer = "
            << static_cast<void*>(
                       hnsw.storage)
            << "\n";


    std::cout
            << "NSG storage pointer  = "
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


    // ========================================================
    // Remove NSG's duplicate vector storage
    // ========================================================

    std::cout
            << "\nDeleting NSG duplicate vector storage...\n";


    if (
            nsg.own_fields &&
            nsg.storage !=
                    nullptr) {

        delete nsg.storage;
    }


    // ========================================================
    // Make NSG share HNSW storage
    // ========================================================

    nsg.storage =
            hnsw.storage;


    // IMPORTANT:
    //
    // HNSW owns this object.
    //
    // NSG MUST NOT delete it when its destructor executes.
    //
    nsg.own_fields =
            false;


    // ========================================================
    // Verify shared storage
    // ========================================================

    std::cout
            << "\n========================================\n"
            << "After sharing storage\n"
            << "========================================\n";


    std::cout
            << "HNSW storage pointer = "
            << static_cast<void*>(
                       hnsw.storage)
            << "\n";


    std::cout
            << "NSG storage pointer  = "
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
    // Print final statistics
    // ========================================================

    std::cout
            << "\n========================================\n"
            << "Final index information\n"
            << "========================================\n";


    std::cout
            << "Vectors              = "
            << n
            << "\n";

    std::cout
            << "Dimension            = "
            << d
            << "\n";

    std::cout
            << "HNSW ntotal          = "
            << hnsw.ntotal
            << "\n";

    std::cout
            << "NSG ntotal           = "
            << nsg.ntotal
            << "\n";

    std::cout
            << "Shared storage total = "
            << hnsw.storage->ntotal
            << "\n";

    std::cout
            << "HNSW build time      = "
            << hnsw_seconds
            << " sec\n";

    std::cout
            << "NSG build time       = "
            << nsg_seconds
            << " sec\n";


    // ========================================================
    // Small search test
    //
    // Use first vector as query.
    // ========================================================

    const faiss::idx_t k =
            10;


    std::vector<float>
            hnsw_distances(k);

    std::vector<faiss::idx_t>
            hnsw_labels(k);


    std::vector<float>
            nsg_distances(k);

    std::vector<faiss::idx_t>
            nsg_labels(k);


    std::cout
            << "\n========================================\n"
            << "Testing searches\n"
            << "========================================\n";


    // --------------------------------------------------------
    // HNSW search
    // --------------------------------------------------------

    hnsw.search(
            1,
            xb,
            k,
            hnsw_distances.data(),
            hnsw_labels.data());


    // --------------------------------------------------------
    // NSG search
    // --------------------------------------------------------

    nsg.search(
            1,
            xb,
            k,
            nsg_distances.data(),
            nsg_labels.data());


    std::cout
            << "\nHNSW neighbors:\n";

    for (
            faiss::idx_t i = 0;
            i < k;
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
            << "\nNSG neighbors:\n";

    for (
            faiss::idx_t i = 0;
            i < k;
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
    // Cleanup input vectors
    //
    // Faiss IndexFlat stores its own copy, so xb can safely
    // be deleted now.
    // ========================================================

    delete[] xb;


    std::cout
            << "\n========================================\n"
            << "Done\n"
            << "========================================\n";


    return 0;
}