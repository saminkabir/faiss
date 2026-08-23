#include <faiss/IndexHNSW.h>
#include <faiss/IndexNSG.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>

#include <sys/time.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) && "unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 && "weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) && "could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

int d = 128, M = 32, R = 32, GK = 64;
size_t nb, d2;
float* xb = fvecs_read("/home/cc/datasets/datasets/sift/base.fvecs", &d2, &nb);
idx_t n = nb;
d=d2;

// 1. Build HNSW normally — owns storage A, permanent.
faiss::IndexHNSWFlat hnsw(d, M);
hnsw.add(n, xb);

// 2. Build NSG normally too — let it compute its own kNN graph
//    internally (brute force or NNDescent), no manual graph needed.
//    This temporarily owns its own storage B.
faiss::IndexNSGFlat nsg(d, R);
nsg.GK = GK;            // candidate pool size for pruning, same role as before
nsg.build_type = 0;     // 0 = brute force kNN, 1 = NNDescent (faster for large n)
nsg.add(n, xb);          // builds full NSG graph using storage B

// 3. Now that NSG's graph is built, storage B has served its purpose.
//    Free it and repoint NSG at HNSW's storage instead.
delete nsg.storage;
nsg.storage = hnsw.storage;
nsg.own_fields = false;   // hnsw keeps ownership; nsg must not double-free

// Both indexes now share exactly one copy of the vectors:
assert(hnsw.storage == nsg.storage);
assert(hnsw.storage->ntotal == n);
