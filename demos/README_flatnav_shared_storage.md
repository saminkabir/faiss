# FlatNav on Faiss shared storage

FlatNav normally stores each node as `[vector][M links][label]` in one block.
With the `faiss-shared-storage` FlatNav patch, an index can instead be built on
an external, row-major buffer. A `faiss::IndexFlat` already keeps its vectors
that way (`codes`, reachable with `get_xb()`). In that mode FlatNav allocates
only `[M links][label]` per node and reads every vector from Faiss.

* FlatNav node `i` == Faiss storage row `i` == Faiss id `i`, even with a
  multi-threaded build.
* No vector is copied. FlatNav's own memory is `n * (4*M + 4)` bytes.
* Works with float32 `IndexFlatL2` (FlatNav `l2`) and `IndexFlatIP`
  (FlatNav `angular`). Scalar-quantized FlatNav can't share, because its
  internal vector format is different. You get an error if you try.
* Graph reordering (`gorder` / `rcm`) is disabled in this mode, because it
  physically moves vectors and would break the shared ids.
* `saveIndex` writes the graph only. After `loadIndex`, call
  `attachExternalStorage` (C++) or `attach_storage` (Python). Older FlatNav
  index files still load as before.

## C++

```cpp
#include "flatnav_faiss_storage.hpp"

faiss::IndexFlatL2 shared_storage(d);
shared_storage.add(n, xb);                       // the only copy of the vectors

faiss::IndexHNSW hnsw(&shared_storage, M);       // as in the same-storage demo
faiss::IndexNSG  nsg (&shared_storage, R);
auto flatnav = flatnav_faiss::make_flatnav_l2(shared_storage, M);

flatnav_faiss::build(*flatnav, ef_construction, /*num_threads=*/1);
flatnav_faiss::search(*flatnav, nq, xq, k, ef_search, D, I);   // I are faiss ids
```

Build (AVX2, no AVX-512):

```bash
cmake -B build -DFAISS_OPT_LEVEL=avx2 -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
      -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DFLATNAV_ROOT=/path/to/flatnav .
make -C build -j faiss_avx2 flatnav_demos      # builds all six FlatNav demos
```

`FLATNAV_SIMD=avx2` is the default:

* FlatNav is compiled with `-mavx2 -mfma -mf16c -mpopcnt -mno-avx512f`.
* The demos link `faiss_avx2`, so they need `-DFAISS_OPT_LEVEL=avx2`.
* A compile-time `#error` stops the build if AVX-512 gets switched back on,
  for example by `-march=native` in `CMAKE_CXX_FLAGS`.
* At startup, every pair demo prints and logs the SIMD level faiss and
  FlatNav use (`config,faiss_avx2/avx512,flatnav_avx2/avx512` rows in the
  CSV), and aborts if faiss is an AVX-512 build.
* To check a binary yourself, run
  `objdump -d build/demos/<demo> | grep -c %zmm`. It should print 0.

`-DFLATNAV_SIMD=native` restores `-march=native`.

## Pair demos: {HNSW, NSG} x FlatNav, {same, separate} storage

| demo | faiss index | FlatNav | vectors in memory |
|---|---|---|---|
| `demo_paired_hnsw_flatnav_same_storage_parallel` | `IndexHNSW(&shared_storage)` | graph on `shared_storage` | 1 copy |
| `demo_paired_hnsw_flatnav_separate_storage_parallel` | `IndexHNSWFlat` (own storage) | owned FlatNav index | 2 copies |
| `demo_paired_nsg_flatnav_same_storage_parallel` | `IndexNSG(&shared_storage)` | graph on `shared_storage` | 1 copy |
| `demo_paired_nsg_flatnav_separate_storage_parallel` | `IndexNSGFlat` (own storage) | owned FlatNav index | 2 copies |

All four:

* build the two indexes in parallel, one pinned thread each, with OpenMP set
  to 1 inside the faiss thread;
* check that storage is shared or separate, with checks that still run in
  Release builds;
* report storage and graph bytes, then query both indexes with all CPUs;
* write the same CSV metrics as the HNSW+NSG demos (`<index>_qps`,
  `<index>_recall_at_20`, `storage_bytes`, `combined_bytes`, ...).

The shared helpers are in `paired_flatnav_common.hpp`. Its first part is
copied from `demo_paired_hnsw_nsg_same_storage_parallal.cpp`.

```bash
# HNSW + FlatNav:  [HNSW_M HNSW_efC HNSW_efS FLATNAV_M FLATNAV_efC FLATNAV_efS]
taskset -c 0-1 ./build/demos/demo_paired_hnsw_flatnav_same_storage_parallel \
    base.fvecs query.fvecs gt.ivecs log.csv  32 100 64  32 100 64
taskset -c 0-1 ./build/demos/demo_paired_hnsw_flatnav_separate_storage_parallel \
    base.fvecs query.fvecs gt.ivecs log.csv  32 100 64  32 100 64

# NSG + FlatNav:   [NSG_R NSG_search_L FLATNAV_M FLATNAV_efC FLATNAV_efS]  (NSG GK = 64)
taskset -c 0-1 ./build/demos/demo_paired_nsg_flatnav_same_storage_parallel \
    base.fvecs query.fvecs gt.ivecs log.csv  32 64  32 100 64
taskset -c 0-1 ./build/demos/demo_paired_nsg_flatnav_separate_storage_parallel \
    base.fvecs query.fvecs gt.ivecs log.csv  32 64  32 100 64
```

Pass `-` instead of `gt.ivecs` to skip recall.

Other FlatNav demos:

* `demo_flatnav_shared_storage --synthetic 100000 128` runs FlatNav alone on
  a shared `IndexFlatL2`.
* `demo_paired_hnsw_nsg_flatnav_same_storage_parallel` runs HNSW, NSG and
  FlatNav on one storage. It takes the original HNSW/NSG arguments followed
  by `FLATNAV_M FLATNAV_efC FLATNAV_efS`.

Note: the original `demo_paired_hnsw_nsg_*` targets link `faiss`, not
`faiss_avx2`. With `-DFAISS_OPT_LEVEL=avx2`, `faiss` is the generic (non-AVX2)
build. Switch their `target_link_libraries` to `faiss_avx2` if you compare
their numbers with the FlatNav demos.

## Python

Build both libraries for AVX2:

* **FlatNav:** `FLATNAV_SIMD=avx2` is the default in
  `python-bindings/setup.py`. `flatnav.simd_level` reports what was compiled.
  `FLATNAV_SIMD=native` restores `-march=native`.
* **faiss built from source:** use `-DFAISS_OPT_LEVEL=avx2`,
  `make swigfaiss_avx2`, and run with `FAISS_OPT_LEVEL=AVX2`.
* **pip faiss-cpu (dynamic dispatch):** run with `FAISS_SIMD_LEVEL=AVX2`, then
  check `faiss.SIMDConfig.get_level_name()`.

```python
import faiss, flatnav

storage = faiss.IndexFlatL2(d)
storage.add(xb)

index = flatnav.index.create(distance_type="l2", dim=d, dataset_size=storage.ntotal,
                             max_edges_per_node=32, external_storage=storage)
index.set_num_threads(8)
index.add_from_storage(ef_construction=100)        # node i == faiss id i
D, I = index.search(xq, K=10, ef_search=64)

# external_storage also accepts faiss.IndexHNSWFlat / IndexNSGFlat (uses .storage)
# or a C-contiguous numpy array. The index keeps the faiss object alive.
index.save("graph.flatnav")                        # graph only
g = flatnav.index.IndexL2Float.load_index("graph.flatnav")
g.attach_storage(storage)
```

## Caveats

* Don't call `storage.add()` while FlatNav is searching. A `std::vector`
  that grows can move its buffer. If you do grow it, re-attach (Python's
  `add_from_storage` does this for you) before the next build or search.
* With the split layout, each hop touches two memory regions (links and
  vector) instead of one. In a 200k x 128-d test on a 2-core VM, search QPS
  was about 1-6% below FlatNav's own interleaved layout, after prefetching
  the links. Build time was within about 5%. If you compare FlatNav's QPS
  against its default layout, report which layout you used.
