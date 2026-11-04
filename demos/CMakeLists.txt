# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

add_executable(demo_imi_flat EXCLUDE_FROM_ALL demo_imi_flat.cpp)
target_link_libraries(demo_imi_flat PRIVATE faiss)

add_executable(demo_imi_pq EXCLUDE_FROM_ALL demo_imi_pq.cpp)
target_link_libraries(demo_imi_pq PRIVATE faiss)

add_executable(demo_ivfpq_indexing EXCLUDE_FROM_ALL demo_ivfpq_indexing.cpp)
target_link_libraries(demo_ivfpq_indexing PRIVATE faiss)

add_executable(demo_nndescent EXCLUDE_FROM_ALL demo_nndescent.cpp)
target_link_libraries(demo_nndescent PRIVATE faiss)

add_executable(demo_sift1M EXCLUDE_FROM_ALL demo_sift1M.cpp)
target_link_libraries(demo_sift1M PRIVATE faiss)

add_executable(demo_weighted_kmeans EXCLUDE_FROM_ALL demo_weighted_kmeans.cpp)
target_link_libraries(demo_weighted_kmeans PRIVATE faiss)

add_executable(demo_residual_quantizer EXCLUDE_FROM_ALL demo_residual_quantizer.cpp)
target_link_libraries(demo_residual_quantizer PRIVATE faiss)

add_executable(demo_paired_hnsw_nsg EXCLUDE_FROM_ALL demo_paired_hnsw_nsg.cpp)
target_link_libraries(demo_paired_hnsw_nsg PRIVATE faiss)

add_executable(demo_paired_hnsw_nsg_diff_storage EXCLUDE_FROM_ALL demo_paired_hnsw_nsg_diff_storage.cpp)
target_link_libraries(demo_paired_hnsw_nsg_diff_storage PRIVATE faiss)

add_executable(demo_paired_hnsw_nsg_same_storage_parallal EXCLUDE_FROM_ALL demo_paired_hnsw_nsg_same_storage_parallal.cpp)
target_link_libraries(demo_paired_hnsw_nsg_same_storage_parallal PRIVATE faiss)


add_executable(demo_paired_hnsw_nsg_separate_storage_parallal EXCLUDE_FROM_ALL demo_paired_hnsw_nsg_separate_storage_parallal.cpp)
target_link_libraries(demo_paired_hnsw_nsg_separate_storage_parallal PRIVATE faiss)

# ---- FlatNav demos (shared / separate storage with faiss) ------------------
# Point FLATNAV_ROOT at a FlatNav checkout that has ExternalVectorStorage
# (branch `faiss-shared-storage`). AVX2 build (no AVX-512):
#
#   cmake -B build -DFAISS_OPT_LEVEL=avx2 -DFAISS_ENABLE_GPU=OFF \
#         -DCMAKE_BUILD_TYPE=Release -DFLATNAV_ROOT=$HOME/flatnav .
#   make -C build -j demo_paired_hnsw_flatnav_same_storage_parallel ...
#
# FLATNAV_SIMD selects how FlatNav's distance kernels are compiled
# (FlatNav picks SSE/AVX/AVX-512 code from compiler macros, Macros.h):
#   avx2    (default) -mavx2 -mfma -mf16c -mpopcnt, AVX-512 explicitly off
#           (-mno-avx512f + a compile-time #error guard), linked to faiss_avx2
#   native  -march=native (uses AVX-512 on CPUs that have it)
#   generic no extra flags (SSE2 baseline)
set(FLATNAV_ROOT "" CACHE PATH "Path to the FlatNav repository (header-only)")
set(FLATNAV_SIMD avx2 CACHE STRING "SIMD level for the FlatNav demos: avx2, native, generic")
set_property(CACHE FLATNAV_SIMD PROPERTY STRINGS avx2 native generic)
if(FLATNAV_SIMD STREQUAL "avx2")
  set(_flatnav_default_faiss_target faiss_avx2)
else()
  set(_flatnav_default_faiss_target faiss)
endif()
set(FLATNAV_FAISS_TARGET ${_flatnav_default_faiss_target} CACHE STRING
    "faiss library target to link (faiss_avx2 for AVX2; needs -DFAISS_OPT_LEVEL=avx2)")

if(FLATNAV_ROOT)
  find_package(OpenMP REQUIRED)
  find_package(Threads REQUIRED)
  if(NOT TARGET ${FLATNAV_FAISS_TARGET})
    message(FATAL_ERROR
      "FLATNAV_FAISS_TARGET=${FLATNAV_FAISS_TARGET} does not exist. For AVX2 configure "
      "faiss with -DFAISS_OPT_LEVEL=avx2 (creates faiss_avx2), or set FLATNAV_FAISS_TARGET.")
  endif()
  set(FLATNAV_INCLUDE_DIRS
      ${FLATNAV_ROOT}/include
      ${FLATNAV_ROOT}/external/cereal/include)

  if(FLATNAV_SIMD STREQUAL "avx2")
    set(FLATNAV_SIMD_FLAGS -mavx2 -mfma -mf16c -mpopcnt -mno-avx512f)
    set(FLATNAV_SIMD_DEFS FLATNAV_FORBID_AVX512)
  elseif(FLATNAV_SIMD STREQUAL "native")
    set(FLATNAV_SIMD_FLAGS -march=native)
    set(FLATNAV_SIMD_DEFS "")
  else()
    set(FLATNAV_SIMD_FLAGS "")
    set(FLATNAV_SIMD_DEFS "")
  endif()
  message(STATUS "FlatNav demos: SIMD=${FLATNAV_SIMD}, faiss target=${FLATNAV_FAISS_TARGET}")

  set(FLATNAV_DEMOS
      demo_flatnav_shared_storage
      demo_paired_hnsw_nsg_flatnav_same_storage_parallel
      demo_paired_hnsw_flatnav_same_storage_parallel
      demo_paired_hnsw_flatnav_separate_storage_parallel
      demo_paired_nsg_flatnav_same_storage_parallel
      demo_paired_nsg_flatnav_separate_storage_parallel)
  foreach(demo ${FLATNAV_DEMOS})
    add_executable(${demo} EXCLUDE_FROM_ALL ${demo}.cpp)
    target_include_directories(${demo} PRIVATE ${FLATNAV_INCLUDE_DIRS})
    target_compile_features(${demo} PRIVATE cxx_std_17)
    target_link_libraries(${demo} PRIVATE ${FLATNAV_FAISS_TARGET} OpenMP::OpenMP_CXX Threads::Threads)
    target_compile_options(${demo} PRIVATE -O3 ${FLATNAV_SIMD_FLAGS})
    if(FLATNAV_SIMD_DEFS)
      target_compile_definitions(${demo} PRIVATE ${FLATNAV_SIMD_DEFS})
    endif()
  endforeach()
  add_custom_target(flatnav_demos DEPENDS ${FLATNAV_DEMOS})
endif()

add_subdirectory(diversity_filter)
