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

add_subdirectory(diversity_filter)
