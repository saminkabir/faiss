# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Python API of the LEANN-style high-degree preserving graph pruning
(IndexHNSW.prune_leann, IndexNSG.prune_leann). The graph invariants are
tested in tests/test_leann_prune.cpp."""

import unittest

import numpy as np

import faiss


def make_clustered(n, d, seed):
    rs = np.random.RandomState(seed)
    centers = 4 * np.random.RandomState(0).randn(40, d)
    x = centers[rs.randint(0, 40, n)] + rs.randn(n, d)
    return x.astype("float32")


def recall(gt, res):
    hit = sum(len(set(g) & set(r)) for g, r in zip(gt, res))
    return hit / gt.size


class TestLeannPruneHNSW(unittest.TestCase):
    def setUp(self):
        self.d, self.n, self.nq, self.k = 32, 4000, 100, 10
        self.xb = make_clustered(self.n, self.d, 1)
        self.xq = make_clustered(self.nq, self.d, 2)
        flat = faiss.IndexFlatL2(self.d)
        flat.add(self.xb)
        _, self.gt = flat.search(self.xq, self.k)

    def test_prune(self):
        index = faiss.IndexHNSWFlat(self.d, 16)
        index.hnsw.efConstruction = 100
        index.add(self.xb)

        params = faiss.LeannPruneParams()
        params.hub_fraction = 0.02
        params.low_max_degree = 4
        stats = index.prune_leann(params)

        self.assertEqual(stats.n_nodes, self.n)
        self.assertGreaterEqual(stats.n_hubs, int(np.ceil(0.02 * self.n)))
        self.assertEqual(stats.low_max_degree, 4)
        self.assertEqual(stats.hub_max_degree, 2 * 16)
        self.assertLess(stats.edges_after, stats.edges_before)
        self.assertGreater(
            stats.avg_degree_hubs_after, stats.avg_degree_non_hubs_after
        )
        self.assertEqual(stats.edge_bytes_after, 4 * stats.edges_after)

        index.hnsw.efSearch = 128
        _, I = index.search(self.xq, self.k)
        self.assertGreater(recall(self.gt, I), 0.85)

    def test_default_parameters(self):
        index = faiss.IndexHNSWFlat(self.d, 12)
        index.add(self.xb)
        stats = index.prune_leann()
        self.assertLess(stats.edges_after, stats.edges_before)

    def test_hub_score(self):
        index = faiss.IndexHNSWFlat(self.d, 12)
        index.add(self.xb)
        params = faiss.LeannPruneParams()
        params.hub_score = faiss.LEANN_HUB_IN_DEGREE
        stats = index.prune_leann(params)
        self.assertLess(stats.edges_after, stats.edges_before)

    def test_bad_parameters(self):
        index = faiss.IndexHNSWFlat(self.d, 8)
        index.add(self.xb)
        params = faiss.LeannPruneParams()
        params.hub_max_degree = 100  # more than the level-0 slots
        with self.assertRaises(RuntimeError):
            index.prune_leann(params)


class TestLeannPruneNSG(unittest.TestCase):
    def setUp(self):
        self.d, self.n, self.nq, self.k = 32, 3000, 100, 10
        self.xb = make_clustered(self.n, self.d, 3)
        self.xq = make_clustered(self.nq, self.d, 4)
        flat = faiss.IndexFlatL2(self.d)
        flat.add(self.xb)
        _, self.gt = flat.search(self.xq, self.k)

    def test_prune(self):
        index = faiss.IndexNSGFlat(self.d, 24)
        index.build_type = 1
        index.add(self.xb)

        params = faiss.LeannPruneParams()
        params.hub_fraction = 0.05
        params.low_max_degree = 5
        stats = index.prune_leann(params)

        self.assertEqual(stats.n_nodes, self.n)
        self.assertEqual(stats.hub_max_degree, 24)
        self.assertLess(stats.edges_after, stats.edges_before)
        self.assertEqual(stats.n_unreachable_after, 0)

        index.nsg.search_L = 100
        _, I = index.search(self.xq, self.k)
        self.assertGreater(recall(self.gt, I), 0.85)

    def test_not_built(self):
        index = faiss.IndexNSGFlat(self.d, 16)
        with self.assertRaises(RuntimeError):
            index.prune_leann()


if __name__ == "__main__":
    unittest.main()
