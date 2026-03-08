# -*- coding: utf-8 -*-
import numpy as np

from nwf.index import BruteForceIndex


def test_bruteforce_search_batch() -> None:
    idx = BruteForceIndex(metric="l2")
    vectors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    idx.add(vectors)
    queries = np.array([[0.1, 0.1], [0.9, 0.9]])
    dist, ids = idx.search_batch(queries, k=2)
    assert dist.shape[0] == 2
    assert ids.shape[0] == 2
