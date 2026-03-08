# -*- coding: utf-8 -*-
import numpy as np

from nwf.metric import cosine, euclidean, mahalanobis_symmetric


def test_mahalanobis_symmetric_single() -> None:
    z1 = np.array([0.0, 0.0])
    s1 = np.array([1.0, 1.0])
    z2 = np.array([1.0, 0.0])
    s2 = np.array([1.0, 1.0])
    d = mahalanobis_symmetric(z1, s1, z2, s2)
    assert isinstance(d, (float, np.floating))
    expected = np.sqrt(0.5)
    np.testing.assert_almost_equal(d, expected, decimal=5)


def test_euclidean() -> None:
    z1 = np.array([0.0, 0.0])
    z2 = np.array([3.0, 4.0])
    d = euclidean(z1, z2)
    assert abs(float(d) - 5.0) < 1e-6


def test_cosine() -> None:
    z1 = np.array([[1.0, 0.0]])
    z2 = np.array([[1.0, 0.0]])
    d = cosine(z1, z2)
    np.testing.assert_almost_equal(d, 0.0, decimal=6)
