# -*- coding: utf-8 -*-
"""Distance metrics for NWF: Mahalanobis, Euclidean, Cosine."""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, "np.ndarray"]


def mahalanobis_symmetric(
    z1: np.ndarray,
    sigma1: np.ndarray,
    z2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-10,
) -> Union[float, np.ndarray]:
    """Symmetric Mahalanobis distance: d^2 = (z1-z2)^T (sigma1+sigma2)^{-1} (z1-z2).

    For diagonal covariances: d = sqrt(sum_d (z1_d - z2_d)^2 / (sigma1_d + sigma2_d)).

    Supports batches: z1 (N,D), z2 (M,D) -> (N,M) distances.
    """
    z1 = np.atleast_2d(z1)
    z2 = np.atleast_2d(z2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    delta = z1[:, np.newaxis, :] - z2[np.newaxis, :, :]
    cov_sum = sigma1[:, np.newaxis, :] + sigma2[np.newaxis, :, :] + eps
    d_sq = np.sum((delta**2) / cov_sum, axis=-1)
    result = np.sqrt(d_sq + eps)
    return float(result.flat[0]) if result.size == 1 else result


def euclidean(
    z1: np.ndarray, z2: np.ndarray, eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """Euclidean (L2) distance."""
    z1 = np.atleast_2d(z1)
    z2 = np.atleast_2d(z2)
    delta = z1[:, np.newaxis, :] - z2[np.newaxis, :, :]
    d_sq = np.sum(delta**2, axis=-1)
    result = np.sqrt(d_sq + eps)
    return float(result.flat[0]) if result.size == 1 else result


def cosine(
    z1: np.ndarray, z2: np.ndarray, eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """Cosine distance: 1 - cos_sim. Returns distance (0 = identical)."""
    z1 = np.atleast_2d(z1)
    z2 = np.atleast_2d(z2)
    n1 = np.linalg.norm(z1, axis=1, keepdims=True) + eps
    n2 = np.linalg.norm(z2, axis=1, keepdims=True) + eps
    z1n = z1 / n1
    z2n = z2 / n2
    sim = np.dot(z1n, z2n.T)
    sim = np.clip(sim, -1.0, 1.0)
    result = 1.0 - sim
    return float(result.flat[0]) if result.size == 1 else result
