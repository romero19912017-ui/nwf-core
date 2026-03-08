# -*- coding: utf-8 -*-
"""Semantic potential for OOD detection.

Phi(r) = sum_i exp(-0.5 * d_mahalanobis_symmetric(r, z_i, sigma_i)^2)
Higher value = closer to in-distribution; lower = OOD.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from nwf.charge import Charge
from nwf.metric import mahalanobis_symmetric


def potential(
    r: np.ndarray,
    charges: List[Charge],
    eps: float = 1e-10,
) -> float:
    """Semantic potential at point r given charges.

    Phi(r) = sum_i exp(-0.5 * d_i^2) where d_i is Mahalanobis from r to charge i.
    Uses point-to-charge distance: (r-z_i)^T Sigma_i^{-1} (r-z_i).

    Args:
        r: Query point, shape (D,).
        charges: List of charges.
        eps: Numerical stability.

    Returns:
        Scalar potential value. Higher = more in-distribution.
    """
    if not charges:
        return 0.0
    r = np.asarray(r, dtype=np.float64).ravel()
    d = len(r)
    z_all = np.stack([c.z for c in charges], axis=0)
    s_all = np.stack([c.sigma for c in charges], axis=0)
    sigma_r = np.full((1, d), eps)
    dists = mahalanobis_symmetric(r.reshape(1, -1), sigma_r, z_all, s_all, eps=eps)
    d_sq = np.asarray(dists).ravel() ** 2
    phi = np.exp(-0.5 * d_sq).sum()
    return float(phi)


def potential_batch(
    r: np.ndarray,
    z_all: np.ndarray,
    sigma_all: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """Batch semantic potential for multiple query points.

    Args:
        r: Query points, shape (N, D).
        z_all: Charge centers, shape (M, D).
        sigma_all: Charge sigmas, shape (M, D).
        eps: Numerical stability.

    Returns:
        Potential values for each query, shape (N,).
    """
    r = np.atleast_2d(r)
    sigma_r = np.full_like(r, eps)
    dists = mahalanobis_symmetric(r, sigma_r, z_all, sigma_all, eps=eps)
    if dists.ndim == 1:
        dists = dists.reshape(1, -1)
    d_sq = dists**2
    phi = np.exp(-0.5 * d_sq).sum(axis=1)
    return phi
