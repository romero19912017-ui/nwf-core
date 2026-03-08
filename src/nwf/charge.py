# -*- coding: utf-8 -*-
"""Charge - base NWF data structure (z, sigma).

Each semantic object is represented as a charge: center z and diagonal covariance sigma.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, "np.ndarray"]


@dataclass
class Charge:
    """Charge - NWF base object: center z and diagonal covariance sigma.

    Attributes:
        z: Center of the charge, shape (D,).
        sigma: Diagonal of covariance matrix (positive values), shape (D,).
    """

    z: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=np.float64)
        sigma = np.asarray(self.sigma, dtype=np.float64)
        if z.shape != sigma.shape:
            raise ValueError("z and sigma must have the same shape")
        if np.any(sigma <= 0):
            raise ValueError("sigma must contain positive values")
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "sigma", sigma)

    @property
    def dim(self) -> int:
        """Latent dimension."""
        return int(self.z.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (JSON-serializable)."""
        return {"z": self.z.tolist(), "sigma": self.sigma.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Charge":
        """Deserialize from dictionary."""
        return cls(z=np.array(d["z"]), sigma=np.array(d["sigma"]))

    def to_vector(self) -> np.ndarray:
        """Return concatenated vector (z, log_sigma) for indexing."""
        log_sigma = np.log(self.sigma + 1e-10)
        return np.concatenate([self.z, log_sigma])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "Charge":
        """Create charge from (z, log_sigma) vector. Assumes equal halves."""
        d = len(v) // 2
        z = v[:d].copy()
        log_sigma = v[d:]
        sigma = np.exp(log_sigma) + 1e-10
        return cls(z=z, sigma=sigma)

    def whiten(
        self,
        global_mean: Optional[np.ndarray] = None,
        global_std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform to whitened space (optional, for indexing).

        If global_mean and global_std are None, uses sqrt(sigma) as scale.
        """
        if global_mean is not None and global_std is not None:
            return (self.z - global_mean) / (global_std + 1e-10)
        return self.z / (np.sqrt(self.sigma) + 1e-10)

    def clip_sigma(self, min_val: float = 1e-6) -> "Charge":
        """Return new Charge with sigma clipped to be at least min_val."""
        sigma_new = np.maximum(self.sigma, min_val)
        return Charge(z=self.z.copy(), sigma=sigma_new)

    def __len__(self) -> int:
        return self.dim
