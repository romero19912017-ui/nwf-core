# -*- coding: utf-8 -*-
"""Index implementations for NWF: BruteForce, FAISS (optional)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np

from nwf.metric import cosine, euclidean


class Index(ABC):
    """Abstract base class for vector indices."""

    @abstractmethod
    def add(
        self,
        vectors: np.ndarray,
        ids: Optional[np.ndarray] = None,
    ) -> None:
        """Add vectors to index. vectors: (N, D)."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search k nearest. Returns (distances, indices)."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save index to path."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load index from path."""
        pass


class BruteForceIndex(Index):
    """Exact brute-force search. Stores vectors, computes distances on query.

    Supports metric: 'l2', 'cosine', or custom (for Mahalanobis use Field.search).
    """

    def __init__(
        self,
        metric: Literal["l2", "cosine"] = "l2",
    ) -> None:
        self.metric = metric
        self._vectors: np.ndarray = np.zeros((0, 0))
        self._ids: Optional[np.ndarray] = None

    def add(
        self,
        vectors: np.ndarray,
        ids: Optional[np.ndarray] = None,
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D (N, D)")
        if len(self._vectors) == 0:
            self._vectors = vectors
            self._ids = ids if ids is not None else np.arange(len(vectors))
        else:
            self._vectors = np.vstack([self._vectors, vectors])
            n = len(vectors)
            if ids is not None:
                self._ids = np.concatenate([self._ids, ids])
            else:
                start = int(self._ids.max()) + 1
                self._ids = np.concatenate([self._ids, np.arange(start, start + n)])

    def search_batch(
        self,
        query_vectors: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch search. query_vectors: (B, D). Returns (distances, indices)."""
        return self.search(query_vectors, k)

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search k nearest. Supports single (D,) or batch (B, D) query."""
        q = np.atleast_2d(query_vector)
        if self.metric == "l2":
            d = euclidean(q, self._vectors)
        elif self.metric == "cosine":
            d = cosine(q, self._vectors)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        if d.ndim == 1:
            d = d.reshape(1, -1)
        k_act = min(k, d.shape[1])
        idx = np.argsort(d, axis=1)[:, :k_act]
        dist = np.take_along_axis(d, idx, axis=1)
        out_idx = self._ids[idx] if self._ids is not None else idx
        return dist.squeeze(), out_idx.squeeze()

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p.with_suffix(".npz") if p.suffix != ".npz" else p,
            vectors=self._vectors,
            ids=self._ids,
            metric=np.array([self.metric]),
        )

    def load(self, path: Union[str, Path]) -> None:
        data = np.load(path, allow_pickle=True)
        self._vectors = data["vectors"]
        self._ids = data["ids"]
        m = data["metric"]
        self.metric = str(m.item()) if m.ndim > 0 else "l2"

    def __len__(self) -> int:
        return len(self._vectors)
