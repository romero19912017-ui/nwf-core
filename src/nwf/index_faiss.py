# -*- coding: utf-8 -*-
"""FAISS-based index with two-stage Mahalanobis reranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple, Union

import numpy as np

from nwf.metric import mahalanobis_symmetric

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None  # type: ignore


class FAISSIndex:
    """FAISS-based approximate nearest neighbor search.

    Supports metrics: l2, cosine, ip (inner product).
    For Mahalanobis: use two-stage search with rerank=True and store
    original (z, sigma) for reranking.
    """

    def __init__(
        self,
        metric: Literal["l2", "cosine", "ip"] = "l2",
        rerank: bool = False,
        rerank_candidates: int = 100,
        z_store: Optional[np.ndarray] = None,
        sigma_store: Optional[np.ndarray] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """Initialize FAISS index.

        Args:
            metric: faiss metric type
            rerank: if True, perform second-stage Mahalanobis rerank
            rerank_candidates: number of L2 candidates to rerank
            z_store: stored z vectors for rerank (half of vector)
            sigma_store: stored sigma for rerank
            transform: optional whitening transform applied before add/search
        """
        if not HAS_FAISS:
            raise ImportError(
                "faiss-cpu required. Install: pip install nwf-core[faiss]"
            )
        self.metric = metric
        self.rerank = rerank
        self.rerank_candidates = rerank_candidates
        self._z_store = z_store
        self._sigma_store = sigma_store
        self._transform = transform
        self._index: Optional[Any] = None
        self._dim: Optional[int] = None
        self._ids: Optional[np.ndarray] = None
        self._vectors_raw: Optional[np.ndarray] = None

    def _ensure_index(self, dim: int) -> None:
        if self._index is not None and self._dim == dim:
            return
        self._dim = dim
        if self.metric == "l2":
            self._index = faiss.IndexFlatL2(dim)
        elif self.metric == "ip":
            self._index = faiss.IndexFlatIP(dim)
        elif self.metric == "cosine":
            self._index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _apply_transform(self, v: np.ndarray) -> np.ndarray:
        if self._transform is not None:
            return self._transform(v)
        return v

    def add(
        self,
        vectors: np.ndarray,
        ids: Optional[np.ndarray] = None,
        z_store: Optional[np.ndarray] = None,
        sigma_store: Optional[np.ndarray] = None,
    ) -> None:
        """Add vectors. For rerank: pass z_store, sigma_store (each half of vector)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D (N, D)")
        v = self._apply_transform(vectors)
        if self.metric == "cosine":
            norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
            v = v / norms
        self._ensure_index(v.shape[1])
        self._index.add(v)
        n = len(v)
        if ids is None:
            start = 0 if self._ids is None else int(self._ids.max()) + 1
            ids = np.arange(start, start + n)
        if self._ids is None:
            self._ids = ids
        else:
            self._ids = np.concatenate([self._ids, ids])
        if self.rerank:
            d = vectors.shape[1] // 2
            if z_store is not None and sigma_store is not None:
                self._z_store = (
                    z_store
                    if self._z_store is None
                    else np.vstack([self._z_store, z_store])
                )
                self._sigma_store = (
                    sigma_store
                    if self._sigma_store is None
                    else np.vstack([self._sigma_store, sigma_store])
                )
            else:
                z = vectors[:, :d]
                log_s = vectors[:, d:]
                sigma = np.exp(log_s) + 1e-10
                self._z_store = (
                    z if self._z_store is None else np.vstack([self._z_store, z])
                )
                self._sigma_store = (
                    sigma
                    if self._sigma_store is None
                    else np.vstack([self._sigma_store, sigma])
                )
        self._vectors_raw = (
            vectors
            if self._vectors_raw is None
            else np.vstack([self._vectors_raw, vectors])
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        query_z: Optional[np.ndarray] = None,
        query_sigma: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search k nearest. For rerank pass query_z, query_sigma (or derived)."""
        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        q = self._apply_transform(q)
        if self.metric == "cosine":
            norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-10
            q = q / norms
        k_act = min(k, self._index.ntotal)
        if self.rerank and self._z_store is not None:
            k_cand = min(self.rerank_candidates, self._index.ntotal)
            dist_cand, idx_cand = self._index.search(q, k_cand)
            d = self._z_store.shape[1]
            if query_z is None or query_sigma is None:
                query_z = q[:, :d]
                query_sigma = np.exp(q[:, d:]) + 1e-10
            dists_out = np.zeros((q.shape[0], k_act))
            idx_out = np.zeros((q.shape[0], k_act), dtype=np.int64)
            for i in range(q.shape[0]):
                idx_i = idx_cand[i]
                z_q = query_z[i] if query_z.ndim > 1 else query_z
                s_q = query_sigma[i] if query_sigma.ndim > 1 else query_sigma
                z_s = self._z_store[idx_i]
                s_s = self._sigma_store[idx_i]
                d_m = mahalanobis_symmetric(z_q[np.newaxis], s_q[np.newaxis], z_s, s_s)
                order = np.argsort(d_m)[:k_act]
                dists_out[i] = d_m[order]
                idx_out[i] = idx_i[order]
            out_ids = self._ids[idx_out] if self._ids is not None else idx_out
            return dists_out.squeeze(), out_ids.squeeze()
        dist, idx = self._index.search(q, k_act)
        if self.metric == "ip":
            dist = -dist
        out_ids = self._ids[idx] if self._ids is not None else idx
        return dist.squeeze(), out_ids.squeeze()

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        base = p.with_suffix("")
        faiss.write_index(self._index, str(base) + ".faiss")
        np.savez(
            str(base) + "_meta.npz",
            ids=self._ids,
            metric=np.array([self.metric]),
            rerank=np.array([self.rerank]),
            z_store=self._z_store if self._z_store is not None else np.zeros((0, 0)),
            sigma_store=(
                self._sigma_store if self._sigma_store is not None else np.zeros((0, 0))
            ),
        )

    def load(self, path: Union[str, Path]) -> None:
        p = Path(path)
        base = p.with_suffix("")
        if base.suffix == ".npz":
            base = base.with_suffix("")
        self._index = faiss.read_index(str(base) + ".faiss")
        meta = np.load(str(base) + "_meta.npz", allow_pickle=True)
        self._ids = meta["ids"]
        self.metric = str(meta["metric"].item())
        self.rerank = bool(meta["rerank"].item())
        zs = meta["z_store"]
        ss = meta["sigma_store"]
        self._z_store = zs if zs.size > 0 else None
        self._sigma_store = ss if ss.size > 0 else None
        self._dim = self._index.d

    def __len__(self) -> int:
        return self._index.ntotal if self._index is not None else 0
