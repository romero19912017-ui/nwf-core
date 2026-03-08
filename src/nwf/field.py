# -*- coding: utf-8 -*-
"""Field - container for charges with labels and ids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np

from nwf.charge import Charge
from nwf.index import Index
from nwf.metric import mahalanobis_symmetric


class Field:
    """Container for NWF charges with labels and ids.

    Supports add, remove, get, build_index, save, load.
    """

    def __init__(self) -> None:
        self._charges: List[Charge] = []
        self._labels: List[Any] = []
        self._ids: List[Any] = []
        self._id_to_idx: dict = {}
        self._index: Optional[Index] = None

    def add(
        self,
        charges: Union[Charge, List[Charge]],
        labels: Optional[List[Any]] = None,
        ids: Optional[List[Any]] = None,
    ) -> None:
        """Add charge(s) to field."""
        if isinstance(charges, Charge):
            charges = [charges]
        n = len(charges)
        if labels is None:
            labels = [None] * n
        if ids is None:
            start = len(self._ids)
            ids = list(range(start, start + n))
        if len(labels) != n or len(ids) != n:
            raise ValueError("charges, labels, ids must have same length")
        for c, lab, iid in zip(charges, labels, ids):
            idx = len(self._charges)
            self._charges.append(c)
            self._labels.append(lab)
            self._ids.append(iid)
            self._id_to_idx[iid] = idx
        self._index = None

    def remove(self, ids: List[Any]) -> None:
        """Remove charges by ids."""
        to_remove = set(ids)
        new_charges, new_labels, new_ids = [], [], []
        for c, lab, iid in zip(self._charges, self._labels, self._ids):
            if iid not in to_remove:
                new_charges.append(c)
                new_labels.append(lab)
                new_ids.append(iid)
        self._charges = new_charges
        self._labels = new_labels
        self._ids = new_ids
        self._id_to_idx = {iid: i for i, iid in enumerate(new_ids)}
        self._index = None

    def get_charges(self) -> List[Charge]:
        """Return all charges."""
        return list(self._charges)

    def get_labels(self) -> List[Any]:
        """Return all labels."""
        return list(self._labels)

    def get_ids(self) -> List[Any]:
        """Return all ids."""
        return list(self._ids)

    def build_index(self, index: Index) -> None:
        """Build index from current charges. Uses to_vector() for storage."""
        vectors = np.stack([c.to_vector() for c in self._charges], axis=0)
        index.add(vectors, ids=np.array(self._ids))
        self._index = index

    def search(
        self,
        query: Union[Charge, np.ndarray],
        k: int = 10,
        metric: str = "symmetric",
    ) -> tuple:
        """Search k nearest charges. Returns (distances, indices, labels)."""
        if len(self._charges) == 0:
            raise RuntimeError("Field is empty")
        if isinstance(query, Charge):
            qz = query.z
            qs = query.sigma
        else:
            v = np.asarray(query)
            d = v.shape[0] // 2
            qz = v[:d]
            qs = np.exp(v[d:]) + 1e-10

        z_all = np.stack([c.z for c in self._charges], axis=0)
        s_all = np.stack([c.sigma for c in self._charges], axis=0)
        qz = np.atleast_2d(qz)
        qs = np.atleast_2d(qs)

        if metric == "symmetric":
            dists = mahalanobis_symmetric(qz, qs, z_all, s_all)
        else:
            raise ValueError("Only symmetric metric supported for Charge search")

        if dists.ndim == 1:
            dists = dists.reshape(1, -1)
        k_act = min(k, dists.shape[1])
        idx = np.argsort(dists, axis=1)[:, :k_act]
        dist_out = np.take_along_axis(dists, idx, axis=1)
        dist_out = dist_out.squeeze()
        idx = np.atleast_1d(idx.squeeze())
        if idx.ndim == 1:
            lab = [[self._labels[int(i)] for i in idx]]
        else:
            lab = [[self._labels[int(i)] for i in row] for row in idx]
        return dist_out, idx, lab

    def save(self, path: Union[str, Path]) -> None:
        """Save field to directory (charges + metadata)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        vectors = np.stack([c.to_vector() for c in self._charges], axis=0)
        np.save(p / "vectors.npy", vectors)
        alphas = [float(c.alpha) for c in self._charges]
        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {"labels": self._labels, "ids": self._ids, "alphas": alphas},
                f,
                ensure_ascii=False,
            )

    def load(self, path: Union[str, Path]) -> None:
        """Load field from directory. Backward compatible: missing alphas default to 1.0."""
        p = Path(path)
        vectors = np.load(p / "vectors.npy")
        with open(p / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        alphas = meta.get("alphas", [1.0] * len(vectors))
        if len(alphas) != len(vectors):
            alphas = [1.0] * len(vectors)
        self._charges = [
            Charge.from_vector(vectors[i], alpha=float(alphas[i]))
            for i in range(len(vectors))
        ]
        self._labels = meta["labels"]
        self._ids = meta["ids"]
        self._id_to_idx = {iid: i for i, iid in enumerate(self._ids)}
        self._index = None

    def __len__(self) -> int:
        return len(self._charges)

    def __iter__(self) -> Iterator[Tuple[Charge, Any, Any]]:
        """Iterate over (charge, label, id) tuples."""
        return iter(zip(self._charges, self._labels, self._ids))

    def __getitem__(self, idx: int) -> Charge:
        """Get charge by index."""
        return self._charges[idx]
