# -*- coding: utf-8 -*-
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from nwf.index_faiss import FAISSIndex

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
def test_faiss_index_l2() -> None:
    idx = FAISSIndex(metric="l2")
    v = np.random.randn(20, 10).astype(np.float32)
    idx.add(v)
    d, i = idx.search(v[0], k=5)
    assert len(d) == 5
    assert 0 in i


@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
def test_faiss_save_load() -> None:
    idx = FAISSIndex(metric="l2")
    idx.add(np.random.randn(10, 5).astype(np.float32))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "idx"
        idx.save(p)
        idx2 = FAISSIndex(metric="l2")
        idx2.load(p)
        assert len(idx2) == 10
