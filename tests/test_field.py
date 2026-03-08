# -*- coding: utf-8 -*-
import tempfile
from pathlib import Path

import numpy as np

from nwf.charge import Charge
from nwf.field import Field


def test_field_add_search() -> None:
    f = Field()
    c1 = Charge(z=np.array([0.0, 0.0]), sigma=np.array([0.1, 0.1]))
    c2 = Charge(z=np.array([1.0, 1.0]), sigma=np.array([0.1, 0.1]))
    f.add([c1, c2], labels=[0, 1])
    dist, idx, lab = f.search(c1, k=2)
    assert len(lab[0]) == 2
    assert 0 in lab[0]


def test_field_save_load() -> None:
    f = Field()
    f.add(
        [Charge(z=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]))],
        labels=[42],
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "field"
        f.save(p)
        f2 = Field()
        f2.load(p)
        assert len(f2) == 1
        assert f2.get_labels()[0] == 42


def test_field_remove() -> None:
    f = Field()
    f.add(
        [Charge(z=np.array([1.0]), sigma=np.array([0.1])) for _ in range(3)],
        ids=[10, 20, 30],
    )
    f.remove([20])
    assert len(f) == 2
    assert f.get_ids() == [10, 30]


def test_field_iter_getitem() -> None:
    f = Field()
    c1 = Charge(z=np.array([0.0]), sigma=np.array([0.1]))
    c2 = Charge(z=np.array([1.0]), sigma=np.array([0.1]))
    f.add([c1, c2], labels=["a", "b"], ids=[0, 1])
    items = list(f)
    assert len(items) == 2
    assert items[0][1] == "a"
    assert f[0].z[0] == 0.0
