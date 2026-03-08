# -*- coding: utf-8 -*-
import numpy as np
import pytest

from nwf.charge import Charge


def test_charge_basic() -> None:
    z = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.1, 0.2, 0.3])
    c = Charge(z=z, sigma=sigma)
    assert c.dim == 3


def test_charge_to_from_dict() -> None:
    c = Charge(z=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]))
    d = c.to_dict()
    c2 = Charge.from_dict(d)
    np.testing.assert_array_almost_equal(c.z, c2.z)
    np.testing.assert_array_almost_equal(c.sigma, c2.sigma)


def test_charge_to_from_vector() -> None:
    c = Charge(z=np.array([1.0, 2.0]), sigma=np.array([0.5, 1.0]))
    v = c.to_vector()
    assert v.shape == (4,)
    c2 = Charge.from_vector(v)
    np.testing.assert_array_almost_equal(c.z, c2.z)
    np.testing.assert_array_almost_equal(c.sigma, c2.sigma)


def test_charge_sigma_positive() -> None:
    with pytest.raises(ValueError):
        Charge(z=np.array([1.0]), sigma=np.array([0.0]))


def test_charge_clip_sigma() -> None:
    c = Charge(z=np.array([1.0, 2.0]), sigma=np.array([1e-10, 0.5]))
    c2 = c.clip_sigma(min_val=1e-6)
    assert c2.sigma[0] >= 1e-6
    assert c2.sigma[1] == 0.5
