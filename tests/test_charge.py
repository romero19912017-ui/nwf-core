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


def test_charge_alpha_positive() -> None:
    with pytest.raises(ValueError):
        Charge(z=np.array([1.0]), sigma=np.array([0.1]), alpha=0.0)


def test_charge_clip_sigma() -> None:
    c = Charge(z=np.array([1.0, 2.0]), sigma=np.array([1e-10, 0.5]))
    c2 = c.clip_sigma(min_val=1e-6)
    assert c2.sigma[0] >= 1e-6
    assert c2.sigma[1] == 0.5


def test_charge_alpha_default() -> None:
    c = Charge(z=np.array([1.0]), sigma=np.array([0.1]))
    assert c.alpha == 1.0


def test_charge_alpha_explicit() -> None:
    c = Charge(z=np.array([1.0]), sigma=np.array([0.1]), alpha=2.5)
    assert c.alpha == 2.5


def test_charge_alpha_serialization() -> None:
    c = Charge(z=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]), alpha=2.0)
    d = c.to_dict()
    assert "alpha" in d
    assert d["alpha"] == 2.0
    c2 = Charge.from_dict(d)
    assert c2.alpha == 2.0
    np.testing.assert_array_almost_equal(c.z, c2.z)


def test_charge_from_dict_backward_compat() -> None:
    d = {"z": [1.0, 2.0], "sigma": [0.5, 0.5]}
    c = Charge.from_dict(d)
    assert c.alpha == 1.0


def test_charge_to_vector_unchanged() -> None:
    c1 = Charge(z=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]))
    c2 = Charge(z=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]), alpha=3.0)
    np.testing.assert_array_almost_equal(c1.to_vector(), c2.to_vector())


def test_charge_with_alpha() -> None:
    c = Charge(z=np.array([1.0]), sigma=np.array([0.1]), alpha=1.0)
    c2 = c.with_alpha(5.0)
    assert c2.alpha == 5.0
    assert c.alpha == 1.0
    np.testing.assert_array_almost_equal(c.z, c2.z)
