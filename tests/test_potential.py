# -*- coding: utf-8 -*-
"""Tests for potential module."""

import numpy as np
import pytest

from nwf import Charge, potential, potential_batch


def test_potential_single_charge() -> None:
    c = Charge(z=np.array([0.0, 0.0]), sigma=np.array([1.0, 1.0]))
    phi = potential(np.array([0.0, 0.0]), [c])
    assert phi > 0.5
    phi_far = potential(np.array([10.0, 10.0]), [c])
    assert phi_far < 0.01


def test_potential_batch() -> None:
    charges = [
        Charge(z=np.array([0.0, 0.0]), sigma=np.array([1.0, 1.0])),
        Charge(z=np.array([1.0, 0.0]), sigma=np.array([1.0, 1.0])),
    ]
    z_all = np.stack([c.z for c in charges])
    s_all = np.stack([c.sigma for c in charges])
    r = np.array([[0.0, 0.0], [5.0, 5.0]])
    phi = potential_batch(r, z_all, s_all)
    assert len(phi) == 2
    assert phi[0] > phi[1]
