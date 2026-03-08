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


def test_potential_alpha_linearity() -> None:
    c = Charge(z=np.array([0.0, 0.0]), sigma=np.array([1.0, 1.0]))
    r = np.array([0.0, 0.0])
    phi1 = potential(r, [c])
    c_alpha2 = c.with_alpha(2.0)
    phi2 = potential(r, [c_alpha2])
    assert abs(phi2 - 2 * phi1) < 1e-10


def test_potential_alpha_default_unchanged() -> None:
    c = Charge(z=np.array([0.0, 0.0]), sigma=np.array([1.0, 1.0]))
    c_explicit = Charge(z=np.array([0.0, 0.0]), sigma=np.array([1.0, 1.0]), alpha=1.0)
    r = np.array([0.5, 0.5])
    phi1 = potential(r, [c])
    phi2 = potential(r, [c_explicit])
    assert abs(phi1 - phi2) < 1e-12


def test_potential_batch_with_alpha() -> None:
    z_all = np.array([[0.0, 0.0], [1.0, 0.0]])
    s_all = np.array([[1.0, 1.0], [1.0, 1.0]])
    r = np.array([[0.0, 0.0]])
    phi_default = potential_batch(r, z_all, s_all)
    alpha_all = np.array([2.0, 1.0])
    phi_weighted = potential_batch(r, z_all, s_all, alpha_all=alpha_all)
    assert phi_weighted[0] > phi_default[0]
