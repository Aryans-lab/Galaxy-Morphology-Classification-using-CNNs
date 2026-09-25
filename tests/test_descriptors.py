"""Sanity tests for the photometric descriptors on synthetic galaxies.

These verify the PHYSICS of each descriptor (concentration, asymmetry,
ellipticity) using toy images, not the full photutils pipeline. The low
level descriptor functions all take a (N, H, W) luminance stack.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from classical_baseline import (  # noqa: E402
    asymmetry,
    concentration,
    ellipticity,
    extract_descriptors,
    gini_m20,
)


def _lum(arr2d):
    """2-D float luminance -> (1, H, W) stack as the descriptor fns expect."""
    return arr2d[None]


def _gaussian(shape=(64, 64), sigma=2.0, sx=None, sy=None):
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H / 2 - 0.5, W / 2 - 0.5
    sx = sigma if sx is None else sx
    sy = sigma if sy is None else sy
    return np.exp(-(((x - cx) ** 2) / (2 * sx**2) + ((y - cy) ** 2) / (2 * sy**2)))


def _powerlaw(shape=(64, 64), sigma=8.0, k=1.0):
    """Axisymmetric profile I(r) ~ exp(-(r/sigma)^k).

    k=1 approximates an exponential disk (extended); k=4 a de Vaucouleurs
    elliptical (steep). Same scale, different SHAPE - exactly what the
    concentration index r90/r50 is supposed to measure.
    """
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H / 2 - 0.5, W / 2 - 0.5
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.exp(-(r / sigma) ** k)


def test_gini_higher_for_concentrated_source():
    g_point, _ = gini_m20(_lum(_gaussian(sigma=1.5)))
    g_diffuse, _ = gini_m20(_lum(_gaussian(sigma=12.0)))
    assert g_point[0] > g_diffuse[0]
    assert 0 <= g_diffuse[0] <= 1


def test_m20_more_negative_for_concentrated_source():
    _, m20_point = gini_m20(_lum(_gaussian(sigma=1.5)))
    _, m20_diffuse = gini_m20(_lum(_gaussian(sigma=12.0)))
    assert m20_point[0] < m20_diffuse[0]


def test_asymmetry_zero_for_symmetric_positive():
    sym = _gaussian(sigma=8.0)
    r_sym = asymmetry(_lum(sym))[0]
    assert r_sym < 0.02

    # add a bright lopsided arm -> asymmetry must increase
    arm = sym.copy()
    arm[:20, 40:64] += 0.5
    r_arm = asymmetry(_lum(arm))[0]
    assert r_arm > r_sym + 0.05


def test_concentration_higher_for_extended_disk():
    # exponential disk (k=1) must be more extended than a de Vaucouleurs
    # elliptical (k=4) at the same scale
    c_ellip = concentration(_lum(_powerlaw(k=4)))[0]
    c_disk = concentration(_lum(_powerlaw(k=1)))[0]
    assert c_disk > c_ellip


def test_ellipticity_higher_for_stretched_source():
    e_circ = ellipticity(_lum(_gaussian(sigma=6.0)))[0]
    e_stretch = ellipticity(_lum(_gaussian(sx=4.0, sy=12.0)))[0]
    assert e_stretch > e_circ
    assert e_circ < 0.05


def test_extract_descriptors_shapes_and_names():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    F, names = extract_descriptors(X, with_sersic=False)
    assert F.shape == (3, 5)
    assert names == ["gini", "m20", "asymmetry", "concentration", "ellipticity"]
    assert np.all(np.isfinite(F))
