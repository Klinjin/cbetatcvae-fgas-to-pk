"""Tests for the outer-bin f_gas reconstructor (``fgas_spk.outer_profile``).

Coverage: (i) the outer-bin radius law reproduces the design's endpoints and
log spacing; (ii) ``mpch_per_arcmin`` matches the closed-form Einstein-de Sitter
distance when radiation is switched off, and radiation shortens the distance;
(iii) with heavy-tailed (Student-t) stack noise plus gross failed-stack
outliers, the Student-t GP recovers the noise-free synthetic inner -> outer map,
learns a heavy tail (small nu) and a scale near the true one, gives calibrated
predictive intervals, and survives a save / load round-trip.
"""

import numpy as np
import pytest

from fgas_spk.outer_profile import (
    INNER_RADII_ARCMIN,
    N_OUTER,
    OUTER_EDGE_MPCH,
    OuterProfileReconstructor,
    mpch_per_arcmin,
    outer_radii,
)


def test_outer_radii_law():
    c = np.array([0.17, 0.35])
    r_m, r_a = outer_radii(c)
    assert r_m.shape == (2, N_OUTER)
    np.testing.assert_allclose(r_m[:, -1], OUTER_EDGE_MPCH)
    np.testing.assert_allclose(r_a, r_m / c[:, None])
    # Constant log step from r(6') = 6' c (exclusive) to the 20 Mpc/h edge.
    steps = np.diff(np.log(np.c_[INNER_RADII_ARCMIN[-1] * c, r_m]), axis=1)
    np.testing.assert_allclose(steps, steps[:, :1] * np.ones_like(steps))


def test_mpch_per_arcmin_eds_closed_form():
    z = 0.5
    chi = 2 * 2997.92458 * (1 - 1 / np.sqrt(1 + z))  # EdS, Mpc/h
    got = mpch_per_arcmin(z, omega_m=1.0, h=0.7, tcmb0=0.0)
    np.testing.assert_allclose(got, chi * np.pi / 10800, rtol=1e-10)
    assert mpch_per_arcmin(z, 0.3, 0.7) < mpch_per_arcmin(z, 0.3, 0.7, tcmb0=0.0)


def test_student_t_gp_recovers_map_under_heavy_tails(tmp_path):
    pytest.importorskip("gpytorch")
    rng = np.random.default_rng(1)
    n = 400
    a, b = rng.uniform(0.3, 0.9, n), rng.uniform(-0.5, 0.5, n)
    c = rng.uniform(0.17, 0.18, n)
    inner = a[:, None] + 0.05 * b[:, None] * np.linspace(0, 1, 9)
    x_out = np.linspace(0, 1, N_OUTER)
    truth = a[:, None] + 0.1 * np.sin(3 * b)[:, None] * x_out + 0.2 * np.log(c / 0.175)[:, None]
    scale, nu = 0.01, 3.0
    outer = truth + scale * rng.standard_t(nu, truth.shape)
    outer[:8] += rng.choice([-20.0, 10.0], (8, N_OUTER))  # failed stacks
    n_tr = 320
    m = OuterProfileReconstructor(n_iter_exact=200, n_restarts=2, n_iter_t=1000).fit(
        inner[:n_tr], c[:n_tr], outer[:n_tr])
    mu, sd = m.predict(inner[n_tr:], c[n_tr:])
    assert np.sqrt(np.mean((mu - truth[n_tr:]) ** 2)) < scale
    nm = m.noise_model()
    assert np.all(nm["nu"] < 10)
    assert np.all((nm["scale"] > 0.5 * scale) & (nm["scale"] < 2 * scale))
    lo, hi = m.predictive_interval(inner[n_tr:], c[n_tr:], quantiles=(0.025, 0.975))
    cov = np.mean((outer[n_tr:] > lo) & (outer[n_tr:] < hi))
    assert 0.9 < cov < 0.99
    path = m.save(tmp_path / "gp.pt")
    mu2, sd2 = OuterProfileReconstructor.load(path).predict(inner[n_tr:], c[n_tr:])
    np.testing.assert_allclose(mu2, mu)
    np.testing.assert_allclose(sd2, sd)
