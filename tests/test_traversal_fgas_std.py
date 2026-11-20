"""Latent-traversal profile row with the fgas_std input X = [f_gas, log10 sigma_fgas].

The row must draw f_gas at the radii with sigma_fgas error bars (one errorbar call per
profile column), not the concatenated 2 x n_r vector, and must keep the plain line
whenever the extra columns are not exactly log10(X_std).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.axes import Axes  # noqa: E402

import fgas_spk.train as T  # noqa: E402
from fgas_spk.loader import TrainingData  # noqa: E402
from fgas_spk.models.cond_beta_tcvae import CondBetaTcVae  # noqa: E402


def _td(n=60, n_r=6, n_k=8, with_log_err=True, seed=0):
    rng = np.random.default_rng(seed)
    fgas = 0.5 + 0.4 * rng.random((n, n_r))
    sigma = 10 ** rng.uniform(-1.5, -0.5, (n, n_r))
    X = np.hstack([fgas, np.log10(sigma)]) if with_log_err else fgas
    k = np.logspace(-0.7, 1.0, n_k)
    y = 1.0 - 0.3 * fgas.mean(1, keepdims=True) * (k / k.max())[None, :] ** 1.5
    return TrainingData(
        X=X, X_cond=rng.random((n, 2)), X_params=None, y=y, nd=rng.random(n),
        sim_index=np.arange(n), k=k, radii_mpch=np.linspace(0.3, 2.0, n_r),
        source_path="synthetic", X_std=sigma,
    )


def _count_errorbars(monkeypatch):
    calls = []
    orig = Axes.errorbar

    def spy(self, x, y, *a, **kw):
        calls.append((np.asarray(x).copy(), np.asarray(y).copy(), np.asarray(kw.get("yerr")).copy()))
        return orig(self, x, y, *a, **kw)

    monkeypatch.setattr(Axes, "errorbar", spy)
    return calls


def _fit(td):
    model = CondBetaTcVae(latent_dim=2, hidden=16, n_layers=1, epochs=3, batch_size=16,
                          val_frac=0.0, seed=0, device="cpu")
    model.fit(td)
    return model


def test_profile_row_draws_fgas_with_sigma_error_bars(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    calls = _count_errorbars(monkeypatch)
    td = _td()
    out = T._write_latent_traversal_figure(_fit(td), td, "fgas_std")
    assert out is not None and Path(out).exists()
    n_r = td.radii_mpch.size
    assert len(calls) == 5  # one per profile column of the p10..p90 ladder
    for x, y, yerr in calls:
        assert np.allclose(x, td.radii_mpch)  # f_gas against the radii, not a bin index
        row = np.flatnonzero(np.all(np.isclose(td.X[:, :n_r], y), axis=1))
        assert row.size >= 1
        assert np.allclose(yerr, td.X_std[row[0]])  # linear sigma_fgas, not its log10


def test_profile_row_keeps_line_without_log_error_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    calls = _count_errorbars(monkeypatch)
    td = _td(with_log_err=False)  # X_std present but not an input column
    assert T._write_latent_traversal_figure(_fit(td), td, "fgas_only") is not None
    assert calls == []
