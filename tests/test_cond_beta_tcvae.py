"""Tests for the conditional-prior beta-TCVAE (``fgas_spk.models.cond_beta_tcvae``).

Covers: the prior-whitened MSS decomposition telescopes exactly to the
single-sample KL to the *conditional* prior, it reduces term-for-term to the
parent's fixed-prior decomposition at ``mu_p = 0, logvar_p = 0``, MSS is exact
at ``M = N``; and the fitted model's inference paths use the conditional prior
(predict = decode at ``mu_p``) and the baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fgas_spk.loader import TrainingData  # noqa: E402
from fgas_spk.models.base import REGISTRY  # noqa: E402
from fgas_spk.models.beta_tcvae import BetaTcVae  # noqa: E402
from fgas_spk.models.cond_beta_tcvae import CondBetaTcVae  # noqa: E402


def _training_data(n=40, n_radii=5, n_k=4, n_cond=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_radii))
    return TrainingData(
        X=X, X_cond=rng.random((n, n_cond)), X_params=None,
        y=X[:, :n_k] ** 2 + 0.05 * rng.standard_normal((n, n_k)),
        nd=rng.random(n), sim_index=np.arange(n), k=np.linspace(0.1, 1.0, n_k),
        radii_mpch=np.linspace(0.1, 3.0, n_radii), source_path="synthetic",
    )


def _random_gaussians(m=6, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu_q, lv_q = torch.randn(m, d, generator=g), 0.5 * torch.randn(m, d, generator=g)
    mu_p, lv_p = torch.randn(m, d, generator=g), 0.5 * torch.randn(m, d, generator=g)
    z = mu_q + torch.exp(0.5 * lv_q) * torch.randn(m, d, generator=g)
    return z.double(), mu_q.double(), lv_q.double(), mu_p.double(), lv_p.double()


def test_registered():
    assert REGISTRY["cond_beta_tcvae"] is CondBetaTcVae


@pytest.mark.parametrize("n_data", [6, 50])
def test_decomposition_telescopes_to_conditional_kl(n_data):
    z, mu_q, lv_q, mu_p, lv_p = _random_gaussians()
    model = CondBetaTcVae(latent_dim=3, device="cpu")
    mi, tc, dwkl = model._conditional_kl_decomposition(z, mu_q, lv_q, mu_p, lv_p, n_data)
    lq = BetaTcVae._log_normal(z, mu_q, lv_q).sum(1)
    lp = BetaTcVae._log_normal(z, mu_p, lv_p).sum(1)
    assert torch.allclose(mi + tc + dwkl, (lq - lp).mean(), atol=1e-10)


def test_reduces_to_fixed_prior_decomposition():
    z, mu_q, lv_q, _, _ = _random_gaussians(seed=1)
    model = CondBetaTcVae(latent_dim=3, device="cpu")
    zeros = torch.zeros_like(z)
    cond = model._conditional_kl_decomposition(z, mu_q, lv_q, zeros, zeros, 20)
    fixed = model._kl_decomposition(z, mu_q, lv_q, 20)
    for a, b in zip(cond, fixed):
        assert torch.allclose(a, b, atol=1e-12)


def test_mss_exact_at_full_batch():
    # With M = N the aggregate over u is the exact mixture (1/N) sum_m q(u|m).
    z, mu_q, lv_q, mu_p, lv_p = _random_gaussians(m=5, seed=2)
    model = CondBetaTcVae(latent_dim=3, device="cpu")
    mi, _, _ = model._conditional_kl_decomposition(z, mu_q, lv_q, mu_p, lv_p, 5)
    sd_p = torch.exp(0.5 * lv_p)
    u, mu_u, lv_u = (z - mu_p) / sd_p, (mu_q - mu_p) / sd_p, lv_q - lv_p
    dens = torch.stack([
        torch.stack([BetaTcVae._log_normal(u[i], mu_u[m], lv_u[m]).sum().exp()
                     for m in range(5)]).mean() for i in range(5)])
    log_q_cond = BetaTcVae._log_normal(u, mu_u, lv_u).sum(1)
    assert torch.allclose(mi, (log_q_cond - dens.log()).mean(), atol=1e-10)


@pytest.mark.parametrize("recurrent", [True, False])
def test_fit_predict_uses_conditional_prior(recurrent):
    td = _training_data()
    model = CondBetaTcVae(latent_dim=2, hidden=16, n_layers=1, epochs=4, batch_size=8,
                          recurrent=recurrent, val_frac=0.2, seed=0, device="cpu")
    model.fit(td)
    pred = model.predict(td.X, td.X_cond)
    assert pred.shape == td.y.shape and np.all(np.isfinite(pred))
    mu_p, sd_p = model.latent_dists(td.X, td.X_cond)
    # The prior depends on the input (not the parent's constant N(0, I)).
    assert np.ptp(mu_p, axis=0).max() > 0
    assert np.allclose(model.decode(mu_p, td.X, td.X_cond), pred, atol=1e-5)
    samples = model.predict_samples(td.X, td.X_cond, n_samples=3)
    assert samples.shape == (3,) + td.y.shape
    h = model.history[-1]
    assert np.isclose(h["kl"], h["mi"] + h["tc"] + h["dwkl"])
    if recurrent:
        assert model.predict_baseline(td.X, td.X_cond).shape == td.y.shape
        assert h["baseline_nll"] is not None and h["val_baseline_nll"] is not None
    else:
        assert model._baseline is None and h["baseline_nll"] is None


def test_baseline_detach_blocks_kl_gradient():
    td = _training_data()
    model = CondBetaTcVae(latent_dim=2, hidden=8, n_layers=1, epochs=1, batch_size=8,
                          val_frac=0.0, seed=0, device="cpu")
    model.fit(td)
    x = torch.as_tensor(model._apply_norm(td.X, model._x_mean, model._x_std), dtype=torch.float32)
    c = torch.as_tensor(model._context(td.X_cond, None), dtype=torch.float32)
    for net in model._networks().values():
        net.zero_grad(set_to_none=True)  # drop gradients left by the last training step
    _, mu_p, lv_p = model._prior_and_baseline(x, c)
    (mu_p.sum() + lv_p.sum()).backward()
    assert all(p.grad is None for p in model._baseline.parameters())


def test_rejects_prior_without_input():
    td = _training_data()
    td = TrainingData(X=td.X, X_cond=None, X_params=None, y=td.y, nd=td.nd,
                      sim_index=td.sim_index, k=td.k, radii_mpch=td.radii_mpch,
                      source_path="synthetic")
    model = CondBetaTcVae(use_profile=False, epochs=1, device="cpu")
    with pytest.raises(ValueError, match="needs an input"):
        model.fit(td)
    with pytest.raises(ValueError, match="baseline_weight"):
        CondBetaTcVae(baseline_weight=-1.0)
