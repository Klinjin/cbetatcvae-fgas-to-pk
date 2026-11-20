"""Reconstruct the outer f_gas bins of the arcmin design from the inner bins.

The eighth-gen ARCMIN radial design (``test_CAMELS_eighth_gen.ipynb``) has 18
bins per profile:

* bins 1-9 sit at fixed angular radii, 1, 1.625, ..., 6.0 arcmin, identical in
  every simulation and the only range covered by the real (DESI x ACT / HSC)
  measurements;
* bins 10-18 are log-spaced in comoving radius from r(6') to an edge fixed at
  20 comoving Mpc/h,

      r_j = r6 * (20 / r6) ** (j / 9),   j = 1, ..., 9,   r6 = 6' * c,

  where ``c`` [comoving Mpc/h per arcmin] = D_M(z) * h * pi / 10800 is set by
  the cosmology and redshift (verified to machine precision against the
  producer's per-sim ``r12_to_mpch`` with astropy ``FlatLambdaCDM(H0=100 h,
  Om0, Tcmb0=2.7255 K)`` at each snapshot's exact redshift).

Because the outer radii vary with cosmology, the reconstruction is conditioned
on the geometry through ``log c``. For a real survey ``c`` is fixed by the
cosmology assumed for the angular -> comoving conversion; use
:func:`mpch_per_arcmin`. The design keeps the outer edge at 20 comoving Mpc/h,
so the outer radii of a reconstruction are those of :func:`outer_radii`. No
simulation (astrophysics / cosmology) parameter is an input: only quantities
that are available for real data.

The regressor is a Gaussian process per outer bin (GPyTorch, batched over the
nine bins): inputs are the nine inner f_gas values plus ``log c``,
standardised; each output bin is standardised and given its own linear mean
(the GP models the nonlinear residual about a linear map) and ARD Matern-5/2
kernel. The stack noise of the simulated outer bins is heavy-tailed: stacks
whose total Delta Sigma approaches zero at large R scatter the ratio to
|f_gas| >> 1, and the learned Student-t degrees of freedom come out at
nu ~ 2-6. The likelihood is therefore Student-t (scale and nu learned per
bin), fit as a variational GP whose inducing points are *all* training inputs
(no sparse approximation; the only approximation is the Gaussian variational
posterior required by the non-Gaussian likelihood). Its kernel and mean are
warm-started from an exact Gaussian-likelihood GP fit by maximising the exact
marginal likelihood.

Alternatives tested on 5-fold CV and rejected: a Gaussian likelihood (the tail
inflates the noise and, in the noisiest bins, the error, e.g. 10% vs 3.4%
median in the 20 Mpc/h bin of the 12-halo z=0.47 sample); iterative trimming
of outliers (comparable accuracy but removed 15-25% of simulations,
preferentially high-h ones, biasing the training set); fixed per-sim noise
from the 3-projection scatter (a 2-dof variance estimate over-weights stacks
whose scatter is small by chance; smoothing it via a regression on the inputs
explains only ~10% of its variance).

Usage on a real measurement (one model per redshift and halo sample; trained
models from ``scripts/reconstruct_outer_bins.py``)::

    from fgas_spk.outer_profile import OuterProfileReconstructor, mpch_per_arcmin, outer_radii
    m = OuterProfileReconstructor.load(".../gp_snap74_nd3.pt")
    c = mpch_per_arcmin(z=0.47, omega_m=0.3153, h=0.6736)   # assumed cosmology
    mean, std = m.predict(fgas_1_to_6_arcmin[None, :], c)     # (1, 9) each
    r_mpch, r_arcmin = outer_radii(c)                         # where they sit

The input profile must be on the same 1, 1.625, ..., 6 arcmin grid and use
the same f_gas definition (ionised / total Delta Sigma ratio over f_b) as the
simulations.

This is an opt-in module: torch, gpytorch and scipy are imported lazily inside
the functions that need them, so importing it costs numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_INNER = 9
N_OUTER = 9
INNER_RADII_ARCMIN = np.linspace(1.0, 6.0, N_INNER)  # 1, 1.625, ..., 6.0
OUTER_EDGE_MPCH = 20.0  # comoving Mpc/h, fixed outer edge of the design
TCMB0_K = 2.7255  # producer's FlatLambdaCDM CMB temperature
NEFF = 3.04  # astropy FlatLambdaCDM default, massless


def mpch_per_arcmin(z: float, omega_m: float, h: float, tcmb0: float = TCMB0_K,
                    neff: float = NEFF) -> float:
    """Comoving Mpc/h subtended by one arcmin at redshift ``z``.

    Reproduces the producer's conversion (``SimulationStacker`` ``utils.
    arcmin_to_comoving`` with astropy ``FlatLambdaCDM(H0=100 h, Om0,
    Tcmb0=2.7255 K)``, i.e. astropy's default Neff = 3.04 massless neutrinos)
    without importing astropy: flat, with photons (Omega_gamma from Tcmb0),
    massless neutrinos (Omega_nu = 7/8 (4/11)^(4/3) Neff Omega_gamma) and
    Omega_de = 1 - Omega_m - Omega_gamma - Omega_nu. Verified against the
    stored per-sim ``r12_to_mpch`` to machine precision.

    Args:
        z (float): Lens redshift.
        omega_m (float): Total matter density today.
        h (float): Dimensionless Hubble parameter.
        tcmb0 (float): CMB temperature [K].
        neff (float): Effective number of (massless) neutrino species.

    Returns:
        float: ``c`` in comoving Mpc/h per arcmin.
    """
    import scipy.constants as const
    from scipy.integrate import quad

    h0_si = 100.0 * h * 1e3 / (const.parsec * 1e6)
    rho_crit = 3.0 * h0_si**2 / (8.0 * np.pi * const.G)
    omega_gamma = 4.0 * const.Stefan_Boltzmann / const.c**3 * tcmb0**4 / rho_crit
    omega_r = omega_gamma * (1.0 + neff * 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0))
    omega_de = 1.0 - omega_m - omega_r

    def inv_e(zp):
        return 1.0 / np.sqrt(omega_m * (1 + zp) ** 3 + omega_r * (1 + zp) ** 4 + omega_de)

    chi_mpch = quad(inv_e, 0.0, z, epsabs=0.0, epsrel=1e-13)[0] * const.c / 1e5  # c/H0 in Mpc/h
    return float(chi_mpch * np.pi / 10800.0)


def outer_radii(c_mpch_per_arcmin: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Radii of the nine outer bins for a given angular -> comoving factor.

    Args:
        c_mpch_per_arcmin (array-like): ``c`` per profile, shape (N,) or scalar.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(r_mpch, r_arcmin)``, each (N, 9).
    """
    c = np.atleast_1d(np.asarray(c_mpch_per_arcmin, dtype=float))[:, None]
    r6 = INNER_RADII_ARCMIN[-1] * c
    j = np.arange(1, N_OUTER + 1)[None, :]
    r_mpch = r6 * (OUTER_EDGE_MPCH / r6) ** (j / N_OUTER)
    return r_mpch, r_mpch / c


def _features(fgas_inner: np.ndarray, c: np.ndarray) -> np.ndarray:
    fgas_inner = np.asarray(fgas_inner, dtype=float)
    c = np.atleast_1d(np.asarray(c, dtype=float))
    if fgas_inner.ndim != 2 or fgas_inner.shape[1] != N_INNER:
        raise ValueError(f"fgas_inner must be (N, {N_INNER}); got {fgas_inner.shape}")
    if c.shape != (fgas_inner.shape[0],):
        raise ValueError(f"c must be (N,) = ({fgas_inner.shape[0]},); got {c.shape}")
    if not (np.all(np.isfinite(fgas_inner)) and np.all(c > 0)):
        raise ValueError("fgas_inner must be finite and c positive")
    return np.column_stack([fgas_inner, np.log(c)])


class _Batch:
    """Shared kernel / mean construction for the exact and variational GPs."""

    @staticmethod
    def modules(d: int, n_out: int):
        import gpytorch
        import torch

        batch = torch.Size([n_out])
        mean = gpytorch.means.LinearMean(d, batch_shape=batch)
        covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d, batch_shape=batch),
            batch_shape=batch)
        return mean, covar


def _exact_gp(x, y, n_out: int):
    """Batched exact GP with Gaussian likelihood (warm start)."""
    import gpytorch
    import torch

    class _GP(gpytorch.models.ExactGP):
        def __init__(self, x, y, lik):
            super().__init__(x, y, lik)
            self.mean_module, self.covar_module = _Batch.modules(x.shape[-1], n_out)

        def forward(self, xb):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(xb), self.covar_module(xb))

    lik = gpytorch.likelihoods.GaussianLikelihood(
        batch_shape=torch.Size([n_out]),
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    return _GP(x, y, lik), lik


def _student_t_gp(z, n_out: int):
    """Batched variational GP, inducing points fixed at the training inputs ``z``."""
    import gpytorch
    import torch

    batch = torch.Size([n_out])

    class _VGP(gpytorch.models.ApproximateGP):
        def __init__(self, z):
            q = gpytorch.variational.CholeskyVariationalDistribution(z.shape[-2], batch_shape=batch)
            strategy = gpytorch.variational.VariationalStrategy(
                self, z, q, learn_inducing_locations=False)
            super().__init__(strategy)
            self.mean_module, self.covar_module = _Batch.modules(z.shape[-1], n_out)

        def forward(self, xb):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(xb), self.covar_module(xb))

    # nu > 2 keeps the noise variance finite.
    lik = gpytorch.likelihoods.StudentTLikelihood(
        batch_shape=batch, deg_free_constraint=gpytorch.constraints.GreaterThan(2.05),
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
    return _VGP(z), lik


@dataclass
class OuterProfileReconstructor:
    """Student-t GP map (inner nine f_gas bins, log c) -> outer nine f_gas bins.

    Fit one per (redshift, halo sample): the outer profile shape depends on the
    halo selection, so models are not shared across number densities.

    Attributes:
        n_iter_exact (int): Adam iterations per restart of the exact warm-start GP.
        n_restarts (int): Lengthscale restarts of the warm start; the best exact
            marginal likelihood per bin is kept.
        n_iter_t (int): Adam iterations of the variational Student-t fit.
        lr_exact (float): Adam learning rate, warm start.
        lr_t (float): Adam learning rate, Student-t fit.
        seed (int): Torch seed for the restarts.
        device (str | None): Torch device; None picks CUDA when available.
    """

    n_iter_exact: int = 400
    n_restarts: int = 3
    n_iter_t: int = 1500
    lr_exact: float = 0.05
    lr_t: float = 0.02
    seed: int = 0
    device: str | None = None
    meta: dict = field(default_factory=dict)
    _state: dict | None = field(default=None, repr=False)

    def _device(self):
        import torch

        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ------------------------------------------------------------------ fit
    def _fit_exact(self, x, y):
        """Exact Gaussian GP, best of ``n_restarts`` per bin (kernel + mean warm start)."""
        import gpytorch
        import torch

        dev, d = x.device, x.shape[-1]
        gen = torch.Generator().manual_seed(self.seed)
        best_loss = torch.full((N_OUTER,), np.inf, dtype=torch.float64)
        best_sd = None
        for restart in range(self.n_restarts):
            model, lik = _exact_gp(x, y, N_OUTER)
            model, lik = model.to(dev).double(), lik.to(dev).double()
            ls0 = (torch.full((N_OUTER, 1, d), float(np.sqrt(d)), dtype=torch.float64)
                   if restart == 0 else
                   torch.exp(torch.empty(N_OUTER, 1, d, dtype=torch.float64).uniform_(
                       np.log(0.5), np.log(20.0), generator=gen)))
            model.covar_module.base_kernel.lengthscale = ls0.to(dev)
            model.covar_module.outputscale = torch.full((N_OUTER,), 0.5, device=dev,
                                                        dtype=torch.float64)
            lik.noise = torch.full((N_OUTER, 1), 0.1, device=dev, dtype=torch.float64)
            model.train(); lik.train()
            opt = torch.optim.Adam(model.parameters(), lr=self.lr_exact)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
            for _ in range(self.n_iter_exact):
                opt.zero_grad()
                loss_b = -mll(model(x), y)  # (n_out,) per-bin negative MLL / N
                loss_b.sum().backward()
                opt.step()
            with torch.no_grad():
                loss_b = (-mll(model(x), y)).detach().cpu()
            sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if best_sd is None:
                best_sd, best_loss = sd, loss_b
                continue
            better = loss_b < best_loss
            for k, v in sd.items():  # learned tensors carry the bin batch dim first
                if v.ndim >= 1 and v.shape[0] == N_OUTER:
                    best_sd[k][better.to(v.device)] = v[better.to(v.device)]
            best_loss = torch.where(better, loss_b, best_loss)
        return best_sd

    def _fit_student_t(self, x, y, exact_sd):
        import gpytorch
        import torch

        model, lik = _student_t_gp(x, N_OUTER)
        model, lik = model.to(x.device).double(), lik.to(x.device).double()
        for name in ("mean_module.weights", "mean_module.bias",
                     "covar_module.raw_outputscale", "covar_module.base_kernel.raw_lengthscale"):
            dict(model.named_parameters())[name].data.copy_(exact_sd[name])
        # Student-t scale starts below the Gaussian noise std (the Gaussian fit
        # inflates it to absorb the tail); nu starts at a moderately heavy 4.
        g_noise = torch.nn.functional.softplus(exact_sd["likelihood.noise_covar.raw_noise"])
        lik.noise = (0.5 * g_noise.reshape(-1)).reshape(lik.noise.shape)
        lik.deg_free = torch.full_like(lik.deg_free, 4.0)
        model.train(); lik.train()
        opt = torch.optim.Adam([{"params": model.parameters()}, {"params": lik.parameters()}],
                               lr=self.lr_t)
        elbo = gpytorch.mlls.VariationalELBO(lik, model, num_data=y.shape[-1])
        for _ in range(self.n_iter_t):
            opt.zero_grad()
            loss = -elbo(model(x), y).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = (-elbo(model(x), y)).detach().cpu().numpy()
        return ({k: v.detach().cpu() for k, v in model.state_dict().items()},
                {k: v.detach().cpu() for k, v in lik.state_dict().items()}, final)

    def fit(self, fgas_inner: np.ndarray, c: np.ndarray, fgas_outer: np.ndarray):
        """Fit on simulated profiles.

        Args:
            fgas_inner (np.ndarray): (N, 9) f_gas at 1'-6'.
            c (np.ndarray): (N,) comoving Mpc/h per arcmin for each profile.
            fgas_outer (np.ndarray): (N, 9) f_gas in the outer bins.

        Returns:
            OuterProfileReconstructor: self.
        """
        import torch

        X = _features(fgas_inner, c)
        Y = np.asarray(fgas_outer, dtype=float)
        if Y.shape != (X.shape[0], N_OUTER) or not np.all(np.isfinite(Y)):
            raise ValueError(f"fgas_outer must be finite (N, {N_OUTER}); got {Y.shape}")
        dev = self._device()
        # Robust standardisation of the targets (median / scaled MAD): the
        # heavy-tailed stacks must not set the scale.
        x_mu, x_sd = X.mean(0), X.std(0)
        y_mu = np.median(Y, axis=0)
        y_sd = 1.4826 * np.median(np.abs(Y - y_mu), axis=0)
        x = torch.as_tensor((X - x_mu) / x_sd, dtype=torch.float64, device=dev)
        x = x.unsqueeze(0).expand(N_OUTER, *x.shape).contiguous()
        y = torch.as_tensor(((Y - y_mu) / y_sd).T.copy(), dtype=torch.float64, device=dev)

        exact_sd = self._fit_exact(x, y)
        model_sd, lik_sd, neg_elbo = self._fit_student_t(x, y, exact_sd)
        self._state = {"x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd,
                       "x_train": x[0].cpu(), "model": model_sd, "likelihood": lik_sd,
                       "neg_elbo_per_point": neg_elbo}
        self._model = None
        return self

    # -------------------------------------------------------------- predict
    def _ready_model(self):
        if self._state is None:
            raise RuntimeError("OuterProfileReconstructor is not fitted")
        if getattr(self, "_model", None) is None:
            dev = self._device()
            s = self._state
            z = s["x_train"].to(dev)
            z = z.unsqueeze(0).expand(N_OUTER, *z.shape).contiguous()
            model, lik = _student_t_gp(z, N_OUTER)
            model, lik = model.to(dev).double(), lik.to(dev).double()
            model.load_state_dict({k: v.to(dev) for k, v in s["model"].items()})
            lik.load_state_dict({k: v.to(dev) for k, v in s["likelihood"].items()})
            model.eval(); lik.eval()
            self._model, self._lik, self._dev = model, lik, dev
        return self._model, self._lik, self._dev

    def predict(self, fgas_inner: np.ndarray, c: np.ndarray):
        """Reconstruct the outer bins (posterior of the noise-free outer profile).

        Args:
            fgas_inner (np.ndarray): (M, 9) f_gas at 1'-6'.
            c (np.ndarray | float): (M,) or scalar Mpc/h per arcmin.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(mean, std)``, each (M, 9), on the
            outer radii given by :func:`outer_radii` for the same ``c``. ``std``
            is the latent-function uncertainty; for the spread of a noisy
            simulated stack use :meth:`predictive_interval`.
        """
        import torch

        fgas_inner = np.atleast_2d(fgas_inner)
        c = np.broadcast_to(np.asarray(c, dtype=float), (fgas_inner.shape[0],)).copy()
        s = self._state
        model, _, dev = self._ready_model()
        X = (_features(fgas_inner, c) - s["x_mu"]) / s["x_sd"]
        xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
        xt = xt.unsqueeze(0).expand(N_OUTER, *xt.shape)
        with torch.no_grad():
            f = model(xt)
            mean = f.mean.cpu().numpy().T
            std = f.variance.clamp_min(0).sqrt().cpu().numpy().T
        return mean * s["y_sd"] + s["y_mu"], std * s["y_sd"]

    def noise_model(self) -> dict:
        """Student-t stack noise per outer bin: ``scale`` (data units) and ``nu``."""
        _, lik, _ = self._ready_model()
        return {"scale": lik.noise.detach().reshape(-1).sqrt().cpu().numpy() * self._state["y_sd"],
                "nu": lik.deg_free.detach().reshape(-1).cpu().numpy()}

    def predictive_interval(self, fgas_inner: np.ndarray, c: np.ndarray,
                            quantiles=(0.16, 0.84), n_draws: int = 4000, seed: int = 0):
        """Quantiles of a new noisy simulated stack: latent normal + Student-t noise.

        Args:
            fgas_inner (np.ndarray): (M, 9) f_gas at 1'-6'.
            c (np.ndarray | float): (M,) or scalar Mpc/h per arcmin.
            quantiles (sequence of float): Probabilities in (0, 1).
            n_draws (int): Monte Carlo draws of the convolution.
            seed (int): RNG seed.

        Returns:
            np.ndarray: (len(quantiles), M, 9).
        """
        mean, std = self.predict(fgas_inner, c)
        nm = self.noise_model()
        rng = np.random.default_rng(seed)
        draws = (mean[None] + std[None] * rng.standard_normal((n_draws,) + mean.shape)
                 + nm["scale"] * rng.standard_t(nm["nu"], (n_draws,) + mean.shape))
        return np.quantile(draws, quantiles, axis=0)

    def hyperparameters(self) -> dict:
        """Fitted kernel / mean hyperparameters (standardised units) and noise model."""
        model, _, _ = self._ready_model()
        return {
            "lengthscale": model.covar_module.base_kernel.lengthscale.detach().squeeze(1).cpu().numpy(),
            "outputscale": model.covar_module.outputscale.detach().cpu().numpy(),
            "linear_weights": model.mean_module.weights.detach().squeeze(-1).cpu().numpy(),
            **self.noise_model(),
        }

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> Path:
        """Save the fitted model (training inputs included: inducing points)."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {k: getattr(self, k) for k in ("n_iter_exact", "n_restarts", "n_iter_t",
                                             "lr_exact", "lr_t", "seed")}
        torch.save({"config": cfg, "meta": self.meta, "state": self._state}, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "OuterProfileReconstructor":
        """Load a model written by :meth:`save`."""
        import torch

        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(**blob["config"], device=device, meta=blob["meta"])
        obj._state = blob["state"]
        obj._model = None
        return obj
