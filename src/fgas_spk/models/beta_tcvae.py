"""Conditional beta-TCVAE for SP(k) (= T^2) with a sigma-VAE likelihood (PyTorch).

A variant of :mod:`~fgas_spk.models.cvae` that follows the beta-TCVAE of Lin et al.
(2026, ApJL 996 L41, arXiv:2509.01881; Appendices C--E), which autoencodes the
squared transfer function ``T^2(k) = P_hyd / P_dmo`` -- this project's SP(k), up
to the DM-as-DMO proxy noted in CLAUDE.md. Two changes from the Sohn-style cVAE
are what the beta-TC loss requires:

1. **A fixed, factorised prior** ``p(z) = N(0, I) = prod_j p(z_j)``. The
   dimension-wise KL ``sum_j KL(q(z_j) || p(z_j))`` and the total correlation
   ``KL(q(z) || prod_j q(z_j))`` are defined against the *aggregate* posterior
   and a factorised reference, so the cVAE's learned, input-dependent prior
   network is dropped. Inference samples ``z ~ N(0, I)``.
2. **The ELBO's KL is decomposed** (Chen et al. 2018; Lin et al. Eq. D6) and each
   piece weighted separately.

**Networks** (shared ``n_layers``-deep GELU MLPs, as in the cVAE):

- recognition ``q(z | y, [x], ctx)`` -- input ``concat([X], y, ctx)``, output
  ``(mu_q, logvar_q)``; used in training and by :meth:`latent_dists` with ``y``;
- decoder ``p(y | z, [x], ctx)`` -- input ``concat(z, [X], ctx)``, output the
  standardised SP(k) mean of width ``n_k``.

``ctx`` is the cVAE's context: ``X_params`` (cosmology) and ``X_cond`` (e.g.
``[log10 n, z]`` as built by ``scripts/joint_cvae.py``), each standardised on its
own scale and concatenated -- so cosmology, redshift and number density condition
**both** the encoder and the decoder. ``use_profile`` (default True, as in the
cVAE) also feeds the f_gas(R) profile ``X`` to both networks; with
``use_profile=False`` the model is Lin et al.'s pure conditional autoencoder of
T^2, and :meth:`predict` then depends on the context alone (``X`` is accepted
for interface compatibility and ignored). Note that without the profile, a
number-density column carries no information about the target -- every
number-density row of a simulation shares one SP(k) -- so conditioning on it is
only meaningful with ``use_profile=True``.

**Loss** (to be minimised; the negative of Lin et al. Eq. D6) per minibatch of
``M`` rows drawn from ``N`` training rows::

    z_i ~ q(z | y_i, [x_i], ctx_i)                     (reparameterised)
    y'_i = decode(z_i, [x_i], ctx_i)
    sigma^2_k = mean_i (y'_ik - y_ik)^2                (stop-gradient; Eq. C12)
    recon = mean_i 0.5 * sum_k [ log sigma^2_k + (y'_ik - y_ik)^2 / sigma^2_k ]
    loss  = recon + lambda_eff * ( alpha * MI + beta * TC + gamma * dwKL )

with, estimated by minibatch stratified sampling (MSS; below),

    MI   = mean_i [ log q(z_i | n_i) - log q(z_i) ]          index-code MI
    TC   = mean_i [ log q(z_i) - sum_j log q(z_ij) ]         total correlation
    dwKL = mean_i [ sum_j log q(z_ij) - log p(z_i) ]         dimension-wise KL

``MI + TC + dwKL = mean_i [log q(z_i | n_i) - log p(z_i)]`` exactly (the
estimates telescope), i.e. the single-sample Monte-Carlo estimate of the ELBO's
KL; ``alpha = beta = gamma = lambda = 1`` is therefore the plain VAE objective
(Lin et al. Eq. C14) and is the default here -- Lin et al. tuned these by
Optuna over ``lambda in [1e-4, 1e-1]``, ``alpha in [0.1, 10]``, ``beta in
[0.1, 1]``, ``gamma in [1e-3, 10]`` (their Table 4) without reporting the
selected values, so no tuned setting is imported.

**The sigma-VAE likelihood** (Rybkin et al. 2021, as adopted by Lin et al. Eq.
C11--C13). ``p(y | z) = N(y', diag(sigma^2))`` with one variance per k bin, set
in closed form to the per-bin reconstruction MSE over the minibatch -- the
maximum-likelihood variance given the reconstructions -- and **detached**, so it
acts as a per-k weight on the squared error rather than a free parameter. The
``log sigma^2`` term therefore carries no gradient, but it is kept in the value
so ``recon`` is the actual negative log-likelihood (the additive
``0.5 * log 2*pi`` per bin is dropped, as in the cVAE's ``_nll``). Eq. C12 sets
the *variance* to the MSE; Lin et al.'s Appendix E writes ``sigma = MSE``
without the square, which is inconsistent with C12 and with maximum likelihood,
so the C12 form is used. After training, the variance that ships (read via
:meth:`obs_sigma` / :meth:`obs_variance`, and added as Gaussian observation noise
by :meth:`predict_samples`) is the same per-bin MSE evaluated over the **whole**
training set at the restored weights, with ``z`` drawn from the posterior exactly
as in the training objective (seeded). ``obs_calibrate_on_val`` optionally
replaces it by the cVAE's closed-form held-out calibration
(:meth:`~fgas_spk.models.cvae.Cvae._calibrate_obs_logvar`).

**Minibatch stratified sampling** (Chen et al. 2018; Lin et al. Eq. D3). The
aggregate posterior ``q(z) = (1/N) sum_n q(z | n)`` is estimated, for a ``z_i``
drawn from row ``i`` of the batch, as::

    q(z_i) ~= (1/N) q(z_i | i) + (N - 1) / (N (M - 1)) * sum_{m != i} q(z_i | m)

which is unbiased for ``q(z_i)`` given ``i`` (the ``M - 1`` other rows are a
uniform draw from the remaining ``N - 1``) and exact when ``M = N``. Each
marginal ``q(z_ij)`` uses the same weights on the per-dimension densities. The
logarithm of an unbiased estimate is biased low (Jensen), with the bias shrinking
as ``M`` grows; Lin et al. trained full-batch for this reason. The cost is
``O(M^2 * latent_dim)`` memory per step. ``batch_size=None`` selects full-batch
training (``M = N``); the default is a 512-row minibatch, a compromise between
the estimator's bias and the number of optimiser steps per epoch. A minibatch
needs ``M >= 2``; a size-1 remainder is merged into the previous batch.

**Schedule of ``lambda``** (``kl_lambda``). Without ``anneal_cycle_epochs`` it is
the cVAE's linear warm-up: ``lambda_eff`` ramps from 0 to ``kl_lambda`` over
``anneal_epochs`` (default ``epochs // 4``) and then holds. With
``anneal_cycle_epochs = C`` it is the cyclical schedule Lin et al. use (Fu et al.
2019): the ramp restarts every ``C`` epochs, rising over the first
``anneal_epochs`` of each cycle and holding for the rest. In Lin et al.'s Table 4
parameterisation the cycle is ``1 / (mu * nu)`` epochs and the ramp ``1 / mu``.

**Validation and best-epoch restore.** As in the cVAE (``val_frac``,
``val_grouped``), with the validation loss evaluated at the **terminal**
``kl_lambda`` on the full holdout as one batch (so MSS is exact there, with
``N = M = n_val``), using its own per-bin MSE as ``sigma^2`` -- the same
functional the training batches optimise -- and a fixed, seeded ``eps`` for the
posterior draw so successive epochs are compared on one sample.

**History** (one dict per epoch): ``recon``, ``mi``, ``tc``, ``dwkl``, ``kl`` (=
``mi + tc + dwkl``) and ``train_loss`` are row-weighted averages over the epoch's
optimisation minibatches; ``recon_post`` and ``recon_prior`` are full-training-set
evaluations (eval mode) of the sigma-VAE NLL with ``z = mu_q`` and ``z = 0``
(the prior mean, as :meth:`predict` uses), and ``latent_gap = recon_prior -
recon_post``; ``lambda_eff`` is the KL scale (duplicated as ``beta_eff``, the key
the runner's history figure reads); ``val_loss``, ``val_mi``, ``val_tc`` and
``val_dwkl`` are the holdout values (None without validation).

**Inherited inference API.** :meth:`predict` decodes at the prior mean ``z = 0``;
:meth:`predict_samples` draws ``z ~ N(0, I)`` and adds the sigma-VAE Gaussian
noise per draw; :meth:`decode` decodes a caller-supplied ``z``;
:meth:`latent_dists` / :meth:`latents` return the recognition posterior when a
target is passed and otherwise the fixed prior (zero mean, unit width, for every
row) -- for this model the informative latents are the posterior ones, so pass
``y``.

torch is imported lazily inside methods, as for every plugin, so importing this
module stays torch-free. It is deliberately **not** listed in
:mod:`fgas_spk.models`'s side-effect imports (the other model files are left
untouched), so ``REGISTRY["beta_tcvae"]`` exists only after
``import fgas_spk.models.beta_tcvae``; a run config with ``model: beta_tcvae``
needs that import in the driving script.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np

from fgas_spk.models.base import register
from fgas_spk.models.cvae import _OBS_VAR_FLOOR, Cvae

if TYPE_CHECKING:  # import only for type hints; never pulls torch/loader at runtime
    from fgas_spk.loader import TrainingData

_LOG_2PI = math.log(2.0 * math.pi)
# Seed offset for the fixed validation eps and the post-fit variance draw, so
# neither reuses the stream that seeds the weight initialisation.
_EVAL_SEED_OFFSET = 7919


@register("beta_tcvae")
class BetaTcVae(Cvae):
    """Conditional beta-TCVAE mapping (f_gas(R), cosmology, z, n) to p(SP(k)).

    Trains a recognition / decoder pair under the decomposed, ``lambda``-scaled
    ELBO of Lin et al. Eq. D6 with a fixed ``N(0, I)`` prior and a sigma-VAE
    Gaussian likelihood (see the module docstring). Shares the cVAE's context
    handling, standardisation, validation holdout and inference API.

    Args:
        latent_dim (int): Width of the latent. Defaults to 2 (Lin et al.).
        hidden (int): Hidden-layer width in each network. Defaults to 128.
        n_layers (int): Hidden layers in each network. Defaults to 2.
        epochs (int): Training epochs. Defaults to 500.
        lr (float): Adam/AdamW learning rate. Defaults to 5e-4.
        weight_decay (float): Weight decay; > 0 selects AdamW (as Lin et al.),
            0 plain Adam. Defaults to 1e-4.
        dropout (float): Dropout after each hidden activation. Defaults to 0.0.
        batch_size (int | None): Minibatch size ``M`` (capped at the training
            set); None trains full-batch, as Lin et al. Defaults to 512.
        kl_lambda (float): Overall KL scale ``lambda`` (terminal value of the
            schedule). Defaults to 1.0.
        alpha_mi (float): Weight ``alpha`` of the index-code mutual information.
            Defaults to 1.0.
        beta_tc (float): Weight ``beta`` of the total correlation. Defaults to
            1.0.
        gamma_dwkl (float): Weight ``gamma`` of the dimension-wise KL. Defaults
            to 1.0.
        anneal_epochs (int | None): Ramp length of ``lambda`` (per cycle when
            cyclical). None selects ``epochs // 4``; <= 0 disables the ramp.
            Defaults to None.
        anneal_cycle_epochs (int | None): Cycle length for the cyclical
            schedule; None keeps a single warm-up. Must be >= ``anneal_epochs``.
            Defaults to None.
        use_profile (bool): Feed the f_gas(R) profile to the encoder and
            decoder. False gives Lin et al.'s context-only conditional
            autoencoder of T^2. Defaults to True.
        val_frac (float): Internal validation fraction (0 disables). Defaults
            to 0.1.
        val_grouped (bool): Hold out whole simulations (needed when several
            rows share a simulation's target). Defaults to False.
        obs_calibrate_on_val (bool): Replace the training-set sigma^2 by the
            cVAE's closed-form held-out calibration after training. Defaults to
            False.
        seed (int): Reproducibility seed. Defaults to 0.
        device (str | None): Torch device; None auto-selects. Defaults to None.
        **_: Extra keyword arguments are accepted and ignored.

    Attributes:
        history (list[dict]): Per-epoch trace (see the module docstring).
        best_epoch (int | None): Restored epoch, or None without validation.

    Raises:
        ValueError: If a weight or ``kl_lambda`` is negative, or the cyclical
            schedule's ramp is longer than its cycle.
    """

    def __init__(
        self,
        latent_dim: int = 2,
        hidden: int = 128,
        n_layers: int = 2,
        epochs: int = 500,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.0,
        batch_size: int | None = 512,
        kl_lambda: float = 1.0,
        alpha_mi: float = 1.0,
        beta_tc: float = 1.0,
        gamma_dwkl: float = 1.0,
        anneal_epochs: int | None = None,
        anneal_cycle_epochs: int | None = None,
        use_profile: bool = True,
        val_frac: float = 0.1,
        val_grouped: bool = False,
        obs_calibrate_on_val: bool = False,
        seed: int = 0,
        device: str | None = None,
        **_: object,
    ) -> None:
        # The sigma-VAE variance is the cVAE's Gaussian observation-noise head
        # with a closed-form (not learned) value, so the head is always on and
        # Gaussian; the inherited sampling / accessors / calibration then apply.
        super().__init__(
            latent_dim=latent_dim,
            hidden=hidden,
            n_layers=n_layers,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            batch_size=batch_size,
            beta=kl_lambda,
            anneal_epochs=anneal_epochs,
            val_frac=val_frac,
            val_grouped=val_grouped,
            obs_noise_head=True,
            obs_likelihood="gaussian",
            obs_calibrate_on_val=obs_calibrate_on_val,
            seed=seed,
            device=device,
        )
        for name, value in (("kl_lambda", kl_lambda), ("alpha_mi", alpha_mi),
                            ("beta_tc", beta_tc), ("gamma_dwkl", gamma_dwkl)):
            if value < 0:
                raise ValueError(f"{name} must be non-negative; got {value!r}.")
        if anneal_cycle_epochs is not None:
            if anneal_cycle_epochs < 1:
                raise ValueError(
                    f"anneal_cycle_epochs must be >= 1; got {anneal_cycle_epochs!r}."
                )
            if self.anneal_epochs > anneal_cycle_epochs:
                raise ValueError(
                    f"anneal_epochs ({self.anneal_epochs}) exceeds "
                    f"anneal_cycle_epochs ({anneal_cycle_epochs}): lambda would "
                    "never reach kl_lambda within a cycle."
                )
        self.kl_lambda = kl_lambda
        self.alpha_mi = alpha_mi
        self.beta_tc = beta_tc
        self.gamma_dwkl = gamma_dwkl
        self.anneal_cycle_epochs = anneal_cycle_epochs
        self.use_profile = use_profile

    # --- public API --------------------------------------------------------

    def fit(self, training_data: "TrainingData") -> None:
        """Fit by minimising the negative of Lin et al.'s Eq. D6 (see module docs).

        Consumes ``X``, ``y`` (curve or 1-D ``single_k``), and whichever of
        ``X_cond`` / ``X_params`` are present (then required at predict). A
        seeded ``val_frac`` holdout selects the restored epoch; standardisation
        is fitted on the training rows only. torch is imported lazily here.

        Args:
            training_data (TrainingData): The model-ready arrays to fit on.

        Raises:
            ValueError: If fewer than two training rows remain (MSS needs
                ``M >= 2``).
        """
        import torch

        X = np.asarray(training_data.X, dtype=np.float64)
        y = np.asarray(training_data.y, dtype=np.float64)
        X_cond = training_data.X_cond
        X_params = training_data.X_params
        self._uses_cond = X_cond is not None
        if self._uses_cond:
            X_cond = np.asarray(X_cond, dtype=np.float64)
        self._uses_params = X_params is not None
        if self._uses_params:
            X_params = np.asarray(X_params, dtype=np.float64)
        self._y_was_1d = y.ndim == 1
        y2d = y[:, None] if self._y_was_1d else y

        n_total = X.shape[0]
        train_rows, val_rows, use_val = self._holdout_rows(training_data, n_total)
        if train_rows.size < 2:
            raise ValueError(
                f"BetaTcVae needs at least 2 training rows; got {train_rows.size}."
            )

        self._x_mean, self._x_std = self._fit_norm(X[train_rows])
        self._y_mean, self._y_std = self._fit_norm(y2d[train_rows])
        if X_cond is not None:
            self._cond_mean, self._cond_std = self._fit_norm(X_cond[train_rows])
        if X_params is not None:
            self._params_mean, self._params_std = self._fit_norm(X_params[train_rows])

        n_profile = X.shape[1]
        n_cond = X_cond.shape[1] if X_cond is not None else 0
        n_params = X_params.shape[1] if X_params is not None else 0
        n_k = y2d.shape[1]

        torch.manual_seed(self.seed)
        self._device = self._resolve_device()
        self._build_modules(n_profile, n_cond + n_params, n_k)
        assert self._recognition is not None and self._decoder is not None
        self._recognition.to(self._device)
        self._decoder.to(self._device)

        Xs = self._to_tensor(self._apply_norm(X, self._x_mean, self._x_std))
        ys = self._to_tensor(self._apply_norm(y2d, self._y_mean, self._y_std))
        context = self._context(X_cond, X_params)
        conds = self._to_tensor(context) if context is not None else None

        train_t = torch.as_tensor(train_rows, dtype=torch.long, device=self._device)
        X_tr, y_tr = Xs[train_t], ys[train_t]
        c_tr = conds[train_t] if conds is not None else None
        n_train = X_tr.shape[0]
        if use_val:
            val_t = torch.as_tensor(val_rows, dtype=torch.long, device=self._device)
            X_va, y_va = Xs[val_t], ys[val_t]
            c_va = conds[val_t] if conds is not None else None
            # One fixed posterior draw for the whole run: the validation loss is
            # then a deterministic function of the weights, comparable across
            # epochs. Drawn on a seeded CPU generator and moved (MPS-safe).
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.seed) + _EVAL_SEED_OFFSET)
            eps_va = torch.randn((X_va.shape[0], self.latent_dim), generator=gen).to(
                self._device
            )

        params = list(self._recognition.parameters()) + list(self._decoder.parameters())
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(params, lr=self.lr)

        batch_size = n_train if self.batch_size is None else min(self.batch_size, n_train)
        bounds = list(range(0, n_train, batch_size)) + [n_train]
        if len(bounds) > 2 and bounds[-1] - bounds[-2] < 2:
            del bounds[-2]  # merge a size-1 remainder: MSS needs M >= 2

        self.history = []
        self.best_epoch = None
        best_val = math.inf
        best_state = None
        for epoch in range(self.epochs):
            lam = self._lambda_eff(epoch)
            self._train_mode()
            perm = torch.randperm(n_train, device=self._device)
            sums = dict.fromkeys(("recon", "mi", "tc", "dwkl", "loss"), 0.0)
            for start, stop in zip(bounds[:-1], bounds[1:]):
                idx = perm[start:stop]
                xb, yb = X_tr[idx], y_tr[idx]
                cb = c_tr[idx] if c_tr is not None else None
                optimizer.zero_grad()
                mu_q, logvar_q = self._recognition_dist(xb, yb, cb)
                z = self._reparameterise(mu_q, logvar_q)
                recon = self._sigma_vae_nll(self._decode(z, xb, cb), yb)
                mi, tc, dwkl = self._kl_decomposition(z, mu_q, logvar_q, n_train)
                loss = recon + lam * (
                    self.alpha_mi * mi + self.beta_tc * tc + self.gamma_dwkl * dwkl
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                w = (stop - start) / n_train
                for key, val in (("recon", recon), ("mi", mi), ("tc", tc),
                                 ("dwkl", dwkl), ("loss", loss)):
                    sums[key] += w * float(val.item())

            self._eval_mode()
            with torch.no_grad():
                mu_q_tr, _ = self._recognition_dist(X_tr, y_tr, c_tr)
                recon_post = float(
                    self._sigma_vae_nll(self._decode(mu_q_tr, X_tr, c_tr), y_tr).item()
                )
                z0 = torch.zeros_like(mu_q_tr)
                recon_prior = float(
                    self._sigma_vae_nll(self._decode(z0, X_tr, c_tr), y_tr).item()
                )
                if use_val:
                    mu_q_va, logvar_q_va = self._recognition_dist(X_va, y_va, c_va)
                    z_va = mu_q_va + torch.exp(0.5 * logvar_q_va) * eps_va
                    recon_va = self._sigma_vae_nll(self._decode(z_va, X_va, c_va), y_va)
                    mi_va, tc_va, dwkl_va = self._kl_decomposition(
                        z_va, mu_q_va, logvar_q_va, X_va.shape[0]
                    )
                    val_terms = [float(t.item()) for t in (mi_va, tc_va, dwkl_va)]
                    val_loss = float(recon_va.item()) + self.kl_lambda * (
                        self.alpha_mi * val_terms[0]
                        + self.beta_tc * val_terms[1]
                        + self.gamma_dwkl * val_terms[2]
                    )
                else:
                    val_terms = [None, None, None]
                    val_loss = None
            self.history.append(
                {
                    "epoch": epoch,
                    "recon": sums["recon"],
                    "mi": sums["mi"],
                    "tc": sums["tc"],
                    "dwkl": sums["dwkl"],
                    "kl": sums["mi"] + sums["tc"] + sums["dwkl"],
                    "lambda_eff": lam,
                    "beta_eff": lam,
                    "train_loss": sums["loss"],
                    "recon_post": recon_post,
                    "recon_prior": recon_prior,
                    "latent_gap": recon_prior - recon_post,
                    "val_loss": val_loss,
                    "val_mi": val_terms[0],
                    "val_tc": val_terms[1],
                    "val_dwkl": val_terms[2],
                }
            )
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                self.best_epoch = epoch
                best_state = {
                    "recognition": copy.deepcopy(self._recognition.state_dict()),
                    "decoder": copy.deepcopy(self._decoder.state_dict()),
                }

        if best_state is not None:
            self._recognition.load_state_dict(best_state["recognition"])
            self._decoder.load_state_dict(best_state["decoder"])

        # Ship the sigma-VAE variance of the restored weights: per-bin MSE over
        # the whole training set, z drawn from the posterior as in training.
        self._eval_mode()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.seed) + 2 * _EVAL_SEED_OFFSET)
        with torch.no_grad():
            mu_q_tr, logvar_q_tr = self._recognition_dist(X_tr, y_tr, c_tr)
            eps = torch.randn(mu_q_tr.shape, generator=gen).to(self._device)
            z = mu_q_tr + torch.exp(0.5 * logvar_q_tr) * eps
            var = self._sigma_vae_variance(self._decode(z, X_tr, c_tr), y_tr)
            self._obs_logvar.copy_(torch.log(var))

        if self.obs_calibrate_on_val:
            if use_val:
                self._calibrate_obs_logvar(X_va, y_va, c_va)
            else:
                import warnings

                warnings.warn(
                    "BetaTcVae: obs_calibrate_on_val requires val_frac > 0; the "
                    "training-set sigma-VAE variance was kept.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _holdout_rows(self, training_data: "TrainingData", n_total: int):
        """Split the rows into the training set and the internal validation holdout.

        The same seeded scheme as :meth:`Cvae.fit` (reimplemented here so the
        cVAE is untouched): identical rows for identical ``seed`` / ``val_frac``
        / ``val_grouped``.

        Seeded validation holdout (backend-independent NumPy shuffle); the first
        ``ceil(val_frac * n)`` rows are held out, or with ``val_grouped`` the
        first ``ceil(val_frac * n_sims)`` simulations with all their rows.
        ``val_frac == 0`` (or a degenerate split leaving no training rows) means
        "no validation".

        Args:
            training_data (TrainingData): The bundle being fit (its ``sim_index``
                is read for the grouped holdout).
            n_total (int): Number of rows.

        Returns:
            tuple[np.ndarray, np.ndarray, bool]: ``(train_rows, val_rows,
                use_val)``; ``val_rows`` is empty when ``use_val`` is False.

        Raises:
            ValueError: If ``val_grouped`` leaves no training or validation
                simulations.
        """
        n_val = math.ceil(self.val_frac * n_total) if self.val_frac > 0 else 0
        use_val = 0 < n_val < n_total
        if use_val and self.val_grouped:
            # Hold out whole simulations so no validation target is also trained on.
            sims = np.unique(np.asarray(training_data.sim_index))
            n_val_sims = math.ceil(self.val_frac * len(sims))
            if not 0 < n_val_sims < len(sims):
                raise ValueError(
                    f"val_grouped with val_frac={self.val_frac} leaves no "
                    f"training or validation simulations ({len(sims)} sims)."
                )
            val_sims = np.random.default_rng(self.seed).permutation(sims)[:n_val_sims]
            is_val = np.isin(np.asarray(training_data.sim_index), val_sims)
            return np.where(~is_val)[0], np.where(is_val)[0], True
        if use_val:
            shuffled = np.random.default_rng(self.seed).permutation(n_total)
            return shuffled[n_val:], shuffled[:n_val], True
        return np.arange(n_total), np.empty(0, dtype=int), False

    def obs_variance(self) -> np.ndarray:
        """The sigma-VAE per-k variance ``sigma^2_k`` on the raw SP(k) scale.

        Returns:
            np.ndarray: ``obs_sigma() ** 2``, shape (n_k,).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        return self.obs_sigma() ** 2

    # --- internals ---------------------------------------------------------

    def _build_modules(self, n_profile: int, n_context: int, n_k: int) -> None:
        """Build the recognition and decoder networks and the variance buffer.

        The fixed ``N(0, I)`` prior has no parameters; an empty
        :class:`torch.nn.Module` stands in for the cVAE's prior network so the
        inherited train/eval switching and fitted-state checks apply unchanged.
        ``_obs_logvar`` is a plain tensor (not a Parameter): it is set in closed
        form, never by the optimiser.
        """
        import torch

        n_x = n_profile if self.use_profile else 0
        self._recognition = self._mlp(n_x + n_k + n_context, 2 * self.latent_dim)
        self._prior = torch.nn.Module()
        self._decoder = self._mlp(self.latent_dim + n_x + n_context, n_k)
        self._obs_logvar = torch.zeros(n_k, device=self._device)
        self._obs_raw_df = None

    def _recognition_dist(self, x, y, cond):
        """Recognition ``q(z|y,[x],ctx)`` -> ``(mu_q, logvar_q)`` (logvar clamped)."""
        import torch

        parts = ([x] if self.use_profile else []) + [y] + ([] if cond is None else [cond])
        assert self._recognition is not None
        mu, logvar = torch.chunk(self._recognition(torch.cat(parts, dim=1)), 2, dim=1)
        return mu, torch.clamp(logvar, -8.0, 8.0)

    def _prior_dist(self, x, cond):
        """The fixed prior ``N(0, I)`` as ``(mu, logvar)`` zeros, one row per input."""
        import torch

        zeros = torch.zeros((x.shape[0], self.latent_dim), dtype=x.dtype, device=x.device)
        return zeros, zeros.clone()

    def _decode(self, z, x, cond):
        """Decode ``concat(z, [x], ctx)`` to the standardised SP(k) mean."""
        import torch

        parts = [z] + ([x] if self.use_profile else []) + ([] if cond is None else [cond])
        assert self._decoder is not None
        return self._decoder(torch.cat(parts, dim=1))

    @staticmethod
    def _sigma_vae_variance(pred, y):
        """Per-bin MSE over the batch (Lin et al. Eq. C12), detached and floored."""
        return ((pred - y) ** 2).mean(dim=0).detach().clamp_min(_OBS_VAR_FLOOR)

    def _sigma_vae_nll(self, pred, y):
        """sigma-VAE Gaussian NLL, summed over k and averaged over the batch.

        ``0.5 * sum_k [log sigma^2_k + (pred - y)_k^2 / sigma^2_k]`` with
        ``sigma^2`` from :meth:`_sigma_vae_variance` (stop-gradient), i.e. the
        negative of Lin et al. Eq. C13 without the ``log 2*pi`` constant.
        """
        import torch

        var = self._sigma_vae_variance(pred, y)
        return 0.5 * ((pred - y) ** 2 / var + torch.log(var)).sum(dim=1).mean()

    @staticmethod
    def _log_normal(z, mu, logvar):
        """Elementwise ``log N(z; mu, e^logvar)`` (broadcasting)."""
        return -0.5 * (_LOG_2PI + logvar + (z - mu) ** 2 * (-logvar).exp())

    @staticmethod
    def _mss_log_weights(m: int, n: int, device):
        """``(m, m)`` log-weights of the MSS estimator (Lin et al. Eq. D3).

        Diagonal (the row ``z`` was drawn from): ``1/N``; off-diagonal:
        ``(N-1) / (N (M-1))``. Rows sum to 1, and ``M = N`` gives ``1/N``
        everywhere (the exact aggregate posterior).
        """
        import torch

        if m < 2:
            raise ValueError("Minibatch stratified sampling needs a batch of >= 2 rows.")
        if n < m:
            raise ValueError(f"Dataset size N={n} is smaller than the batch M={m}.")
        w = torch.full((m, m), (n - 1) / (n * (m - 1)), device=device)
        w.fill_diagonal_(1.0 / n)
        return w.log()

    def _kl_decomposition(self, z, mu, logvar, n_data: int):
        """MSS estimates of ``(MI, TC, dwKL)`` for posterior draws ``z``.

        Args:
            z: Posterior draws, shape ``(M, D)``; row ``i`` drawn from
                ``q(z | n_i)`` with parameters ``mu[i]``, ``logvar[i]``.
            mu: Posterior means, shape ``(M, D)``.
            logvar: Posterior log-variances, shape ``(M, D)``.
            n_data (int): Size ``N`` of the dataset the batch was drawn from.

        Returns:
            tuple: Scalar tensors ``(mi, tc, dwkl)``, batch means; their sum is
                ``mean_i [log q(z_i|n_i) - log p(z_i)]`` exactly.
        """
        import torch

        log_qz_cond = self._log_normal(z, mu, logvar).sum(dim=1)            # (M,)
        log_pz = self._log_normal(z, torch.zeros_like(z), torch.zeros_like(z)).sum(dim=1)
        # log q(z_i,d | n_j) for every pair (i, j) and dimension d: (M, M, D).
        pair = self._log_normal(z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0))
        log_w = self._mss_log_weights(z.shape[0], n_data, z.device)          # (M, M)
        log_qz = torch.logsumexp(pair.sum(dim=2) + log_w, dim=1)            # (M,)
        log_prod_qzj = torch.logsumexp(pair + log_w.unsqueeze(2), dim=1).sum(dim=1)
        mi = (log_qz_cond - log_qz).mean()
        tc = (log_qz - log_prod_qzj).mean()
        dwkl = (log_prod_qzj - log_pz).mean()
        return mi, tc, dwkl

    def _lambda_eff(self, epoch: int) -> float:
        """KL scale at ``epoch``: linear warm-up, or cyclical when configured."""
        if self.anneal_cycle_epochs is None:
            return self._beta_eff(epoch)
        if self.anneal_epochs <= 0:
            return self.kl_lambda
        return self.kl_lambda * min(1.0, (epoch % self.anneal_cycle_epochs) / self.anneal_epochs)
