"""Conditional-prior beta-TCVAE with a recurrent baseline connection (PyTorch).

:class:`~fgas_spk.models.beta_tcvae.BetaTcVae` follows Lin et al. (2026) in using
a **fixed** prior ``p(z) = N(0, I)``. That prior carries no information about the
f_gas(R) profile ``x``: at inference ``z ~ N(0, I)`` is drawn identically for every
input, so ``x`` reaches the prediction only through the decoder's direct input and
the latent cannot encode "which SP(k) is plausible *for this profile*". This module
replaces it by the conditional prior of a Sohn, Lee & Yan (2015) CVAE, including
their **recurrent connection**: a deterministic baseline network makes an initial
guess ``y_hat`` of SP(k) from the inputs, and the prior network sees ``x`` *and*
``y_hat`` (Sohn et al. Sec. 4, Fig. 2d; the baseline is their CNN, an MLP here
because the 9-bin profile has no spatial structure a convolution would exploit).

**Graphical model** (``ctx`` = standardised ``[X_cond | X_params]``, ``[x]`` present
iff ``use_profile``)::

    baseline     y_hat     = f_b([x], ctx)                       deterministic
    prior        p(z | [x], y_hat, ctx) = N(mu_p, diag e^{logvar_p})
    recognition  q(z | [x], y, ctx)     = N(mu_q, diag e^{logvar_q})   (training only)
    decoder      p(y | z, [x], ctx)     = N(dec(z, [x], ctx), diag sigma^2)

``recurrent=False`` drops the baseline and gives the plain Sohn prior
``p(z | [x], ctx)``.

**Why the KL decomposition is done in prior-whitened coordinates.** The
beta-TC split of Chen et al. (2018) needs an aggregate posterior. Writing it as a
conditional aggregate ``q(z | x) = E_{p(y|x)} q(z | x, y)`` -- i.e. replacing every
``q(z)``, ``q(z_j)``, ``p(z)`` by its ``| x`` version -- is exact but **degenerate
on this data**: every row has its own profile (each number-density row of a
simulation has a different f_gas(R)), so ``p(y | x)`` is one point per ``x``. Then
``q(z | x) = q(z | n)``, the conditional MI ``I(z; y | x)`` is identically 0, the
conditional TC of a diagonal Gaussian is identically 0, and all of the KL lands in
the dimension-wise term -- alpha and beta would weight nothing.

The non-degenerate exact form uses, per row ``n``, the affine bijection that
whitens ``z`` against that row's own prior::

    u = (z - mu_p(n)) / sigma_p(n)            (elementwise)
    q(u | n) = N( (mu_q(n) - mu_p(n)) / sigma_p(n),  sigma_q(n)^2 / sigma_p(n)^2 )
    p(z | x_n, y_hat_n)  ->  N(0, I)

KL is invariant under a bijection applied to both arguments, so
``KL(q(z|n) || p(z|x_n)) = KL(q(u|n) || N(0, I))`` row by row, and Chen et al.'s
identity applied to ``u`` (whose aggregate ``q(u) = (1/N) sum_n q(u|n)`` is a proper
mixture) is exact::

    E_n KL(q(z|n) || p(z|x_n)) = I_q(n; u)                         index-code MI
                               + KL(q(u) || prod_j q(u_j))          total correlation
                               + sum_j KL(q(u_j) || N(0, 1))        dimension-wise KL

``u`` is the part of the code that the conditional prior does *not* predict. MI
measures how much of it identifies the individual row; TC penalises dependence
among its dimensions; dwKL asks its population to be unit normal per dimension,
which is exactly the condition that sampling ``z ~ p(z | x, y_hat)`` at inference
lands where the decoder was trained. With ``mu_p = 0``, ``logvar_p = 0`` this is
term-for-term the fixed-prior decomposition of the parent class.

**Loss** (minimised) per minibatch of ``M`` rows from ``N`` training rows::

    y_hat_i            = f_b([x_i], ctx_i)
    mu_p, logvar_p     = prior([x_i], sg(y_hat_i), ctx_i)        (sg: stop-gradient)
    z_i ~ q(z | y_i, [x_i], ctx_i)                               (reparameterised)
    u_i = (z_i - mu_p,i) / sigma_p,i
    recon    = sigma-VAE NLL of dec(z_i, [x_i], ctx_i) vs y_i     (as parent, Eq. C12)
    MI, TC, dwKL = MSS estimates on (u_i, q(u|n_m))               (parent's estimator)
    base     = sigma-VAE NLL of y_hat_i vs y_i                    (own per-k variance)
    loss = recon + lambda_eff * (alpha MI + beta TC + gamma dwKL) + w_b * base

``MI + TC + dwKL = mean_i [log q(z_i|n_i) - log p(z_i|x_i, y_hat_i)]`` exactly (the
``sigma_p`` Jacobians cancel), so ``alpha = beta = gamma = lambda = 1`` is the plain
conditional ELBO of Sohn et al. plus the baseline's regression likelihood.

The baseline is fitted by its own Gaussian maximum likelihood (the sigma-VAE form,
per-k MSE variance, detached), so ``y_hat`` stays an estimate of ``E[y | x, ctx]``.
With ``baseline_detach=True`` (default) ``y_hat`` enters the prior with a
stop-gradient: the KL then trains the prior *given* the guess but cannot bend the
guess toward whatever makes the prior match the posterior, which would make
``y_hat`` a hidden prior feature rather than an initial guess of SP(k).
``baseline_detach=False`` trains it end-to-end through the KL as well. The
baseline's and the VAE's gradient norms are clipped separately (max 1 each), so
the baseline's early large regression gradients do not shrink the VAE's steps
through a shared clip.

**Minibatching.** Unchanged from the parent: minibatch stratified sampling
(Chen et al. 2018; Lin et al. Eq. D3) with ``N = n_train``, now applied to
``(u, mu_u, logvar_u)``; the per-row prior parameters enter every ``(i, m)`` pair
through ``mu_u, logvar_u`` of row ``m`` and through ``u_i``. Cost stays
``O(M^2 latent_dim)``; the log-of-an-unbiased-estimate bias shrinks with ``M``.
The baseline loss is a per-row mean and needs no stratification.

**Validation.** As the parent: whole holdout as one batch (``N = M = n_val``),
terminal ``kl_lambda``, fixed seeded posterior draw; ``val_loss`` now includes
``w_b * base`` so best-epoch selection uses the same functional as training.

**Inference** (inherited through :meth:`_prior_dist`): :meth:`predict` decodes at
the conditional-prior mean ``z = mu_p(x, y_hat, ctx)``; :meth:`predict_samples`
draws ``z ~ p(z | x, y_hat, ctx)`` and adds the sigma-VAE noise; ``latent_dists``
with ``y=None`` returns the conditional prior. :meth:`predict_baseline` returns
``y_hat`` itself (the deterministic initial guess) for comparison.

**History** adds to the parent's keys: ``baseline_nll`` (train minibatch average),
``val_baseline_nll``, ``recon_baseline`` (full-training-set sigma-VAE NLL of
``y_hat``), and ``recon_prior`` is now evaluated at ``z = mu_p`` (what
:meth:`predict` uses) instead of ``z = 0``.

Registered as ``"cond_beta_tcvae"``; like the parent, it is not imported by
:mod:`fgas_spk.models`, so a driver must ``import fgas_spk.models.cond_beta_tcvae``.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np

from fgas_spk.models.base import register
from fgas_spk.models.beta_tcvae import _EVAL_SEED_OFFSET, BetaTcVae

if TYPE_CHECKING:  # import only for type hints; never pulls torch/loader at runtime
    from fgas_spk.loader import TrainingData


@register("cond_beta_tcvae")
class CondBetaTcVae(BetaTcVae):
    """beta-TCVAE with a conditional prior ``p(z | x, y_hat, ctx)`` (see module docs).

    Args:
        baseline_weight (float): Weight ``w_b`` of the baseline's sigma-VAE NLL in
            the loss. Defaults to 1.0 (the joint likelihood of the two models).
        recurrent (bool): Build the baseline and feed its ``y_hat`` to the prior
            (Sohn et al.'s recurrent connection). False gives the prior
            ``p(z | [x], ctx)``. Defaults to True.
        baseline_detach (bool): Stop-gradient on ``y_hat`` where it enters the
            prior. Defaults to True.
        **kwargs: Forwarded to :class:`~fgas_spk.models.beta_tcvae.BetaTcVae`
            (``latent_dim``, ``hidden``, ``n_layers``, ``epochs``, ``lr``,
            ``weight_decay``, ``dropout``, ``batch_size``, ``kl_lambda``,
            ``alpha_mi``, ``beta_tc``, ``gamma_dwkl``, ``anneal_epochs``,
            ``anneal_cycle_epochs``, ``use_profile``, ``val_frac``,
            ``val_grouped``, ``obs_calibrate_on_val``, ``seed``, ``device``).

    Raises:
        ValueError: If ``baseline_weight`` is negative, or (at fit) if the prior
            would have no input (``use_profile=False`` and no context).
    """

    def __init__(
        self,
        baseline_weight: float = 1.0,
        recurrent: bool = True,
        baseline_detach: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if baseline_weight < 0:
            raise ValueError(f"baseline_weight must be non-negative; got {baseline_weight!r}.")
        self.baseline_weight = baseline_weight
        self.recurrent = recurrent
        self.baseline_detach = baseline_detach
        self._baseline = None

    # --- public API --------------------------------------------------------

    def fit(self, training_data: "TrainingData") -> None:
        """Fit by minimising the loss of the module docstring.

        Same data handling as :meth:`BetaTcVae.fit` (standardisation on the
        training rows, seeded/grouped holdout, best-epoch restore, post-fit
        sigma-VAE variance); torch is imported lazily here.

        Args:
            training_data (TrainingData): The model-ready arrays to fit on.

        Raises:
            ValueError: If fewer than two training rows remain, or if the prior
                would have no input.
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
                f"CondBetaTcVae needs at least 2 training rows; got {train_rows.size}."
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
        for net in self._networks().values():
            net.to(self._device)

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
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.seed) + _EVAL_SEED_OFFSET)
            eps_va = torch.randn((X_va.shape[0], self.latent_dim), generator=gen).to(
                self._device
            )

        nets = self._networks()
        vae_params = [p for name, net in nets.items() if name != "baseline"
                      for p in net.parameters()]
        base_params = list(nets["baseline"].parameters()) if "baseline" in nets else []
        params = vae_params + base_params
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
            sums = dict.fromkeys(("recon", "mi", "tc", "dwkl", "base", "loss"), 0.0)
            for start, stop in zip(bounds[:-1], bounds[1:]):
                idx = perm[start:stop]
                xb, yb = X_tr[idx], y_tr[idx]
                cb = c_tr[idx] if c_tr is not None else None
                optimizer.zero_grad()
                loss, terms = self._objective(xb, yb, cb, n_train, lam)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae_params, max_norm=1.0)
                if base_params:
                    torch.nn.utils.clip_grad_norm_(base_params, max_norm=1.0)
                optimizer.step()
                w = (stop - start) / n_train
                for key, val in terms.items():
                    sums[key] += w * float(val.item())
                sums["loss"] += w * float(loss.item())

            self._eval_mode()
            with torch.no_grad():
                y_hat_tr, mu_p_tr, _ = self._prior_and_baseline(X_tr, c_tr)
                mu_q_tr, _ = self._recognition_dist(X_tr, y_tr, c_tr)
                recon_post = float(
                    self._sigma_vae_nll(self._decode(mu_q_tr, X_tr, c_tr), y_tr).item()
                )
                recon_prior = float(
                    self._sigma_vae_nll(self._decode(mu_p_tr, X_tr, c_tr), y_tr).item()
                )
                recon_base = (float(self._sigma_vae_nll(y_hat_tr, y_tr).item())
                              if y_hat_tr is not None else None)
                if use_val:
                    val_loss_t, val_t_terms = self._objective(
                        X_va, y_va, c_va, X_va.shape[0], self.kl_lambda, eps=eps_va
                    )
                    val_loss = float(val_loss_t.item())
                    val_terms = {k: float(v.item()) for k, v in val_t_terms.items()}
                else:
                    val_loss = None
                    val_terms = {}
            self.history.append(
                {
                    "epoch": epoch,
                    "recon": sums["recon"],
                    "mi": sums["mi"],
                    "tc": sums["tc"],
                    "dwkl": sums["dwkl"],
                    "kl": sums["mi"] + sums["tc"] + sums["dwkl"],
                    "baseline_nll": sums["base"] if self.recurrent else None,
                    "lambda_eff": lam,
                    "beta_eff": lam,
                    "train_loss": sums["loss"],
                    "recon_post": recon_post,
                    "recon_prior": recon_prior,
                    "recon_baseline": recon_base,
                    "latent_gap": recon_prior - recon_post,
                    "val_loss": val_loss,
                    "val_mi": val_terms.get("mi"),
                    "val_tc": val_terms.get("tc"),
                    "val_dwkl": val_terms.get("dwkl"),
                    "val_baseline_nll": val_terms.get("base") if self.recurrent else None,
                }
            )
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                self.best_epoch = epoch
                best_state = {name: copy.deepcopy(net.state_dict())
                              for name, net in self._networks().items()}

        if best_state is not None:
            for name, net in self._networks().items():
                net.load_state_dict(best_state[name])

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
                    "CondBetaTcVae: obs_calibrate_on_val requires val_frac > 0; the "
                    "training-set sigma-VAE variance was kept.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def predict_baseline(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """The baseline's initial guess ``y_hat`` on the raw SP(k) scale.

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): As for :meth:`predict`.
            X_params (np.ndarray | None): As for :meth:`predict`.

        Returns:
            np.ndarray: ``y_hat``, shape (n_examples, n_k) or (n_examples,) for a
                1-D target.

        Raises:
            RuntimeError: If called before :meth:`fit` or with ``recurrent=False``.
        """
        import torch

        if self._baseline is None:
            raise RuntimeError(
                "CondBetaTcVae.predict_baseline needs a fitted model with recurrent=True."
            )
        self._check_modality(X_cond, X_params)
        x_t, cond_t = self._prepare_inputs(X, X_cond, X_params)
        self._eval_mode()
        with torch.no_grad():
            out = self._baseline_mean(x_t, cond_t).detach().cpu().numpy()
        assert self._y_mean is not None and self._y_std is not None
        y = self._invert_norm(out, self._y_mean, self._y_std)
        return y[:, 0] if self._y_was_1d else y

    # --- internals ---------------------------------------------------------

    def _objective(self, x, y, cond, n_data: int, lam: float, eps=None):
        """Loss and its terms for one batch (``eps`` fixes the posterior draw).

        Returns:
            tuple: ``(loss, {"recon", "mi", "tc", "dwkl", "base"})``; ``base`` is a
                zero tensor without the baseline.
        """
        import torch

        y_hat, mu_p, logvar_p = self._prior_and_baseline(x, cond)
        mu_q, logvar_q = self._recognition_dist(x, y, cond)
        if eps is None:
            z = self._reparameterise(mu_q, logvar_q)
        else:
            z = mu_q + torch.exp(0.5 * logvar_q) * eps
        recon = self._sigma_vae_nll(self._decode(z, x, cond), y)
        mi, tc, dwkl = self._conditional_kl_decomposition(
            z, mu_q, logvar_q, mu_p, logvar_p, n_data
        )
        loss = recon + lam * (self.alpha_mi * mi + self.beta_tc * tc + self.gamma_dwkl * dwkl)
        if y_hat is not None:
            base = self._sigma_vae_nll(y_hat, y)
            loss = loss + self.baseline_weight * base
        else:
            base = torch.zeros((), device=recon.device)
        return loss, {"recon": recon, "mi": mi, "tc": tc, "dwkl": dwkl, "base": base}

    def _conditional_kl_decomposition(self, z, mu_q, logvar_q, mu_p, logvar_p, n_data: int):
        """MSS ``(MI, TC, dwKL)`` of the KL to the conditional prior (module docs).

        Maps each row to its prior-whitened coordinates ``u = (z - mu_p) /
        sigma_p``, under which ``q(u | n) = N((mu_q - mu_p) / sigma_p,
        sigma_q^2 / sigma_p^2)`` and the prior is ``N(0, I)``, then applies the
        parent's fixed-prior estimator unchanged. The three terms sum to
        ``mean_i [log q(z_i | n_i) - log p(z_i | x_i, y_hat_i)]`` exactly.
        """
        import torch

        inv_sd_p = torch.exp(-0.5 * logvar_p)
        u = (z - mu_p) * inv_sd_p
        mu_u = (mu_q - mu_p) * inv_sd_p
        logvar_u = logvar_q - logvar_p
        return self._kl_decomposition(u, mu_u, logvar_u, n_data)

    def _build_modules(self, n_profile: int, n_context: int, n_k: int) -> None:
        """Build recognition, conditional prior, decoder, baseline and sigma^2 buffer.

        - recognition ``q(z|[x],y,ctx)``: ``n_x + n_k + n_context -> 2*latent``;
        - prior ``p(z|[x],y_hat,ctx)``: ``n_x + [n_k] + n_context -> 2*latent``,
          logvar head zero-initialised (unit-variance start, as the cVAE);
        - decoder ``p(y|z,[x],ctx)``: ``latent + n_x + n_context -> n_k``;
        - baseline ``f_b([x],ctx)``: ``n_x + n_context -> n_k`` (``recurrent`` only).
        """
        import torch

        n_x = n_profile if self.use_profile else 0
        if n_x + n_context == 0:
            raise ValueError(
                "CondBetaTcVae: the conditional prior needs an input; with "
                "use_profile=False supply X_cond and/or X_params."
            )
        self._recognition = self._mlp(n_x + n_k + n_context, 2 * self.latent_dim)
        self._prior = self._mlp(n_x + (n_k if self.recurrent else 0) + n_context,
                                2 * self.latent_dim)
        self._decoder = self._mlp(self.latent_dim + n_x + n_context, n_k)
        self._baseline = self._mlp(n_x + n_context, n_k) if self.recurrent else None
        final = self._prior[-1]
        with torch.no_grad():
            final.weight[self.latent_dim:].zero_()
            final.bias[self.latent_dim:].zero_()
        self._obs_logvar = torch.zeros(n_k, device=self._device)
        self._obs_raw_df = None

    def _networks(self) -> dict:
        """The trainable networks by name (the baseline only when ``recurrent``)."""
        nets = {"recognition": self._recognition, "prior": self._prior,
                "decoder": self._decoder}
        if self._baseline is not None:
            nets["baseline"] = self._baseline
        return nets

    def _baseline_mean(self, x, cond):
        """Baseline initial guess ``y_hat = f_b([x], ctx)`` (standardised scale)."""
        import torch

        parts = ([x] if self.use_profile else []) + ([] if cond is None else [cond])
        assert self._baseline is not None
        return self._baseline(torch.cat(parts, dim=1))

    def _prior_and_baseline(self, x, cond):
        """``(y_hat, mu_p, logvar_p)``; ``y_hat`` is None without the baseline."""
        import torch

        y_hat = self._baseline_mean(x, cond) if self._baseline is not None else None
        parts = [x] if self.use_profile else []
        if y_hat is not None:
            parts.append(y_hat.detach() if self.baseline_detach else y_hat)
        if cond is not None:
            parts.append(cond)
        assert self._prior is not None
        mu, logvar = torch.chunk(self._prior(torch.cat(parts, dim=1)), 2, dim=1)
        return y_hat, mu, torch.clamp(logvar, -8.0, 8.0)

    def _prior_dist(self, x, cond):
        """Conditional prior ``p(z|[x],y_hat,ctx)`` -> ``(mu_p, logvar_p)``.

        Overrides the parent's fixed ``N(0, I)``; every inherited inference path
        (:meth:`predict`, :meth:`predict_samples`, :meth:`latent_dists`, the
        held-out calibration) goes through here.
        """
        _, mu, logvar = self._prior_and_baseline(x, cond)
        return mu, logvar

    def _train_mode(self) -> None:
        """Put every network (incl. the baseline) in train mode."""
        for net in self._networks().values():
            net.train()

    def _eval_mode(self) -> None:
        """Put every network (incl. the baseline) in eval mode."""
        for net in self._networks().values():
            net.eval()
