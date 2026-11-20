"""Apply a joint model (default ``experiments/runs/joint_cvae``) to an observed f_gas(theta) profile.

The joint model (``scripts/joint_cvae.py``) is one cVAE over both snapshots and all
number densities, conditioned on ``X_cond = [log10 n, z]`` plus the 5 cosmological
parameters. This script decodes an observed profile at one snapshot's redshift and,
in turn, at each of that snapshot's 5 training number densities, so the output is
directly comparable with the separate per-(snapshot, n) cVAEs decoded by
``scripts/predict_observed_fgas.py``. The decoding follows that script exactly:

* every seed of the run group is reloaded and must reproduce its recorded test RMSE;
* **at the measurement** -- ``predict_samples`` at ``fgas_obs`` (latent draws from the
  model's prior plus its observation-noise term), pooled over the seeds;
* **with the measurement error** -- the same, pooled over ``n_realisations`` draws of
  the profile from ``N(fgas_obs, diag(fgas_obs_err^2))`` (``q_err_diag``) and, when the
  input carries ``fgas_obs_cov``, from ``N(fgas_obs, fgas_obs_cov)`` (``q_err_cov``),
  with the same unit normals (shared ``--rng-seed``) as ``predict_observed_fgas.py``.

Other joint models trained on the same stacked data by ``joint_cvae.train`` are
decoded through the same code path, selected by ``--group`` / ``--run-cfg`` /
``--fgas-std`` (see :data:`MODELS` for the recorded combinations):

* ``beta_tcvae`` (``scripts/joint_beta_tcvae.py``) -- ``BetaTcVae`` subclasses ``Cvae``
  and inherits its ``predict`` / ``predict_samples``; with the fixed ``N(0, I)`` prior
  of the beta-TCVAE, ``predict_samples`` draws ``z ~ N(0, I)`` and adds the sigma-VAE
  Gaussian observation noise (the per-k training reconstruction variance), and
  ``predict`` decodes at ``z = 0``.
* ``cond_beta_tcvae`` (``scripts/joint_cond_beta_tcvae.py``) -- the beta-TCVAE with the
  conditional prior ``p(z | x, y_hat, ctx)`` and Sohn et al.'s recurrent baseline
  ``y_hat = f_b(x, ctx)``; ``predict_samples`` draws ``z`` from that conditional prior
  (evaluated at the observed profile) plus the sigma-VAE noise, and ``predict`` decodes
  at its mean ``mu_p``. The baseline ``y_hat`` is a training component only (it feeds
  the prior) and is never stored or reported as a prediction.
* ``joint_cvae_fgas_std`` (``scripts/joint_cvae_fgas_std.py``, ``--fgas-std``) -- the
  profile input is ``[f_gas, log10 sigma_fgas]`` (18 columns). For the observation the
  error columns are ``log10(fgas_obs_err)``. When the profile is redrawn from
  ``N(fgas_obs, C)`` only the f_gas columns are perturbed; the error columns stay at
  ``log10(fgas_obs_err)``, since the measurement error does not change between
  realisations. **Flag (rule 3):** the training sigma_fgas is a *halo-to-halo scatter*
  of the stacked simulated profiles (``np.std`` over halos x 3 projections, not divided
  by sqrt(N)), whereas ``fgas_obs_err`` is the measurement error of a stacked mean;
  the two are different quantities and are not rescaled to match. Their comparison,
  per (snapshot, n) and bin, is recorded in the sidecar (``fgas_err_vs_train_sigma``):
  the percentile rank of ``log10 fgas_obs_err`` within the training
  ``log10 sigma_fgas``, the ratio ``fgas_obs_err / median(sigma_fgas)``, and whether
  each bin lies inside the training range.

Conventions (flagged, passed in explicitly):

* ``fgas_obs`` is ``Delta Sigma_kSZ / Delta Sigma_GGL / f_b``; the joint models are trained
  on the f_b-normalised datasets (checked from the dataset metadata), so it is used as-is.
* The redshift entering ``X_cond`` is the training snapshot's (``DataConfig.redshift``:
  0.47 for snap74, 0.21 for snap82), not the observed sample's mean redshift, matching
  the snapshot-to-tracer pairing used for the separate models.
* The number density entering ``X_cond`` is one of the snapshot's training values; it
  labels which halo-sample stack the observed profile is treated as.
* The cosmology is fixed (default: Planck 2018 ``FIDUCIAL``).
* The observed angular bins must equal the training arcmin grid exactly.

The per-bin percentile rank of the observed f_gas profile is computed within the
training f_gas inputs of the same (snapshot, n) rows, as for the separate models.

Output keys are ``<group>/snap{S}_nd{i}__<quantity>`` with the quantities of
``predict_observed_fgas.py``; a ``.json`` sidecar records the provenance. With the
defaults (``--group joint_cvae``) the outputs are those of the original joint-cVAE
version of this script, unchanged.

Usage::

    python scripts/predict_observed_fgas_joint.py --obs obs.npz --out pred.npz --snap 74
    python scripts/predict_observed_fgas_joint.py --obs obs.npz --out pred.npz --snap 74 \\
        --group joint_cvae_fgas_std --fgas-std
    python scripts/predict_observed_fgas_joint.py --obs obs.npz --out pred.npz --snap 74 \\
        --group beta_tcvae --run-cfg scripts/configs/run/beta_tcvae_joint.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cosmo_agnostic_predictions import FIDUCIAL  # noqa: E402
from joint_cvae import (  # noqa: E402
    GROUP,
    RUN_CFG,
    X_PROFILE_DEFINITION_FGAS_STD,
    _configs,
    _joint_runs,
    _load_joint_model,
    _predict,
    _rmse,
    load_joint,
)
from predict_observed_fgas import PERCENTILES, _draw_realisations, _percentile_rank  # noqa: E402

from fgas_spk.train import _subset, grouped_split  # noqa: E402

# Recorded joint models: group -> (run config, fgas_std). Used to check that a
# --group is decoded with the data/run configuration it was trained with.
MODELS = {
    GROUP: (RUN_CFG, False),
    "joint_cvae_fgas_std": (RUN_CFG, True),
    "beta_tcvae": ("scripts/configs/run/beta_tcvae_joint.yaml", False),
    "cond_beta_tcvae": ("scripts/configs/run/cond_beta_tcvae_joint.yaml", False),
}


def _check_grid(meta: dict, n_bins: int, theta_obs: np.ndarray, snap: int) -> None:
    """Raise unless the snapshot's (cropped) training arcmin grid equals the observed one."""
    grid = meta["radial_grid"]
    radii = np.asarray(grid.get("radii_arcmin", grid.get("radii_arcmin_median")), dtype=float)[:n_bins]
    if radii.shape != theta_obs.shape or not np.allclose(radii, theta_obs):
        raise RuntimeError(f"snap{snap}: observed theta {theta_obs} != training grid {radii}.")


def _err_vs_train_sigma(sigma_train: np.ndarray, err_obs: np.ndarray) -> dict:
    """Compare the observed f_gas error with the training sigma_fgas of one (snapshot, n).

    Args:
        sigma_train: Training sigma_fgas (linear, > 0) of the (snapshot, n) rows,
            shape (n_rows, n_bins).
        err_obs: Observed ``fgas_obs_err`` (linear, > 0), shape (n_bins,).

    Returns:
        dict: Per-bin lists -- ``percentile_rank_log10_err`` (0--100, rank of
        ``log10 err_obs`` within ``log10 sigma_train``), ``ratio_err_to_median_sigma``,
        the training ``sigma_min`` / ``sigma_median`` / ``sigma_max``, ``inside_train_range``
        -- and ``n_bins_outside_train_range``.
    """
    log_tr, log_obs = np.log10(sigma_train), np.log10(err_obs)
    med = np.median(sigma_train, axis=0)
    inside = (log_obs >= log_tr.min(axis=0)) & (log_obs <= log_tr.max(axis=0))
    return {
        "percentile_rank_log10_err": _percentile_rank(log_tr, log_obs).round(2).tolist(),
        "ratio_err_to_median_sigma": (err_obs / med).tolist(),
        "sigma_min": sigma_train.min(axis=0).tolist(),
        "sigma_median": med.tolist(),
        "sigma_max": sigma_train.max(axis=0).tolist(),
        "fgas_obs_err": err_obs.tolist(),
        "inside_train_range": inside.tolist(),
        "n_bins_outside_train_range": int((~inside).sum()),
    }


def predict(snap: int, obs: dict, cosmology: dict, n_samples: int, n_realisations: int,
            n_per_realisation: int, rng_seed: int, group: str = GROUP, run_cfg: str = RUN_CFG,
            fgas_std: bool = False) -> tuple[dict, dict]:
    """Decode the observed profile at every training number density of ``snap``.

    Args:
        snap: Snapshot whose redshift and number densities condition the decode.
        obs: ``theta_arcmin``, ``fgas_obs``, ``fgas_obs_err`` and (or None) ``fgas_obs_cov``.
        cosmology: The conditioning cosmology (parameter name -> value).
        n_samples: Predictive draws per seed at ``fgas_obs``.
        n_realisations: Profile draws from ``N(fgas_obs, C)``.
        n_per_realisation: Predictive draws per profile draw and seed.
        rng_seed: Seed of the profile draws.
        group: Run group under ``experiments/runs`` (one run per seed 0/1/2).
        run_cfg: Run config the group was trained with (selects the outer split).
        fgas_std: The group was trained on ``X = [f_gas, log10 sigma_fgas]``.

    Returns:
        (arrays, sidecar): the npz arrays and the per-(snapshot, n) provenance.
    """
    dcs, rc = _configs(run_cfg, fgas_std=fgas_std)
    td, snap_row = load_joint(dcs)
    meta = td.meta[str(snap)]
    if not meta.get("fgas_fb_normalised", True):
        raise RuntimeError(f"snap{snap}: joint dataset is not f_b-normalised.")
    # Number of f_gas columns: the profile input is [f_gas, log10 sigma_fgas] with fgas_std.
    n_r = td.X.shape[1] // 2 if fgas_std else td.X.shape[1]
    if fgas_std and td.X.shape[1] != 2 * n_r:
        raise RuntimeError(f"fgas_std profile input has {td.X.shape[1]} columns, expected 2 x {n_r}.")
    _check_grid(meta, n_r, obs["theta_arcmin"], snap)
    z = float(dcs[snap].redshift)
    nds = np.unique(td.nd[snap_row == snap])

    theta = np.array([cosmology[n] for n in td.param_names], dtype=float)[None, :]
    if np.any(theta < td.X_params.min(axis=0)) or np.any(theta > td.X_params.max(axis=0)):
        raise RuntimeError(f"Cosmology {cosmology} lies outside the training box.")

    x_in, err_in = obs["fgas_obs"], obs["fgas_obs_err"]
    covs = {"diag": np.diag(err_in ** 2)}
    if obs.get("fgas_obs_cov") is not None:
        covs["cov"] = obs["fgas_obs_cov"]
    real = {name: _draw_realisations(x_in, c, n_realisations, np.random.default_rng(rng_seed))
            for name, c in covs.items()}
    if fgas_std:
        if not (np.all(np.isfinite(err_in)) and np.all(err_in > 0)):
            raise RuntimeError("fgas_obs_err must be finite and > 0 to enter as log10.")
        # Error columns: log10 of the measurement error, identical for every realisation.
        log_err = np.log10(err_in)
        x_model = np.concatenate([x_in, log_err])
        real_model = {name: np.hstack([r, np.tile(log_err, (r.shape[0], 1))]) for name, r in real.items()}
    else:
        x_model, real_model = x_in, real

    runs = _joint_runs(group)
    te = _subset(td, grouped_split(td.sim_index, rc.split)["test"])
    models = {}
    for seed, run_dir in runs.items():
        model = _load_joint_model(run_dir)
        rec = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
        if not np.isclose(_rmse(te.y, _predict(model, te)), rec, rtol=1e-4):
            raise RuntimeError(f"{run_dir.name}: reloaded test RMSE does not match the record")
        models[seed] = model

    out, side = {}, {}
    for i, n in enumerate(nds):
        rows = (snap_row == snap) & (td.nd == n)
        X_train = td.X[rows][:, :n_r]
        rank = _percentile_rank(X_train, x_in)
        frac_outside = {name: float(np.mean(np.any(
            (r < X_train.min(axis=0)) | (r > X_train.max(axis=0)), axis=1))) for name, r in real.items()}
        cond = np.array([[np.log10(n), z]])
        pooled_obs, point, pooled_err = [], [], {name: [] for name in real}
        for seed, model in models.items():
            point.append(np.asarray(model.predict(x_model[None, :], cond, theta), dtype=float)[0])
            pooled_obs.append(model.predict_samples(
                x_model[None, :], cond, theta, n_samples=n_samples, seed=seed)[:, 0])
            for name, x_real in real_model.items():
                pooled_err[name].append(model.predict_samples(
                    x_real, np.repeat(cond, n_realisations, axis=0), np.repeat(theta, n_realisations, axis=0),
                    n_samples=n_per_realisation, seed=seed).reshape(-1, td.k.size))
        key = f"{group}/snap{snap}_nd{i}"
        out[f"{key}__q_obs"] = np.percentile(np.concatenate(pooled_obs), PERCENTILES, axis=0)
        out[f"{key}__point_per_seed"] = np.array(point)
        out[f"{key}__train_percentile_rank"] = rank
        out[f"{key}__fgas_input"] = x_in
        out[f"{key}__fgas_input_err"] = err_in
        for name in real:
            out[f"{key}__q_err_{name}"] = np.percentile(np.concatenate(pooled_err[name]), PERCENTILES, axis=0)
        side[key] = {"run_ids": [d.name for d in runs.values()], "x_cond": [float(np.log10(n)), z],
                     "number_density": float(n), "fb_normalised": True,
                     "train_percentile_rank": rank.round(1).tolist(),
                     "frac_realisations_outside_train_range": frac_outside}
        if fgas_std:
            X_full = td.X[rows]
            diag = _err_vs_train_sigma(td.X_std[rows], err_in)
            out[f"{key}__err_percentile_rank_log10"] = np.asarray(diag["percentile_rank_log10_err"])
            out[f"{key}__err_ratio_to_median_sigma"] = np.asarray(diag["ratio_err_to_median_sigma"])
            side[key]["fgas_err_vs_train_sigma"] = diag
            # Fraction of realisations with any of the 18 input columns outside the training range.
            side[key]["frac_realisations_outside_train_range_full_input"] = {
                name: float(np.mean(np.any((r < X_full.min(axis=0)) | (r > X_full.max(axis=0)), axis=1)))
                for name, r in real_model.items()}
        print(key, json.dumps(side[key]), flush=True)
    out["k"] = np.asarray(td.k, dtype=float)
    return out, side


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--obs", required=True, help="npz with theta_arcmin, fgas_obs, fgas_obs_err.")
    ap.add_argument("--out", required=True, help="Output .npz path (a .json sidecar is written beside it).")
    ap.add_argument("--snap", type=int, required=True, choices=(74, 82), help="Snapshot whose z and n to condition on.")
    ap.add_argument("--cosmology", default=None,
                    help="JSON dict of the conditioning cosmology (default: Planck 2018 FIDUCIAL).")
    ap.add_argument("--n-samples", type=int, default=4000, help="Predictive draws per seed at fgas_obs.")
    ap.add_argument("--n-realisations", type=int, default=2000, help="fgas draws from N(obs, err).")
    ap.add_argument("--n-per-realisation", type=int, default=20, help="Predictive draws per fgas draw and seed.")
    ap.add_argument("--rng-seed", type=int, default=20261002)
    ap.add_argument("--cov-note", default="", help="Provenance of fgas_obs_cov, copied to the sidecar.")
    ap.add_argument("--group", default=GROUP, choices=sorted(MODELS),
                    help=f"Run group under experiments/runs (default {GROUP}).")
    ap.add_argument("--run-cfg", default=None,
                    help="Run config the group was trained with (default: the group's, see MODELS).")
    ap.add_argument("--fgas-std", action="store_true",
                    help="The group's profile input is [f_gas, log10 sigma_fgas]; feed log10(fgas_obs_err).")
    ap.add_argument("--obs-source", default=None,
                    help="Provenance of the --obs file (e.g. its source CSV and series); when given, the "
                         "sidecar records it with the resolved --obs path under 'obs_input'.")
    args = ap.parse_args()
    run_cfg = args.run_cfg if args.run_cfg is not None else MODELS[args.group][0]
    if (run_cfg, args.fgas_std) != MODELS[args.group]:
        raise SystemExit(f"{args.group} was trained with run config / fgas_std {MODELS[args.group]}, "
                         f"not {(run_cfg, args.fgas_std)}")

    d = np.load(args.obs)
    obs = {key: np.asarray(d[key], dtype=float) for key in ("theta_arcmin", "fgas_obs", "fgas_obs_err")}
    obs["fgas_obs_cov"] = np.asarray(d["fgas_obs_cov"], dtype=float) if "fgas_obs_cov" in d.files else None
    if obs["fgas_obs_cov"] is not None and not np.allclose(
            np.sqrt(np.diag(obs["fgas_obs_cov"])), obs["fgas_obs_err"], rtol=1e-6):
        raise RuntimeError("sqrt(diag(fgas_obs_cov)) does not reproduce fgas_obs_err.")
    cosmology = json.loads(args.cosmology) if args.cosmology else FIDUCIAL

    out, groups = predict(args.snap, obs, cosmology, args.n_samples, args.n_realisations,
                          args.n_per_realisation, args.rng_seed, group=args.group, run_cfg=run_cfg,
                          fgas_std=args.fgas_std)
    side = {"snapshot": args.snap, "cosmology": cosmology, "percentiles": PERCENTILES,
            "n_samples": args.n_samples, "n_realisations": args.n_realisations,
            "n_per_realisation": args.n_per_realisation, "rng_seed": args.rng_seed,
            "cov_note": args.cov_note, "groups": groups}
    if args.obs_source is not None:
        side["obs_input"] = {"path": str(Path(args.obs).resolve()), "source": args.obs_source,
                             "covariance": "fgas_obs_cov" if obs["fgas_obs_cov"] is not None
                             else "none; diag(fgas_obs_err^2) only"}
    if args.group != GROUP:
        # Model provenance; absent for the default joint cVAE so its sidecar is unchanged.
        side["model"] = {"group": args.group, "run_cfg": run_cfg, "fgas_std": args.fgas_std}
        if args.fgas_std:
            side["model"]["x_profile"] = X_PROFILE_DEFINITION_FGAS_STD
            side["model"]["x_profile_obs"] = (
                "fgas_obs (9 bins), then log10(fgas_obs_err) on the same bins; realisations perturb "
                "only the fgas_obs columns")
            side["model"]["flag"] = (
                "training sigma_fgas is the halo-to-halo scatter of the simulated stack (not divided by "
                "sqrt(N)); fgas_obs_err is the measurement error of the observed stacked mean. They are "
                "different quantities; neither is rescaled. See groups[*].fgas_err_vs_train_sigma.")
    np.savez(args.out, groups=np.array(list(groups)), percentiles=np.array(PERCENTILES), **out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(side, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
