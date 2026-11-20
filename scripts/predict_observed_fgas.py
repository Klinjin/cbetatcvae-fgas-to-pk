"""Apply recorded CVAE run groups to an observed f_gas(theta) profile.

Reloads every seed of each requested run group (e.g. ``arcmin9_cosmo_kmax10_snap74_nd2``
and ``arcmin9nofb_noparams_kmax10_snap74_nd2``), checks that each reload reproduces its
recorded test RMSE, and decodes the observed profile into SP(k) = P_tot / P_DM
predictive samples in two ways:

* **at the measurement** -- the profile is fixed at ``fgas_obs``; the spread comes
  from the CVAE's own predictive distribution (conditional-prior latent draws plus
  the learned observation-noise head, :meth:`Cvae.predict_samples`) pooled over
  the training seeds (seed ensemble);
* **with the measurement error** -- the same, but the profile is first drawn
  ``n_realisations`` times from the measurement's error distribution and the
  decoded samples of all realisations are pooled. Two variants are produced:
  ``q_err_diag`` draws ``N(fgas_obs, diag(fgas_obs_err^2))`` (bins independent),
  and, when the input file carries ``fgas_obs_cov``, ``q_err_cov`` draws
  ``N(fgas_obs, fgas_obs_cov)`` (bin-to-bin correlations kept).

Input conventions (flagged, not resolved here -- they are passed in explicitly):

* ``fgas_obs`` is the observed ratio ``Delta Sigma_kSZ / Delta Sigma_GGL / f_b``.
  Run groups trained on the f_b-normalised dataset use it as-is. Run groups whose
  dataset carries ``fgas_fb_normalised = False`` (the ``*nofb*`` groups) get
  ``fgas_obs * fb_obs`` (and ``fgas_obs_err * fb_obs``), where ``fb_obs`` must be
  the f_b the measurement was normalised by.
* Cosmology-conditioned groups are decoded at the fixed cosmology passed in
  ``--cosmology`` (default: the Planck 2018 ``FIDUCIAL`` of
  ``scripts/cosmo_agnostic_predictions.py``).
* The observed angular bins must equal the training arcmin grid exactly; there is
  no interpolation.
* The covariance, when given, is used as supplied; how it was assembled (e.g.
  from bootstrap resamples or by propagating the kSZ and lensing covariances) is
  the caller's responsibility and is recorded in the ``.json`` sidecar only via
  ``--cov-note``.

Writes one ``.npz`` (``--out``) with ``k``, the per-group 2.5/16/50/84/97.5
percentiles for every case, the per-seed point predictions, and the per-bin
percentile rank of the observed profile within the training inputs (an
extrapolation diagnostic), plus a ``.json`` sidecar with the provenance.

Usage::

    python scripts/predict_observed_fgas.py --obs obs.npz --out pred.npz \\
        --groups arcmin9_cosmo_kmax10_snap74_nd2 arcmin9nofb_noparams_kmax10_snap74_nd2 \\
        --fb-obs 0.158
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cosmo_agnostic_predictions import FIDUCIAL, _check_rmse, _load_run  # noqa: E402

from fgas_spk.paths import repo_root  # noqa: E402

PERCENTILES = (2.5, 16.0, 50.0, 84.0, 97.5)


def _group_runs(group: str) -> list[Path]:
    """All run-record directories of ``group`` (one per training seed)."""
    base = repo_root() / "experiments" / "runs" / group
    runs = sorted(d for d in base.iterdir() if (d / "config_run.yaml").exists())
    if not runs:
        raise RuntimeError(f"No runs found in {base}.")
    return runs


def _train_radii_arcmin(td) -> np.ndarray:
    """Training arcmin grid of the loaded (cropped) profile input."""
    grid = td.meta["radial_grid"]
    radii = np.asarray(grid.get("radii_arcmin", grid.get("radii_arcmin_median")), dtype=float)
    return radii[: td.X.shape[1]]


def _percentile_rank(X_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-bin percentile of ``x`` within the training inputs (0-100)."""
    return 100.0 * np.mean(X_train <= x[None, :], axis=0)


def _draw_realisations(x: np.ndarray, cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` draws from ``N(x, cov)`` via a Cholesky factor (raises if ``cov`` is not PD)."""
    L = np.linalg.cholesky(cov)
    return x[None, :] + rng.standard_normal((n, x.size)) @ L.T


def _predict_group(group: str, obs: dict, fb_obs: float, cosmology: dict,
                   n_samples: int, n_realisations: int, n_per_realisation: int,
                   rng_seed: int) -> dict:
    """Decode the observed profile with every seed of ``group``."""
    pooled_obs, point, run_ids = [], [], []
    pooled_err = {"diag": [], "cov": []}
    k = rank = x_in = err_in = None
    fb_normalised = None
    real, frac_outside = {}, {}
    for run_dir in _group_runs(group):
        model, td, test = _load_run(run_dir)
        y_test = np.asarray(test.y, dtype=float)
        p_test = model.predict(test.X, test.X_cond, test.X_params)
        _check_rmse(np.asarray(p_test, dtype=float), y_test, run_dir)

        if td.X_cond is not None:
            raise RuntimeError(f"{run_dir.name}: X_cond conditioning has no observed counterpart.")
        radii = _train_radii_arcmin(td)
        if radii.shape != obs["theta_arcmin"].shape or not np.allclose(radii, obs["theta_arcmin"]):
            raise RuntimeError(
                f"{run_dir.name}: observed theta {obs['theta_arcmin']} != training grid {radii}."
            )

        norm = td.meta.get("fgas_fb_normalised", True)
        if fb_normalised is None:
            fb_normalised = norm
            scale = 1.0 if norm else fb_obs
            x_in = obs["fgas_obs"] * scale
            err_in = obs["fgas_obs_err"] * scale
            k = np.asarray(td.k, dtype=float)
            X_train = np.asarray(td.X, dtype=float)
            rank = _percentile_rank(X_train, x_in)
            # Same realisations for every seed of the group (and the same unit
            # normals for every group, via the shared rng seed).
            covs = {"diag": np.diag(err_in ** 2)}
            if obs.get("fgas_obs_cov") is not None:
                covs["cov"] = obs["fgas_obs_cov"] * scale ** 2
            for name, c in covs.items():
                real[name] = _draw_realisations(x_in, c, n_realisations,
                                                np.random.default_rng(rng_seed))
                frac_outside[name] = float(np.mean(np.any(
                    (real[name] < X_train.min(axis=0)) | (real[name] > X_train.max(axis=0)),
                    axis=1)))
        elif norm != fb_normalised:
            raise RuntimeError(f"{group}: runs disagree on the f_b normalisation.")

        params_obs = params_real = None
        if td.X_params is not None:
            theta = np.array([cosmology[n] for n in td.param_names], dtype=float)[None, :]
            lo, hi = td.X_params.min(axis=0), td.X_params.max(axis=0)
            if np.any(theta < lo) or np.any(theta > hi):
                raise RuntimeError(f"Cosmology {cosmology} lies outside the training box.")
            params_obs = theta
            params_real = np.repeat(theta, n_realisations, axis=0)

        seed = int(json.loads((run_dir / "env.json").read_text())["seed"])
        point.append(np.asarray(model.predict(x_in[None, :], None, params_obs), dtype=float)[0])
        pooled_obs.append(model.predict_samples(
            x_in[None, :], None, params_obs, n_samples=n_samples, seed=seed)[:, 0])
        for name, x_real in real.items():
            pooled_err[name].append(model.predict_samples(
                x_real, None, params_real, n_samples=n_per_realisation, seed=seed
            ).reshape(-1, k.size))
        run_ids.append(run_dir.name)

    out = {
        "k": k,
        "q_obs": np.percentile(np.concatenate(pooled_obs), PERCENTILES, axis=0),
        "point_per_seed": np.array(point),
        "fgas_input": x_in,
        "fgas_input_err": err_in,
        "train_percentile_rank": rank,
        "frac_realisations_outside_train_range": frac_outside,
        "fb_normalised": bool(fb_normalised),
        "run_ids": run_ids,
    }
    for name in real:
        out[f"q_err_{name}"] = np.percentile(np.concatenate(pooled_err[name]), PERCENTILES, axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--obs", required=True, help="npz with theta_arcmin, fgas_obs, fgas_obs_err.")
    ap.add_argument("--out", required=True, help="Output .npz path (a .json sidecar is written beside it).")
    ap.add_argument("--groups", nargs="+", required=True, help="Run groups under experiments/runs/.")
    ap.add_argument("--fb-obs", type=float, required=True,
                    help="f_b the observed ratio was normalised by (used only for nofb groups).")
    ap.add_argument("--cosmology", default=None,
                    help="JSON dict of the conditioning cosmology (default: Planck 2018 FIDUCIAL).")
    ap.add_argument("--n-samples", type=int, default=4000, help="Predictive draws per seed at fgas_obs.")
    ap.add_argument("--n-realisations", type=int, default=2000, help="fgas draws from N(obs, err).")
    ap.add_argument("--n-per-realisation", type=int, default=20, help="Predictive draws per fgas draw and seed.")
    ap.add_argument("--rng-seed", type=int, default=20261002)
    ap.add_argument("--cov-note", default="", help="Provenance of fgas_obs_cov, copied to the sidecar.")
    ap.add_argument("--obs-source", default=None,
                    help="Provenance of the --obs file (e.g. its source CSV and series); when given, the "
                         "sidecar records it with the resolved --obs path under 'obs_input'.")
    args = ap.parse_args()

    d = np.load(args.obs)
    obs = {key: np.asarray(d[key], dtype=float) for key in ("theta_arcmin", "fgas_obs", "fgas_obs_err")}
    obs["fgas_obs_cov"] = np.asarray(d["fgas_obs_cov"], dtype=float) if "fgas_obs_cov" in d.files else None
    if obs["fgas_obs_cov"] is not None and not np.allclose(
            np.sqrt(np.diag(obs["fgas_obs_cov"])), obs["fgas_obs_err"], rtol=1e-6):
        raise RuntimeError("sqrt(diag(fgas_obs_cov)) does not reproduce fgas_obs_err.")
    cosmology = json.loads(args.cosmology) if args.cosmology else FIDUCIAL

    out, side = {}, {"fb_obs": args.fb_obs, "cosmology": cosmology, "percentiles": PERCENTILES,
                     "n_samples": args.n_samples, "n_realisations": args.n_realisations,
                     "n_per_realisation": args.n_per_realisation, "rng_seed": args.rng_seed,
                     "cov_note": args.cov_note, "groups": {}}
    for g in args.groups:
        r = _predict_group(g, obs, args.fb_obs, cosmology, args.n_samples,
                           args.n_realisations, args.n_per_realisation, args.rng_seed)
        out["k"] = r["k"]
        for key in ("q_obs", "q_err_diag", "q_err_cov", "point_per_seed", "fgas_input", "fgas_input_err",
                    "train_percentile_rank"):
            if key in r:
                out[f"{g}__{key}"] = r[key]
        side["groups"][g] = {"run_ids": r["run_ids"], "fb_normalised": r["fb_normalised"],
                             "train_percentile_rank": r["train_percentile_rank"].round(1).tolist(),
                             "frac_realisations_outside_train_range":
                                 r["frac_realisations_outside_train_range"]}
        print(g, json.dumps(side["groups"][g]))
    if args.obs_source is not None:
        side["obs_input"] = {"path": str(Path(args.obs).resolve()), "source": args.obs_source,
                             "covariance": "fgas_obs_cov" if obs["fgas_obs_cov"] is not None
                             else "none; diag(fgas_obs_err^2) only"}

    np.savez(args.out, groups=np.array(args.groups), percentiles=np.array(PERCENTILES), **out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(side, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
