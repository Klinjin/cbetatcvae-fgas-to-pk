"""Cosmology-agnostic SP(k) predictions from the cosmology-conditioned CVAE.

Question: if the five SB35 cosmological parameters of an f_gas(R) curve are
known only through a noisy measurement (or not at all), how much does the
cosmology-conditioned ``cvae`` (``latent_dim=1``, full-arcmin-range profiles)
prediction move, compared with its error when the true cosmology is supplied?

For each snapshot (74, z=0.47; 82, z=0.21) the recorded run is reloaded from its
checkpoint, the run's own grouped split is rebuilt, and the held-out **test**
curves at the p10/p30/p50/p70/p90 percentiles of suppression depth
(``SP(k_max)``, the same ladder as the run's residual-map figure) are decoded:

* with the **true** cosmological parameters of that simulation, and
* with ``N_DRAWS`` cosmologies drawn from a **per-simulation posterior**
  (summarised per k by its central 68% interval),
* with the fixed **Planck 2018 fiducial** cosmology (``FIDUCIAL``), and
* by the matched **profile-only** CVAE (same data, split, seed and
  architecture, trained without the cosmology vector; ``RUN_GROUPS_NOPARAMS``).

Per-simulation prior. Each test simulation gets its own cosmology prior, built
from a mock measurement whose uncertainty is set by the SB35 prior itself:

1. Covariance ``Sigma``: the covariance of the SB35 prior. The SB35
   cosmological parameters are a Sobol fill of an independent uniform box
   (verified on the 1024 sims: KS p = 1.0 against uniform on [min, max] for
   each parameter, all pairwise correlations ~0), so
   ``Sigma = diag((hi - lo)^2 / 12)``, the variance of a uniform distribution on
   the box spanned by the suite's parameters, with no off-diagonal terms.
2. Measurement: ``theta_obs ~ N(theta_true, Sigma)``, one realisation per
   simulation, seeded by ``(DRAW_SEED, sim_index)``. Both snapshots of a
   simulation (the same universe) therefore share their measurement and draws.
   ``theta_obs`` itself is not confined to the box.
3. Posterior: the Gaussian likelihood ``N(theta_obs, Sigma)`` times the SB35
   flat prior, i.e. that Gaussian truncated to the box. ``Sigma`` is diagonal,
   so the posterior factorises into independent 1-D truncated normals, which
   are sampled exactly (``scipy.stats.truncnorm``), giving ``N_DRAWS`` draws
   per simulation.

Every decode is the deterministic conditional-prior-mean prediction
(:meth:`Cvae.predict`), so the spread between the draws is due to the
cosmology input alone (no latent or observation-noise sampling).

Caveat (flagged, not resolved here): the input ``fgas`` is
``Delta Sigma_ion / Delta Sigma_tot / f_b`` normalised by each simulation's
*true* ``f_b = Omega_b / Omega_m``. The profile itself therefore still carries
the true baryon fraction; only the explicit conditioning vector is replaced.

Outputs (figures are gitignored): ``figures/<YYYY-MM>/<MM-DD>/cosmo_agnostic/``
``cvae_ld1_arcmin_cosmo_agnostic_seed<seed>.png`` and a ``.json`` summary with
the per-pick numbers and test-fold aggregates.

Usage::

    python scripts/cosmo_agnostic_predictions.py --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fgas_spk.experiment import load_configs  # noqa: E402
from fgas_spk.loader import load_training_data  # noqa: E402
from fgas_spk.paths import figure_dir, fill_data_root, repo_root  # noqa: E402
from fgas_spk.train import (  # noqa: E402
    _LADDER_PERCENTILES,
    _ladder_picks,
    _subset,
    _suppression_depth,
    grouped_split,
)

# Run-record groups for the cosmology-conditioned, latent_dim=1, full-arcmin runs
# (tag 20260930-arcmin-all), keyed by snapshot.
RUN_GROUPS = {74: "arcmin_snap74_nd2", 82: "arcmin_snap82_nd2"}
# Matched profile-only runs (no cosmology conditioning) on the same arcmin data.
RUN_GROUPS_NOPARAMS = {74: "arcmin_noparams_snap74_nd2", 82: "arcmin_noparams_snap82_nd2"}
from fgas_spk.pub_style import REDSHIFT  # noqa: E402
N_DRAWS = 1024
DRAW_SEED = 20260930
# Interior k edges [h/Mpc] for the binned head-to-head comparisons: below 1 (onset of
# suppression), 1-3, 3-10 (deepest suppression) and above 10 (beyond k_max = 10).
K_BIN_EDGES = (1.0, 3.0, 10.0)

# Fiducial Planck 2018 cosmology (Planck Collaboration 2020, A&A 641, A6, Table 2,
# TT,TE,EE+lowE+lensing)
_H_FID = 0.6736
FIDUCIAL = {
    "Omega0": 0.3153,
    "sigma8": 0.8111,
    "OmegaBaryon": 0.02237 / _H_FID**2,
    "HubbleParam": _H_FID,
    "n_s": 0.9649,
}


def fiducial_vector(param_names: list[str]) -> np.ndarray:
    """FIDUCIAL in the model's ``X_params`` column order (raises on a missing name)."""
    return np.array([FIDUCIAL[n] for n in param_names], dtype=float)


def sb35_covariance(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Covariance of the SB35 prior: independent uniforms on ``[lo, hi]``."""
    return np.diag((hi - lo) ** 2 / 12.0)


def _posterior_draws(rng: np.random.Generator, theta_true: np.ndarray, sd: np.ndarray,
                     lo: np.ndarray, hi: np.ndarray, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
    """One simulated measurement and exact draws from its box-truncated posterior.

    ``theta_obs ~ N(theta_true, diag(sd^2))``. The draws come from
    ``N(theta_obs, diag(sd^2))`` restricted to ``[lo, hi]``; with a diagonal
    covariance this is a product of independent 1-D truncated normals.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``theta_obs`` (n,) and the draws (n_draws, n).
    """
    from scipy.stats import truncnorm

    theta_obs = theta_true + sd * rng.standard_normal(theta_true.shape[0])
    a, b = (lo - theta_obs) / sd, (hi - theta_obs) / sd
    draws = truncnorm.rvs(a, b, loc=theta_obs, scale=sd,
                          size=(n_draws, theta_true.shape[0]), random_state=rng)
    return theta_obs, draws


def _find_run(group: str, seed: int) -> Path:
    """Return the run-record directory in ``group`` trained with ``seed``."""
    import yaml

    base = repo_root() / "experiments" / "runs" / group
    hits = []
    for d in sorted(base.iterdir()):
        cfg = yaml.safe_load((d / "config_run.yaml").read_text())
        if cfg["seed"] == seed:
            hits.append(d)
    if len(hits) != 1:
        raise RuntimeError(f"Expected one run with seed {seed} in {base}, found {hits}.")
    return hits[0]


def _prior_box(td) -> tuple[np.ndarray, np.ndarray]:
    """Per-parameter [min, max] over the suite's simulations (one row per sim)."""
    _, first = np.unique(td.sim_index, return_index=True)
    P = np.asarray(td.X_params, dtype=float)[first]
    return P.min(axis=0), P.max(axis=0)


def _load_run(run_dir: Path):
    """Reload a run's model, full dataset and test subset (dataset path checked)."""
    env = json.loads((run_dir / "env.json").read_text())
    dc, rc = load_configs(run_dir / "config_data.yaml", run_dir / "config_run.yaml")
    td = load_training_data(fill_data_root(dc))
    if td.source_path != env["dataset"]["path"]:
        raise RuntimeError(
            f"Dataset mismatch: run used {env['dataset']['path']}, loaded {td.source_path}."
        )
    model = joblib.load(Path(env["roots"]["scratch_root"]) / "models" / run_dir.name / "model.joblib")
    masks = grouped_split(td.sim_index, rc.split)
    return model, td, _subset(td, masks["test"])


def _check_rmse(pred: np.ndarray, y: np.ndarray, run_dir: Path) -> float:
    """Test RMSE of ``pred``; raises unless it reproduces the run's recorded value."""
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    recorded = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
    if not np.isclose(rmse, recorded, rtol=1e-4):
        raise RuntimeError(f"{run_dir.name}: reloaded test RMSE {rmse} != recorded {recorded}.")
    return rmse


def _run_snapshot(snap: int, seed: int) -> dict:
    """Load one snapshot's run, decode the ladder picks and test-fold aggregates."""
    run_dir = _find_run(RUN_GROUPS[snap], seed)
    model, td, test = _load_run(run_dir)
    y_true = np.asarray(test.y, dtype=float)
    pred_true = np.asarray(model.predict(test.X, test.X_cond, test.X_params), dtype=float)
    rmse_check = _check_rmse(pred_true, y_true, run_dir)

    # Matched profile-only model: must see the identical test curves and profiles.
    run_dir_np = _find_run(RUN_GROUPS_NOPARAMS[snap], seed)
    model_np, _, test_np = _load_run(run_dir_np)
    if not (np.array_equal(test.sim_index, test_np.sim_index)
            and np.allclose(test.X, test_np.X) and np.allclose(test.y, test_np.y)):
        raise RuntimeError(f"{run_dir_np.name} test fold differs from {run_dir.name}.")
    pred_nocosmo = np.asarray(model_np.predict(test_np.X, test_np.X_cond, None), dtype=float)
    _check_rmse(pred_nocosmo, y_true, run_dir_np)

    lo, hi = _prior_box(td)
    cov = sb35_covariance(lo, hi)
    sd = np.sqrt(np.diag(cov))

    # One SB35-width measurement and box-truncated posterior per test simulation.
    sims, first, inv = np.unique(test.sim_index, return_index=True, return_inverse=True)
    theta_sim = np.asarray(test.X_params, dtype=float)[first]
    if not np.allclose(np.asarray(test.X_params, dtype=float), theta_sim[inv]):
        raise RuntimeError("Rows of one simulation carry different cosmological parameters.")
    theta_obs = np.empty_like(theta_sim)
    draws_sim = np.empty((len(sims), N_DRAWS, theta_sim.shape[1]))
    for s, (sim, th) in enumerate(zip(sims, theta_sim)):
        rng = np.random.default_rng([DRAW_SEED, int(sim)])
        theta_obs[s], draws_sim[s] = _posterior_draws(rng, th, sd, lo, hi, N_DRAWS)
    if np.any(draws_sim < lo) or np.any(draws_sim > hi):
        raise RuntimeError("Posterior draws fall outside the SB35 box.")
    draws_row = draws_sim[inv].transpose(1, 0, 2)  # (N_DRAWS, n_test, n_params), raw units

    # Decode every test curve at each of its own drawn cosmologies (profile held fixed).
    n = test.X.shape[0]
    # float32 storage keeps (N_DRAWS, n_test, n_k) manageable at N_DRAWS = 1024.
    pred_draw = np.stack([
        np.asarray(model.predict(test.X, test.X_cond, c), dtype=np.float32) for c in draws_row
    ])  # (N_DRAWS, n_test, n_k)
    # Point prediction at the measured cosmology itself.
    pred_obs = np.asarray(model.predict(test.X, test.X_cond, theta_obs[inv]), dtype=float)
    # Per-k median over the drawn cosmologies: the prior-marginalised point prediction.
    pred_draw_med = np.median(pred_draw, axis=0).astype(float)  # (n_test, n_k)

    theta_fid = fiducial_vector(list(td.param_names))
    if np.any(theta_fid < lo) or np.any(theta_fid > hi):
        raise RuntimeError(f"Fiducial cosmology {FIDUCIAL} lies outside the SB35 box.")
    pred_fid = np.asarray(
        model.predict(test.X, test.X_cond, np.repeat(theta_fid[None], n, axis=0)), dtype=float
    )

    rows, labels, _ = _ladder_picks(_suppression_depth(y_true), _LADDER_PERCENTILES)

    def per_curve_rms(a):
        return np.sqrt(np.mean(a ** 2, axis=-1))

    err_true = per_curve_rms(pred_true - y_true)                       # (n_test,)
    shift_draw = per_curve_rms(pred_draw - pred_true[None])            # (N_DRAWS, n_test)
    err_draw = per_curve_rms(pred_draw - y_true[None])                 # (N_DRAWS, n_test)
    err_draw_med = per_curve_rms(pred_draw_med - y_true)               # (n_test,)
    err_obs = per_curve_rms(pred_obs - y_true)                         # (n_test,)
    err_fid = per_curve_rms(pred_fid - y_true)                         # (n_test,)
    err_nocosmo = per_curve_rms(pred_nocosmo - y_true)                 # (n_test,)

    # Head-to-head per curve: is the profile-only error below the theta_obs / Planck
    # error, overall and within each k bin (per-curve RMS over that bin's k)?
    k_arr = np.asarray(test.k, dtype=float)
    edges = np.concatenate([[k_arr.min()], K_BIN_EDGES, [k_arr.max()]])
    k_bins = [{"k_range": "all", "n_k": int(k_arr.size),
               "frac_no_cosmo_better_than_measured": float(np.mean(err_nocosmo < err_obs)),
               "frac_no_cosmo_better_than_fid": float(np.mean(err_nocosmo < err_fid))}]
    for b, (k0, k1) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (k_arr >= k0) & ((k_arr <= k1) if b == len(edges) - 2 else (k_arr < k1))
        e_n, e_o, e_f = (per_curve_rms((p_ - y_true)[:, sel]) for p_ in (pred_nocosmo, pred_obs, pred_fid))
        k_bins.append({"k_range": [float(k0), float(k1)], "n_k": int(sel.sum()),
                       "frac_no_cosmo_better_than_measured": float(np.mean(e_n < e_o)),
                       "frac_no_cosmo_better_than_fid": float(np.mean(e_n < e_f))})

    picks = []
    names = list(td.param_names)
    for r, pct in zip(rows, labels):
        s = inv[r]
        picks.append({
            "percentile": int(pct),
            "sim_index": int(test.sim_index[r]),
            "true_params": dict(zip(names, map(float, test.X_params[r]))),
            "measured_params": dict(zip(names, map(float, theta_obs[s]))),
            "posterior_draws_mean": dict(zip(names, map(float, draws_sim[s].mean(axis=0)))),
            "posterior_draws_std": dict(zip(names, map(float, draws_sim[s].std(axis=0, ddof=1)))),
            "SP_kmax_true": float(y_true[r, -1]),
            "rms_err_true_cosmo": float(err_true[r]),
            "rms_err_measured_cosmo": float(err_obs[r]),
            "rms_shift_draws_median": float(np.median(shift_draw[:, r])),
            "rms_err_draws_median": float(np.median(err_draw[:, r])),
            "rms_err_draws_median_pred": float(err_draw_med[r]),
            "rms_err_fid_cosmo": float(err_fid[r]),
            "rms_err_no_cosmo": float(err_nocosmo[r]),
        })

    return {
        "snap": snap,
        "run_id": run_dir.name,
        "run_id_no_cosmo": run_dir_np.name,
        "k": np.asarray(test.k, dtype=float),
        "y_true": y_true[rows],
        "pred_true": pred_true[rows],
        "pred_draw": pred_draw[:, rows].astype(float),
        "pred_draw_med": pred_draw_med[rows],
        "pred_fid": pred_fid[rows],
        "pred_nocosmo": pred_nocosmo[rows],
        "picks": picks,
        "param_names": names,
        "prior_box": {n_: [float(a), float(b)] for n_, a, b in zip(names, lo, hi)},
        "prior_sd": dict(zip(names, map(float, sd))),
        "test_fold": {
            "n_test": int(n),
            "n_test_sims": int(len(sims)),
            "frac_sims_measurement_outside_box": float(np.mean(np.any((theta_obs < lo) | (theta_obs > hi), axis=1))),
            "test_rmse_measured_cosmo": float(np.sqrt(np.mean((pred_obs - y_true) ** 2))),
            "median_curve_rms_err_measured_cosmo": float(np.median(err_obs)),
            "test_rmse_true_cosmo": rmse_check,
            "test_rmse_draws_mean": float(np.mean([
                np.sqrt(np.mean((p - y_true) ** 2)) for p in pred_draw
            ])),
            "median_curve_rms_err_true_cosmo": float(np.median(err_true)),
            "median_curve_rms_err_draws": float(np.median(err_draw)),
            "test_rmse_draws_median_pred": float(np.sqrt(np.mean((pred_draw_med - y_true) ** 2))),
            "median_curve_rms_err_draws_median_pred": float(np.median(err_draw_med)),
            "test_rmse_fid_cosmo": float(np.sqrt(np.mean((pred_fid - y_true) ** 2))),
            "median_curve_rms_err_fid_cosmo": float(np.median(err_fid)),
            "frac_curves_fid_worse_than_true": float(np.mean(err_fid > err_true)),
            "test_rmse_no_cosmo": float(np.sqrt(np.mean((pred_nocosmo - y_true) ** 2))),
            "median_curve_rms_err_no_cosmo": float(np.median(err_nocosmo)),
            "frac_curves_no_cosmo_worse_than_true": float(np.mean(err_nocosmo > err_true)),
            "frac_curves_no_cosmo_better_than_fid": float(np.mean(err_nocosmo < err_fid)),
            "frac_curves_no_cosmo_better_than_measured": float(np.mean(err_nocosmo < err_obs)),
            "no_cosmo_head_to_head_k_bins": k_bins,
            "median_curve_err_ratio_no_cosmo_over_fid": float(np.median(err_nocosmo / err_fid)),
            "median_curve_rms_shift_draws": float(np.median(shift_draw)),
            "frac_curves_draw_shift_gt_err": float(np.mean(np.median(shift_draw, axis=0) > err_true)),
        },
    }


from fgas_spk.pub_style import COLOURMAP, PLOT_RC, TRACER  # noqa: E402,F401


def _plot(results: list[dict], seed: int, out: Path) -> None:
    """2 snapshots x 5 ladder picks; SP(k) panel over a residual panel for each.

    Each snapshot row is a subfigure titled by the tracer it stands in for; each
    column is titled by the true cosmology of the plotted test simulation.
    """
    with matplotlib.rc_context(PLOT_RC):
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c_draw, c_true, c_pred, c_fid, c_noc = "#a3a9b3", "black", cycle[0], cycle[3], cycle[1]
        n_cols = max(len(r["picks"]) for r in results)
        fig = plt.figure(figsize=(4.9 * n_cols, 16.5), layout="constrained")
        subfigs = fig.subfigures(len(results), 1, hspace=0.04)

        for res, sf in zip(results, np.atleast_1d(subfigs)):
            snap, k = res["snap"], res["k"]
            sf.suptitle(f"{TRACER[snap]} ($z={REDSHIFT[snap]}$)", fontsize=22, fontweight="bold")
            gs = sf.add_gridspec(2, n_cols, height_ratios=[3, 1.2])
            for j, pick in enumerate(res["picks"]):
                ax = sf.add_subplot(gs[0, j])
                axr = sf.add_subplot(gs[1, j], sharex=ax)
                # Central 68% interval over this sim's posterior draws.
                q16, q84 = np.percentile(res["pred_draw"][:, j], [16, 84], axis=0)
                yt = res["y_true"][j]
                for a, off in ((ax, 0.0), (axr, yt)):
                    a.fill_between(k, q16 - off, q84 - off, color=c_draw, alpha=0.5, lw=0,
                                   label="per-sim posterior, 68%" if a is ax else None)
                ax.plot(k, res["y_true"][j], color=c_true, lw=2.2, label="true")
                ax.plot(k, res["pred_true"][j], color=c_pred, lw=2.2, ls="--",
                        label="pred, true cosmology")
                axr.plot(k, res["pred_true"][j] - res["y_true"][j], color=c_pred, lw=2.2, ls="--")
                ax.plot(k, res["pred_fid"][j], color=c_fid, lw=2.0, ls="-.",
                        label="pred, Planck 2018 cosmology")
                axr.plot(k, res["pred_fid"][j] - res["y_true"][j], color=c_fid, lw=2.0, ls="-.")
                ax.plot(k, res["pred_nocosmo"][j], color=c_noc, lw=2.2, ls=":",
                        label="pred, no cosmology")
                axr.plot(k, res["pred_nocosmo"][j] - res["y_true"][j], color=c_noc, lw=2.2, ls=":")
                axr.axhline(0, color=c_true, lw=0.8)

                p = pick["true_params"]
                ax.set_title(
                    f"$\\Omega_m={p['Omega0']:.2f}$, $\\sigma_8={p['sigma8']:.2f}$\n"
                    f"$\\Omega_b={p['OmegaBaryon']:.3f}$, $h={p['HubbleParam']:.2f}$, "
                    f"$n_s={p['n_s']:.2f}$",
                    fontsize=16,
                )
                # Enhanced curves (SP > 1 at k_max) rise to the right and leave the
                # lower-left corner crowded, so their annotation goes top-left.
                enhanced = res["y_true"][j, -1] > 1.0
                ax.text(0.04, 0.96 if enhanced else 0.04,
                        f"true {pick['rms_err_true_cosmo']:.3f}\n"
                        f"Planck {pick['rms_err_fid_cosmo']:.3f}\n"
                        f"none {pick['rms_err_no_cosmo']:.3f}",
                        transform=ax.transAxes, fontsize=13,
                        va="top" if enhanced else "bottom", color="#333333")
                ax.set_xscale("log")
                ax.tick_params(labelbottom=False)
                ax.grid(True)
                axr.grid(True)
                axr.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
                if j == 0:
                    ax.set_ylabel(r"$P_\mathrm{tot}/P_\mathrm{DM}$")
                    axr.set_ylabel(r"pred $-$ true")

        handles, labels = fig.axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        order = ["true", "pred, true cosmology", "pred, Planck 2018 cosmology",
                 "pred, no cosmology", "per-sim posterior, 68%"]
        fig.legend([by_label[o] for o in order], order,
                   loc="outside lower center", ncol=5, frameon=False, fontsize=16)
        fig.savefig(out, dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0, help="Training seed of the runs to load.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: today's figures/<YYYY-MM>/<MM-DD>/cosmo_agnostic).")
    args = ap.parse_args()

    results = [_run_snapshot(s, args.seed) for s in (74, 82)]
    if results[0]["prior_box"] != results[1]["prior_box"]:
        raise RuntimeError("The two snapshots disagree on the SB35 parameter box.")

    out_dir = Path(args.out_dir) if args.out_dir else figure_dir("cosmo_agnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cvae_ld1_arcmin_cosmo_agnostic_seed{args.seed}"
    _plot(results, args.seed, out_dir / f"{stem}.png")

    summary = {
        "seed": args.seed,
        "draw_seed": DRAW_SEED,
        "fiducial": FIDUCIAL,
        "prior_box": results[0]["prior_box"],
        "n_draws_per_sim": N_DRAWS,
        "prior_sd": results[0]["prior_sd"],
        "prior_cov": "diag((hi - lo)^2 / 12), independent uniforms on prior_box",
        "snapshots": {
            r["snap"]: {"run_id": r["run_id"], "run_id_no_cosmo": r["run_id_no_cosmo"], "picks": r["picks"], "test_fold": r["test_fold"]}
            for r in results
        },
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({s: v["test_fold"] for s, v in summary["snapshots"].items()}, indent=2))
    for s, v in summary["snapshots"].items():
        for p in v["picks"]:
            print(s, p["percentile"], p["sim_index"], round(p["SP_kmax_true"], 3),
                  round(p["rms_err_true_cosmo"], 4), round(p["rms_err_measured_cosmo"], 4),
                  round(p["rms_shift_draws_median"], 4),
                  round(p["rms_err_draws_median"], 4), round(p["rms_err_fid_cosmo"], 4),
                  round(p["rms_err_no_cosmo"], 4))
    print(out_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
