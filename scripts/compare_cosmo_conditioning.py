"""Compare cosmology-conditioned vs profile-only CVAEs on matched data and splits.

Arms (all ``cvae``, ``latent_dim=1``, ``obs_noise_head=True``, identical run
config ``scripts/configs/run/cvae_noise.yaml``, seeds 0/1/2, same grouped split):

* ``true_theta``  -- cosmology-conditioned model given each simulation's TRUE
  (Omega_m, sigma_8, Omega_b, h, n_s): the oracle, cosmology-known case.
* ``no_theta``    -- profile-only model (``include_camels_params: false``): it
  learns ``p(SP | f_gas)`` from the joint training distribution, so it is the
  properly cosmology-agnostic model.
* ``prior_theta`` -- the cosmology-conditioned model used without knowing the
  cosmology, marginalised over the SB35 prior:
  ``p(SP | x) ~= E_{theta ~ U(box)} p(SP | x, theta)``. Note this integrates over
  the *prior*, not over ``p(theta | x)``: it ignores what the profile says about
  the cosmology, which ``no_theta`` exploits implicitly. It is the cost of
  plugging an unknown cosmology into the conditioned model, not the best
  achievable agnostic predictor.
* ``fid_theta``   -- the cosmology-conditioned model given the fixed Planck 2018
  fiducial cosmology for every curve (``FIDUCIAL`` in
  ``scripts/cosmo_agnostic_predictions.py``): the common practice of assuming a
  fiducial cosmology when the true one is unknown.

The SB35 cosmological parameters are an independent uniform (Sobol) box; draws
are uniform on the box spanned by the 1024 sims (see
``scripts/cosmo_agnostic_predictions.py`` for the verification).

Point predictions: ``true_theta`` / ``no_theta`` use :meth:`Cvae.predict` (the
recorded headline prediction; checked against ``summary.json``); ``prior_theta``
uses the mean over ``N_THETA`` prior draws of :meth:`Cvae.predict`.
Predictive samples: :meth:`Cvae.predict_samples` (latent + learned observation
noise); for ``prior_theta`` ``N_SAMPLES / N_THETA`` samples at each of the
``N_THETA`` draws, pooled into one mixture.

Metrics on the held-out test fold (205 curves x 255 k bins), per seed then
mean +/- sd over seeds: pooled RMSE and MAE; RMSE in k bands; per-curve RMSE
median / p90 / max; |pred/true - 1| median and p95; fraction of curves with
max |pred/true - 1| < 2% / 5% for k <= 10 h/Mpc; suppressed-bin RMSE (true
SP < 0.9, < 0.8); R^2 over all bins; CRPS (sample estimator); empirical
coverage of the central 68/90/95% intervals; mean 68% interval width.

Outputs: ``figures/<YYYY-MM>/<MM-DD>/cosmo_agnostic/compare_cosmo_conditioning.{png,json}``.

Usage::

    python scripts/compare_cosmo_conditioning.py
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
import yaml  # noqa: E402

from fgas_spk.experiment import load_configs  # noqa: E402
from fgas_spk.loader import load_training_data  # noqa: E402
from fgas_spk.paths import figure_dir, fill_data_root, repo_root  # noqa: E402
from fgas_spk.train import _subset, grouped_split  # noqa: E402

from cosmo_agnostic_predictions import FIDUCIAL, fiducial_vector  # noqa: E402

# (snapshot, design) -> {"with": group with cosmology, "without": profile-only group}.
COMPARISONS = {
    (74, "arcmin-all"): {"with": "arcmin_snap74_nd2", "without": "arcmin_noparams_snap74_nd2"},
    (82, "arcmin-all"): {"with": "arcmin_snap82_nd2", "without": "arcmin_noparams_snap82_nd2"},
    (74, "mpch R<=10"): {"with": "cosmos_true_snap74_nd2", "without": "cosmos_false_snap74_nd2"},
}
SEEDS = (0, 1, 2)
N_SAMPLES = 1000
N_THETA = 50
DRAW_SEED = 20260930
K_BANDS = ((0.0, 1.0), (1.0, 10.0), (10.0, np.inf))
LEVELS = (0.68, 0.90, 0.95)
ARMS = ("true_theta", "no_theta", "prior_theta", "fid_theta")
ARM_LABEL = {
    "true_theta": "with cosmology, true θ",
    "no_theta": "no cosmology (profile only)",
    "prior_theta": "with cosmology, θ marginalised over SB35 prior",
    "fid_theta": "with cosmology, Planck 2018 fiducial θ",
}
ARM_COLOUR = {"true_theta": "#2a6fbb", "no_theta": "#d1622b", "prior_theta": "#7a7f87",
              "fid_theta": "#2e9e6b"}


def _run_dir(group: str, seed: int, latent_dim: int = 1) -> Path:
    """Unique run-record directory in ``group`` with this seed and latent width."""
    hits = []
    for d in sorted((repo_root() / "experiments" / "runs" / group).iterdir()):
        cfg = yaml.safe_load((d / "config_run.yaml").read_text())
        if cfg["seed"] == seed and cfg["model_params"]["latent_dim"] == latent_dim:
            hits.append(d)
    if len(hits) != 1:
        raise RuntimeError(f"{group} seed {seed} ld {latent_dim}: found {hits}")
    return hits[0]


def _load(run_dir: Path):
    """Load (model, test subset, recorded test RMSE, dataset path) for a run."""
    env = json.loads((run_dir / "env.json").read_text())
    dc, rc = load_configs(run_dir / "config_data.yaml", run_dir / "config_run.yaml")
    td = load_training_data(fill_data_root(dc))
    if td.source_path != env["dataset"]["path"]:
        raise RuntimeError(f"Dataset mismatch for {run_dir}")
    model = joblib.load(
        Path(env["roots"]["scratch_root"]) / "models" / run_dir.name / "model.joblib"
    )
    masks = grouped_split(td.sim_index, rc.split)
    recorded = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
    return model, td, _subset(td, masks["test"]), recorded


def _crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-bin CRPS from samples: E|X - y| - 0.5 E|X - X'| (exact for the sample set).

    Args:
        samples: shape (S, ...).
        y: shape (...).
    """
    s = np.sort(samples, axis=0)
    n = s.shape[0]
    term1 = np.mean(np.abs(s - y[None]), axis=0)
    w = (2 * np.arange(1, n + 1) - n - 1).reshape((n,) + (1,) * (s.ndim - 1))
    term2 = np.sum(w * s, axis=0) / n**2  # = 0.5 E|X - X'|
    return term1 - term2


def _metrics(y: np.ndarray, pred: np.ndarray, samples: np.ndarray, k: np.ndarray) -> dict:
    """All point and probabilistic metrics for one arm on one test fold."""
    err = pred - y
    frac = np.abs(pred / y - 1.0)
    curve_rmse = np.sqrt(np.mean(err**2, axis=1))
    k10 = k <= 10.0
    out = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
        "curve_rmse_median": float(np.median(curve_rmse)),
        "curve_rmse_p90": float(np.percentile(curve_rmse, 90)),
        "curve_rmse_max": float(curve_rmse.max()),
        "frac_err_median": float(np.median(frac)),
        "frac_err_p95": float(np.percentile(frac, 95)),
        "frac_curves_maxfrac_lt2pct_k_le10": float(np.mean(frac[:, k10].max(axis=1) < 0.02)),
        "frac_curves_maxfrac_lt5pct_k_le10": float(np.mean(frac[:, k10].max(axis=1) < 0.05)),
    }
    for lo, hi in K_BANDS:
        m = (k > lo) & (k <= hi)
        out[f"rmse_k_{lo:g}_{hi:g}"] = float(np.sqrt(np.mean(err[:, m] ** 2)))
    for t in (0.9, 0.8):
        m = y < t
        out[f"rmse_true_lt_{t:g}"] = float(np.sqrt(np.mean(err[m] ** 2)))
    out["crps"] = float(np.mean(_crps(samples, y)))
    for lev in LEVELS:
        lo_q, hi_q = np.quantile(samples, [(1 - lev) / 2, (1 + lev) / 2], axis=0)
        out[f"coverage_{int(lev * 100)}"] = float(np.mean((y >= lo_q) & (y <= hi_q)))
        if lev == 0.68:
            out["width_68"] = float(np.mean(hi_q - lo_q))
    # Per-k curves for the figure.
    lo_q, hi_q = np.quantile(samples, [0.16, 0.84], axis=0)
    out["_rmse_k"] = np.sqrt(np.mean(err**2, axis=0))
    out["_cov68_k"] = np.mean((y >= lo_q) & (y <= hi_q), axis=0)
    return out


def _evaluate(comp: dict, draws_unit: np.ndarray, seed: int) -> tuple[dict, np.ndarray]:
    """Metrics for the three arms at one training seed."""
    m_with, td_w, te_w, rec_w = _load(_run_dir(comp["with"], seed))
    m_wo, td_wo, te_wo, rec_wo = _load(_run_dir(comp["without"], seed))
    # Same curves, same order, same profiles in both arms.
    assert np.array_equal(te_w.sim_index, te_wo.sim_index)
    assert np.allclose(te_w.y, te_wo.y) and np.allclose(te_w.X, te_wo.X)
    y, k = np.asarray(te_w.y, float), np.asarray(te_w.k, float)
    n = y.shape[0]

    res = {}
    pred = m_with.predict(te_w.X, te_w.X_cond, te_w.X_params)
    assert np.isclose(np.sqrt(np.mean((pred - y) ** 2)), rec_w, rtol=1e-4)
    samp = m_with.predict_samples(te_w.X, te_w.X_cond, te_w.X_params, n_samples=N_SAMPLES, seed=seed)
    res["true_theta"] = _metrics(y, pred, samp, k)

    pred = m_wo.predict(te_wo.X, te_wo.X_cond, None)
    assert np.isclose(np.sqrt(np.mean((pred - y) ** 2)), rec_wo, rtol=1e-4)
    samp = m_wo.predict_samples(te_wo.X, te_wo.X_cond, None, n_samples=N_SAMPLES, seed=seed)
    res["no_theta"] = _metrics(y, pred, samp, k)

    _, first = np.unique(td_w.sim_index, return_index=True)
    P = np.asarray(td_w.X_params, float)[first]
    lo, hi = P.min(0), P.max(0)
    thetas = lo + draws_unit * (hi - lo)  # (N_THETA, 5)
    per = N_SAMPLES // N_THETA
    preds, samps = [], []
    for i, th in enumerate(thetas):
        Xp = np.repeat(th[None], n, axis=0)
        preds.append(m_with.predict(te_w.X, te_w.X_cond, Xp))
        samps.append(m_with.predict_samples(te_w.X, te_w.X_cond, Xp, n_samples=per, seed=seed * 1000 + i))
    res["prior_theta"] = _metrics(y, np.mean(preds, axis=0), np.concatenate(samps, axis=0), k)

    Xf = np.repeat(fiducial_vector(list(td_w.param_names))[None], n, axis=0)
    if np.any(Xf[0] < lo) or np.any(Xf[0] > hi):
        raise RuntimeError("Fiducial cosmology lies outside the SB35 box.")
    pred = m_with.predict(te_w.X, te_w.X_cond, Xf)
    samp = m_with.predict_samples(te_w.X, te_w.X_cond, Xf, n_samples=N_SAMPLES, seed=seed)
    res["fid_theta"] = _metrics(y, pred, samp, k)
    return res, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: today's figures/<YYYY-MM>/<MM-DD>/cosmo_agnostic).")
    args = ap.parse_args()
    rng = np.random.default_rng(DRAW_SEED)
    draws_unit = rng.uniform(size=(N_THETA, 5))

    table, curves = {}, {}
    for (snap, design), comp in COMPARISONS.items():
        key = f"snap{snap} {design}"
        per_seed = []
        for seed in SEEDS:
            r, k = _evaluate(comp, draws_unit, seed)
            per_seed.append(r)
            print(key, seed, {a: round(r[a]["rmse"], 4) for a in ARMS}, flush=True)
        table[key] = {}
        curves[key] = {"k": k}
        for a in ARMS:
            names = [m for m in per_seed[0][a] if not m.startswith("_")]
            table[key][a] = {
                m: {"mean": float(np.mean([p[a][m] for p in per_seed])),
                    "sd": float(np.std([p[a][m] for p in per_seed], ddof=1))}
                for m in names
            }
            curves[key][a] = {c: np.stack([p[a][c] for p in per_seed]) for c in ("_rmse_k", "_cov68_k")}

    out_dir = Path(args.out_dir) if args.out_dir else figure_dir("cosmo_agnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare_cosmo_conditioning.json").write_text(json.dumps(
        {"arms": ARM_LABEL, "fiducial": FIDUCIAL, "seeds": SEEDS, "n_samples": N_SAMPLES, "n_theta": N_THETA,
         "groups": {f"snap{s} {d}": c for (s, d), c in COMPARISONS.items()}, "metrics": table},
        indent=2))

    # Text table.
    metric_names = list(table[next(iter(table))]["true_theta"])
    for key in table:
        print(f"\n=== {key}  (mean ± sd over seeds {SEEDS}) ===")
        print(f"{'metric':36s}" + "".join(f"{a:>22s}" for a in ARMS))
        for m in metric_names:
            print(f"{m:36s}" + "".join(
                f"{table[key][a][m]['mean']:>14.4f} ±{table[key][a][m]['sd']:.4f}" for a in ARMS))

    # Figure: per-k RMSE and 68% coverage, one row per comparison.
    fig, axes = plt.subplots(len(curves), 2, figsize=(11, 3.2 * len(curves)),
                             constrained_layout=True, sharex=True)
    for i, (key, c) in enumerate(curves.items()):
        k = c["k"]
        for a in ARMS:
            for j, (name, ylab) in enumerate((("_rmse_k", "RMSE(k)"), ("_cov68_k", "68% coverage(k)"))):
                v = c[a][name]
                ax = axes[i, j]
                ax.fill_between(k, v.min(0), v.max(0), color=ARM_COLOUR[a], alpha=0.25, lw=0)
                ax.plot(k, v.mean(0), color=ARM_COLOUR[a], lw=1.8, label=ARM_LABEL[a])
                ax.set_ylabel(ylab)
        axes[i, 0].set_title(f"{key}: test-fold RMSE per k", fontsize=10)
        axes[i, 1].set_title(f"{key}: empirical coverage of central 68% interval", fontsize=10)
        axes[i, 1].axhline(0.68, color="black", lw=0.8, ls=":")
        for ax in axes[i]:
            ax.set_xscale("log")
            ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("CVAE latent_dim=1: cosmology-conditioned vs profile-only "
                 "(line: mean over 3 seeds; band: seed min–max)", fontsize=11)
    fig.savefig(out_dir / "compare_cosmo_conditioning.png", dpi=150)
    print(out_dir / "compare_cosmo_conditioning.png")


if __name__ == "__main__":
    main()
