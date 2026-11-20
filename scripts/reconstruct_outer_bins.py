"""Reconstruct the outer nine arcmin-design f_gas bins from the inner nine (1'-6').

Real measurements cover only the fixed angular bins 1'-6'; the simulated
18-bin profiles continue with nine bins log-spaced from r(6') to 20 comoving
Mpc/h, whose positions vary with each simulation's cosmology. This script
trains and validates :class:`fgas_spk.outer_profile.OuterProfileReconstructor`
(one model per redshift and halo sample) on the 1024 SB35 simulations.

Subcommands:

``fit --snapshot S --nd I``
    5-fold cross-validation (shuffled, seed 0; every simulation is predicted
    once by a model that never saw it) of the GP and of a ridge-regression
    baseline on the same inputs, then the final Student-t GP on all 1024 sims. Writes
    ``cv_snap{S}_nd{I}.npz`` and ``gp_snap{S}_nd{I}.pt`` under
    ``$FGAS_SCRATCH_ROOT/models/outer_profile_gp/<tag>/``.

``noise --snapshot S``
    Projection-scatter noise of each simulated stack: f_gas recomputed per
    projection (xy, xz, yz) from the raw per-sim arcmin profiles with halos
    ranked by saved stellar mass (as the compiled files), sigma = std / sqrt(3).
    The three projections share their halos, so this is a lower bound on the
    stack noise (no halo-sample variance). Used as a reference level in the
    figures. Writes ``noise_snap{S}.npz``.

``figures``
    Percentage reconstruction error across the outer r bins, one figure per
    redshift, plus a summary of median |error| for both redshifts.

Usage::

    export FGAS_SCRATCH_ROOT=/pscratch/sd/l/lindajin/fgas-to-pk-ml
    python scripts/reconstruct_outer_bins.py fit --snapshot 74 --nd 0
    python scripts/reconstruct_outer_bins.py noise --snapshot 74
    python scripts/reconstruct_outer_bins.py figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fgas_spk.outer_profile import N_INNER, OuterProfileReconstructor, mpch_per_arcmin, outer_radii
from fgas_spk.paths import figure_dir, resolve_scratch_root

DATA_DIR = Path("/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data")
RAW = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"
TAG = "20260930-arcmin-all"  # the 18-bin arcmin dataset this is trained on
BOX_SIZE_MPCH = 50.0
PROJECTIONS = ("xy", "xz", "yz")
N_FOLDS = 5
SEED = 0
# snapshot -> (redshift, number-density bins [(Mpc/h)^-3]); producer notebook config.
SNAPSHOTS = {
    74: (0.47, (1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3)),
    82: (0.21, (1.0e-3, 4.0e-3, 8.0e-3, 2.0e-2, 3.0e-2)),
}


def _n_halos(snapshot: int) -> list[int]:
    return [int(nd * BOX_SIZE_MPCH**3) for nd in SNAPSHOTS[snapshot][1]]


def _out_dir(scratch_root: str | None) -> Path:
    out = resolve_scratch_root(scratch_root) / "models" / "outer_profile_gp" / TAG
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_profiles(snapshot: int, nd_idx: int) -> dict:
    """Compiled 18-bin arcmin f_gas plus the per-sim geometry, verified.

    Returns:
        dict: ``fgas`` (1024, 18), ``c`` (1024,) Mpc/h per arcmin,
        ``r_mpch`` / ``r_arcmin`` (1024, 18), ``nd_label``.
    """
    n = _n_halos(snapshot)[nd_idx]
    d = np.load(DATA_DIR / ("DeltaSigma_kSZ_over_DeltaSigma_total_profiles_arcmin"
                            f"_snap{snapshot}_nd_{nd_idx}_n_{n}_eighth_gen.npz"))
    r_arcmin, r_mpch = d["radii_arcmin"], d["radii_mpch"]
    c = r_mpch[:, 0] / r_arcmin[:, 0]
    # The inner bins are fixed in angle, the outer bins follow the documented law.
    assert np.allclose(r_arcmin[:, :N_INNER], np.linspace(1, 6, N_INNER))
    ro_m, ro_a = outer_radii(c)
    assert np.allclose(ro_m, r_mpch[:, N_INNER:]) and np.allclose(ro_a, r_arcmin[:, N_INNER:])
    return {"fgas": d["prof"], "c": c, "r_mpch": r_mpch, "r_arcmin": r_arcmin,
            "nd_label": str(d["nd_label"]), "n_halos": n}


def _check_geometry(snapshot: int, c: np.ndarray, sims=(0, 341, 682, 1023)) -> float:
    """Max relative difference between stored c and mpch_per_arcmin (per-sim z, h)."""
    om = np.load(DATA_DIR / "camels_params_matrix.npy")[:, 0]
    worst = 0.0
    for i in sims:
        raw = np.load(Path(RAW.format(i)) /
                      f"Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma_snap{snapshot}.npz")
        ca = mpch_per_arcmin(float(raw["redshift"]), om[i], float(raw["HubbleParam"]))
        worst = max(worst, abs(ca / c[i] - 1))
    return worst


def _ridge_cv(Xtr, Ytr, Xte):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    sx = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-4, 3, 50)).fit(sx.transform(Xtr), Ytr)
    return m.predict(sx.transform(Xte))


def cmd_fit(args) -> int:
    from sklearn.model_selection import KFold

    snapshot, nd_idx = args.snapshot, args.nd
    out = _out_dir(args.scratch_root)
    p = load_profiles(snapshot, nd_idx)
    geo = _check_geometry(snapshot, p["c"])
    print(f"snap{snapshot} nd{nd_idx}: geometry check max rel diff {geo:.1e}", flush=True)
    assert geo < 1e-10
    inner, outer, c = p["fgas"][:, :N_INNER], p["fgas"][:, N_INNER:], p["c"]
    n_sims = inner.shape[0]

    gp_mu, gp_sd = np.full_like(outer, np.nan), np.full_like(outer, np.nan)
    q = np.full((4,) + outer.shape, np.nan)  # predictive 2.5/16/84/97.5 % of a noisy stack
    ridge = np.full_like(outer, np.nan)
    fold_id = np.full(n_sims, -1)
    for k, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(inner)):
        m = OuterProfileReconstructor(seed=SEED).fit(inner[tr], c[tr], outer[tr])
        gp_mu[te], gp_sd[te] = m.predict(inner[te], c[te])
        q[:, te] = m.predictive_interval(inner[te], c[te], quantiles=(0.025, 0.16, 0.84, 0.975))
        X = np.column_stack([inner, np.log(c)])
        ridge[te] = _ridge_cv(X[tr], outer[tr], X[te])
        fold_id[te] = k
        err = 100 * np.median(np.abs(gp_mu[te] / outer[te] - 1), axis=0)
        print(f"  fold {k}: nu {np.array2string(m.noise_model()['nu'], precision=1)}; "
              f"median |err| % {np.array2string(err, precision=2)}", flush=True)
    assert np.all(fold_id >= 0)

    final = OuterProfileReconstructor(seed=SEED).fit(inner, c, outer)
    final.meta = {
        "snapshot": snapshot, "redshift": SNAPSHOTS[snapshot][0], "nd_idx": nd_idx,
        "number_density": SNAPSHOTS[snapshot][1][nd_idx], "n_halos": p["n_halos"],
        "dataset_tag": TAG,
        "inputs": "fgas at 1, 1.625, ..., 6 arcmin (9) + log(c [Mpc/h per arcmin])",
        "outputs": "fgas at the 9 outer bins r_j = r6 (20/r6)^(j/9), r6 = 6' c",
        "likelihood": "Student-t (scale, nu per bin); variational GP, inducing = all training inputs",
        "geometry": "c = D_M(z) h pi/10800, flat LCDM with Tcmb0=2.7255 K, Neff=3.04 massless",
    }
    model_path = final.save(out / f"gp_snap{snapshot}_nd{nd_idx}.pt")
    # Round-trip: the reloaded model reproduces the in-memory predictions.
    re = OuterProfileReconstructor.load(model_path)
    a, b = final.predict(inner[:16], c[:16]), re.predict(inner[:16], c[:16])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])

    nm = final.noise_model()
    np.savez(out / f"cv_snap{snapshot}_nd{nd_idx}.npz",
             fgas=p["fgas"], c=c, r_mpch=p["r_mpch"], r_arcmin=p["r_arcmin"],
             gp_mean=gp_mu, gp_std_latent=gp_sd, gp_pred_q=q, ridge=ridge,
             fold_id=fold_id, nd_label=p["nd_label"], n_halos=p["n_halos"],
             final_nu=nm["nu"], final_scale=nm["scale"])
    pe = 100 * (gp_mu / outer - 1)
    summary = {
        "snapshot": snapshot, "nd": nd_idx,
        "gp_median_abs_pct": np.median(np.abs(pe), 0).round(3).tolist(),
        "ridge_median_abs_pct": np.median(np.abs(100 * (ridge / outer - 1)), 0).round(3).tolist(),
        "gp_cov68": np.mean((outer > q[1]) & (outer < q[2]), 0).round(3).tolist(),
        "gp_cov95": np.mean((outer > q[0]) & (outer < q[3]), 0).round(3).tolist(),
        "final_nu": nm["nu"].round(2).tolist(),
    }
    print(json.dumps(summary), flush=True)
    print("saved:", model_path, flush=True)
    return 0


def cmd_noise(args) -> int:
    snapshot = args.snapshot
    n_halos = _n_halos(snapshot)
    fname = f"Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma_snap{snapshot}.npz"
    per_proj = np.empty((1024, len(n_halos), 3, 18))
    for sim in range(1024):
        d = np.load(Path(RAW.format(sim)) / fname)
        order = np.argsort(-np.asarray(d["stellar_masses"]), kind="stable")
        fb = float(d["fb"])
        for q, proj in enumerate(PROJECTIONS):
            ion = d[f"prof2_ionized_gas_DSigma_arcmin2_{proj}"][:, order]
            tot = d[f"prof1_total_DSigma_arcmin2_{proj}"][:, order]
            for i, n in enumerate(n_halos):
                per_proj[sim, i, q] = (np.nanmean(ion[:, :n], 1) / np.nanmean(tot[:, :n], 1) / fb)
        if (sim + 1) % 128 == 0:
            print(f"  snap{snapshot}: {sim + 1}/1024", flush=True)
    sigma = per_proj.std(axis=2, ddof=1) / np.sqrt(3)
    out = _out_dir(args.scratch_root) / f"noise_snap{snapshot}.npz"
    np.savez(out, fgas_per_projection=per_proj, sigma_proj=sigma)
    # Consistency: the 3-projection stack is the mean of per-projection ratios
    # only to first order; report how close the compiled f_gas is to it.
    for i in range(len(n_halos)):
        comp = load_profiles(snapshot, i)["fgas"]
        rel = np.median(np.abs(per_proj[:, i].mean(1) / comp - 1), axis=0)
        print(f"  nd{i}: median |mean_proj / compiled - 1| per bin max {rel.max():.2e}")
    print("saved:", out)
    return 0


def _percentile_band(pe, q):
    return np.nanpercentile(pe, q, axis=0)


def cmd_figures(args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    def _radius_axis(ax, ticks):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

    src = _out_dir(args.scratch_root)
    fig_root = figure_dir("outer_bin_reconstruction")
    # Reference palette (dataviz skill, light mode): series-1 blue for the GP,
    # series-2 orange for the ridge baseline, neutral ink for the noise floor.
    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    ND_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal, light->dark

    summary = {}
    for snapshot, (z, _) in SNAPSHOTS.items():
        noise_path = src / f"noise_snap{snapshot}.npz"
        sig = np.load(noise_path)["sigma_proj"] if noise_path.exists() else None
        fig, axes = plt.subplots(1, 5, figsize=(19, 4.4), sharey=True)
        for i, ax in enumerate(axes):
            d = np.load(src / f"cv_snap{snapshot}_nd{i}.npz")
            outer = d["fgas"][:, N_INNER:]
            r = np.median(d["r_mpch"][:, N_INNER:], axis=0)
            r_lo, r_hi = d["r_mpch"][:, N_INNER:].min(0), d["r_mpch"][:, N_INNER:].max(0)
            pe = 100 * (d["gp_mean"] / outer - 1)
            pr = 100 * (d["ridge"] / outer - 1)
            lo95, lo68, med, hi68, hi95 = _percentile_band(pe, [2.5, 16, 50, 84, 97.5])
            ax.fill_between(r, lo95, hi95, color=BLUE, alpha=0.15, lw=0,
                            label="GP 95% of sims")
            ax.fill_between(r, lo68, hi68, color=BLUE, alpha=0.35, lw=0,
                            label="GP 68% of sims")
            ax.plot(r, med, color=BLUE, lw=2, marker="o", ms=4, label="GP median")
            rl, rh = _percentile_band(pr, [16, 84])
            ax.plot(r, rl, color=ORANGE, lw=1.5, ls="--", label="ridge baseline 68%")
            ax.plot(r, rh, color=ORANGE, lw=1.5, ls="--")
            if sig is not None:
                nf = 100 * np.median(sig[:, i, N_INNER:] / np.abs(outer), axis=0)
                ax.plot(r, nf, color=MUTED, lw=1.2, ls=":",
                        label=r"projection noise $\pm1\sigma$ (lower bound)")
                ax.plot(r, -nf, color=MUTED, lw=1.2, ls=":")
            ax.axhline(0, color=INK, lw=0.6)
            _radius_axis(ax, [3, 5, 10, 20] if snapshot == 74 else [2, 5, 10, 20])
            ax.set_xlim(r_lo[0] * 0.95, r_hi[-1] * 1.05)
            qq = d["gp_pred_q"]
            cov = np.mean((outer > qq[1]) & (outer < qq[2]))
            ax.set_title(f"nd{i}: top {int(d['n_halos'])} halos\n"
                         f"median |err|: GP {np.median(np.abs(pe)):.2f}%, "
                         f"ridge {np.median(np.abs(pr)):.2f}%\n"
                         f"GP 68% interval coverage {cov:.2f}", fontsize=9.5)
            ax.grid(alpha=0.25, lw=0.5)
            summary[(snapshot, i)] = (r, np.median(np.abs(pe), 0), d)
        axes[0].set_ylabel(r"$100\,(f_{\rm gas}^{\rm rec}-f_{\rm gas}^{\rm true})/f_{\rm gas}^{\rm true}$ [%]")
        axes[0].set_ylim(-15, 15)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, 1.07), fontsize=10)
        fig.suptitle(f"Outer-bin f_gas reconstructed from 1'-6' — snap{snapshot}, z={z} "
                     "(5-fold CV over 1024 SB35 sims)", y=1.17, fontsize=13)
        fig.supxlabel(r"outer-bin radius $r$ [comoving Mpc/$h$] (median over sims)",
                      fontsize=11)
        fig.tight_layout()
        path = fig_root / f"outer_bin_recon_pct_error_snap{snapshot}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("saved:", path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (snapshot, (z, _)) in zip(axes, SNAPSHOTS.items()):
        for i in range(5):
            r, mabs, d = summary[(snapshot, i)]
            ax.plot(r, mabs, color=ND_RAMP[i], lw=2, marker="o", ms=4,
                    label=f"nd{i} (top {int(d['n_halos'])})")
        _radius_axis(ax, [3, 5, 10, 20] if snapshot == 74 else [2, 5, 10, 20])
        ax.set_title(f"snap{snapshot}, z={z}", fontsize=12)
        ax.set_xlabel(r"outer-bin radius $r$ [comoving Mpc/$h$] (median over sims)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=9, title="halo sample", title_fontsize=9,
                  loc="lower right", ncol=2)
    axes[0].set_ylabel("median |percentage error| [%]")
    axes[0].set_ylim(bottom=0)
    fig.suptitle("GP outer-bin reconstruction: median |error| per bin (5-fold CV)", fontsize=13)
    fig.tight_layout()
    path = fig_root / "outer_bin_recon_median_abs_error_both_z.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scratch-root", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--snapshot", type=int, choices=sorted(SNAPSHOTS), required=True)
    f.add_argument("--nd", type=int, choices=range(5), required=True)
    n = sub.add_parser("noise")
    n.add_argument("--snapshot", type=int, choices=sorted(SNAPSHOTS), required=True)
    sub.add_parser("figures")
    args = parser.parse_args(argv)
    return {"fit": cmd_fit, "noise": cmd_noise, "figures": cmd_figures}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
