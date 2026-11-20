"""Test RMSE(k) of the seven joint / separate model arms on identical held-out SP(k) curves.

Extends the comparison of ``scripts/joint_dual_vae.py`` (``report``: joint dual_vae
vs joint cVAE vs separate per-(snapshot, n) cVAEs) with four further joint arms trained
on the same stacked data (1024 sims x 10 (snapshot, n) rows; snap74 DESI LRG z = 0.47,
snap82 DESI BGS z = 0.21):

* ``beta_tcvae`` -- ``scripts/joint_beta_tcvae.py``, the conditional beta-TCVAE;
* ``cond_beta_tcvae`` -- ``scripts/joint_cond_beta_tcvae.py``, the beta-TCVAE with the
  conditional prior ``p(z | x, y_hat, ctx)`` and the recurrent baseline ``y_hat``,
  scored at its ``predict`` (decoded at the conditional-prior mean ``mu_p``);
* ``joint_cvae_fgas_std`` -- ``scripts/joint_cvae_fgas_std.py``, the joint cVAE with
  profile input ``[f_gas, log10 sigma_fgas]`` (18 columns);
* ``joint_cvae_nocond`` -- ``scripts/joint_cvae_nocond.py``, the joint cVAE with no
  ``X_cond`` (no n, no z) and no ``X_params`` (no cosmology), the expected worst case.
  Its f_gas/f_b input still carries the per-simulation f_b = Omega_b / Omega_m, so some
  cosmological information leaks in through the input normalisation.

Every arm is scored on the outer test fold of the joint cVAE (``grouped_split`` on
``sim_index``, seed 10). Checks before scoring: each joint arm's run-config split
equals the joint cVAE's; the 18-column fgas_std data and the unconditioned nocond data
have the 9-column data's rows, snapshots, ``sim_index``, ``y`` and f_gas columns; the separate runs hold exactly the
joint test fold's simulations and targets per (snapshot, n) (``_separate_arm``); and
every reloaded seed reproduces its recorded test RMSE (``_joint_arm`` / ``_evaluate``).

Outputs, to ``figures/<YYYY-MM>/<MM-DD>/joint_models_compare/``:

* ``joint_models_rmse_vs_k.png`` (+ ``.pdf``) -- per-(tracer, n) per-k test RMSE, seed
  mean with seed min--max band, seven arms;
* ``joint_models_rmse.{md,json}`` -- RMSE over all k and per ``K_BANDS``, mean +/- sd
  over the three seeds, per (tracer, n), for the seven arms, plus each arm's rank.

Usage::

    python scripts/compare_joint_models.py
    python scripts/compare_joint_models.py --out-dir figures/2026-11/11-17/joint_models_compare --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fgas_spk.models.beta_tcvae  # noqa: E402,F401  (registers "beta_tcvae" for unpickling)
import joint_beta_tcvae as B  # noqa: E402
import joint_cond_beta_tcvae as CB  # noqa: E402  (also registers "cond_beta_tcvae")
import joint_cvae as J  # noqa: E402
import joint_cvae_fgas_std as F  # noqa: E402
import joint_cvae_nocond as N  # noqa: E402
import joint_dual_vae as D  # noqa: E402
from ablate_cosmo_params import K_BANDS, plot_seed_band, style_rmse_axis  # noqa: E402

from fgas_spk.paths import figure_dir  # noqa: E402
from fgas_spk.train import _subset, grouped_split  # noqa: E402

OUT_SUBDIR = "joint_models_compare"
# (arm, legend label, colour, line style). The first three keep the colours and styles of
# joint_dual_vae_rmse_vs_k.png. The seven colours pass the dataviz palette validator on all
# pairs (lightness band, chroma floor, normal-vision dE >= 15, contrast >= 3:1); the one
# CVD WARN (olive vs orange, protan dE 6.1) is separated by line style.
ARMS = (("joint_dual_vae", "joint dual_vae (Arm P)", "#2a78d6", "-"),
        ("joint_cvae", "joint cVAE", "#1a9e77", "-."),
        ("separate", "separate cVAEs", "#eb6834", "--"),
        ("joint_cvae_fgas_std", r"joint cVAE, $[f_\mathrm{gas}, \log_{10}\sigma_{f_\mathrm{gas}}]$", "#c2185b", ":"),
        ("beta_tcvae", "joint β-TCVAE", "#8a6d00", (0, (6, 1.5, 1.2, 1.5, 1.2, 1.5))),
        ("joint_cvae_nocond", r"joint cVAE, $f_\mathrm{gas}$ only, no $n$, $z$, cosmology", "#5c3d99",
         (0, (9, 3))),
        ("cond_beta_tcvae", "joint cond. β-TCVAE (conditional prior)", "#b576af", (0, (2, 1, 2, 1, 8, 1))))
# Joint arm -> (run group, run config, fgas_std, nocond).
JOINT_ARMS = {
    "joint_dual_vae": (D.GROUP, D.RUN_CFG, False, False),
    "joint_cvae": (J.GROUP, J.RUN_CFG, False, False),
    "joint_cvae_fgas_std": (F.GROUP, J.RUN_CFG, True, False),
    "beta_tcvae": (B.GROUP, B.RUN_CFG, False, False),
    "cond_beta_tcvae": (CB.GROUP, CB.RUN_CFG, False, False),
    "joint_cvae_nocond": (N.GROUP, J.RUN_CFG, False, True),
}


def compare(out_dir: Path) -> dict:
    """Score the joint and separate arms on the joint cVAE's test curves; write the figure and table.

    Args:
        out_dir: Output directory (created if needed).

    Returns:
        dict: The per-(snapshot, n) RMSE table (as written to the json).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from fgas_spk.pub_style import PLOT_RC, REDSHIFT, TRACER

    dcs, rc = J._configs(J.RUN_CFG)
    td, snap = J.load_joint(dcs)
    te_mask = grouped_split(td.sim_index, rc.split)["test"]
    te, snap_te = _subset(td, te_mask), snap[te_mask]
    k = np.asarray(td.k, float)
    nds = {s: np.unique(td.nd[snap == s]) for s in J.SNAPS}

    # The unconditioned data: same rows, targets and profile, no X_cond / X_params.
    dcs_nc, _ = J._configs(J.RUN_CFG, nocond=True)
    td_nc, snap_nc = J.load_joint(dcs_nc)
    if not (td_nc.X_cond is None and td_nc.X_params is None and np.array_equal(snap_nc, snap)
            and np.array_equal(td_nc.X, td.X) and np.array_equal(td_nc.y, td.y)
            and np.array_equal(td_nc.sim_index, td.sim_index) and np.array_equal(td_nc.nd, td.nd)):
        raise RuntimeError("nocond joint data is not row-aligned with the conditioned joint data")
    # The 18-column fgas_std data: same rows, targets and f_gas columns as the 9-column data.
    dcs_std, _ = J._configs(J.RUN_CFG, fgas_std=True)
    td_std, snap_std = J.load_joint(dcs_std)
    n_r = td.X.shape[1]
    if not (td_std.X.shape == (td.X.shape[0], 2 * n_r) and np.array_equal(snap_std, snap)
            and np.array_equal(td_std.sim_index, td.sim_index) and np.array_equal(td_std.y, td.y)
            and np.array_equal(td_std.nd, td.nd) and np.array_equal(td_std.X[:, :n_r], td.X)
            and np.array_equal(td_std.X_cond, td.X_cond) and np.array_equal(td_std.X_params, td.X_params)
            and np.array_equal(td_std.X[:, n_r:], np.log10(td_std.X_std))):
        raise RuntimeError("fgas_std joint data is not row-aligned with the 9-column joint data")

    stores, runs = {}, {}
    for arm, (group, run_cfg, fgas_std, nocond) in JOINT_ARMS.items():
        _, rc_arm = J._configs(run_cfg, fgas_std=fgas_std, nocond=nocond)
        if rc_arm.split != rc.split:
            raise RuntimeError(f"{arm}: outer split {rc_arm.split} differs from the joint cVAE's {rc.split}")
        if not np.array_equal(grouped_split(td.sim_index, rc_arm.split)["test"], te_mask):
            raise RuntimeError(f"{arm}: test fold differs from the joint cVAE's")
        te_arm = _subset(td_std, te_mask) if fgas_std else _subset(td_nc, te_mask) if nocond else te
        _, stores[arm] = D._joint_arm(group, te_arm, snap_te, nds)
        runs[arm] = {s: d.name for s, d in J._joint_runs(group).items()}
    stores["separate"] = D._separate_arm(te, snap_te, nds, k)
    stores = {arm: stores[arm] for arm, *_ in ARMS}
    table = D._rmse_table(stores, nds, k)
    # Rank of each arm (1 = lowest seed-mean RMSE) per (snap, n) and metric.
    metric_names = ["rmse"] + [f"rmse_k_{lo:g}_{hi:g}" for lo, hi in K_BANDS]
    for row in table.values():
        for m in metric_names:
            order = sorted(stores, key=lambda a: row[f"{a}/{m}"]["mean"])
            for a in stores:
                row[f"{a}/{m}"]["rank"] = order.index(a) + 1

    # Overall ranking: per metric, the mean over the 10 (tracer, n) panels of each arm's seed-mean RMSE.
    overall = {}
    for m in metric_names:
        means = {a: float(np.mean([row[f"{a}/{m}"]["mean"] for row in table.values()])) for a in stores}
        ranks = {a: float(np.mean([row[f"{a}/{m}"]["rank"] for row in table.values()])) for a in stores}
        overall[m] = [{"arm": a, "mean_rmse_over_panels": means[a], "mean_rank_over_panels": ranks[a]}
                      for a in sorted(stores, key=means.get)]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "joint_models_rmse.json").write_text(json.dumps(
        {"setting": {"design": "arcmin9", "k_max": J.KMAX, "seeds": J.SEEDS, "split_seed": rc.split.seed,
                     "n_test_rows": int(te_mask.sum()), "n_test_sims": int(np.unique(te.sim_index).size),
                     "x_cond": J.X_COND_COLUMNS, "k": k.tolist(), "k_bands": [list(b) for b in K_BANDS],
                     "arms": {arm: {"label": label, "run_cfg": JOINT_ARMS[arm][1] if arm in JOINT_ARMS else None,
                                    "fgas_std": JOINT_ARMS[arm][2] if arm in JOINT_ARMS else False,
                                    "nocond": JOINT_ARMS[arm][3] if arm in JOINT_ARMS else False,
                                    "runs": runs.get(arm)} for arm, label, *_ in ARMS},
                     "nocond_note": "joint_cvae_nocond has no X_cond and no X_params, but its f_gas/f_b input "
                                    "carries the per-simulation f_b = Omega_b/Omega_m",
                     "rmse_definition": "per seed: sqrt(mean over the band's k bins of the per-k test RMSE^2); "
                                        "mean and sd (ddof=1) over the 3 seeds"},
         "overall_ranking": overall, "metrics": table}, indent=2))
    cols = [("rmse", "RMSE (all k)")] + [(f"rmse_k_{lo:g}_{hi:g}", f"k∈({lo:g},{hi:g}]") for lo, hi in K_BANDS]
    short = {"joint_dual_vae": "dual_vae", "joint_cvae": "joint cVAE", "separate": "separate",
             "joint_cvae_fgas_std": "cVAE+σ_fgas", "beta_tcvae": "β-TCVAE", "joint_cvae_nocond": "cVAE nocond",
             "cond_beta_tcvae": "cond. β-TCVAE"}
    lines = ["### Overall ranking (mean over the 10 (tracer, n) panels of the seed-mean test RMSE)", "",
             "| metric | " + " | ".join(f"#{i + 1}" for i in range(len(stores))) + " |",
             "|---|" + "---|" * len(stores)]
    for m, c in cols:
        lines.append(f"| {c} | " + " | ".join(
            f"{short[e['arm']]} {e['mean_rmse_over_panels']:.4f} (rank {e['mean_rank_over_panels']:.1f})"
            for e in overall[m]) + " |")
    lines.append("")
    for m, c in cols:
        lines += [f"### {c} (test, mean ± sd over seeds 0/1/2)", "",
                  "| tracer | n [(Mpc/h)⁻³] | " + " | ".join(short[a] for a in stores) + " |",
                  "|---|---|" + "---|" * len(stores)]
        for t in table.values():
            lines.append(f"| {t['tracer']} | {t['number_density']:.2g} | " + " | ".join(
                f"{t[f'{a}/{m}']['mean']:.4f} ± {t[f'{a}/{m}']['sd']:.4f}" for a in stores) + " |")
        lines.append("")
    md = "\n".join(lines)
    (out_dir / "joint_models_rmse.md").write_text(md)
    print(md)

    def nd_title(n):
        e = int(np.floor(np.log10(n)))
        return rf"$n={n / 10**e:.1f}\times10^{{{e}}}$"

    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 5, figsize=(26, 11.5), sharex=True, sharey=True, layout="constrained")
        for r, s in enumerate(J.SNAPS):
            for c, n in enumerate(nds[s]):
                ax = axes[r, c]
                t = table[f"snap{s}/nd{c}"]
                for arm, _, col, ls in ARMS:
                    # Panel legend: the arm's all-k RMSE (seed mean); arm names are in the figure legend.
                    plot_seed_band(ax, k, np.stack(stores[arm][(s, c)]), col, ls, 2.4,
                                   f"{t[f'{arm}/rmse']['mean']:.4f}")
                style_rmse_axis(ax, k)
                if r == 0:
                    ax.set_xlabel("")
                ax.set_title(f"{TRACER[s]} ($z={REDSHIFT[s]}$), {nd_title(n)}", fontsize=15)
                ax.legend(loc="upper left", frameon=False, fontsize=12, title="RMSE (all k)",
                          title_fontsize=12, handlelength=2.6)
            axes[r, 0].set_ylabel("Test RMSE")
        axes[0, 0].set_ylim(bottom=0)
        handles = [Line2D([], [], color=col, ls=ls, lw=2.4, label=label) for _, label, col, ls in ARMS]
        fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=15,
                   handlelength=3.0)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"joint_models_rmse_vs_k.{ext}", dpi=150)
        plt.close(fig)
    print(out_dir / "joint_models_rmse_vs_k.png")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None, help=f"Default: today's figures/.../{OUT_SUBDIR}.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace an existing figure/table in --out-dir (e.g. after adding an arm).")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else figure_dir(OUT_SUBDIR)
    for name in ("joint_models_rmse_vs_k.png", "joint_models_rmse.json"):
        if (out_dir / name).exists() and not args.overwrite:
            raise SystemExit(f"{out_dir / name} exists; pass another --out-dir rather than overwrite it")
    compare(out_dir)


if __name__ == "__main__":
    main()
