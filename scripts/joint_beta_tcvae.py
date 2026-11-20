"""Joint beta-TCVAE: f_gas(R) + cosmology + z + n -> SP(k), against the joint cVAE.

Trains ``fgas_spk.models.beta_tcvae`` (``scripts/configs/run/beta_tcvae_joint.yaml``)
on exactly the data, outer split (seed 10) and seeds (0/1/2) of the joint cVAE of
``scripts/joint_cvae.py``: both snapshots (snap74 DESI LRG z=0.47, snap82 DESI BGS
z=0.21), all five number densities each, X = the 9-bin arcmin f_gas/f_b profile,
X_params = the 5 cosmological parameters, X_cond = [log10 n, z], target SP(k) for
k <= 10 h/Mpc. The data assembly, training/recording and standard diagnostics are
``joint_cvae``'s own functions (imported, not copied, so the two cannot drift);
only the group name, run config and the comparison report are this script's.

Outputs, all under the model's name ``beta_tcvae``:

* ``experiments/runs/beta_tcvae/<run_id>/`` -- per-seed records (configs, x_cond
  definition, env, per-epoch history incl. the MI / TC / dwKL terms, summary);
  one ledger line per run is appended to ``experiments/runs.jsonl`` as for every
  run; the checkpoint goes to ``$FGAS_SCRATCH_ROOT/models/<run_id>/model.joblib``.
* ``figures/<YYYY-MM>/<MM-DD>/beta_tcvae/`` -- ``--report``: per-(tracer, n) test
  RMSE(k) beta-TCVAE vs joint cVAE (seed mean, seed min--max band), the table
  (``beta_tcvae_vs_cvae_rmse.{md,json}``), the ELBO-decomposition history
  (``beta_tcvae_history_terms.png``) and the posterior latent plane coloured by
  each conditioning variable (``beta_tcvae_latent_posterior.png``);
  ``--diagnostics``: the runner's standard per-run figures per seed and tracer.

Usage::

    python scripts/joint_beta_tcvae.py --train
    python scripts/joint_beta_tcvae.py --report --diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fgas_spk.models.beta_tcvae  # noqa: E402,F401  (registers "beta_tcvae")
import joint_cvae as J  # noqa: E402
from ablate_cosmo_params import K_BANDS, plot_seed_band, style_rmse_axis  # noqa: E402

from fgas_spk.paths import figure_dir  # noqa: E402
from fgas_spk.train import _subset, grouped_split  # noqa: E402

GROUP = "beta_tcvae"
RUN_CFG = "scripts/configs/run/beta_tcvae_joint.yaml"
CVAE_GROUP = J.GROUP


def describe_dimensions() -> dict:
    """Shapes of the joint data and the resulting network widths (printed and recorded)."""
    dcs, _ = J._configs(RUN_CFG)
    td, snap = J.load_joint(dcs)
    n_x, n_c, n_p, n_k = td.X.shape[1], td.X_cond.shape[1], td.X_params.shape[1], td.y.shape[1]
    dims = {
        "rows": int(td.X.shape[0]), "sims": int(np.unique(td.sim_index).size),
        "rows_per_snapshot": {int(s): int((snap == s).sum()) for s in J.SNAPS},
        "X_profile": n_x, "X_cond": n_c, "X_params": n_p, "y_n_k": n_k,
        "param_names": td.param_names, "x_cond": J.X_COND_COLUMNS,
        "k_min_max": [float(td.k.min()), float(td.k.max())],
    }
    print(json.dumps(dims, indent=2))
    return dims


def report(out_dir: Path) -> None:
    """beta-TCVAE vs joint cVAE on the identical test fold, plus beta-TC-specific figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fgas_spk.pub_style import PLOT_RC, REDSHIFT, TRACER

    dcs, rc = J._configs(RUN_CFG)
    td, snap = J.load_joint(dcs)
    te_mask = grouped_split(td.sim_index, rc.split)["test"]
    te, snap_te = _subset(td, te_mask), snap[te_mask]
    k = np.asarray(td.k, float)
    nds = {s: np.unique(td.nd[snap == s]) for s in J.SNAPS}

    def per_k(group: str) -> tuple[dict, dict]:
        runs, out = J._joint_runs(group), {}
        models = {}
        for seed, run_dir in runs.items():
            model = J._load_joint_model(run_dir)
            models[seed] = model
            pred = J._predict(model, te)
            rec = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
            if not np.isclose(J._rmse(te.y, pred), rec, rtol=1e-4):
                raise RuntimeError(f"{run_dir.name}: reloaded test RMSE does not match the record")
            for s in J.SNAPS:
                for i, n in enumerate(nds[s]):
                    m = (snap_te == s) & (te.nd == n)
                    out.setdefault((s, i), []).append(np.sqrt(np.mean((pred[m] - te.y[m]) ** 2, axis=0)))
            print(f"{group} seed {seed} {run_dir.name} test rmse {rec:.4f}", flush=True)
        return out, {"runs": runs, "models": models}

    tc_rmse, tc_info = per_k(GROUP)
    cv_rmse, cv_info = per_k(CVAE_GROUP)

    table = {}
    for (s, i) in tc_rmse:
        row = {"snapshot": s, "tracer": TRACER[s], "nd_index": i, "number_density": float(nds[s][i])}
        for arm, store in (("beta_tcvae", tc_rmse), ("cvae", cv_rmse)):
            v = np.stack(store[(s, i)])
            for name, sel in [("rmse", np.ones_like(k, bool))] + [
                    (f"rmse_k_{lo:g}_{hi:g}", (k > lo) & (k <= hi)) for lo, hi in K_BANDS]:
                per_seed = np.sqrt(np.mean(v[:, sel] ** 2, axis=1))
                row[f"{arm}/{name}"] = {"mean": float(per_seed.mean()), "sd": float(per_seed.std(ddof=1))}
            row[f"{arm}/rmse_k_per_seed"] = v.tolist()
        table[f"snap{s}/nd{i}"] = row

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "beta_tcvae_vs_cvae_rmse.json").write_text(json.dumps(
        {"setting": {"k": k.tolist(), "seeds": J.SEEDS, "split_seed": rc.split.seed,
                     "beta_tcvae_runs": {s: d.name for s, d in tc_info["runs"].items()},
                     "cvae_runs": {s: d.name for s, d in cv_info["runs"].items()}},
         "metrics": table}, indent=2))
    cols = [("rmse", "RMSE (all k)")] + [(f"rmse_k_{lo:g}_{hi:g}", f"k∈({lo:g},{hi:g}]") for lo, hi in K_BANDS]
    lines = ["| tracer | n [(Mpc/h)⁻³] | " + " | ".join(f"{c} β-TCVAE | {c} cVAE" for _, c in cols) + " |",
             "|---|---|" + "---|---|" * len(cols)]
    for t in table.values():
        lines.append(f"| {t['tracer']} | {t['number_density']:.2g} | " + " | ".join(
            f"{t[f'beta_tcvae/{m}']['mean']:.4f} ± {t[f'beta_tcvae/{m}']['sd']:.4f} | "
            f"{t[f'cvae/{m}']['mean']:.4f} ± {t[f'cvae/{m}']['sd']:.4f}" for m, _ in cols) + " |")
    md = "\n".join(lines)
    (out_dir / "beta_tcvae_vs_cvae_rmse.md").write_text(md + "\n")
    print(md)

    def nd_title(n):
        e = int(np.floor(np.log10(n)))
        return rf"$n={n / 10**e:.1f}\times10^{{{e}}}$"

    # ---- Figure 1: per-k test RMSE, beta-TCVAE vs joint cVAE, per (tracer, n) ----
    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 5, figsize=(26, 10.5), sharex=True, sharey=True, layout="constrained")
        for r, s in enumerate(J.SNAPS):
            for c, n in enumerate(nds[s]):
                ax = axes[r, c]
                t = table[f"snap{s}/nd{c}"]
                plot_seed_band(ax, k, np.stack(tc_rmse[(s, c)]), "#2a78d6", "-", 2.4,
                               f"β-TCVAE ({t['beta_tcvae/rmse']['mean']:.4f})")
                plot_seed_band(ax, k, np.stack(cv_rmse[(s, c)]), "#eb6834", "--", 2.4,
                               f"joint cVAE ({t['cvae/rmse']['mean']:.4f})")
                style_rmse_axis(ax, k)
                if r == 0:
                    ax.set_xlabel("")
                ax.set_title(f"{TRACER[s]} ($z={REDSHIFT[s]}$), {nd_title(n)}", fontsize=15)
                ax.legend(loc="upper left", frameon=False, fontsize=13)
            axes[r, 0].set_ylabel("Test RMSE")
        axes[0, 0].set_ylim(bottom=0)
        fig.savefig(out_dir / "beta_tcvae_rmse_vs_k.png", dpi=150)
        plt.close(fig)
    print(out_dir / "beta_tcvae_rmse_vs_k.png")

    # ---- Figure 2: ELBO decomposition per epoch (train minibatch avg, val full-holdout) ----
    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 4, figsize=(26, 6), layout="constrained")
        for seed, run_dir in tc_info["runs"].items():
            hist = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            ep = [h["epoch"] for h in hist]
            ls = ("-", "--", ":")[seed % 3]
            for ax, key in zip(axes[:3], ("mi", "tc", "dwkl")):
                ax.plot(ep, [h[key] for h in hist], color="#2a78d6", ls=ls, label=f"train, seed {seed}")
                ax.plot(ep, [h[f"val_{key}"] for h in hist], color="#eb6834", ls=ls, label=f"val, seed {seed}")
            axes[3].plot(ep, [h["train_loss"] for h in hist], color="#2a78d6", ls=ls, label=f"train, seed {seed}")
            axes[3].plot(ep, [h["val_loss"] for h in hist], color="#eb6834", ls=ls, label=f"val, seed {seed}")
            be = json.loads((run_dir / "summary.json").read_text())["best_epoch"]
            for ax in axes:
                ax.axvline(be, color="0.5", ls=ls, lw=1.0)
        for ax, title in zip(axes, ("index-code MI [nats]", "total correlation [nats]",
                                    "dimension-wise KL [nats]", "loss (−ELBO, no 2π const.)")):
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(True)
        axes[0].legend(frameon=False, fontsize=11)
        fig.savefig(out_dir / "beta_tcvae_history_terms.png", dpi=150)
        plt.close(fig)
    print(out_dir / "beta_tcvae_history_terms.png")

    # ---- Figure 3: posterior means on the test fold, coloured by each conditioning variable ----
    model0 = tc_info["models"][0]
    mu, _ = model0.latent_dists(te.X, te.X_cond, te.X_params, y=te.y)
    colour_by = [(te.X_params[:, j], name) for j, name in enumerate(te.param_names)]
    colour_by += [(te.X_cond[:, 0], r"$\log_{10} n$"), (te.X_cond[:, 1], "$z$"),
                  (te.y[:, np.argmin(np.abs(k - 3.0))], r"SP($k\approx3$)")]
    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 4, figsize=(24, 11), layout="constrained")
        for ax, (cval, name) in zip(axes.flat, colour_by):
            sc = ax.scatter(mu[:, 0], mu[:, 1], c=cval, s=6, cmap="viridis", rasterized=True)
            fig.colorbar(sc, ax=ax)
            ax.set_title(name)
            ax.set_xlabel(r"$\mu_{q,1}$")
            ax.set_ylabel(r"$\mu_{q,2}$")
        fig.suptitle("β-TCVAE seed 0: posterior latent means, test fold")
        fig.savefig(out_dir / "beta_tcvae_latent_posterior.png", dpi=150)
        plt.close(fig)
    print(out_dir / "beta_tcvae_latent_posterior.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dims", action="store_true", help="Print the data shapes / network widths.")
    ap.add_argument("--train", action="store_true", help="Train + record seeds 0/1/2.")
    ap.add_argument("--report", action="store_true", help="Comparison vs the joint cVAE + beta-TC figures.")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Runner's standard diagnostic figures per seed and tracer.")
    ap.add_argument("--out-dir", default=None, help="Default: today's figures/.../beta_tcvae.")
    args = ap.parse_args()
    if args.dims:
        describe_dimensions()
    if args.train:
        gdir = J._runs_root() / GROUP
        if gdir.exists() and any(gdir.iterdir()):
            raise SystemExit(f"{gdir} already has runs; remove them deliberately to retrain")
        for seed in J.SEEDS:
            J.train(seed, group=GROUP, run_cfg=RUN_CFG)
    out = Path(args.out_dir) if args.out_dir else None
    if args.report:
        report(out or figure_dir(GROUP))
    if args.diagnostics:
        J.diagnostics(out or figure_dir(GROUP), group=GROUP, run_cfg=RUN_CFG)


if __name__ == "__main__":
    main()
