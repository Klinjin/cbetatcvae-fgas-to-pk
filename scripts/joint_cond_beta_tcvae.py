"""Joint conditional-prior beta-TCVAE vs the fixed-prior beta-TCVAE and the joint cVAE.

Trains ``fgas_spk.models.cond_beta_tcvae``
(``scripts/configs/run/cond_beta_tcvae_joint.yaml``) on exactly the data, outer
split (seed 10) and seeds (0/1/2) of ``scripts/joint_beta_tcvae.py`` and
``scripts/joint_cvae.py``. Data assembly, training/recording and the standard
diagnostics are ``joint_cvae``'s own functions (imported, not copied).

Outputs under the model's name ``cond_beta_tcvae``:

* ``experiments/runs/cond_beta_tcvae/<run_id>/`` -- per-seed records (history incl.
  MI / TC / dwKL in prior-whitened coordinates and the baseline NLL), checkpoint
  under ``$FGAS_SCRATCH_ROOT/models/<run_id>/``;
* ``figures/<YYYY-MM>/<MM-DD>/cond_beta_tcvae/`` -- ``--report``: per-(tracer, n)
  test RMSE(k) for the conditional beta-TCVAE (decode at ``mu_p``; the baseline
  ``y_hat`` is training-only and not reported), the fixed-prior beta-TCVAE and the joint cVAE (seed mean, min--max
  band), the table ``cond_beta_tcvae_rmse.{md,json}``, the loss-term history and
  the test-fold latent plane (conditional prior means vs posterior means);
  ``--diagnostics``: the runner's standard per-run figures.

Usage::

    python scripts/joint_cond_beta_tcvae.py --train
    python scripts/joint_cond_beta_tcvae.py --report --diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fgas_spk.models.beta_tcvae  # noqa: E402,F401  (registers "beta_tcvae" for unpickling)
import fgas_spk.models.cond_beta_tcvae  # noqa: E402,F401  (registers "cond_beta_tcvae")
import joint_beta_tcvae as B  # noqa: E402
import joint_cvae as J  # noqa: E402
from ablate_cosmo_params import K_BANDS, plot_seed_band, style_rmse_axis  # noqa: E402

from fgas_spk.paths import figure_dir  # noqa: E402
from fgas_spk.train import _subset, grouped_split  # noqa: E402

GROUP = "cond_beta_tcvae"
RUN_CFG = "scripts/configs/run/cond_beta_tcvae_joint.yaml"
# arm key -> (label, colour, linestyle)
ARMS = {
    "cond_beta_tcvae": ("cond. β-TCVAE (z=μ_p)", "#2a78d6", "-"),
    "beta_tcvae": ("β-TCVAE, N(0,I) prior", "#8a6d00", "-."),
    "cvae": ("joint cVAE", "#eb6834", "--"),
}


def report(out_dir: Path) -> None:
    """Per-(tracer, n) test RMSE(k) of the four arms, history terms and latent planes."""
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

    def per_cell(pred) -> dict:
        return {(s, i): np.sqrt(np.mean((pred[m] - te.y[m]) ** 2, axis=0))
                for s in J.SNAPS for i, n in enumerate(nds[s])
                for m in [(snap_te == s) & (te.nd == n)]}

    rmse: dict[str, dict] = {a: {} for a in ARMS}
    runs, models = {}, {}
    for arm, group in (("cond_beta_tcvae", GROUP), ("beta_tcvae", B.GROUP), ("cvae", J.GROUP)):
        runs[arm] = J._joint_runs(group)
        for seed, run_dir in runs[arm].items():
            model = J._load_joint_model(run_dir)
            pred = J._predict(model, te)
            rec = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
            if not np.isclose(J._rmse(te.y, pred), rec, rtol=1e-4):
                raise RuntimeError(f"{run_dir.name}: reloaded test RMSE does not match the record")
            for key, v in per_cell(pred).items():
                rmse[arm].setdefault(key, []).append(v)
            if arm == "cond_beta_tcvae":
                models[seed] = model
            print(f"{group} seed {seed} {run_dir.name} test rmse {rec:.4f}", flush=True)

    table = {}
    for (s, i) in rmse["cond_beta_tcvae"]:
        row = {"snapshot": s, "tracer": TRACER[s], "nd_index": i, "number_density": float(nds[s][i])}
        for arm in ARMS:
            v = np.stack(rmse[arm][(s, i)])
            for name, sel in [("rmse", np.ones_like(k, bool))] + [
                    (f"rmse_k_{lo:g}_{hi:g}", (k > lo) & (k <= hi)) for lo, hi in K_BANDS]:
                per_seed = np.sqrt(np.mean(v[:, sel] ** 2, axis=1))
                row[f"{arm}/{name}"] = {"mean": float(per_seed.mean()), "sd": float(per_seed.std(ddof=1))}
            row[f"{arm}/rmse_k_per_seed"] = v.tolist()
        table[f"snap{s}/nd{i}"] = row

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cond_beta_tcvae_rmse.json").write_text(json.dumps(
        {"setting": {"k": k.tolist(), "seeds": J.SEEDS, "split_seed": rc.split.seed,
                     "runs": {a: {s: d.name for s, d in r.items()} for a, r in runs.items()}},
         "metrics": table}, indent=2))
    cols = [("rmse", "all k")] + [(f"rmse_k_{lo:g}_{hi:g}", f"k∈({lo:g},{hi:g}]") for lo, hi in K_BANDS]
    lines = ["| tracer | n [(Mpc/h)⁻³] | " + " | ".join(f"{c} {ARMS[a][0]}" for _, c in cols for a in ARMS) + " |",
             "|---|---|" + "---|" * (len(cols) * len(ARMS))]
    for t in table.values():
        lines.append(f"| {t['tracer']} | {t['number_density']:.2g} | " + " | ".join(
            f"{t[f'{a}/{m}']['mean']:.4f} ± {t[f'{a}/{m}']['sd']:.4f}" for m, _ in cols for a in ARMS) + " |")
    md = "\n".join(lines)
    (out_dir / "cond_beta_tcvae_rmse.md").write_text(md + "\n")
    print(md)

    def nd_title(n):
        e = int(np.floor(np.log10(n)))
        return rf"$n={n / 10**e:.1f}\times10^{{{e}}}$"

    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 5, figsize=(26, 10.5), sharex=True, sharey=True, layout="constrained")
        for r, s in enumerate(J.SNAPS):
            for c, n in enumerate(nds[s]):
                ax = axes[r, c]
                t = table[f"snap{s}/nd{c}"]
                for arm, (label, col, ls) in ARMS.items():
                    plot_seed_band(ax, k, np.stack(rmse[arm][(s, c)]), col, ls, 2.2,
                                   f"{label} ({t[f'{arm}/rmse']['mean']:.4f})")
                style_rmse_axis(ax, k)
                if r == 0:
                    ax.set_xlabel("")
                ax.set_title(f"{TRACER[s]} ($z={REDSHIFT[s]}$), {nd_title(n)}", fontsize=15)
                ax.legend(loc="upper left", frameon=False, fontsize=10)
            axes[r, 0].set_ylabel("Test RMSE")
        axes[0, 0].set_ylim(bottom=0)
        fig.savefig(out_dir / "cond_beta_tcvae_rmse_vs_k.png", dpi=150)
        plt.close(fig)
    print(out_dir / "cond_beta_tcvae_rmse_vs_k.png")

    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 5, figsize=(32, 6), layout="constrained")
        for seed, run_dir in runs["cond_beta_tcvae"].items():
            hist = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            ep = [h["epoch"] for h in hist]
            ls = ("-", "--", ":")[seed % 3]
            for ax, (key, vkey) in zip(axes, (("mi", "val_mi"), ("tc", "val_tc"), ("dwkl", "val_dwkl"),
                                               ("baseline_nll", "val_baseline_nll"),
                                               ("train_loss", "val_loss"))):
                ax.plot(ep, [h[key] for h in hist], color="#2a78d6", ls=ls, label=f"train, seed {seed}")
                ax.plot(ep, [h[vkey] for h in hist], color="#eb6834", ls=ls, label=f"val, seed {seed}")
            be = json.loads((run_dir / "summary.json").read_text())["best_epoch"]
            for ax in axes:
                ax.axvline(be, color="0.5", ls=ls, lw=1.0)
        for ax, title in zip(axes, ("index-code MI I(n;u) [nats]", "total correlation of u [nats]",
                                    "dimension-wise KL of u [nats]", "baseline ŷ NLL [nats]",
                                    "loss (no 2π const.)")):
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(True)
        axes[0].legend(frameon=False, fontsize=11)
        fig.savefig(out_dir / "cond_beta_tcvae_history_terms.png", dpi=150)
        plt.close(fig)
    print(out_dir / "cond_beta_tcvae_history_terms.png")

    model0 = models[0]
    mu_p, _ = model0.latent_dists(te.X, te.X_cond, te.X_params)
    mu_q, _ = model0.latent_dists(te.X, te.X_cond, te.X_params, y=te.y)
    colour_by = [(te.X_params[:, j], name) for j, name in enumerate(te.param_names)]
    colour_by += [(te.X_cond[:, 0], r"$\log_{10} n$"), (te.X_cond[:, 1], "$z$"),
                  (te.y[:, np.argmin(np.abs(k - 3.0))], r"SP($k\approx3$)")]
    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(4, 4, figsize=(24, 22), layout="constrained")
        for row0, (mu, tag) in enumerate(((mu_p, "p"), (mu_q, "q"))):
            for ax, (cval, name) in zip(axes[2 * row0:2 * row0 + 2].flat, colour_by):
                sc = ax.scatter(mu[:, 0], mu[:, 1], c=cval, s=6, cmap="viridis", rasterized=True)
                fig.colorbar(sc, ax=ax)
                ax.set_title(f"{name}  (μ_{tag})")
                ax.set_xlabel(rf"$\mu_{{{tag},1}}$")
                ax.set_ylabel(rf"$\mu_{{{tag},2}}$")
        fig.suptitle("cond. β-TCVAE seed 0, test fold: conditional-prior means (top) vs posterior means (bottom)")
        fig.savefig(out_dir / "cond_beta_tcvae_latent_prior_posterior.png", dpi=150)
        plt.close(fig)
    print(out_dir / "cond_beta_tcvae_latent_prior_posterior.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dims", action="store_true", help="Print the data shapes / network widths.")
    ap.add_argument("--train", action="store_true", help="Train + record seeds 0/1/2.")
    ap.add_argument("--report", action="store_true",
                    help="Comparison vs the fixed-prior beta-TCVAE and the joint cVAE.")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Runner's standard diagnostic figures per seed and tracer.")
    ap.add_argument("--out-dir", default=None, help="Default: today's figures/.../cond_beta_tcvae.")
    args = ap.parse_args()
    if args.dims:
        B.RUN_CFG = RUN_CFG
        B.describe_dimensions()
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
