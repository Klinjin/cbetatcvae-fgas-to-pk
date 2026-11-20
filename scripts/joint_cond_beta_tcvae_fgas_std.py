"""Joint conditional-prior beta-TCVAE with the propagated f_gas error as input.

The fgas_std arm of ``scripts/joint_cond_beta_tcvae.py``: the same model and run config
(``scripts/configs/run/cond_beta_tcvae_joint.yaml``; ``fgas_spk.models.cond_beta_tcvae``),
the same stacked data (1024 sims x 10 (snapshot, n) rows, snap74 DESI LRG z = 0.47 and
snap82 DESI BGS z = 0.21), the same outer split (seed 10) and seeds 0/1/2, with one
change to the profile input -- exactly the change of ``scripts/joint_cvae_fgas_std.py``:

* ``X``        -- 18 columns: f_gas(R)/f_b on the 9 angle-fixed bins 1--6 arcmin, then
                  log10 sigma_fgas(R) on the same bins (``joint_cvae.X_PROFILE_DEFINITION_FGAS_STD``);
* ``X_params`` -- the 5 cosmological parameters (unchanged);
* ``X_cond``   -- ``[log10 n, z]`` (unchanged).

sigma_fgas is the dataset's ``fgas_std``: the halo-to-halo standard deviation of the
stacked profile propagated from Delta Sigma_kSZ and Delta Sigma_total (see
``joint_cvae_fgas_std.py``). It feeds every network that reads x: the recognition
network, the prior network, the baseline and the generation network.

Outputs under ``cond_beta_tcvae_fgas_std``:

* ``experiments/runs/cond_beta_tcvae_fgas_std/<run_id>/`` -- per-seed records (with
  ``x_profile.json``), checkpoint under ``$FGAS_SCRATCH_ROOT/models/<run_id>/``;
* ``figures/<YYYY-MM>/<MM-DD>/cond_beta_tcvae_fgas_std/`` -- ``--report``: per-(tracer, n)
  test RMSE(k) of this model (decode at ``mu_p``; the baseline ``y_hat`` is training-only
  and not reported), the f_gas-only
  conditional beta-TCVAE and the joint cVAE with sigma_fgas (seed mean, min--max band),
  the table ``cond_beta_tcvae_fgas_std_rmse.{md,json}`` and the loss-term history;
  ``--diagnostics``: the runner's standard per-run figures (the latent traversal's
  profile row then shows f_gas with its sigma_fgas error bars).

Usage::

    python scripts/joint_cond_beta_tcvae_fgas_std.py --train
    python scripts/joint_cond_beta_tcvae_fgas_std.py --report --diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fgas_spk.models.cond_beta_tcvae  # noqa: E402,F401  (registers "cond_beta_tcvae")
import joint_cond_beta_tcvae as C  # noqa: E402
import joint_cvae as J  # noqa: E402
import joint_cvae_fgas_std as JS  # noqa: E402
from ablate_cosmo_params import K_BANDS, plot_seed_band, style_rmse_axis  # noqa: E402

from fgas_spk.paths import figure_dir  # noqa: E402
from fgas_spk.train import _subset, grouped_split  # noqa: E402

GROUP = "cond_beta_tcvae_fgas_std"
RUN_CFG = C.RUN_CFG  # identical model settings; only the profile input differs
# arm key -> (label, colour, linestyle, run group, profile input)
ARMS = {
    "cond_fgas_std": ("cond. β-TCVAE, [f_gas, log10 σ_fgas] (z=μ_p)", "#2a78d6", "-", GROUP, "std"),
    "cond": ("cond. β-TCVAE, f_gas only (z=μ_p)", "#b576af", "-.", C.GROUP, "base"),
    "cvae_fgas_std": ("joint cVAE, [f_gas, log10 σ_fgas]", "#eb6834", "--", JS.GROUP, "std"),
}


def describe_dimensions() -> dict:
    """Shapes of the joint data with sigma_fgas and the resulting input widths."""
    dcs, _ = J._configs(RUN_CFG, fgas_std=True)
    td, snap = J.load_joint(dcs)
    dims = {
        "rows": int(td.X.shape[0]), "sims": int(np.unique(td.sim_index).size),
        "rows_per_snapshot": {int(s): int((snap == s).sum()) for s in J.SNAPS},
        "X_profile": int(td.X.shape[1]), "X_profile_definition": J.X_PROFILE_DEFINITION_FGAS_STD,
        "X_cond": int(td.X_cond.shape[1]), "X_params": int(td.X_params.shape[1]),
        "y_n_k": int(td.y.shape[1]), "param_names": td.param_names, "x_cond": J.X_COND_COLUMNS,
        "log10_sigma_fgas_min_max": [float(np.log10(td.X_std).min()), float(np.log10(td.X_std).max())],
    }
    print(json.dumps(dims, indent=2))
    return dims


def report(out_dir: Path) -> None:
    """Per-(tracer, n) test RMSE(k) of the three arms on identical held-out curves, and history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fgas_spk.pub_style import PLOT_RC, REDSHIFT, TRACER

    dcs_std, rc = J._configs(RUN_CFG, fgas_std=True)
    dcs_base, _ = J._configs(RUN_CFG)
    td_std, snap = J.load_joint(dcs_std)
    td_base, snap_base = J.load_joint(dcs_base)
    # The arms differ only in the profile input: same rows, f_gas columns and targets.
    n_r = td_base.X.shape[1]
    if not (np.array_equal(snap, snap_base) and np.array_equal(td_std.sim_index, td_base.sim_index)
            and np.array_equal(td_std.y, td_base.y) and np.array_equal(td_std.X[:, :n_r], td_base.X)
            and td_std.X.shape[1] == 2 * n_r):
        raise RuntimeError("fgas_std and f_gas-only joint data are not row-aligned")
    te_mask = grouped_split(td_std.sim_index, rc.split)["test"]
    tes = {"std": _subset(td_std, te_mask), "base": _subset(td_base, te_mask)}
    snap_te, y_te, nd_te = snap[te_mask], td_std.y[te_mask], td_std.nd[te_mask]
    k = np.asarray(td_std.k, float)
    nds = {s: np.unique(td_std.nd[snap == s]) for s in J.SNAPS}

    def per_cell(pred) -> dict:
        return {(s, i): np.sqrt(np.mean((pred[m] - y_te[m]) ** 2, axis=0))
                for s in J.SNAPS for i, n in enumerate(nds[s])
                for m in [(snap_te == s) & (nd_te == n)]}

    rmse: dict[str, dict] = {a: {} for a in ARMS}
    runs = {}
    for arm, (_, _, _, group, inp) in ARMS.items():
        runs[arm] = J._joint_runs(group)
        te = tes[inp]
        for seed, run_dir in runs[arm].items():
            model = J._load_joint_model(run_dir)
            pred = J._predict(model, te)
            rec = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
            if not np.isclose(J._rmse(te.y, pred), rec, rtol=1e-4):
                raise RuntimeError(f"{run_dir.name}: reloaded test RMSE does not match the record")
            for key, v in per_cell(pred).items():
                rmse[arm].setdefault(key, []).append(v)
            print(f"{group} seed {seed} {run_dir.name} test rmse {rec:.4f}", flush=True)

    table = {}
    for (s, i) in rmse["cond_fgas_std"]:
        row = {"snapshot": s, "tracer": TRACER[s], "nd_index": i, "number_density": float(nds[s][i])}
        for arm in ARMS:
            v = np.stack(rmse[arm][(s, i)])  # (n_seeds, n_k)
            for name, sel in [("rmse", np.ones_like(k, bool))] + [
                    (f"rmse_k_{lo:g}_{hi:g}", (k > lo) & (k <= hi)) for lo, hi in K_BANDS]:
                # pooled over the k bins of the band: sqrt(mean_k rmse_k^2) per seed
                per_seed = np.sqrt(np.mean(v[:, sel] ** 2, axis=1))
                row[f"{arm}/{name}"] = {"mean": float(per_seed.mean()), "sd": float(per_seed.std(ddof=1))}
            row[f"{arm}/rmse_k_per_seed"] = v.tolist()
        table[f"snap{s}/nd{i}"] = row

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cond_beta_tcvae_fgas_std_rmse.json").write_text(json.dumps(
        {"setting": {"k": k.tolist(), "seeds": J.SEEDS, "split_seed": rc.split.seed,
                     "x_profile": J.X_PROFILE_DEFINITION_FGAS_STD, "model": rc.model,
                     "model_params": rc.model_params,
                     "runs": {a: {s: d.name for s, d in r.items()} for a, r in runs.items()}},
         "metrics": table}, indent=2))
    cols = [("rmse", "all k")] + [(f"rmse_k_{lo:g}_{hi:g}", f"k∈({lo:g},{hi:g}]") for lo, hi in K_BANDS]
    lines = ["| tracer | n [(Mpc/h)⁻³] | " + " | ".join(f"{c} {ARMS[a][0]}" for _, c in cols for a in ARMS) + " |",
             "|---|---|" + "---|" * (len(cols) * len(ARMS))]
    for t in table.values():
        lines.append(f"| {t['tracer']} | {t['number_density']:.2g} | " + " | ".join(
            f"{t[f'{a}/{m}']['mean']:.4f} ± {t[f'{a}/{m}']['sd']:.4f}" for m, _ in cols for a in ARMS) + " |")
    md = "\n".join(lines)
    (out_dir / "cond_beta_tcvae_fgas_std_rmse.md").write_text(md + "\n")
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
                for arm, (label, col, ls, _, _) in ARMS.items():
                    plot_seed_band(ax, k, np.stack(rmse[arm][(s, c)]), col, ls, 2.2,
                                   f"{label} ({t[f'{arm}/rmse']['mean']:.4f})")
                style_rmse_axis(ax, k)
                if r == 0:
                    ax.set_xlabel("")
                ax.set_title(f"{TRACER[s]} ($z={REDSHIFT[s]}$), {nd_title(n)}", fontsize=15)
                ax.legend(loc="upper left", frameon=False, fontsize=9)
            axes[r, 0].set_ylabel("Test RMSE")
        axes[0, 0].set_ylim(bottom=0)
        fig.savefig(out_dir / "cond_beta_tcvae_fgas_std_rmse_vs_k.png", dpi=150)
        plt.close(fig)
    print(out_dir / "cond_beta_tcvae_fgas_std_rmse_vs_k.png")

    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 5, figsize=(32, 6), layout="constrained")
        for seed, run_dir in runs["cond_fgas_std"].items():
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
        fig.savefig(out_dir / "cond_beta_tcvae_fgas_std_history_terms.png", dpi=150)
        plt.close(fig)
    print(out_dir / "cond_beta_tcvae_fgas_std_history_terms.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dims", action="store_true", help="Print the data shapes / input widths.")
    ap.add_argument("--train", action="store_true", help="Train + record seeds 0/1/2.")
    ap.add_argument("--report", action="store_true",
                    help="Comparison vs the f_gas-only cond. beta-TCVAE and the joint cVAE with sigma_fgas.")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Runner's standard diagnostic figures per seed and tracer.")
    ap.add_argument("--out-dir", default=None, help="Default: today's figures/.../cond_beta_tcvae_fgas_std.")
    args = ap.parse_args()
    if args.dims:
        describe_dimensions()
    if args.train:
        gdir = J._runs_root() / GROUP
        if gdir.exists() and any(gdir.iterdir()):
            raise SystemExit(f"{gdir} already has runs; remove them deliberately to retrain")
        for seed in J.SEEDS:
            J.train(seed, group=GROUP, run_cfg=RUN_CFG, fgas_std=True)
    out = Path(args.out_dir) if args.out_dir else None
    if args.report:
        report(out or figure_dir(GROUP))
    if args.diagnostics:
        J.diagnostics(out or figure_dir(GROUP), group=GROUP, run_cfg=RUN_CFG, fgas_std=True)


if __name__ == "__main__":
    main()
