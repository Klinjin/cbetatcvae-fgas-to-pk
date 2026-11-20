"""One dual_vae over both tracers and all number densities: f_gas(R) + cosmology + z + n -> SP(k).

The dual-codec counterpart of ``scripts/joint_cvae.py``: a **single** ``dual_vae``
composite (Arm P, ``scripts/configs/run/dual_vae_pca_vae_joint.yaml``) trained on
the same stacked data -- 1024 sims x 10 (snapshot, n) rows, snap74 DESI LRG
(z = 0.47) and snap82 DESI BGS (z = 0.21), five number densities each:

* ``X``        -- the stacked f_gas(R)/f_b profile, 9 angle-fixed bins 1--6 arcmin;
* ``X_params`` -- the 5 cosmological parameters;
* ``X_cond``   -- ``[log10 n, z]``.

The data loading (:func:`joint_cvae.load_joint`), the outer split (seed 10, so the
test simulations are those of the joint and per-nd cVAEs), the run record and the
checkpointing are joint_cvae.py's, shared verbatim. Arm P routes the context
``[X_cond, X_params]`` through the VAE-Y encoder and decoder only: the PCA x-codec
is unconditional and the MDN maps x-codes to y-codes. ``val_grouped: true`` draws
the VAE-Y and MDN best-epoch holdouts by simulation (10 rows share 2 targets).

Records go to ``experiments/runs/joint_dual_vae/<run_id>/`` plus one ledger line;
the checkpoint to ``$FGAS_SCRATCH_ROOT/models/<run_id>/model.joblib``.

Report (``--report``), to ``figures/<YYYY-MM>/<MM-DD>/joint_dual_vae/``:

* ``joint_dual_vae_spk_pred.png`` -- SP(k) predicted (MDN mixture mean decoded,
  with the 16--84% band of ``predict_samples``: mapping density + decoder-Y
  observation noise) vs truth for the four test simulations of the joint-cVAE
  figure, for every (tracer, n), with the fractional residual below;
* ``joint_dual_vae_rmse_vs_k.png`` -- test per-k RMSE of the joint dual_vae vs the
  joint cVAE vs the separate per-(snapshot, n) cVAEs, per (tracer, n) (seed mean,
  seed min--max band);
* ``joint_dual_vae_rmse.{md,json}`` -- the corresponding table.

Diagnostics (``--diagnostics``), to the same folder: the runner's standard per-run
figures per seed and per tracer (:func:`joint_cvae.diagnostics`).

Usage::

    python scripts/joint_dual_vae.py --train --seeds 0   # GPU; one seed per debug job
    python scripts/joint_dual_vae.py --report
    python scripts/joint_dual_vae.py --diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_cvae as J  # noqa: E402
from ablate_cosmo_params import K_BANDS, plot_seed_band, style_rmse_axis  # noqa: E402
from nd_compare import _evaluate as _evaluate_separate  # noqa: E402
from nd_compare import _seed_runs  # noqa: E402

from fgas_spk.paths import figure_dir  # noqa: E402
from fgas_spk.train import _subset, grouped_split  # noqa: E402

GROUP = "joint_dual_vae"
RUN_CFG = "scripts/configs/run/dual_vae_pca_vae_joint.yaml"
ARMS = (("joint_dual_vae", "joint dual_vae (Arm P)", "#2a78d6", "-"),
        ("joint_cvae", "joint cVAE", "#1a9e77", "-."),
        ("separate", "separate cVAEs", "#eb6834", "--"))


def _per_k_rmse(pred, y, snap_te, te_nd, nds) -> dict:
    """Per-(snap, nd index) per-k RMSE of one model's test predictions."""
    out = {}
    for s in J.SNAPS:
        for i, n in enumerate(nds[s]):
            m = (snap_te == s) & (te_nd == n)
            out[(s, i)] = np.sqrt(np.mean((pred[m] - y[m]) ** 2, axis=0))
    return out


def _joint_arm(group: str, te, snap_te, nds) -> tuple[dict, dict]:
    """Load a joint group's seeds, check each against its record, return (models, rmse_k lists)."""
    models, rmse_k = {}, {}
    for seed, run_dir in J._joint_runs(group).items():
        model = J._load_joint_model(run_dir)
        pred = J._predict(model, te)
        rec = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
        if not np.isclose(J._rmse(te.y, pred), rec, rtol=1e-4):
            raise RuntimeError(f"{run_dir.name}: reloaded test RMSE does not match the record")
        for key, v in _per_k_rmse(pred, te.y, snap_te, te.nd, nds).items():
            rmse_k.setdefault(key, []).append(v)
        models[seed] = model
        print(f"{group} seed {seed} {run_dir.name} test rmse {rec:.4f}", flush=True)
    return models, rmse_k


def _separate_arm(te, snap_te, nds, k) -> dict:
    """Per-k test RMSE lists of the separate per-(snap, n) cVAEs, on the joint test curves.

    Each separate run is scored by :func:`nd_compare._evaluate` (record checked) and
    must hold exactly the joint test fold's simulations, targets and k grid for its
    (snapshot, n).
    """
    sep_rmse_k = {}
    for s in J.SNAPS:
        for i, n in enumerate(nds[s]):
            m = (snap_te == s) & (te.nd == n)
            for seed, run_dir in sorted(_seed_runs(J._separate_group(s, i)).items()):
                r = _evaluate_separate(run_dir)
                if not (np.isclose(r["nd_value"], n) and np.array_equal(r["sim_index"], te.sim_index[m])
                        and np.array_equal(r["y"], te.y[m]) and np.array_equal(r["k"], k)):
                    raise RuntimeError(f"{run_dir}: test fold differs from the joint model's (snap{s}, n={n:g})")
                sep_rmse_k.setdefault((s, i), []).append(r["rmse_k"])
    return sep_rmse_k


def _rmse_table(stores: dict, nds: dict, k) -> dict:
    """Per-(snap, n) RMSE table: all-k and per K_BAND, seed mean and sd, for every arm of ``stores``."""
    from fgas_spk.pub_style import TRACER

    table = {}
    for (s, i) in next(iter(stores.values())):
        row = {"snapshot": s, "tracer": TRACER[s], "nd_index": i, "number_density": float(nds[s][i])}
        for arm, store in stores.items():
            v = np.stack(store[(s, i)])  # (n_seeds, n_k)
            for name, sel in [("rmse", np.ones_like(k, bool))] + [
                    (f"rmse_k_{lo:g}_{hi:g}", (k > lo) & (k <= hi)) for lo, hi in K_BANDS]:
                # pooled over the k bins of the band: sqrt(mean_k rmse_k^2) per seed
                per_seed = np.sqrt(np.mean(v[:, sel] ** 2, axis=1))
                row[f"{arm}/{name}"] = {"mean": float(per_seed.mean()), "sd": float(per_seed.std(ddof=1))}
            row[f"{arm}/rmse_k_per_seed"] = v.tolist()
        table[f"snap{s}/nd{i}"] = row
    return table


def report(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from fgas_spk.pub_style import PLOT_RC, REDSHIFT, TRACER

    dcs, rc = J._configs(RUN_CFG)
    _, rc_cvae = J._configs(J.RUN_CFG)
    if rc.split != rc_cvae.split:
        raise RuntimeError(f"outer split differs from the joint cVAE's: {rc.split} vs {rc_cvae.split}")
    td, snap = J.load_joint(dcs)
    te_mask = grouped_split(td.sim_index, rc.split)["test"]
    te, snap_te = _subset(td, te_mask), snap[te_mask]
    k = np.asarray(td.k, float)
    nds = {s: np.unique(td.nd[snap == s]) for s in J.SNAPS}

    models, dual_rmse_k = _joint_arm(GROUP, te, snap_te, nds)
    _, cvae_rmse_k = _joint_arm(J.GROUP, te, snap_te, nds)

    # Separate per-(snap, n) cVAEs, scored on the identical test curves.
    sep_rmse_k = _separate_arm(te, snap_te, nds, k)

    stores = {"joint_dual_vae": dual_rmse_k, "joint_cvae": cvae_rmse_k, "separate": sep_rmse_k}
    table = _rmse_table(stores, nds, k)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "joint_dual_vae_rmse.json").write_text(json.dumps(
        {"setting": {"design": "arcmin9", "k_max": J.KMAX, "seeds": J.SEEDS, "split_seed": rc.split.seed,
                     "x_cond": J.X_COND_COLUMNS, "k": k.tolist(), "model": rc.model,
                     "model_params": rc.model_params,
                     "joint_dual_vae_runs": {s: d.name for s, d in J._joint_runs(GROUP).items()},
                     "joint_cvae_runs": {s: d.name for s, d in J._joint_runs(J.GROUP).items()}},
         "metrics": table}, indent=2))
    cols = [("rmse", "RMSE (all k)")] + [(f"rmse_k_{lo:g}_{hi:g}", f"k∈({lo:g},{hi:g}]") for lo, hi in K_BANDS]
    lines = ["| tracer | n [(Mpc/h)⁻³] | " + " | ".join(
                 f"{c} dual_vae | {c} joint cVAE | {c} separate" for _, c in cols) + " |",
             "|---|---|" + "---|---|---|" * len(cols)]
    for t in table.values():
        lines.append(f"| {t['tracer']} | {t['number_density']:.2g} | " + " | ".join(
            " | ".join(f"{t[f'{arm}/{m}']['mean']:.4f} ± {t[f'{arm}/{m}']['sd']:.4f}" for arm in stores)
            for m, _ in cols) + " |")
    md = "\n".join(lines)
    (out_dir / "joint_dual_vae_rmse.md").write_text(md + "\n")
    print(md)

    def nd_title(n):
        e = int(np.floor(np.log10(n)))
        return rf"$n={n / 10**e:.1f}\times10^{{{e}}}$"

    # ---- Figure 1: SP(k) predictions for the joint-cVAE figure's four test sims ----
    model0 = models[0]
    m74 = snap_te == J.SNAPS[0]
    sims_te = np.unique(te.sim_index)
    depth = {sid: te.y[m74 & (te.sim_index == sid)][0].min() for sid in sims_te}
    order = sorted(sims_te, key=lambda sid: depth[sid])
    picks = [order[int(round(p / 100 * (len(order) - 1)))] for p in J.PICK_PERCENTILES]
    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(4, 5, figsize=(26, 17), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [2.2, 1, 2.2, 1]})
        for r, s in enumerate(J.SNAPS):
            for c, n in enumerate(nds[s]):
                ax, axr = axes[2 * r, c], axes[2 * r + 1, c]
                for sid, col in zip(picks, J.SIM_COLOURS):
                    sub = _subset(te, (snap_te == s) & (te.nd == n) & (te.sim_index == sid))
                    y_true = sub.y[0]
                    y_pred = J._predict(model0, sub)[0]
                    draws = np.asarray(model0.predict_samples(sub.X, sub.X_cond, sub.X_params,
                                                              n_samples=500, seed=0), float)[:, 0, :]
                    lo, hi = np.percentile(draws, [16, 84], axis=0)
                    ax.fill_between(k, lo, hi, color=col, alpha=0.2, lw=0)
                    ax.plot(k, y_pred, color=col, lw=2.0)
                    ax.plot(k, y_true, color=col, lw=2.0, ls="--")
                    axr.fill_between(k, lo / y_true - 1, hi / y_true - 1, color=col, alpha=0.2, lw=0)
                    axr.plot(k, y_pred / y_true - 1, color=col, lw=2.0)
                ax.axhline(1.0, color="0.5", lw=1.0)
                axr.axhline(0.0, color="0.5", lw=1.0)
                for a in (ax, axr):
                    a.set_xscale("log")
                    a.grid(True)
                ax.set_title(f"{TRACER[s]} ($z={REDSHIFT[s]}$), {nd_title(n)}", fontsize=15)
                if c == 0:
                    ax.set_ylabel(r"$P_\mathrm{tot}/P_\mathrm{DM}$")
                    axr.set_ylabel("pred/true $-$ 1")
                if r == 1:
                    axr.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
            ymin = min(axes[2 * r, c].get_ylim()[0] for c in range(5))
            ymax = max(axes[2 * r, c].get_ylim()[1] for c in range(5))
            rres = max(max(abs(v) for v in axes[2 * r + 1, c].get_ylim()) for c in range(5))
            for c in range(5):
                axes[2 * r, c].set_ylim(ymin, ymax)
                axes[2 * r + 1, c].set_ylim(-rres, rres)
        handles = [Line2D([], [], color="0.2", lw=2, label="joint dual_vae (MDN mean; band 16–84%)"),
                   Line2D([], [], color="0.2", lw=2, ls="--", label="simulation")]
        for sid, col, p in zip(picks, J.SIM_COLOURS, J.PICK_PERCENTILES):
            ps = te.X_params[te.sim_index == sid][0]
            par = ", ".join(rf"{nm}={v:.3g}" for nm, v in zip(td.param_names, ps) if nm in ("Omega0", "sigma8"))
            handles.append(Line2D([], [], color=col, lw=3, label=f"sim {sid} (LRG depth p{p}; {par})"))
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=14)
        fig.savefig(out_dir / "joint_dual_vae_spk_pred.png", dpi=150)
        plt.close(fig)
    print(out_dir / "joint_dual_vae_spk_pred.png")

    # ---- Figure 2: per-k test RMSE, joint dual_vae vs joint cVAE vs separate cVAEs ----
    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 5, figsize=(26, 10.5), sharex=True, sharey=True, layout="constrained")
        for r, s in enumerate(J.SNAPS):
            for c, n in enumerate(nds[s]):
                ax = axes[r, c]
                t = table[f"snap{s}/nd{c}"]
                for arm, label, col, ls in ARMS:
                    plot_seed_band(ax, k, np.stack(stores[arm][(s, c)]), col, ls, 2.4,
                                   f"{label} ({t[f'{arm}/rmse']['mean']:.4f})")
                style_rmse_axis(ax, k)
                if r == 0:
                    ax.set_xlabel("")
                ax.set_title(f"{TRACER[s]} ($z={REDSHIFT[s]}$), {nd_title(n)}", fontsize=15)
                ax.legend(loc="upper left", frameon=False, fontsize=12)
            axes[r, 0].set_ylabel("Test RMSE")
        axes[0, 0].set_ylim(bottom=0)
        fig.savefig(out_dir / "joint_dual_vae_rmse_vs_k.png", dpi=150)
        plt.close(fig)
    print(out_dir / "joint_dual_vae_rmse_vs_k.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", action="store_true", help="Train + record the seeds of --seeds.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(J.SEEDS),
                    help="Seeds to train (default 0 1 2); one per job fits the debug queue.")
    ap.add_argument("--report", action="store_true", help="Prediction + RMSE figures and table.")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Runner's standard diagnostic figures per seed and tracer.")
    ap.add_argument("--out-dir", default=None, help="Default: today's figures/.../joint_dual_vae.")
    args = ap.parse_args()
    if args.train:
        import yaml

        gdir = J._runs_root() / GROUP
        done = {yaml.safe_load((d / "config_run.yaml").read_text())["seed"]
                for d in gdir.iterdir()} if gdir.exists() else set()
        if done & set(args.seeds):
            raise SystemExit(f"{gdir} already has seeds {sorted(done & set(args.seeds))}; "
                             "remove them deliberately to retrain")
        for seed in args.seeds:
            J.train(seed, group=GROUP, run_cfg=RUN_CFG)
    out_dir = Path(args.out_dir) if args.out_dir else figure_dir(GROUP)
    if args.report:
        report(out_dir)
    if args.diagnostics:
        J.diagnostics(out_dir, group=GROUP, run_cfg=RUN_CFG)


if __name__ == "__main__":
    main()
