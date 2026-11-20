"""Joint cVAE with the propagated f_gas error as input: [f_gas, log10 sigma_fgas](R) + cosmology + z + n -> SP(k).

The fgas_std arm of ``scripts/joint_cvae.py``: the same single conditional VAE
(``scripts/configs/run/cvae_noise_joint.yaml``), the same stacked data (1024 sims x
10 (snapshot, n) rows, snap74 DESI LRG z = 0.47 and snap82 DESI BGS z = 0.21), the
same outer split (seed 10) and seeds, with one change to the profile input:

* ``X``        -- 18 columns: the stacked f_gas(R)/f_b on the 9 angle-fixed bins
                  1--6 arcmin, then log10 sigma_fgas(R) on the same bins;
* ``X_params`` -- the 5 cosmological parameters (unchanged);
* ``X_cond``   -- ``[log10 n, z]`` (unchanged).

sigma_fgas is the dataset's ``fgas_std``, i.e. the compiled file's ``prof_std`` as
read by :func:`fgas_spk.builder.build_fgas_spk_from_compiled`: the producer's
(``SimulationStacker/tests/test_CAMELS_eighth_gen.ipynb``, ``_stack_ratio``)
propagation of the halo-to-halo standard deviations of the stacked Delta Sigma_kSZ
(ionised) and Delta Sigma_total profiles,
``|f_gas| sqrt((sigma_ion/DS_ion)^2 + (sigma_tot/DS_tot)^2)``. It enters as log10
because it is strictly positive and spans more than a decade across n and R.

Records go to ``experiments/runs/joint_cvae_fgas_std/<run_id>/`` (with an extra
``x_profile.json`` defining the 18 profile columns) plus one ledger line; the
checkpoint to ``$FGAS_SCRATCH_ROOT/models/<run_id>/model.joblib``.

Report (``--report``), to ``figures/<YYYY-MM>/<MM-DD>/joint_cvae_fgas_std/``:

* ``joint_cvae_fgas_std_rmse_vs_k.png`` -- test per-k RMSE with vs without
  sigma_fgas, per (tracer, n) (seed mean, seed min--max band), on identical
  held-out curves;
* ``joint_cvae_fgas_std_rmse.{md,json}`` -- the corresponding table.

Diagnostics (``--diagnostics``): the runner's standard per-run figures per seed and
tracer (:func:`joint_cvae.diagnostics`).

Usage::

    python scripts/joint_cvae_fgas_std.py --train     # GPU, seeds 0/1/2
    python scripts/joint_cvae_fgas_std.py --report
    python scripts/joint_cvae_fgas_std.py --diagnostics
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
from joint_dual_vae import _joint_arm  # noqa: E402

from fgas_spk.paths import figure_dir  # noqa: E402
from fgas_spk.train import _subset, grouped_split  # noqa: E402

GROUP = "joint_cvae_fgas_std"
ARMS = (("fgas_std", "joint cVAE, [f_gas, log10 σ_fgas]", "#2a78d6", "-"),
        ("baseline", "joint cVAE, f_gas only", "#eb6834", "--"))


def report(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fgas_spk.pub_style import PLOT_RC, REDSHIFT, TRACER

    dcs_std, rc = J._configs(J.RUN_CFG, fgas_std=True)
    dcs_base, _ = J._configs(J.RUN_CFG)
    td_std, snap = J.load_joint(dcs_std)
    td_base, snap_base = J.load_joint(dcs_base)
    # The two arms differ only in the profile input: same rows, f_gas columns, targets.
    n_r = td_base.X.shape[1]
    if not (np.array_equal(snap, snap_base) and np.array_equal(td_std.sim_index, td_base.sim_index)
            and np.array_equal(td_std.y, td_base.y) and np.array_equal(td_std.X[:, :n_r], td_base.X)
            and td_std.X.shape[1] == 2 * n_r):
        raise RuntimeError("fgas_std and baseline joint data are not row-aligned")
    te_mask = grouped_split(td_std.sim_index, rc.split)["test"]
    snap_te = snap[te_mask]
    te_std, te_base = _subset(td_std, te_mask), _subset(td_base, te_mask)
    k = np.asarray(td_std.k, float)
    nds = {s: np.unique(td_std.nd[snap == s]) for s in J.SNAPS}

    _, std_rmse_k = _joint_arm(GROUP, te_std, snap_te, nds)
    _, base_rmse_k = _joint_arm(J.GROUP, te_base, snap_te, nds)
    stores = {"fgas_std": std_rmse_k, "baseline": base_rmse_k}

    table = {}
    for (s, i) in std_rmse_k:
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

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "joint_cvae_fgas_std_rmse.json").write_text(json.dumps(
        {"setting": {"design": "arcmin9", "k_max": J.KMAX, "seeds": J.SEEDS, "split_seed": rc.split.seed,
                     "x_cond": J.X_COND_COLUMNS, "x_profile": J.X_PROFILE_DEFINITION_FGAS_STD,
                     "k": k.tolist(), "model": rc.model, "model_params": rc.model_params,
                     "fgas_std_runs": {s: d.name for s, d in J._joint_runs(GROUP).items()},
                     "baseline_runs": {s: d.name for s, d in J._joint_runs(J.GROUP).items()}},
         "metrics": table}, indent=2))
    cols = [("rmse", "RMSE (all k)")] + [(f"rmse_k_{lo:g}_{hi:g}", f"k∈({lo:g},{hi:g}]") for lo, hi in K_BANDS]
    lines = ["| tracer | n [(Mpc/h)⁻³] | " + " | ".join(
                 f"{c} with σ_fgas | {c} f_gas only" for _, c in cols) + " |",
             "|---|---|" + "---|---|" * len(cols)]
    for t in table.values():
        lines.append(f"| {t['tracer']} | {t['number_density']:.2g} | " + " | ".join(
            " | ".join(f"{t[f'{arm}/{m}']['mean']:.4f} ± {t[f'{arm}/{m}']['sd']:.4f}" for arm in stores)
            for m, _ in cols) + " |")
    md = "\n".join(lines)
    (out_dir / "joint_cvae_fgas_std_rmse.md").write_text(md + "\n")
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
                for arm, label, colour, ls in ARMS:
                    plot_seed_band(ax, k, np.stack(stores[arm][(s, c)]), colour, ls, 2.4,
                                   f"{label} ({t[f'{arm}/rmse']['mean']:.4f})")
                style_rmse_axis(ax, k)
                if r == 0:
                    ax.set_xlabel("")
                ax.set_title(f"{TRACER[s]} ($z={REDSHIFT[s]}$), {nd_title(n)}", fontsize=15)
                ax.legend(loc="upper left", frameon=False, fontsize=12)
            axes[r, 0].set_ylabel("Test RMSE")
        axes[0, 0].set_ylim(bottom=0)
        fig.savefig(out_dir / "joint_cvae_fgas_std_rmse_vs_k.png", dpi=150)
        plt.close(fig)
    print(out_dir / "joint_cvae_fgas_std_rmse_vs_k.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", action="store_true", help="Train + record the seeds of --seeds.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(J.SEEDS),
                    help="Seeds to train (default 0 1 2).")
    ap.add_argument("--report", action="store_true", help="RMSE figure and table vs the f_gas-only joint cVAE.")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Runner's standard diagnostic figures per seed and tracer.")
    ap.add_argument("--out-dir", default=None, help="Default: today's figures/.../joint_cvae_fgas_std.")
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
            J.train(seed, group=GROUP, run_cfg=J.RUN_CFG, fgas_std=True)
    out_dir = Path(args.out_dir) if args.out_dir else figure_dir(GROUP)
    if args.report:
        report(out_dir)
    if args.diagnostics:
        J.diagnostics(out_dir, group=GROUP, run_cfg=J.RUN_CFG, fgas_std=True)


if __name__ == "__main__":
    main()
