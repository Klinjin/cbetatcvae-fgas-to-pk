"""Number-density comparison for the arcmin9 k <= 10 h/Mpc CVAE: nd_2 baseline vs another nd bin.

Question: does the halo-sample number density used to stack the input f_gas(R)
profile change the reconstruction of SP(k <= 10 h/Mpc)?

Four reference arms, all at number-density index 2 (seeds 0/1/2, split seed 10,
``cvae_noise.yaml``), each paired with a copy at number-density index ``--nd``
(default 4, the largest number density: snap74 2.4e-3, snap82 3e-2 (Mpc/h)^-3):

=========================================  =====================================  ========
reference group (nd_2)                     input                                  snapshot
=========================================  =====================================  ========
``arcmin9_cosmo_kmax10_snap74_nd2``        f_gas / f_b + 5 cosmological params    74
``arcmin9nofb_noparams_kmax10_snap74_nd2`` f_gas (not / f_b), profile only        74
``arcmin9_cosmo_kmax10_snap82_nd2``        f_gas / f_b + 5 cosmological params    82
``arcmin9nofb_noparams_kmax10_snap82_nd2`` f_gas (not / f_b), profile only        82
=========================================  =====================================  ========

The new arm's data config is the reference run's recorded ``config_data.yaml`` with
``number_density_indices`` changed from ``[2]`` to ``[nd]`` and nothing else (checked
field-by-field on the loaded ``DataConfig``); the run config is the reference
run's recorded ``config_run.yaml`` (seed replaced per seed by ``_train_group``).
Configs are written to ``scripts/configs/data/nd_compare/`` and run records to
``experiments/runs/nd_compare/<base>_nd<nd>/``, where ``<base>`` is the reference group
without its ``_nd2`` suffix (e.g. ``arcmin9_cosmo_kmax10_snap74_nd4``).

The split is grouped by simulation and there is one curve per simulation, so both
arms score the same held-out simulations and the same SP(k) target; only the input
profile differs (checked in ``--report``).

Report (``--report``): two panels (snap74, snap82) overplotting the test per-k RMSE
of the 4 arms of each snapshot (line: seed mean; band: seed min--max; the drawing
helpers of ``scripts/ablate_cosmo_params.py``), plus a table of pooled and k-band
test RMSE (mean +/- sd over seeds). Outputs:
``figures/<YYYY-MM>/<MM-DD>/nd_compare/nd_compare_rmse_vs_k.{png,json,md}``.

Usage::

    python scripts/nd_compare.py --train     # GPU (debug qos), 4 groups x 3 seeds
    python scripts/nd_compare.py --report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablate_cosmo_params import K_BANDS, _load, _predict, plot_seed_band, style_rmse_axis  # noqa: E402
from k3_final import SEEDS, _train_group  # noqa: E402

from fgas_spk.experiment import load_configs  # noqa: E402
from fgas_spk.paths import figure_dir, repo_root  # noqa: E402

REF_ND = 2
KMAX = 10.0
# (snapshot, input) -> reference nd_2 group.
REFERENCE_GROUPS = {
    (74, "cosmo"): "arcmin9_cosmo_kmax10_snap74_nd2",
    (74, "nofb"): "arcmin9nofb_noparams_kmax10_snap74_nd2",
    (82, "cosmo"): "arcmin9_cosmo_kmax10_snap82_nd2",
    (82, "nofb"): "arcmin9nofb_noparams_kmax10_snap82_nd2",
}
SUBDIR = "nd_compare"
CFG_DIR = Path("scripts/configs/data") / SUBDIR
INPUT_LABEL = {"cosmo": r"$f_\mathrm{gas}/f_\mathrm{b}$ + 5 cosmo. params",
               "nofb": r"$f_\mathrm{gas}$ (no $f_\mathrm{b}$), no params"}
# Colour follows the input; line style the number density (reference dashed).
# Colours as the fb (blue) / nofb (orange) pair of ablate_cosmo_params.compare.
COLOUR = {"cosmo": "#2a78d6", "nofb": "#eb6834"}


def _runs_root() -> Path:
    return repo_root() / "experiments" / "runs"


def _group(snap: int, inp: str, nd: int) -> str:
    """Run-record group (relative to ``experiments/runs``) for an arm."""
    ref = REFERENCE_GROUPS[(snap, inp)]
    return ref if nd == REF_ND else f"{SUBDIR}/{_base(ref)}_nd{nd}"


def _base(ref: str) -> str:
    """Reference group name without its ``_nd<REF_ND>`` suffix."""
    sfx = f"_nd{REF_ND}"
    if not ref.endswith(sfx):
        raise ValueError(f"{ref}: expected suffix {sfx}")
    return ref[: -len(sfx)]


def _seed_runs(group: str) -> dict[int, Path]:
    """seed -> run dir for a group (exactly one run per seed in SEEDS)."""
    import yaml

    out: dict[int, list[Path]] = {}
    for d in sorted((_runs_root() / group).iterdir()):
        out.setdefault(yaml.safe_load((d / "config_run.yaml").read_text())["seed"], []).append(d)
    if sorted(out) != sorted(SEEDS) or any(len(v) != 1 for v in out.values()):
        raise RuntimeError(f"{group}: expected one run per seed {SEEDS}, found {out}")
    return {s: v[0] for s, v in out.items()}


def _write_data_config(snap: int, inp: str, nd: int) -> tuple[Path, Path]:
    """Reference run's data config with only the number-density index changed; returns (data, run) cfg."""
    ref_dir = _seed_runs(REFERENCE_GROUPS[(snap, inp)])[0]
    src = ref_dir / "config_data.yaml"
    lines = src.read_text().splitlines()
    i = lines.index("number_density_indices:")
    if lines[i + 1] != f"- {REF_ND}" or lines[i + 2].startswith("- "):
        raise RuntimeError(f"{src}: expected number_density_indices: [{REF_ND}]")
    lines[i + 1] = f"- {nd}"
    head = [f"# {REFERENCE_GROUPS[(snap, inp)]}/{ref_dir.name}/config_data.yaml with",
            f"# number_density_indices [{REF_ND}] -> [{nd}] and nothing else (scripts/nd_compare.py)."]
    dst = repo_root() / CFG_DIR / f"{_base(REFERENCE_GROUPS[(snap, inp)])}_nd{nd}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(head + lines) + "\n")

    # Field-by-field check: the only DataConfig difference is the nd index.
    run_cfg = ref_dir / "config_run.yaml"
    dc_ref, rc = load_configs(src, run_cfg)
    dc_new, _ = load_configs(dst, run_cfg)
    diff = {f.name for f in dataclasses.fields(dc_ref) if getattr(dc_ref, f.name) != getattr(dc_new, f.name)}
    if diff != {"number_density_indices"} or list(dc_new.number_density_indices) != [nd]:
        raise RuntimeError(f"{dst}: unexpected DataConfig differences {diff}")
    assert dc_new.snapshot == snap and dc_new.include_camels_params == (inp == "cosmo")
    assert dc_new.target_mode == "k_range" and tuple(dc_new.k_range) == (0.0, KMAX)
    assert rc.seed == 0 and rc.model == "cvae"
    return dst, run_cfg


def _evaluate(run_dir: Path) -> dict:
    """Test-fold per-k RMSE and summary metrics for a recorded run (record checked)."""
    model, td, te = _load(run_dir)
    y, k = np.asarray(te.y, float), np.asarray(te.k, float)
    if k.max() > KMAX:
        raise RuntimeError(f"{run_dir}: k window exceeds {KMAX}")
    nds = np.unique(np.asarray(te.nd, float))
    if len(nds) != 1:
        raise RuntimeError(f"{run_dir}: expected one number density, found {nds}")
    err = _predict(model, te) - y
    rec = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
    if not np.isclose(np.sqrt(np.mean(err**2)), rec, rtol=1e-4):
        raise RuntimeError(f"{run_dir.name}: reloaded test RMSE does not match the record")
    out = {"rmse": float(np.sqrt(np.mean(err**2))), "rmse_k": np.sqrt(np.mean(err**2, axis=0)),
           "k": k, "nd_value": float(nds[0]), "sim_index": np.asarray(te.sim_index), "y": y}
    for lo, hi in K_BANDS:
        m = (k > lo) & (k <= hi)
        out[f"rmse_k_{lo:g}_{hi:g}"] = float(np.sqrt(np.mean(err[:, m] ** 2)))
    return out


def report(nd: int, out_dir: Path, inputs: tuple[str, ...] = ("cosmo", "nofb")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fgas_spk.pub_style import PLOT_RC, REDSHIFT, TRACER

    metric_names = ["rmse", *(f"rmse_k_{lo:g}_{hi:g}" for lo, hi in K_BANDS)]
    table, curves = {}, {}
    for (snap, inp) in REFERENCE_GROUPS:
        if inp not in inputs:
            continue
        ref_fold = None
        for n in (REF_ND, nd):
            group = _group(snap, inp, n)
            per_seed = []
            for seed, run_dir in sorted(_seed_runs(group).items()):
                r = _evaluate(run_dir)
                # Both nd arms must score the same held-out sims and SP(k) target.
                if ref_fold is None:
                    ref_fold = (r["sim_index"], r["y"])
                elif not (np.array_equal(ref_fold[0], r["sim_index"]) and np.array_equal(ref_fold[1], r["y"])):
                    raise RuntimeError(f"{group} seed {seed}: test fold differs from the reference")
                per_seed.append(r)
                print(f"snap{snap} {inp:5s} nd_{n} seed {seed} rmse {r['rmse']:.4f}", flush=True)
            key = f"snap{snap}/{inp}/nd{n}"
            curves[key] = {"k": per_seed[0]["k"], "rmse_k": np.stack([p["rmse_k"] for p in per_seed])}
            table[key] = {"group": group, "snapshot": snap, "input": inp, "nd_index": n,
                          "number_density": per_seed[0]["nd_value"],
                          "runs": [p_dir.name for _, p_dir in sorted(_seed_runs(group).items())],
                          **{m: {"mean": float(np.mean([p[m] for p in per_seed])),
                                 "sd": float(np.std([p[m] for p in per_seed], ddof=1))} for m in metric_names},
                          "rmse_k_per_seed": curves[key]["rmse_k"].tolist()}

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"nd_compare_rmse_vs_k_nd{REF_ND}_vs_nd{nd}"
    (out_dir / f"{stem}.json").write_text(json.dumps(
        {"setting": {"design": "arcmin9", "k_max": KMAX, "seeds": SEEDS, "split_seed": 10,
                     "k": {s: curves[f"snap{s}/{inputs[0]}/nd{REF_ND}"]["k"].tolist() for s in (74, 82)}},
         "metrics": table}, indent=2))

    cols = [("rmse", "RMSE (all k)")] + [(f"rmse_k_{lo:g}_{hi:g}", f"RMSE k∈({lo:g},{hi:g}]") for lo, hi in K_BANDS]
    lines = ["| snap | input | nd | n̄ [(Mpc/h)⁻³] | " + " | ".join(c for _, c in cols) + " |",
             "|---|---|---|---|" + "---|" * len(cols)]
    for t in table.values():
        lines.append(f"| {t['snapshot']} | {t['input']} | nd_{t['nd_index']} | {t['number_density']:.3g} | "
                     + " | ".join(f"{t[m]['mean']:.4f} ± {t[m]['sd']:.4f}" for m, _ in cols) + " |")
    md = "\n".join(lines)
    (out_dir / f"{stem}.md").write_text(md + "\n")
    print(md)

    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(16, 7.5), layout="constrained")
        for ax, snap in zip(axes, (74, 82)):
            for inp in inputs:
                for n, ls in ((REF_ND, "--"), (nd, "-")):
                    key = f"snap{snap}/{inp}/nd{n}"
                    t = table[key]
                    plot_seed_band(ax, curves[key]["k"], curves[key]["rmse_k"], COLOUR[inp], ls, 2.4,
                                   f"{INPUT_LABEL[inp]}, nd$_{n}$ "
                                   rf"($\bar n={t['number_density']:.2g}$; RMSE {t['rmse']['mean']:.4f})")
            style_rmse_axis(ax, curves[f"snap{snap}/{inputs[0]}/nd{REF_ND}"]["k"])
            ax.set_title(f"{TRACER[snap]} ($z={REDSHIFT[snap]}$)")
            ax.legend(loc="upper left", frameon=False, fontsize=12)
        axes[0].set_ylabel("Test reconstruction RMSE")
        axes[0].set_ylim(bottom=0)
        axes[1].text(0.97, 0.03, "9 arcmin bins, " r"$k\leq10\ h\,\mathrm{Mpc}^{-1}$" "\n"
                     r"$\bar n$ in $(h^{-1}\mathrm{Mpc})^{-3}$",
                     transform=axes[1].transAxes, ha="right", va="bottom", fontsize=15)
        fig.savefig(out_dir / f"{stem}.png", dpi=200)
        plt.close(fig)
    print(out_dir / f"{stem}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nd", type=int, default=4, choices=(0, 1, 3, 4),
                    help="Number-density index of the new arms (default 4: the largest).")
    ap.add_argument("--train", action="store_true", help="Train + record seeds 0/1/2 for the new arms of --inputs.")
    ap.add_argument("--report", action="store_true", help="Evaluate nd_2 vs nd_<nd>; write figure + table.")
    ap.add_argument("--inputs", nargs="+", default=["cosmo", "nofb"], choices=("cosmo", "nofb"),
                    help="Input arms to train/report (default both).")
    ap.add_argument("--out-dir", default=None, help="Default: today's figures/.../nd_compare.")
    args = ap.parse_args()
    if args.train:
        for (snap, inp) in REFERENCE_GROUPS:
            if inp not in args.inputs:
                continue
            group = _group(snap, inp, args.nd)
            gdir = _runs_root() / group
            if gdir.exists() and any(gdir.iterdir()):
                print(f"skip {group}: already has runs", flush=True)
                continue
            data_cfg, run_cfg = _write_data_config(snap, inp, args.nd)
            for d in _train_group(data_cfg, run_cfg, group):
                print(group, d.name, flush=True)
    if args.report:
        report(args.nd, Path(args.out_dir) if args.out_dir else figure_dir(SUBDIR), tuple(args.inputs))


if __name__ == "__main__":
    main()
