"""Cosmological-parameter ablation for the CVAE on SP(k <= 10 h/Mpc), arcmin9 design.

Setting: snap74, the 9 arcmin-design bins <= 6 arcmin (fixed in angle across
sims), ``cvae_noise.yaml``, target cut to k in [0, 10] h/Mpc, grouped split seed
10, training seeds 0/1/2 -- the ``arcmin9`` arm of ``scripts/kmax10_noparams.py``.
The only thing varied between arms is which CAMELS cosmological parameters are fed
to the model as ``X_params``:

* ``all``   -- all five (Omega0, sigma8, OmegaBaryon, HubbleParam, n_s); the
  existing ``arcmin9_cosmo_kmax10_snap74_nd2`` runs.
* ``none``  -- profile only; the existing ``arcmin9_noparams_kmax10_snap74_nd2`` runs.
* ``drop-<p>`` -- leave-one-out: all five except ``p`` (what is lost when ``p``
  is withheld and the others are kept; a parameter whose information is
  redundant with the profile or the other parameters scores ~0 here).
* ``only-<p>`` -- single parameter: ``p`` alone (what ``p`` supplies on its own).
* ``drop-Omega0+OmegaBaryon`` / ``only-Omega0+OmegaBaryon`` -- the pair. The input
  f_gas is divided by the per-sim f_b = Omega_b / Omega_m
  (``fgas_spk.builder``), so this pair tests whether the normalisation already
  carries their information, or instead removes it from the profile so that
  only the conditioning can supply the cosmic baryon fraction.

New arms are recorded in ``experiments/runs/arcmin9_cosmo-<arm>_kmax10_snap74_nd2/``;
their data configs are written to ``scripts/configs/data/ablation/``.

Report (``--report``): test-fold metrics per seed, mean +/- sd over seeds:
pooled RMSE; RMSE in k bands (0, 1], (1, 3], (3, 10] h/Mpc; median over k of
RMSE(k); fraction of k bins with RMSE(k) < 0.01; and the recovered fraction
``(RMSE_none - RMSE_arm) / (RMSE_none - RMSE_all)`` of the full-conditioning gain
(computed from the seed-mean pooled RMSEs). It also computes permutation importance on
the trained ``all`` models: the test-fold pooled-RMSE increase when one parameter
column (or the Omega0+OmegaBaryon pair, jointly) is permuted across test sims,
averaged over ``N_PERM`` permutations. The figure overplots the test-split
per-k RMSE (the top panel of the runner's ``__rmse_vs_k_pub`` figure,
:func:`fgas_spk.train.plot_rmse_vs_k_pub`) for every arm: left, leave-one-out;
right, single parameter (line: seed mean; band: seed min--max).

Outputs: ``figures/<YYYY-MM>/<MM-DD>/cosmo_ablation/ablate_cosmo_params.{png,json,md}``.

``--snap 82`` runs the same ablation at snap82 (DESI BGS, z = 0.21); groups are then
``..._snap82_nd2`` and the report goes to ``figures/.../cosmo_ablation_snap82/``.

Usage::

    python scripts/ablate_cosmo_params.py --train     # GPU (debug qos), ~10 min
    python scripts/ablate_cosmo_params.py --report
    python scripts/ablate_cosmo_params.py --snap 82 --train --arms only-Omega0+OmegaBaryon
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k3_final import _train_group  # noqa: E402

from fgas_spk.experiment import load_configs  # noqa: E402
from fgas_spk.loader import load_training_data  # noqa: E402
from fgas_spk.paths import figure_dir, fill_data_root, repo_root  # noqa: E402
from fgas_spk.train import _RMSE_PUB_K_REF, _RMSE_PUB_REF_PCT, _subset, grouped_split  # noqa: E402

SNAP = 74
KMAX = 10.0
SEEDS = (0, 1, 2)
N_PERM = 20
PERM_SEED = 20261001
def _src_cfg(snap: int) -> Path:
    """The arcmin9 (cosmology) k<=10 data config for ``snap`` (written by kmax10_noparams.py)."""
    return Path(f"scripts/configs/data/config_eighth_gen_compiled_snap{snap}_arcmin_le6_kmax10.yaml")


SRC_CFG = _src_cfg(SNAP)
RUN_CFG = Path("scripts/configs/run/cvae_noise.yaml")
CFG_DIR = Path("scripts/configs/data/ablation")
PARAMS = ("Omega0", "sigma8", "OmegaBaryon", "HubbleParam", "n_s")
PAIR = ("Omega0", "OmegaBaryon")
LATEX = {"Omega0": r"\Omega_\mathrm{m}", "sigma8": r"\sigma_8", "OmegaBaryon": r"\Omega_\mathrm{b}",
         "HubbleParam": "h", "n_s": "n_s", "f_b": r"f_\mathrm{b}"}
PAIR_KEY = "+".join(PAIR)
# Colour follows the parameter (the one dropped, or the one kept), the same in both panels.
# Reference categorical slots 1-7, validated (light surface): all checks pass.
COLOUR = {"Omega0": "#2a78d6", "sigma8": "#eb6834", "OmegaBaryon": "#1baf7a",
          "HubbleParam": "#eda100", "n_s": "#e87ba4", PAIR_KEY: "#008300", "f_b": "#4a3aa7"}
# Derived-parameter arms (f_b-normalised input only): conditioning on the cosmic baryon
# fraction f_b = OmegaBaryon / Omega0, stored as an extra camels_params column in the
# 20260930-arcmin-all-fbcol dataset (scripts/build_dataset_arcmin_fbcol.py).
FB_ARMS = {"only-f_b": ("f_b",)}
FBCOL_TAG = "20260930-arcmin-all-fbcol"
BASELINE_GROUPS = {"all": f"arcmin9_cosmo_kmax{KMAX:g}_snap{SNAP}_nd2",
                   "none": f"arcmin9_noparams_kmax{KMAX:g}_snap{SNAP}_nd2"}
K_BANDS = ((0.0, 1.0), (1.0, 3.0), (3.0, KMAX))
# Input variant: "fb" -- f_gas / f_b (the standard 20260930-arcmin-all dataset);
# "nofb" -- f_gas without the f_b = Omega_b / Omega_m normalisation
# (scripts/build_dataset_eighth_gen_arcmin_nofb.py). Set by --nofb.
TAGS = {"fb": "20260930-arcmin-all", "nofb": "20261001-arcmin-all-nofb"}
PREFIX = {"fb": "arcmin9", "nofb": "arcmin9nofb"}
VARIANT = "fb"


def _ablation_arms() -> dict[str, tuple[str, ...]]:
    """New arm name -> the conditioning parameters it keeps (in column order)."""
    arms = {}
    for p in PARAMS:
        arms[f"drop-{p}"] = tuple(q for q in PARAMS if q != p)
    arms[f"drop-{PAIR_KEY}"] = tuple(q for q in PARAMS if q not in PAIR)
    for p in PARAMS:
        arms[f"only-{p}"] = (p,)
    arms[f"only-{PAIR_KEY}"] = PAIR
    return arms


ARMS = _ablation_arms()


def _group(arm: str) -> str:
    if VARIANT == "fb" and arm in BASELINE_GROUPS:
        return BASELINE_GROUPS[arm]
    pre = PREFIX[VARIANT]
    mid = {"all": "cosmo", "none": "noparams"}.get(arm, f"cosmo-{arm}")
    return f"{pre}_{mid}_kmax{KMAX:g}_snap{SNAP}_nd2"


def _kept(arm: str) -> tuple[str, ...]:
    if arm in BASELINE_GROUPS:
        return {"all": PARAMS, "none": ()}[arm]
    return FB_ARMS[arm] if arm in FB_ARMS else ARMS[arm]


def _arms() -> list[str]:
    """Every arm reported for the current input variant."""
    return ["all", "none", *ARMS, *(FB_ARMS if VARIANT == "fb" else ())]


def _tag(arm: str) -> str:
    """Dataset tag an arm trains on: the f_b-column copy when it conditions on f_b."""
    return FBCOL_TAG if "f_b" in _kept(arm) else TAGS[VARIANT]


def _write_data_config(arm: str) -> Path:
    """The arcmin9 k<=10 data config with the arm's conditioning subset and the variant's dataset tag."""
    src = repo_root() / SRC_CFG
    kept = _kept(arm)
    out = [f"# {SRC_CFG.name} with the cosmological conditioning cut to {list(kept)}"
           f" and dataset tag {_tag(arm)} (ablation arm {arm}, input variant {VARIANT};"
           " scripts/ablate_cosmo_params.py)."]
    hits = {"tag:": 0, "include_camels_params:": 0, "camels_param_names:": 0}
    for ln in src.read_text().splitlines():
        if ln.startswith("tag:"):
            ln = f"tag: '{_tag(arm)}'"
        elif ln.startswith("include_camels_params:"):
            ln = f"include_camels_params: {'true' if kept else 'false'}"
        elif ln.startswith("camels_param_names:"):
            ln = f"camels_param_names: [{', '.join(kept)}]" if kept else "camels_param_names: null"
        for key in hits:
            hits[key] += ln.startswith(key)
        out.append(ln)
    if any(v != 1 for v in hits.values()):
        raise RuntimeError(f"{src}: expected one line each of {list(hits)}, found {hits}")
    sfx = "" if VARIANT == "fb" else f"_{VARIANT}"
    dst = repo_root() / CFG_DIR / f"{SRC_CFG.stem}{sfx}_params-{arm}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n")
    return dst


def _run_dir(group: str, seed: int) -> Path:
    hits = [d for d in sorted((repo_root() / "experiments" / "runs" / group).iterdir())
            if yaml.safe_load((d / "config_run.yaml").read_text())["seed"] == seed]
    if len(hits) != 1:
        raise RuntimeError(f"{group} seed {seed}: found {hits}")
    return hits[0]


def _load(run_dir: Path):
    """(model, full data, test subset) for a recorded run; test RMSE checked against the record."""
    env = json.loads((run_dir / "env.json").read_text())
    dc, rc = load_configs(run_dir / "config_data.yaml", run_dir / "config_run.yaml")
    td = load_training_data(fill_data_root(dc))
    if td.source_path != env["dataset"]["path"]:
        raise RuntimeError(f"Dataset mismatch for {run_dir}")
    model = joblib.load(Path(env["roots"]["scratch_root"]) / "models" / run_dir.name / "model.joblib")
    te = _subset(td, grouped_split(td.sim_index, rc.split)["test"])
    return model, td, te


def _predict(model, te, X_params=None) -> np.ndarray:
    y = np.asarray(te.y, float)
    xp = te.X_params if X_params is None else X_params
    return np.asarray(model.predict(te.X, te.X_cond, xp), float).reshape(y.shape)


def _evaluate(arm: str, seed: int, ref: dict | None) -> dict:
    """Test-fold metrics and per-k RMSE for one arm and seed."""
    run_dir = _run_dir(_group(arm), seed)
    model, td, te = _load(run_dir)
    kept = _kept(arm)
    # The recorded fb baselines used the pre-cut 9-bin le6 dataset, bit-identical in
    # the 9 kept bins (the test-fold X equality below checks it).
    ok_tags = (_tag(arm), "20260930-arcmin-le6") if VARIANT == "fb" else (_tag(arm),)
    if not any(f"__{t}.npz" in str(td.source_path) for t in ok_tags):
        raise RuntimeError(f"{arm}: trained on {td.source_path}, expected tag {_tag(arm)}")
    names = tuple(td.param_names) if te.X_params is not None else ()
    if names != kept:
        raise RuntimeError(f"{arm}: model conditioned on {names}, expected {kept}")
    y, k = np.asarray(te.y, float), np.asarray(te.k, float)
    if k.max() > KMAX:
        raise RuntimeError(f"{arm}: k window exceeds {KMAX}")
    # Every arm must score the same held-out curves in the same order.
    if ref is not None and not (np.array_equal(ref["sim_index"], te.sim_index) and np.array_equal(ref["y"], y)
                                and np.array_equal(ref["X"], te.X)):
        raise RuntimeError(f"{arm} seed {seed}: test fold differs from the reference arm")
    pred = _predict(model, te)
    err = pred - y
    rec = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
    if not np.isclose(np.sqrt(np.mean(err**2)), rec, rtol=1e-4):
        raise RuntimeError(f"{run_dir.name}: reloaded test RMSE does not match the record")
    rmse_k = np.sqrt(np.mean(err**2, axis=0))
    out = {"rmse": float(np.sqrt(np.mean(err**2))),
           "rmse_median_over_k": float(np.median(rmse_k)),
           "frac_k_bins_rmse_below_1pct": float(np.mean(rmse_k < _RMSE_PUB_REF_PCT / 100.0))}
    for lo, hi in K_BANDS:
        m = (k > lo) & (k <= hi)
        out[f"rmse_k_{lo:g}_{hi:g}"] = float(np.sqrt(np.mean(err[:, m] ** 2)))
    out["_rmse_k"], out["_k"] = rmse_k, k
    out["_sim_index"], out["_y"] = np.asarray(te.sim_index), y
    out["_model"], out["_te"], out["_names"] = model, te, names
    return out


def _permutation_importance(model, te, names: tuple[str, ...], seed: int) -> dict:
    """Test pooled-RMSE increase when a parameter column (or the pair, jointly) is permuted.

    One test curve per sim (single number-density bin), so permuting rows
    permutes the parameter across test sims while the profile stays fixed.
    """
    y = np.asarray(te.y, float)
    base = np.sqrt(np.mean((_predict(model, te) - y) ** 2))
    Xp = np.asarray(te.X_params, float)
    rng = np.random.default_rng(PERM_SEED + seed)
    groups = {p: [names.index(p)] for p in names}
    groups[PAIR_KEY] = [names.index(p) for p in PAIR]
    out = {}
    for g, cols in groups.items():
        d = []
        for _ in range(N_PERM):
            Xs = Xp.copy()
            Xs[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]  # joint permutation for the pair
            d.append(np.sqrt(np.mean((_predict(model, te, Xs) - y) ** 2)) - base)
        out[g] = float(np.mean(d))
    return out


def _arm_label(arm: str) -> str:
    if arm == "all":
        return "all 5 parameters"
    if arm == "none":
        return "no cosmology"
    kind, key = arm.split("-", 1)
    sym = ",\\ ".join(LATEX[p] for p in key.split("+"))
    return f"without ${sym}$" if kind == "drop" else f"only ${sym}$"


def plot_seed_band(ax, k: np.ndarray, rmse_k_per_seed: np.ndarray, color, ls: str = "-",
                   lw: float = 2.0, label: str | None = None) -> None:
    """Draw per-k test RMSE over seeds: seed-mean line and seed min--max band.

    Args:
        ax: Matplotlib axes.
        k: Wavenumbers [h/Mpc], shape (n_k,).
        rmse_k_per_seed: Per-k test RMSE, shape (n_seeds, n_k).
        color: Line and band colour.
        ls: Line style.
        lw: Line width.
        label: Legend label.
    """
    v = np.asarray(rmse_k_per_seed, float)
    ax.fill_between(k, v.min(0), v.max(0), color=color, alpha=0.15, lw=0)
    ax.plot(k, v.mean(0), color=color, ls=ls, lw=lw, label=label)


def style_rmse_axis(ax, k: np.ndarray) -> None:
    """Shared RMSE(k) axis styling: 1% reference, k = 3 h/Mpc marker, log-k ticks, grid."""
    import matplotlib.ticker

    ax.axhline(_RMSE_PUB_REF_PCT / 100.0, color="0.35", ls="--", lw=1.2)
    ax.axvline(_RMSE_PUB_K_REF, color="0.35", ls=":", lw=1.4)
    ax.set_xscale("log")
    decades = np.arange(np.floor(np.log10(k.min())), np.ceil(np.log10(k.max())) + 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([m * 10.0**d for d in decades for m in (1, 2, 5)]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(k.min() / 1.1, k.max() * 1.1)
    ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax.grid(True)


def _plot(curves: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fgas_spk.pub_style import PLOT_RC, REDSHIFT, TRACER

    k = curves["all"]["k"]
    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(16, 7.5), layout="constrained")
        for ax, kind, title in zip(axes, ("drop", "only"), ("Leave one out", "Single parameter")):
            for arm, c, ls in (("none", "0.45", "--"), ("all", "black", "-")):
                plot_seed_band(ax, k, curves[arm]["rmse_k"], c, ls, 2.6, _arm_label(arm))
            for key in (*PARAMS, PAIR_KEY):
                arm = f"{kind}-{key}"
                plot_seed_band(ax, k, curves[arm]["rmse_k"], COLOUR[key],
                               ":" if key == PAIR_KEY else "-", 2.0, _arm_label(arm))
            for arm, kept in FB_ARMS.items():
                if arm.startswith(kind) and arm in curves:
                    plot_seed_band(ax, k, curves[arm]["rmse_k"], COLOUR[kept[0]], "-.", 2.0, _arm_label(arm))
            style_rmse_axis(ax, k)
            ax.set_title(title)
            ax.legend(loc="upper left", frameon=False, fontsize=13)
        axes[0].set_ylabel("Test reconstruction RMSE")
        axes[0].set_ylim(bottom=0)
        axes[1].text(0.97, 0.03, f"{TRACER[SNAP]} ($z={REDSHIFT[SNAP]}$)\n9 arcmin bins, "
                     r"$k\leq10\ h\,\mathrm{Mpc}^{-1}$"
                     + ("" if VARIANT == "fb" else "\n" + r"$f_\mathrm{gas}$ not divided by $f_\mathrm{b}$"),
                     transform=axes[1].transAxes,
                     ha="right", va="bottom", fontsize=15)
        fig.savefig(out, dpi=200)
        plt.close(fig)


def report(out_dir: Path) -> None:
    arm_order = _arms()
    per_seed, curves, perm = {a: [] for a in arm_order}, {}, []
    for seed in SEEDS:
        ref = None
        for arm in arm_order:
            r = _evaluate(arm, seed, ref)
            if ref is None:
                ref = {"sim_index": r["_sim_index"], "y": r["_y"], "X": np.asarray(r["_te"].X)}
            if arm == "all":
                perm.append(_permutation_importance(r["_model"], r["_te"], r["_names"], seed))
            per_seed[arm].append(r)
            print(f"seed {seed} {arm:28s} rmse {r['rmse']:.4f}", flush=True)

    metric_names = [m for m in per_seed["all"][0] if not m.startswith("_")]
    table = {}
    for arm in arm_order:
        table[arm] = {m: {"mean": float(np.mean([p[m] for p in per_seed[arm]])),
                          "sd": float(np.std([p[m] for p in per_seed[arm]], ddof=1))}
                      for m in metric_names}
        table[arm]["params"] = list(_kept(arm))
        curves[arm] = {"k": per_seed[arm][0]["_k"], "rmse_k": np.stack([p["_rmse_k"] for p in per_seed[arm]])}
        table[arm]["rmse_k_per_seed"] = curves[arm]["rmse_k"].tolist()
    r_all, r_none = table["all"]["rmse"]["mean"], table["none"]["rmse"]["mean"]
    for arm in arm_order:
        table[arm]["recovered_fraction"] = (r_none - table[arm]["rmse"]["mean"]) / (r_none - r_all)
    perm_tab = {g: {"mean": float(np.mean([p[g] for p in perm])), "sd": float(np.std([p[g] for p in perm], ddof=1))}
                for g in perm[0]}

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "ablate_cosmo_params" + ("" if VARIANT == "fb" else f"_{VARIANT}")
    (out_dir / f"{stem}.json").write_text(json.dumps(
        {"setting": {"snapshot": SNAP, "design": "arcmin9", "k_max": KMAX, "seeds": SEEDS,
                     "input_variant": VARIANT, "dataset_tag": TAGS[VARIANT], "k": curves["all"]["k"].tolist(),
                     "groups": {a: _group(a) for a in arm_order}},
         "metrics": table, "permutation_importance_all_model": perm_tab, "n_perm": N_PERM}, indent=2))

    cols = [("rmse", "RMSE (all k)")] + [(f"rmse_k_{lo:g}_{hi:g}", f"RMSE k∈({lo:g},{hi:g}]") for lo, hi in K_BANDS] \
        + [("rmse_median_over_k", "median_k RMSE(k)"), ("frac_k_bins_rmse_below_1pct", "frac k: RMSE<1%")]
    lines = ["| arm | conditioning | " + " | ".join(c for _, c in cols) + " | recovered |",
             "|---|---|" + "---|" * (len(cols) + 1)]
    for arm in arm_order:
        t = table[arm]
        lines.append(f"| {arm} | {', '.join(t['params']) or '—'} | "
                     + " | ".join(f"{t[m]['mean']:.4f} ± {t[m]['sd']:.4f}" for m, _ in cols)
                     + f" | {t['recovered_fraction']:.2f} |")
    lines += ["", f"Permutation importance on the `all` models (ΔRMSE, mean ± sd over seeds; {N_PERM} perms each):", "",
              "| permuted | ΔRMSE |", "|---|---|"]
    lines += [f"| {g} | {v['mean']:.4f} ± {v['sd']:.4f} |" for g, v in perm_tab.items()]
    md = "\n".join(lines)
    (out_dir / f"{stem}.md").write_text(md + "\n")
    print(md)
    _plot(curves, out_dir / f"{stem}.png")
    print(out_dir / f"{stem}.png")


def compare(out_dir: Path) -> None:
    """Overplot test RMSE(k) of the all / none arms for the f_b-normalised and raw f_gas inputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fgas_spk.pub_style import PLOT_RC, REDSHIFT, TRACER

    res = {v: json.loads((out_dir / f"ablate_cosmo_params{'' if v == 'fb' else '_' + v}.json").read_text())
           for v in TAGS}
    k = np.asarray(res["nofb"]["setting"]["k"])
    style = {("fb", "all"): ("#2a78d6", "-"), ("fb", "none"): ("#2a78d6", "--"),
             ("nofb", "all"): ("#eb6834", "-"), ("nofb", "none"): ("#eb6834", "--")}
    lab = {"fb": r"$f_\mathrm{gas}/f_\mathrm{b}$", "nofb": r"$f_\mathrm{gas}$ (no $f_\mathrm{b}$)"}
    with matplotlib.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
        for (v, arm), (c, ls) in style.items():
            m = res[v]["metrics"][arm]
            plot_seed_band(ax, k, m["rmse_k_per_seed"], c, ls, 2.4,
                           f"{lab[v]}, {_arm_label(arm)} (RMSE {m['rmse']['mean']:.4f})")
        style_rmse_axis(ax, k)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Test reconstruction RMSE")
        ax.legend(loc="upper left", frameon=False, fontsize=13)
        ax.text(0.97, 0.03, f"{TRACER[SNAP]} ($z={REDSHIFT[SNAP]}$)\n9 arcmin bins", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=15)
        fig.savefig(out_dir / "ablate_cosmo_params_fb_vs_nofb.png", dpi=200)
        plt.close(fig)
    print(out_dir / "ablate_cosmo_params_fb_vs_nofb.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", action="store_true", help="Train + record seeds 0/1/2 for the new arms.")
    ap.add_argument("--arms", nargs="*", default=None, help="Subset of new arms to train (default: all).")
    ap.add_argument("--report", action="store_true", help="Evaluate all arms; write table + figure.")
    ap.add_argument("--nofb", action="store_true",
                    help="Use the f_gas dataset without the f_b normalisation (trains all/none too).")
    ap.add_argument("--compare", action="store_true",
                    help="Overplot all/none arms of both input variants (needs both reports).")
    ap.add_argument("--out-dir", default=None,
                    help="Default: today's figures/.../cosmo_ablation (cosmo_ablation_snap<NN> off snap74).")
    ap.add_argument("--snap", type=int, choices=(74, 82), default=74,
                    help="Snapshot: 74 (DESI LRG, z=0.47; default) or 82 (DESI BGS, z=0.21).")
    args = ap.parse_args()
    global VARIANT, SNAP, SRC_CFG, BASELINE_GROUPS
    VARIANT = "nofb" if args.nofb else "fb"
    SNAP = args.snap
    SRC_CFG = _src_cfg(SNAP)
    BASELINE_GROUPS = {"all": f"arcmin9_cosmo_kmax{KMAX:g}_snap{SNAP}_nd2",
                       "none": f"arcmin9_noparams_kmax{KMAX:g}_snap{SNAP}_nd2"}
    if not (repo_root() / SRC_CFG).exists():
        raise FileNotFoundError(f"{SRC_CFG} missing; write it with kmax10_noparams._write_data_config"
                                f"('arcmin9', True, {KMAX:g}, {SNAP}).")
    fig_sub = "cosmo_ablation" if SNAP == 74 else f"cosmo_ablation_snap{SNAP}"
    default_arms = [a for a in _arms() if VARIANT == "nofb" or a not in BASELINE_GROUPS]
    if args.train:
        for arm in args.arms or default_arms:
            gdir = repo_root() / "experiments" / "runs" / _group(arm)
            if gdir.exists() and any(gdir.iterdir()):
                print(f"skip {arm}: {gdir} already has runs", flush=True)
                continue
            cfg = _write_data_config(arm)
            dc, rc = load_configs(cfg, repo_root() / RUN_CFG)
            kept = _kept(arm)
            assert dc.tag == _tag(arm) and dc.include_camels_params == bool(kept)
            assert (tuple(dc.camels_param_names) if kept else dc.camels_param_names) == (kept or None)
            assert dc.target_mode == "k_range" and tuple(dc.k_range) == (0.0, KMAX)
            assert tuple(dc.radial_range_arcmin) == (0.0, 6.0) and dc.radial_range_mpch is None
            assert dc.snapshot == SNAP and list(dc.number_density_indices) == [2]
            for d in _train_group(cfg, repo_root() / RUN_CFG, _group(arm)):
                print(arm, d, flush=True)
    if args.report:
        report(Path(args.out_dir) if args.out_dir else figure_dir(fig_sub))
    if args.compare:
        compare(Path(args.out_dir) if args.out_dir else figure_dir(fig_sub))


if __name__ == "__main__":
    main()
