"""One cVAE over both tracers and all number densities: f_gas(R) + cosmology + z + n -> SP(k).

The per-(snapshot, number-density) cVAEs of ``scripts/nd_compare.py`` each see one
halo sample at one redshift. This script trains a **single** conditional VAE
(``fgas_spk.models.cvae``, ``scripts/configs/run/cvae_noise_joint.yaml``) on all of

=========  ===========  =====  =============================================
snapshot   tracer       z      number densities n [(Mpc/h)^-3]
=========  ===========  =====  =============================================
74         DESI LRG     0.47   1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3
82         DESI BGS     0.21   1.0e-3, 4.0e-3, 8.0e-3, 2.0e-2, 3.0e-2
=========  ===========  =====  =============================================

i.e. 1024 sims x 10 (snapshot, n) rows. Inputs, per row:

* ``X``        -- the stacked f_gas(R)/f_b profile, 9 angle-fixed bins 1--6 arcmin
                  (the ``20260930-arcmin-all`` datasets cropped at 6 arcmin, as in
                  the ``arcmin9_cosmo_kmax10`` runs);
* ``X_params`` -- the 5 cosmological parameters (Omega0, sigma8, OmegaBaryon,
                  HubbleParam, n_s);
* ``X_cond``   -- ``[log10 n, z]``. n spans 2.5 dex, so it enters as log10 n (the
                  model z-scores each column linearly); z is the snapshot redshift.

Target: SP(k) = P_total/P_DM within the hydro run (proxy target, see CLAUDE.md) for
k <= 10 h/Mpc (79 bins), at the row's snapshot. Both snapshots share the k grid,
the angular radial bins, the simulation ids and the parameters (checked on load).

The outer split is :func:`fgas_spk.train.grouped_split` on ``sim_index`` with the
run config's split (seed 10): all 10 rows of a simulation land in one fold, and the
test simulations are exactly those of the per-nd runs, so the two are compared on
identical held-out curves (checked in ``--report``).

Records go to ``experiments/runs/joint_cvae/<run_id>/`` (both data configs, the run
config, the X_cond definition, env, per-epoch history, summary) plus one ledger line;
the checkpoint to ``$FGAS_SCRATCH_ROOT/models/<run_id>/model.joblib``.

Report (``--report``), to ``figures/<YYYY-MM>/<MM-DD>/joint_cvae/``:

* ``joint_cvae_spk_pred.png`` -- SP(k) predicted (prior-mean decode, with the 16--84%
  band of ``predict_samples``) vs truth for four test simulations spanning the
  suppression depth, for every (tracer, n), with the fractional residual below;
* ``joint_cvae_rmse_vs_k.png`` -- test per-k RMSE of the joint model vs the
  separate per-(snapshot, n) cVAEs, per (tracer, n) (seed mean, seed min--max band);
* ``joint_cvae_rmse.{md,json}`` -- the corresponding table.

Diagnostics (``--diagnostics``), to the same folder: the runner's standard per-run
figures (pred-vs-true, residual map, coverage, latent, history, ...) per seed and per
tracer (see :func:`diagnostics`).

Usage::

    python scripts/joint_cvae.py --train     # GPU, seeds 0/1/2
    python scripts/joint_cvae.py --report
    python scripts/joint_cvae.py --diagnostics
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablate_cosmo_params import K_BANDS, plot_seed_band, style_rmse_axis  # noqa: E402
from nd_compare import _evaluate as _evaluate_separate  # noqa: E402
from nd_compare import _seed_runs  # noqa: E402

from fgas_spk.experiment import load_configs  # noqa: E402
from fgas_spk.loader import TrainingData, load_training_data  # noqa: E402
from fgas_spk.models import REGISTRY  # noqa: E402
from fgas_spk.paths import fill_data_root, figure_dir, repo_root, resolve_scratch_root  # noqa: E402
from fgas_spk.run_record import (  # noqa: E402
    DEFAULT_SUPPRESSION_DEFINITION,
    _package_versions,
    _read_dataset_meta,
    git_info,
    make_run_id,
)
from fgas_spk.train import _subset, grouped_split, pick_device  # noqa: E402

SNAPS = (74, 82)
SEEDS = (0, 1, 2)
KMAX = 10.0
GROUP = "joint_cvae"
DATA_CFG = "scripts/configs/data/joint_cvae/arcmin9_cosmo_kmax10_snap{snap}_allnd.yaml"
RUN_CFG = "scripts/configs/run/cvae_noise_joint.yaml"
# Reference single-nd data config each joint data config must match except for the nd subset.
REF_GROUP = "arcmin9_cosmo_kmax10_snap{snap}_nd2"
X_COND_COLUMNS = ["log10_number_density", "redshift"]
X_COND_DEFINITION = {
    "log10_number_density": "log10 of the halo-sample number density n [(Mpc/h)^-3] used to stack the profile",
    "redshift": "snapshot redshift (DataConfig.redshift): 0.47 for snap74, 0.21 for snap82",
}
# Profile-input columns when the fgas_std arm is on (``fgas_std=True``): X = [f_gas, log10 sigma_fgas].
X_PROFILE_DEFINITION_FGAS_STD = {
    "columns": "9 x f_gas(R)/f_b (1--6 arcmin), then 9 x log10 sigma_fgas(R) on the same bins",
    "sigma_fgas": "the dataset's fgas_std = the compiled file's prof_std (build_fgas_spk_from_compiled): "
                  "|f_gas| * sqrt((sigma_ion/DS_ion)^2 + (sigma_tot/DS_tot)^2), f_gas = DS_ion/DS_tot/f_b, "
                  "with sigma_* the halo-to-halo std of Delta Sigma_kSZ (ionised) and Delta Sigma_total over the selected "
                  "halos x 3 projections (SimulationStacker test_CAMELS_eighth_gen.ipynb, _stack_ratio)",
    "log10": "sigma_fgas is strictly positive and spans >1 dex across n and R; the model z-scores each column",
}
# Test simulations drawn in the SP(k) figure: these percentiles of the LRG
# suppression depth min_k SP(k) over the test fold (deep -> shallow).
PICK_PERCENTILES = (5, 35, 65, 95)
SIM_COLOURS = ("#2a78d6", "#eb6834", "#1a9e77", "#7b52ab")  # validated categorical (dataviz)


def _runs_root() -> Path:
    return repo_root() / "experiments" / "runs"


def _configs(run_cfg: str = RUN_CFG, fgas_std: bool = False, nocond: bool = False):
    """(data configs per snap, run config), each data config checked against its nd_2 reference.

    With ``fgas_std`` the checked configs are switched to ``include_fgas_std=True``
    (nothing else changes), so :func:`load_joint` also serves the profile error.
    With ``nocond`` they are switched to ``include_camels_params=False`` and
    ``camels_param_names=None`` (nothing else changes), so :func:`load_joint` serves
    no cosmological parameters and no ``X_cond``.
    """
    if fgas_std and nocond:
        raise ValueError("fgas_std and nocond are separate arms; set at most one")
    dcs = {}
    for snap in SNAPS:
        dc, rc = load_configs(repo_root() / DATA_CFG.format(snap=snap), repo_root() / run_cfg)
        ref_dir = _seed_runs(REF_GROUP.format(snap=snap))[0]
        dc_ref, _ = load_configs(ref_dir / "config_data.yaml", ref_dir / "config_run.yaml")
        diff = {f.name for f in dataclasses.fields(dc) if getattr(dc, f.name) != getattr(dc_ref, f.name)}
        if diff != {"number_density_indices"} or dc.number_density_indices is not None:
            raise RuntimeError(f"snap{snap}: unexpected DataConfig differences {diff}")
        if dc.snapshot != snap or dc.include_nd_feature or dc.include_mean_halo_mass:
            raise RuntimeError(f"snap{snap}: data config is not the expected joint arm")
        if fgas_std:
            dc = dataclasses.replace(dc, include_fgas_std=True)
        if nocond:
            dc = dataclasses.replace(dc, include_camels_params=False, camels_param_names=None)
        dcs[snap] = dc
    return dcs, rc


def load_joint(dcs: dict) -> tuple[TrainingData, np.ndarray]:
    """Stack both snapshots into one TrainingData with X_cond = [log10 n, z].

    When the data configs set ``include_fgas_std``, the profile input becomes
    ``X = [f_gas, log10 sigma_fgas]`` (:data:`X_PROFILE_DEFINITION_FGAS_STD`) and the
    linear error is kept in ``X_std``. When they unset ``include_camels_params`` (the
    ``nocond`` arm of :func:`_configs`), ``X_params`` and ``X_cond`` are both None.

    Returns:
        (TrainingData, snap): the stacked data (snap74 rows first) and the snapshot
        of each row, shape (n_rows,).
    """
    tds = {s: load_training_data(fill_data_root(dcs[s])) for s in SNAPS}
    a, b = tds[SNAPS[0]], tds[SNAPS[1]]
    if not np.array_equal(a.k, b.k):
        raise RuntimeError("k grids differ between snapshots")
    if a.param_names != b.param_names:
        raise RuntimeError(f"parameter columns differ: {a.param_names} vs {b.param_names}")
    if a.X.shape[1] != b.X.shape[1]:
        raise RuntimeError("radial bin counts differ between snapshots")
    for td in (a, b):
        rg = td.meta.get("radial_grid") or {}
        if rg.get("design") != "arcmin":
            raise RuntimeError(f"{td.source_path}: expected the arcmin radial design")
    with_params = {dcs[s].include_camels_params for s in SNAPS}
    if len(with_params) != 1:
        raise RuntimeError("include_camels_params differs between snapshots")
    with_params = with_params.pop()
    if any((td.X_params is not None) != with_params for td in (a, b)):
        raise RuntimeError("X_params presence does not match include_camels_params")
    # Same simulations with the same parameters in both snapshots.
    if not (np.array_equal(a.sim_index, b.sim_index)
            and (not with_params or np.array_equal(a.X_params, b.X_params))):
        raise RuntimeError("simulation rows / parameters differ between snapshots")
    for td in (a, b):
        if not (np.isfinite(td.X).all() and np.isfinite(td.y).all()
                and (not with_params or np.isfinite(td.X_params).all())):
            raise RuntimeError(f"{td.source_path}: non-finite inputs or targets")
    with_std = {dcs[s].include_fgas_std for s in SNAPS}
    if len(with_std) != 1:
        raise RuntimeError("include_fgas_std differs between snapshots")
    with_std = with_std.pop()
    if with_std:
        for td in (a, b):
            if not (np.isfinite(td.X_std).all() and (td.X_std > 0).all()):
                raise RuntimeError(f"{td.source_path}: fgas_std must be finite and > 0 for log10")

    def profile(td: TrainingData) -> np.ndarray:
        return np.hstack([td.X, np.log10(td.X_std)]) if with_std else td.X

    def cond(td: TrainingData) -> np.ndarray:
        z = float(td.config.redshift)
        return np.column_stack([np.log10(td.nd), np.full(len(td.nd), z)])

    snap = np.concatenate([np.full(len(tds[s].y), s) for s in SNAPS])
    joint = TrainingData(
        X=np.vstack([profile(a), profile(b)]),
        # No conditioning at all in the nocond arm: neither [log10 n, z] nor cosmology.
        X_cond=np.vstack([cond(a), cond(b)]) if with_params else None,
        X_params=np.vstack([a.X_params, b.X_params]) if with_params else None,
        y=np.vstack([a.y, b.y]),
        nd=np.concatenate([a.nd, b.nd]),
        sim_index=np.concatenate([a.sim_index, b.sim_index]),
        k=a.k,
        radii_mpch=a.radii_mpch,
        source_path=f"{a.source_path};{b.source_path}",
        param_names=a.param_names,
        meta={str(s): tds[s].meta for s in SNAPS},
        config=None,
        X_std=np.vstack([a.X_std, b.X_std]) if with_std else None,
    )
    return joint, snap


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def _predict(model, td: TrainingData) -> np.ndarray:
    return np.asarray(model.predict(td.X, td.X_cond, td.X_params), float).reshape(td.y.shape)


def _x_cond_columns(dcs: dict) -> list[str]:
    """The X_cond columns of the arm: :data:`X_COND_COLUMNS`, or none in the nocond arm."""
    return X_COND_COLUMNS if dcs[SNAPS[0]].include_camels_params else []


def _config_hash(dcs: dict, rc) -> str:
    payload = {"data": {str(s): dataclasses.asdict(dcs[s]) for s in SNAPS}, "run": dataclasses.asdict(rc),
               "x_cond": _x_cond_columns(dcs)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _best_epoch(model):
    """The restored epoch: ``best_epoch`` of a single-network model, per phase for dual_vae."""
    if hasattr(model, "best_epoch"):
        return model.best_epoch
    return {phase: getattr(getattr(model, attr), "best_epoch", None)
            for phase, attr in (("codec_y", "_y_codec"), ("codec_x", "_x_codec"), ("mapping", "_map"))}


def train(seed: int, group: str = GROUP, run_cfg: str = RUN_CFG, fgas_std: bool = False,
          nocond: bool = False) -> Path:
    """Train, record and checkpoint the joint model of ``run_cfg`` for one seed; returns the run dir.

    ``fgas_std`` adds log10 sigma_fgas to the profile input; ``nocond`` drops X_cond
    and X_params (see :func:`load_joint`).
    """
    dcs, rc = _configs(run_cfg, fgas_std=fgas_std, nocond=nocond)
    x_cond_columns = _x_cond_columns(dcs)
    rc = dataclasses.replace(rc, seed=seed)
    scratch_root = resolve_scratch_root(explicit=rc.write_root)
    td, snap = load_joint(dcs)
    masks = grouped_split(td.sim_index, rc.split)
    train_td = _subset(td, masks["train"])

    model = REGISTRY[rc.model](seed=rc.seed, **rc.model_params)
    model.fit(train_td)

    summary = {"n_train": int(masks["train"].sum()), "n_val": int(masks["val"].sum()),
               "n_test": int(masks["test"].sum()), "best_epoch": _best_epoch(model),
               "train_rmse": _rmse(train_td.y, _predict(model, train_td))}
    for name in ("val", "test"):
        sub = _subset(td, masks[name])
        summary[f"{name}_rmse"] = _rmse(sub.y, _predict(model, sub))
    summary["held_out_split"] = "test"
    summary["rmse"] = summary["test_rmse"]
    ledger_metrics = dict(summary)
    # Test RMSE per (snapshot, number density).
    te_mask = masks["test"]
    pred_te = _predict(model, _subset(td, te_mask))
    per = {}
    for s in SNAPS:
        for n in np.unique(td.nd[snap == s]):
            m = (snap[te_mask] == s) & (td.nd[te_mask] == n)
            per[f"snap{s}/n{n:g}"] = _rmse(td.y[te_mask][m], pred_te[m])
    summary["test_rmse_per_snap_nd"] = per

    git_sha, dirty = git_info()
    run_id = make_run_id(_config_hash(dcs, rc), git_sha)
    run_dir = _runs_root() / group / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for s in SNAPS:
        dcs[s].to_yaml(run_dir / f"config_data_snap{s}.yaml")
    rc.to_yaml(run_dir / "config_run.yaml")
    (run_dir / "x_cond.json").write_text(json.dumps(
        {"columns": x_cond_columns, "definition": {c: X_COND_DEFINITION[c] for c in x_cond_columns}}, indent=2))
    if fgas_std:
        (run_dir / "x_profile.json").write_text(json.dumps(
            {"n_columns": int(td.X.shape[1]), **X_PROFILE_DEFINITION_FGAS_STD}, indent=2))
    env = {
        "git": {"sha": git_sha, "dirty": dirty},
        "seed": rc.seed,
        "device": pick_device(),
        "roots": {"data_root": dcs[SNAPS[0]].project_root, "scratch_root": str(scratch_root)},
        "datasets": {str(s): {"path": p, "meta": _read_dataset_meta(p)}
                     for s, p in zip(SNAPS, td.source_path.split(";"))},
        "suppression_target": {"definition": DEFAULT_SUPPRESSION_DEFINITION},
        "target_selection": {"target_mode": dcs[SNAPS[0]].target_mode, "k_range": list(dcs[SNAPS[0]].k_range)},
        "param_selection": {"camels_param_names": dcs[SNAPS[0]].camels_param_names},
        "packages": _package_versions(),
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    with (run_dir / "metrics.jsonl").open("w") as fh:
        for row in model.history:
            fh.write(json.dumps(dict(row)) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (repo_root() / "experiments" / "runs.jsonl").open("a") as fh:
        fh.write(json.dumps({
            "run_id": run_id, "config_hash": run_id.split("__")[1], "git_sha": git_sha[:7], "dirty": dirty,
            "hparams": {"model": rc.model, "model_params": rc.model_params, "seed": rc.seed},
            "data": {"snapshots": list(SNAPS), "number_density_indices": None, "x_cond": x_cond_columns,
                     "include_fgas_std": fgas_std, "n_profile_columns": int(td.X.shape[1]),
                     "include_camels_params": dcs[SNAPS[0]].include_camels_params,
                     "camels_param_names": dcs[SNAPS[0]].camels_param_names,
                     "target_mode": dcs[SNAPS[0]].target_mode, "k_range": list(dcs[SNAPS[0]].k_range)},
            "metrics": ledger_metrics,
            "run_dir": str(run_dir.relative_to(repo_root())),
        }) + "\n")

    ckpt = Path(scratch_root) / "models" / run_id
    ckpt.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ckpt / "model.joblib")
    print(f"seed {seed}: {run_id} test RMSE {summary['test_rmse']:.4f} best epoch {summary['best_epoch']}",
          flush=True)
    return run_dir


def _joint_runs(group: str = GROUP) -> dict[int, Path]:
    import yaml

    out: dict[int, list[Path]] = {}
    for d in sorted((_runs_root() / group).iterdir()):
        out.setdefault(yaml.safe_load((d / "config_run.yaml").read_text())["seed"], []).append(d)
    if sorted(out) != sorted(SEEDS) or any(len(v) != 1 for v in out.values()):
        raise RuntimeError(f"{group}: expected one run per seed {SEEDS}, found {out}")
    return {s: v[0] for s, v in out.items()}


def _load_joint_model(run_dir: Path):
    env = json.loads((run_dir / "env.json").read_text())
    return joblib.load(Path(env["roots"]["scratch_root"]) / "models" / run_dir.name / "model.joblib")


def _separate_group(snap: int, nd_idx: int) -> str:
    base = REF_GROUP.format(snap=snap)
    return base if nd_idx == 2 else f"nd_compare/{base[:-len('_nd2')]}_nd{nd_idx}"


def report(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from fgas_spk.pub_style import PLOT_RC, REDSHIFT, TRACER

    dcs, rc = _configs()
    td, snap = load_joint(dcs)
    te_mask = grouped_split(td.sim_index, rc.split)["test"]
    te, snap_te = _subset(td, te_mask), snap[te_mask]
    k = np.asarray(td.k, float)
    nds = {s: np.unique(td.nd[snap == s]) for s in SNAPS}

    # ---- joint model: per-(snap, n) per-k test RMSE for each seed (record checked) ----
    runs = _joint_runs()
    models, joint_rmse_k = {}, {}
    for seed, run_dir in runs.items():
        model = _load_joint_model(run_dir)
        models[seed] = model
        pred = _predict(model, te)
        rec = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
        if not np.isclose(_rmse(te.y, pred), rec, rtol=1e-4):
            raise RuntimeError(f"{run_dir.name}: reloaded test RMSE does not match the record")
        for s in SNAPS:
            for i, n in enumerate(nds[s]):
                m = (snap_te == s) & (te.nd == n)
                joint_rmse_k.setdefault((s, i), []).append(np.sqrt(np.mean((pred[m] - te.y[m]) ** 2, axis=0)))
        print(f"joint seed {seed} {run_dir.name} test rmse {rec:.4f}", flush=True)

    # ---- separate per-(snap, n) models, scored on the identical test curves ----
    sep_rmse_k = {}
    for s in SNAPS:
        for i, n in enumerate(nds[s]):
            m = (snap_te == s) & (te.nd == n)
            for seed, run_dir in sorted(_seed_runs(_separate_group(s, i)).items()):
                r = _evaluate_separate(run_dir)
                if not (np.isclose(r["nd_value"], n) and np.array_equal(r["sim_index"], te.sim_index[m])
                        and np.array_equal(r["y"], te.y[m]) and np.array_equal(r["k"], k)):
                    raise RuntimeError(f"{run_dir}: test fold differs from the joint model's (snap{s}, n={n:g})")
                sep_rmse_k.setdefault((s, i), []).append(r["rmse_k"])

    table = {}
    for (s, i) in joint_rmse_k:
        row = {"snapshot": s, "tracer": TRACER[s], "nd_index": i, "number_density": float(nds[s][i])}
        for arm, store in (("joint", joint_rmse_k), ("separate", sep_rmse_k)):
            v = np.stack(store[(s, i)])  # (n_seeds, n_k)
            for name, sel in [("rmse", np.ones_like(k, bool))] + [
                    (f"rmse_k_{lo:g}_{hi:g}", (k > lo) & (k <= hi)) for lo, hi in K_BANDS]:
                # pooled over the k bins of the band: sqrt(mean_k rmse_k^2) per seed
                per_seed = np.sqrt(np.mean(v[:, sel] ** 2, axis=1))
                row[f"{arm}/{name}"] = {"mean": float(per_seed.mean()), "sd": float(per_seed.std(ddof=1))}
            row[f"{arm}/rmse_k_per_seed"] = v.tolist()
        table[f"snap{s}/nd{i}"] = row

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "joint_cvae_rmse.json").write_text(json.dumps(
        {"setting": {"design": "arcmin9", "k_max": KMAX, "seeds": SEEDS, "split_seed": rc.split.seed,
                     "x_cond": X_COND_COLUMNS, "k": k.tolist(),
                     "joint_runs": {s: d.name for s, d in runs.items()}},
         "metrics": table}, indent=2))
    cols = [("rmse", "RMSE (all k)")] + [(f"rmse_k_{lo:g}_{hi:g}", f"k∈({lo:g},{hi:g}]") for lo, hi in K_BANDS]
    lines = ["| tracer | n [(Mpc/h)⁻³] | " + " | ".join(f"{c} joint | {c} separate" for _, c in cols) + " |",
             "|---|---|" + "---|---|" * len(cols)]
    for t in table.values():
        lines.append(f"| {t['tracer']} | {t['number_density']:.2g} | " + " | ".join(
            f"{t[f'joint/{m}']['mean']:.4f} ± {t[f'joint/{m}']['sd']:.4f} | "
            f"{t[f'separate/{m}']['mean']:.4f} ± {t[f'separate/{m}']['sd']:.4f}" for m, _ in cols) + " |")
    md = "\n".join(lines)
    (out_dir / "joint_cvae_rmse.md").write_text(md + "\n")
    print(md)

    def nd_title(s, n):
        e = int(np.floor(np.log10(n)))
        return rf"$n={n / 10**e:.1f}\times10^{{{e}}}$"

    # ---- Figure 1: SP(k) predictions for four test sims at every (tracer, n) ----
    model0 = models[0]
    m74 = snap_te == SNAPS[0]
    sims_te = np.unique(te.sim_index)
    depth = {sid: te.y[m74 & (te.sim_index == sid)][0].min() for sid in sims_te}
    order = sorted(sims_te, key=lambda sid: depth[sid])
    picks = [order[int(round(p / 100 * (len(order) - 1)))] for p in PICK_PERCENTILES]
    pnames = td.param_names
    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(4, 5, figsize=(26, 17), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [2.2, 1, 2.2, 1]})
        for r, s in enumerate(SNAPS):
            for c, n in enumerate(nds[s]):
                ax, axr = axes[2 * r, c], axes[2 * r + 1, c]
                for sid, col in zip(picks, SIM_COLOURS):
                    m = np.where((snap_te == s) & (te.nd == n) & (te.sim_index == sid))[0]
                    sub = _subset(te, np.isin(np.arange(len(te.y)), m))
                    y_true = sub.y[0]
                    y_pred = _predict(model0, sub)[0]
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
                ax.set_title(f"{TRACER[s]} ($z={REDSHIFT[s]}$), {nd_title(s, n)}", fontsize=15)
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
        handles = [Line2D([], [], color="0.2", lw=2, label="joint cVAE (prior mean; band 16–84%)"),
                   Line2D([], [], color="0.2", lw=2, ls="--", label="simulation")]
        for sid, col, p in zip(picks, SIM_COLOURS, PICK_PERCENTILES):
            ps = te.X_params[te.sim_index == sid][0]
            par = ", ".join(rf"{nm}={v:.3g}" for nm, v in zip(pnames, ps) if nm in ("Omega0", "sigma8"))
            handles.append(Line2D([], [], color=col, lw=3,
                                  label=f"sim {sid} (LRG depth p{p}; {par})"))
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=14)
        fig.savefig(out_dir / "joint_cvae_spk_pred.png", dpi=150)
        plt.close(fig)
    print(out_dir / "joint_cvae_spk_pred.png")

    # ---- Figure 2: per-k test RMSE, joint vs separate, per (tracer, n) ----
    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 5, figsize=(26, 10.5), sharex=True, sharey=True, layout="constrained")
        for r, s in enumerate(SNAPS):
            for c, n in enumerate(nds[s]):
                ax = axes[r, c]
                t = table[f"snap{s}/nd{c}"]
                plot_seed_band(ax, k, np.stack(joint_rmse_k[(s, c)]), "#2a78d6", "-", 2.4,
                               f"joint ({t['joint/rmse']['mean']:.4f})")
                plot_seed_band(ax, k, np.stack(sep_rmse_k[(s, c)]), "#eb6834", "--", 2.4,
                               f"separate ({t['separate/rmse']['mean']:.4f})")
                style_rmse_axis(ax, k)
                if r == 0:
                    ax.set_xlabel("")
                ax.set_title(f"{TRACER[s]} ($z={REDSHIFT[s]}$), {nd_title(s, n)}", fontsize=15)
                ax.legend(loc="upper left", frameon=False, fontsize=13)
            axes[r, 0].set_ylabel("Test RMSE")
        axes[0, 0].set_ylim(bottom=0)
        fig.savefig(out_dir / "joint_cvae_rmse_vs_k.png", dpi=150)
        plt.close(fig)
    print(out_dir / "joint_cvae_rmse_vs_k.png")


def diagnostics(out_dir: Path, group: str = GROUP, run_cfg: str = RUN_CFG, fgas_std: bool = False,
                nocond: bool = False) -> None:
    """The runner's standard per-run diagnostic figures, per seed and per tracer.

    Calls the same ``fgas_spk.train._write_*`` writers as
    :func:`fgas_spk.train.run_training`, with their output redirected to ``out_dir``.
    Each writer pools the rows of the held-out fold, so it is run on one snapshot
    at a time (that tracer's 5 number densities pooled) rather than mixing LRG and
    BGS; filenames are ``<timestamp>__<config-hash>__<model>__snap<NN>__<figure>.png``.
    """
    import fgas_spk.train as T

    out_dir.mkdir(parents=True, exist_ok=True)
    T.figure_dir = lambda subdir=None: out_dir  # writers call figure_dir() at write time
    dcs, rc = _configs(run_cfg, fgas_std=fgas_std, nocond=nocond)
    td, snap = load_joint(dcs)
    masks = grouped_split(td.sim_index, rc.split)
    for seed, run_dir in _joint_runs(group).items():
        model = _load_joint_model(run_dir)
        for s in SNAPS:
            sel = snap == s
            tds = _subset(td, sel)
            ms = {name: m[sel] for name, m in masks.items()}
            summ = {"held_out_split": "test",
                    "rmse": _rmse(_subset(tds, ms["test"]).y, _predict(model, _subset(tds, ms["test"])))}
            label = f"{T._figure_label(run_dir.name, rc.model)}__snap{s}"
            T._write_pred_vs_true_figure(model, tds, ms, label, summ)
            T._write_rmse_vs_k_pub_figure(model, tds, ms, label, summ, snapshot=s, cosmo=not nocond)
            T._write_frac_err_vs_k_figure(model, tds, ms, label, summ)
            T._write_per_curve_rmse_figure(model, tds, ms, label, summ)
            T._write_worst_curves_figure(model, tds, ms, label, summ)
            T._write_residual_map_figure(model, tds, ms, label, summ, seed=seed)
            # Duck-typed as in fgas_spk.train.run_training.
            if hasattr(model, "predict_samples"):
                T._write_spread_vs_error_figure(model, tds, ms, label, summ, seed=seed)
                T._write_coverage_figure(model, tds, ms, label, summ, seed=seed)
            if hasattr(model, "latents"):
                T._write_latent_norm_figure(model, tds, ms, label)
            if hasattr(model, "latent_dists"):
                T._write_latent_density_figure(model, tds, label, seed=seed)
            if hasattr(model, "decode") and hasattr(model, "latent_dists"):
                T._write_latent_traversal_figure(model, tds, label)
        # Training history is per model, not per tracer.
        T._write_history_figure(model, T._figure_label(run_dir.name, rc.model))
        print(f"seed {seed}: diagnostics written for {run_dir.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", action="store_true", help="Train + record seeds 0/1/2.")
    ap.add_argument("--report", action="store_true", help="Prediction + RMSE figures and table.")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Runner's standard diagnostic figures per seed and tracer.")
    ap.add_argument("--out-dir", default=None, help="Default: today's figures/.../joint_cvae.")
    args = ap.parse_args()
    if args.train:
        gdir = _runs_root() / GROUP
        if gdir.exists() and any(gdir.iterdir()):
            raise SystemExit(f"{gdir} already has runs; remove them deliberately to retrain")
        for seed in SEEDS:
            train(seed)
    if args.report:
        report(Path(args.out_dir) if args.out_dir else figure_dir(GROUP))
    if args.diagnostics:
        diagnostics(Path(args.out_dir) if args.out_dir else figure_dir(GROUP))


if __name__ == "__main__":
    main()
