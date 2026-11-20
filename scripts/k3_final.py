"""Final, recorded runs and test-fold scoring for the SP(k=3) sweep (scripts/k3_sweep.py).

Reads ``<sweep>/selection.json`` (k window + hyperparameters chosen by grouped CV
on the non-test sims only), then for each snapshot and seeds 0/1/2 trains and
records, via the standard runner (:func:`fgas_spk.train.run_training`), on the
standard train split (split seed 10):

* ``k3_default_snap<NN>`` -- default hyperparameters (``cvae_noise.yaml``) on
  the selected k window (isolates the window effect);
* ``k3_tuned_snap<NN>``   -- the selected hyperparameters on the selected window.

The baseline ``arcmin_noparams_snap<NN>`` (default hyperparameters, full curve)
already exists and is reloaded from its checkpoints. All three are then scored
on the held-out **test** fold -- the first time it is looked at in this study --
with :func:`k3_sweep.fractional_metrics`, mean +/- sd over the three seeds.

Writes the window data configs and the tuned run config under
``scripts/configs/``, and ``<sweep>/test_scores.json``.

Usage::

    python scripts/k3_final.py --sweep experiments/sweeps/k3_20261001
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_sweep as K  # noqa: E402

from fgas_spk.experiment import load_configs  # noqa: E402
from fgas_spk.loader import load_training_data  # noqa: E402
from fgas_spk.paths import fill_data_root, repo_root  # noqa: E402
from fgas_spk.train import _subset, grouped_split, run_training  # noqa: E402

SEEDS = (0, 1, 2)


def _window_tag(window) -> str:
    if window == "single":
        return "k3single"
    lo, hi = window
    return f"k{lo:g}-{'max' if hi is None else f'{hi:g}'}"


def _write_data_config(snap: int, window) -> Path:
    """Data config = the arcmin-all no-params config with the selected target window."""
    src = Path(K.DATA_CFG.format(snap=snap))
    lines = src.read_text().splitlines()
    if window == "single":
        mode, kt, kr = "single_k", str(K.K_STAR), "null"
    elif window[0] == 0.0 and window[1] is None:
        mode, kt, kr = "curve", "null", "null"
    else:
        hi = window[1] if window[1] is not None else 1.0e9
        mode, kt, kr = "k_range", "null", f"[{window[0]}, {hi}]"
    out = [f"# SP(k=3) study (scripts/k3_sweep.py): {src.name} with the CV-selected target window {window}."]
    for ln in lines:
        if ln.startswith("target_mode:"):
            ln = f"target_mode: {mode}"
        elif ln.startswith("k_target:"):
            ln = f"k_target: {kt}"
        elif ln.startswith("k_range:"):
            ln = f"k_range: {kr}"
        out.append(ln)
    dst = src.with_name(src.stem + f"_{_window_tag(window)}.yaml")
    dst.write_text("\n".join(out) + "\n")
    return dst


def _write_run_config(hp: dict, tag: str) -> Path:
    """Run config = cvae_noise.yaml with the selected (non-architecture) hyperparameters."""
    cfg = yaml.safe_load(Path(K.RUN_CFG).read_text())
    cfg["model_params"].update(hp)
    cfg["model_params"].update(K.FIXED_ARCH)
    dst = Path(K.RUN_CFG).with_name("cvae_noise_k3_tuned.yaml")
    header = (f"# SP(k=3) study: cvae_noise.yaml with hyperparameters selected by grouped 5-fold CV\n"
              f"# on the non-test sims (scripts/k3_sweep.py, selected trial {tag}). Architecture unchanged.\n")
    dst.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    return dst


def _train_group(data_cfg: Path, run_cfg: Path, group: str) -> list[Path]:
    """Train + record seeds 0/1/2 and move the records into ``experiments/runs/<group>/``."""
    dc, rc = load_configs(data_cfg, run_cfg)
    gdir = repo_root() / "experiments" / "runs" / group
    gdir.mkdir(parents=True, exist_ok=True)
    moved = {}
    for seed in SEEDS:
        res = run_training(dc, dataclasses.replace(rc, seed=seed))
        dst = gdir / res.run_id
        shutil.move(str(res.run_dir), dst)
        moved[res.run_id] = dst
    ledger = repo_root() / "experiments" / "runs.jsonl"
    lines = []
    for ln in ledger.read_text().splitlines():
        d = json.loads(ln)
        if d["run_id"] in moved:
            d["run_dir"] = str(moved[d["run_id"]].relative_to(repo_root()))
            ln = json.dumps(d)
        lines.append(ln)
    ledger.write_text("\n".join(lines) + "\n")
    return list(moved.values())


def _score(run_dir: Path) -> dict:
    """Test-fold fractional metrics for one recorded run (reloaded from its checkpoint)."""
    env = json.loads((run_dir / "env.json").read_text())
    dc, rc = load_configs(run_dir / "config_data.yaml", run_dir / "config_run.yaml")
    td = load_training_data(fill_data_root(dc))
    model = joblib.load(Path(env["roots"]["scratch_root"]) / "models" / run_dir.name / "model.joblib")
    te = _subset(td, grouped_split(td.sim_index, rc.split)["test"])
    y = np.asarray(te.y, float).reshape(len(te.y), -1)
    pred = np.asarray(model.predict(te.X, te.X_cond, te.X_params), float).reshape(y.shape)
    rec = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
    assert np.isclose(np.sqrt(np.mean((pred - y) ** 2)), rec, rtol=1e-4)
    return K.fractional_metrics(y, pred, np.asarray(te.k, float))


def _runs(group: str) -> list[Path]:
    return sorted((repo_root() / "experiments" / "runs" / group).iterdir())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    sweep = Path(args.sweep)
    sel = json.loads((sweep / "selection.json").read_text())
    window = sel["stage4_best_window"]
    window = window if window == "single" else tuple(window)
    hp = sel["selected_hp"]

    run_cfg = _write_run_config(hp, sel["stage3_selected_tag"])
    arms = {}
    for snap in K.SNAPS:
        data_cfg = _write_data_config(snap, window)
        if not args.skip_train:
            _train_group(data_cfg, Path(K.RUN_CFG), f"k3_default_snap{snap}_nd2")
            _train_group(data_cfg, run_cfg, f"k3_tuned_snap{snap}_nd2")
        for arm, group in (("baseline_full_default", f"arcmin_noparams_snap{snap}_nd2"),
                           ("window_default", f"k3_default_snap{snap}_nd2"),
                           ("window_tuned", f"k3_tuned_snap{snap}_nd2")):
            per_seed = [_score(d) for d in _runs(group)]
            keys = [k for k, v in per_seed[0].items() if isinstance(v, float)]
            arms.setdefault(snap, {})[arm] = {
                "group": group,
                **{k: {"mean": float(np.mean([p[k] for p in per_seed])),
                       "sd": float(np.std([p[k] for p in per_seed], ddof=1))} for k in keys},
                **{k: v for k, v in per_seed[0].items() if not isinstance(v, float)},
            }
    (sweep / "test_scores.json").write_text(json.dumps(
        {"window": sel["stage4_best_window"], "hp": hp, "arms": arms}, indent=2))
    for snap, a in arms.items():
        print(f"\n=== snap{snap} TEST (mean ± sd over seeds) window={window} ===")
        names = [k for k in a["window_tuned"] if isinstance(a["window_tuned"][k], dict)]
        print(f"{'metric':24s}" + "".join(f"{n:>26s}" for n in a))
        for m in names:
            print(f"{m:24s}" + "".join(
                (f"{a[n][m]['mean']:>17.4f} ±{a[n][m]['sd']:.4f}" if m in a[n] else f"{'—':>26s}") for n in a))


if __name__ == "__main__":
    main()
