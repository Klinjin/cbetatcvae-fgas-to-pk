"""Target-k-window sweep and hyperparameter tuning of the profile-only CVAE for SP(k=3).

Goal: the profile-only (no CAMELS parameters) ``cvae`` with the architecture
held fixed (``latent_dim=1``, ``hidden=128``, ``n_layers=2``, learned Gaussian
observation-noise head) that predicts SP(k) at k = 3 h/Mpc as accurately as
possible, targeting an absolute RMSE of 0.01 in SP(k=3) (maintainer decision,
2026-10-01; replaces the earlier ~1% median-fractional-error target).

Selection discipline. The held-out **test** fold of the recorded runs (split
seed 10, 70/10/20) is never touched here. Every choice is scored by **5-fold
grouped (by sim) cross-validation over the 819 non-test sims**, pooling the
out-of-fold predictions so each metric is computed on all 819 curves.

Stages (each written to ``<out>/stage<N>.jsonl``):

1. **k-window sweep** at the default hyperparameters (``cvae_noise.yaml``):
   the training target is SP(k) over ``[k_lo, k_hi]`` for every pair in
   ``K_LO x K_HI`` (``k_hi = None`` = to the last bin), plus ``single_k`` at
   k = 3. Both snapshots.
2. **Random hyperparameter search** (``N_RANDOM`` configs + the default) at
   the stage-1 best window. Architecture fixed; tuned: lr, weight_decay,
   dropout, batch_size, epochs, beta, anneal fraction.
3. **Re-evaluation** of the stage-2 top ``N_TOP`` with an independent fold
   assignment and model seeds (guards against a lucky fold draw); the
   selected config minimises the mean over both CV repetitions.
4. **k-window sweep re-run** with the selected hyperparameters, to check the
   window choice is not an artefact of the default hyperparameters.

Objective: the pooled absolute RMSE, sqrt(mean (SP_pred - SP_true)^2), at the
k bin nearest 3 h/Mpc over the out-of-fold curves, averaged over the two
snapshots (one config serves both). Target 0.01.
Also recorded for every fit: median / p68 / p90 / RMS of the fractional error
at that bin, the fraction of curves within 1% / 2%, and median / RMS fractional
error in the k bands of ``BANDS`` that lie inside the trained window.

Usage (on a 4-GPU node)::

    python scripts/k3_sweep.py --out <dir> [--workers-per-gpu 4]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

SNAPS = (74, 82)
DATA_CFG = "scripts/configs/data/config_eighth_gen_compiled_snap{snap}_arcmin_all_noparams.yaml"
RUN_CFG = "scripts/configs/run/cvae_noise.yaml"
K_STAR = 3.0
K_LO = (0.0, 1.0, 2.0)
K_HI = (3.1, 5.0, 10.0, 20.0, None)
BANDS = ((0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, np.inf))
N_FOLDS = 5
CV_SEEDS = (101, 202)
N_RANDOM = 48
N_TOP = 6
FIXED_ARCH = {"latent_dim": 1, "hidden": 128, "n_layers": 2, "obs_noise_head": True}


# ----------------------------------------------------------------- worker side
_GPU_QUEUE = None


def _init_worker(q):
    """Pin this worker to one GPU (before torch is imported) and limit threads."""
    gpu = q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = "2"
    import torch

    torch.set_num_threads(2)


_DATA_CACHE: dict = {}


def _data(snap: int, window):
    """Load (and cache per worker) the training data for one snapshot/window."""
    key = (snap, window)
    if key not in _DATA_CACHE:
        from fgas_spk.experiment import load_configs
        from fgas_spk.loader import load_training_data
        from fgas_spk.paths import fill_data_root
        from fgas_spk.train import grouped_split

        dc, rc = load_configs(DATA_CFG.format(snap=snap), RUN_CFG)
        if window == "single":
            dc = dataclasses.replace(dc, target_mode="single_k", k_target=K_STAR, k_range=None)
        elif window[1] is None and window[0] == 0.0:
            dc = dataclasses.replace(dc, target_mode="curve", k_range=None)
        else:
            hi = window[1] if window[1] is not None else 1e9
            dc = dataclasses.replace(dc, target_mode="k_range", k_range=(window[0], hi))
        td = load_training_data(fill_data_root(dc))
        masks = grouped_split(td.sim_index, rc.split)
        _DATA_CACHE[key] = (td, masks, rc)
    return _DATA_CACHE[key]


def _cv_folds(sim_index: np.ndarray, non_test: np.ndarray, seed: int) -> np.ndarray:
    """Fold id per row (−1 for test rows), grouped by simulation."""
    sims = np.unique(sim_index[non_test])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sims)
    fold_of = {s: i % N_FOLDS for i, s in enumerate(perm)}
    out = np.full(sim_index.shape, -1)
    out[non_test] = [fold_of[s] for s in sim_index[non_test]]
    return out


def _fit_fold(job: dict) -> dict:
    """Fit one CV fold; return out-of-fold predictions (row ids + preds)."""
    from fgas_spk.models import REGISTRY
    from fgas_spk.train import _subset

    window = job["window"] if job["window"] == "single" else tuple(job["window"])
    td, masks, _ = _data(job["snap"], window)
    folds = _cv_folds(td.sim_index, ~masks["test"], job["cv_seed"])
    tr, va = (folds >= 0) & (folds != job["fold"]), folds == job["fold"]
    params = {**job["hp"], **FIXED_ARCH}
    model = REGISTRY["cvae"](seed=job["cv_seed"] + job["fold"], device="cuda", **params)
    model.fit(_subset(td, tr))
    sub = _subset(td, va)
    pred = np.asarray(model.predict(sub.X, None, None), dtype=float).reshape(np.shape(sub.y))
    return {"rows": np.flatnonzero(va).tolist(), "pred": pred.tolist(),
            "best_epoch": model.best_epoch}


# ----------------------------------------------------------------- metrics
def fractional_metrics(y: np.ndarray, pred: np.ndarray, k: np.ndarray) -> dict:
    """Fractional-error metrics at the bin nearest K_STAR and in BANDS.

    Args:
        y, pred: shape (n_curves, n_k) (a 1-D single_k target is reshaped).
        k: shape (n_k,), h/Mpc.
    """
    y = np.asarray(y, float).reshape(len(y), -1)
    pred = np.asarray(pred, float).reshape(y.shape)
    d = pred / y - 1.0
    i = int(np.argmin(np.abs(k - K_STAR)))
    a = np.abs(d[:, i])
    out = {
        "k_star": float(k[i]),
        "k3_median": float(np.median(a)),
        "k3_p68": float(np.percentile(a, 68)),
        "k3_p90": float(np.percentile(a, 90)),
        "k3_rms": float(np.sqrt(np.mean(d[:, i] ** 2))),
        "k3_frac_lt1pct": float(np.mean(a < 0.01)),
        "k3_frac_lt2pct": float(np.mean(a < 0.02)),
        "k3_abs_rmse": float(np.sqrt(np.mean((pred[:, i] - y[:, i]) ** 2))),
    }
    for lo, hi in BANDS:
        m = (k > lo) & (k <= hi)
        if m.any() and k[m].min() >= k.min() and m.sum() >= 1:
            key = f"band_{lo:g}_{hi:g}"
            out[f"{key}_median"] = float(np.median(np.abs(d[:, m])))
            out[f"{key}_rms"] = float(np.sqrt(np.mean(d[:, m] ** 2)))
            out[f"{key}_coverage"] = f"{k[m].min():.2f}-{k[m].max():.2f}"
    return out


# ----------------------------------------------------------------- driver side
def _evaluate(pool, configs: list[dict], cv_seed: int, out_path: Path, stage: str) -> list[dict]:
    """CV-evaluate a list of {window, hp, tag} configs on both snapshots."""
    from fgas_spk.train import _subset  # noqa: F401  (import check on driver)

    jobs = []
    for ci, cfg in enumerate(configs):
        for snap in SNAPS:
            for f in range(N_FOLDS):
                jobs.append({"ci": ci, "snap": snap, "fold": f, "cv_seed": cv_seed,
                             "window": cfg["window"], "hp": cfg["hp"]})
    results = pool.map(_fit_fold, jobs, chunksize=1)

    rows_out = []
    for ci, cfg in enumerate(configs):
        rec = {"stage": stage, "tag": cfg.get("tag"), "window": cfg["window"],
               "hp": cfg["hp"], "cv_seed": cv_seed, "snap": {}}
        for snap in SNAPS:
            window = cfg["window"] if cfg["window"] == "single" else tuple(cfg["window"])
            td, masks, _ = _data(snap, window)
            y = np.asarray(td.y, float).reshape(len(td.y), -1)
            pred = np.full_like(y, np.nan)
            best = []
            for job, r in zip(jobs, results):
                if job["ci"] == ci and job["snap"] == snap:
                    pred[r["rows"]] = np.asarray(r["pred"]).reshape(len(r["rows"]), -1)
                    best.append(r["best_epoch"])
            nt = ~masks["test"]
            assert np.isfinite(pred[nt]).all()
            m = fractional_metrics(y[nt], pred[nt], np.asarray(td.k, float))
            m["best_epochs"] = best
            rec["snap"][snap] = m
        rec["objective"] = float(np.mean([rec["snap"][s]["k3_abs_rmse"] for s in SNAPS]))
        rows_out.append(rec)
        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(stage, cfg.get("tag"), cfg["window"], f"obj(k3 abs RMSE)={rec['objective']:.4f}",
              {s: round(rec["snap"][s]["k3_median"], 4) for s in SNAPS}, flush=True)
    return rows_out


def _default_hp() -> dict:
    import yaml

    mp_ = yaml.safe_load(Path(RUN_CFG).read_text())["model_params"]
    return {k: v for k, v in mp_.items() if k not in FIXED_ARCH}


def _random_hp(rng) -> dict:
    epochs = int(rng.choice([200, 400, 800]))
    return {
        "epochs": epochs,
        "lr": float(10 ** rng.uniform(-4, np.log10(3e-3))),
        "weight_decay": float(10 ** rng.uniform(-6, -2)),
        "dropout": float(rng.choice([0.0, 0.05, 0.1, 0.2])),
        "batch_size": int(rng.choice([16, 32, 64, 128])),
        "beta": float(10 ** rng.uniform(-2, 0)),
        "anneal_epochs": int(epochs * rng.choice([0.0, 0.25, 0.5])),
        "val_frac": 0.1,
    }


def _windows():
    w = [[lo, hi] for lo in K_LO for hi in K_HI]
    return w + ["single"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--n-gpus", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for g in range(args.n_gpus):
        for _ in range(args.workers_per_gpu):
            q.put(g)
    with ctx.Pool(args.n_gpus * args.workers_per_gpu, initializer=_init_worker, initargs=(q,)) as pool:
        hp0 = _default_hp()

        # Stage 1: window sweep at default HP.
        s1 = _evaluate(pool, [{"window": w, "hp": hp0, "tag": "default"} for w in _windows()],
                       CV_SEEDS[0], out / "stage1.jsonl", "stage1")
        best_w = min(s1, key=lambda r: r["objective"])["window"]
        print("STAGE1 best window", best_w, flush=True)

        # Stage 2: random search at that window.
        rng = np.random.default_rng(20261001)
        cfgs = [{"window": best_w, "hp": hp0, "tag": "default"}]
        cfgs += [{"window": best_w, "hp": _random_hp(rng), "tag": f"rand{i:02d}"} for i in range(N_RANDOM)]
        s2 = _evaluate(pool, cfgs, CV_SEEDS[0], out / "stage2.jsonl", "stage2")

        # Stage 3: re-evaluate the top N_TOP with an independent CV repetition.
        top = sorted(s2, key=lambda r: r["objective"])[:N_TOP]
        s3 = _evaluate(pool, [{"window": r["window"], "hp": r["hp"], "tag": r["tag"]} for r in top],
                       CV_SEEDS[1], out / "stage3.jsonl", "stage3")
        combined = {r["tag"]: 0.5 * (r["objective"] + next(t["objective"] for t in top if t["tag"] == r["tag"]))
                    for r in s3}
        best_tag = min(combined, key=combined.get)
        best_hp = next(r["hp"] for r in s3 if r["tag"] == best_tag)
        print("STAGE3 selected", best_tag, best_hp, combined, flush=True)

        # Stage 4: window sweep with the selected HP.
        s4 = _evaluate(pool, [{"window": w, "hp": best_hp, "tag": best_tag} for w in _windows()],
                       CV_SEEDS[0], out / "stage4.jsonl", "stage4")
        best_w4 = min(s4, key=lambda r: r["objective"])["window"]

    (out / "selection.json").write_text(json.dumps({
        "stage1_best_window": best_w, "stage3_selected_tag": best_tag, "selected_hp": best_hp,
        "stage3_combined_objective": combined, "stage4_best_window": best_w4,
        "fixed_arch": FIXED_ARCH}, indent=2))
    print("DONE", best_tag, best_hp, "window", best_w4, flush=True)


if __name__ == "__main__":
    main()
