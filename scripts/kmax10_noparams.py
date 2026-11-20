"""CVAE trained on SP(k <= k_max h/Mpc) (default k_max = 10): train and publication figure.

Arm: the snap74 (``--snap 74``, default; DESI LRG, z = 0.47) or snap82
(``--snap 82``; DESI BGS, z = 0.21) setup with the radial input of one
eighth-gen radial design (``--design``, default ``arcmin``), profile-only by default or additionally
conditioned on the five cosmological parameters (Omega0, sigma8, OmegaBaryon,
HubbleParam, n_s) with ``--cosmo``:

==========  ============  ===========================================================
design      conditioning  source data config (``scripts/configs/data/``; NN = snapshot)
==========  ============  ===========================================================
``arcmin``  none          ``config_eighth_gen_compiled_snapNN_arcmin_all_noparams.yaml``
``arcmin``  cosmology     ``config_eighth_gen_compiled_snapNN_arcmin_all.yaml``
``arcmin9`` none          ``config_eighth_gen_compiled_snapNN_arcmin_le6_noparams.yaml``
``arcmin9`` cosmology     ``config_eighth_gen_compiled_snapNN_arcmin_le6.yaml``
``arcmin9nofb`` none      ``config_eighth_gen_compiled_snapNN_arcmin_le6_nofb_noparams.yaml``
``mpch``    none          ``config_eighth_gen_compiled_snapNN_mpch_fullr_noparams.yaml``
``mpch``    cosmology     ``config_eighth_gen_compiled_snapNN_mpch_fullr.yaml``
``mpch9``   none          ``config_eighth_gen_compiled_snapNN_mpch_9bins_noparams.yaml``
``mpch9``   cosmology     ``config_eighth_gen_compiled_snapNN_mpch_9bins.yaml``
==========  ============  ===========================================================

(a missing source config for the chosen snapshot raises).

(``arcmin``: 18 arcmin-design bins to the 20 cMpc/h edge; ``arcmin9``: the 9
arcmin-design bins <= 6 arcmin, fixed in angle across sims; ``arcmin9nofb``: the
same 9 bins of f_gas NOT divided by f_b = Omega_b / Omega_m, dataset tag
20261001-arcmin-all-nofb, as the ``arcmin9nofb_*`` groups of
scripts/ablate_cosmo_params.py; ``mpch``: 20
comoving bins, 0.1 -- 20 Mpc/h; ``mpch9``: the 9 innermost of those comoving
bins, 0.1 -- 2.678 Mpc/h, cut by ``radial_range_mpch: [0.0, 2.7]`` as the
``mpch9_snapNN`` full-curve groups), each with ``cvae_noise.yaml`` and the target restricted
to the inclusive window k in [0, k_max] h/Mpc (``target_mode: k_range``;
``--kmax``, default 10).
Everything else -- split (seed 10), architecture, hyperparameters, training
seeds 0/1/2 -- is unchanged, so the only difference from the full-curve
baseline is the k window, the only difference between the designs is the
radial binning, and the only difference between the conditionings is the
cosmological-parameter input.

Training records go to
``experiments/runs/<design>_{noparams,cosmo}_kmax<k_max>_snap<NN>_nd2/`` (all source configs use
number-density index 2; checked at training time).

The figure is the runner's ``__rmse_vs_k_pub`` figure
(:func:`fgas_spk.train.plot_rmse_vs_k_pub`), redrawn from the reloaded checkpoint:

* top    -- per-k reconstruction RMSE of SP(k) (absolute) for the train / val /
  test splits, with the 1% threshold drawn at RMSE = 0.01, i.e. 1% of the
  unsuppressed SP = 1 (it is a larger fractional error where SP < 1);
* bottom -- held-out (test) signed residual in percent of the unsuppressed
  SP = 1, ``delta_SP = 100 * (pred - true)`` (the same normalisation as the top
  panel's 1% threshold; NOT divided by the true SP, which would inflate the
  error of the most suppressed curves): median with 16-84 and 5-95 percentile
  bands across curves.

Both panels carry a dotted vertical reference line at k = 3 h/Mpc.

Usage::

    python scripts/kmax10_noparams.py --train        # GPU (debug qos), ~1 min
    python scripts/kmax10_noparams.py --seed 0       # figure only
    python scripts/kmax10_noparams.py --design mpch --train --seed 2
    python scripts/kmax10_noparams.py --design mpch --cosmo --train --seed 2
    python scripts/kmax10_noparams.py --design arcmin9 --kmax 3 --train --seed 0
    python scripts/kmax10_noparams.py --design arcmin9 --snap 82 --train --seed 2
    python scripts/kmax10_noparams.py --design mpch9 --snap 82 --train --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k3_final import _train_group  # noqa: E402

from fgas_spk.experiment import load_configs  # noqa: E402
from fgas_spk.loader import load_training_data  # noqa: E402
from fgas_spk.paths import figure_dir, fill_data_root, repo_root  # noqa: E402
from fgas_spk.train import _subset, grouped_split, plot_rmse_vs_k_pub  # noqa: E402

SNAPS = (74, 82)
K_MAX_DEFAULT = 10.0  # upper edge [h/Mpc] of the inclusive target window [0, k_max]
# (design, cosmology-conditioned) -> source data config stem (all radial bins of
# the design, full-k target; the k window is applied by _write_data_config).
_CFG_DIR = Path("scripts/configs/data")
SRC_DATA_STEMS = {
    ("arcmin", False): "arcmin_all_noparams",
    ("arcmin", True): "arcmin_all",
    ("arcmin9", False): "arcmin_le6_noparams",
    ("arcmin9", True): "arcmin_le6",
    ("arcmin9nofb", False): "arcmin_le6_nofb_noparams",
    ("mpch", False): "mpch_fullr_noparams",
    ("mpch", True): "mpch_fullr",
    ("mpch9", False): "mpch_9bins_noparams",
    ("mpch9", True): "mpch_9bins",
}
# Comoving radial crop [Mpc/h] each design's source config must carry (None: all bins).
RADIAL_RANGE_MPCH = {"mpch9": (0.0, 2.7)}
DESIGNS = sorted({d for d, _ in SRC_DATA_STEMS})
RUN_CFG = Path("scripts/configs/run/cvae_noise.yaml")


def _src_data_cfg(design: str, cosmo: bool, snap: int) -> Path:
    """The arm's source data config for ``snap`` (raises if it does not exist)."""
    src = _CFG_DIR / f"config_eighth_gen_compiled_snap{snap}_{SRC_DATA_STEMS[(design, cosmo)]}.yaml"
    if not (repo_root() / src).exists():
        raise FileNotFoundError(f"No source data config for design={design!r}, cosmo={cosmo}, snap{snap}: {src}")
    return src


def _group(design: str, cosmo: bool, kmax: float, snap: int) -> str:
    return f"{design}_{'cosmo' if cosmo else 'noparams'}_kmax{kmax:g}_snap{snap}_nd2"


def _write_data_config(design: str, cosmo: bool, kmax: float, snap: int) -> Path:
    """The arm's data config for ``snap`` with the target cut to [0, kmax] h/Mpc."""
    src = _src_data_cfg(design, cosmo, snap)
    lines = (repo_root() / src).read_text().splitlines()
    out = [f"# {src.name} with the target restricted to k in "
           f"[0, {kmax:g}] h/Mpc (scripts/kmax10_noparams.py)."]
    for ln in lines:
        if ln.startswith("target_mode:"):
            ln = "target_mode: k_range"
        elif ln.startswith("k_range:"):
            ln = f"k_range: [0.0, {float(kmax)}]"
        out.append(ln)
    dst = (repo_root() / src).with_name(src.stem + f"_kmax{kmax:g}.yaml")
    dst.write_text("\n".join(out) + "\n")
    return dst


def _run_dir(group: str, seed: int) -> Path:
    import yaml

    hits = [d for d in sorted((repo_root() / "experiments" / "runs" / group).iterdir())
            if yaml.safe_load((d / "config_run.yaml").read_text())["seed"] == seed]
    if len(hits) != 1:
        raise RuntimeError(f"{group} seed {seed}: found {hits}")
    return hits[0]


def _predict_splits(run_dir: Path, kmax: float) -> tuple[np.ndarray, dict]:
    """k and {split: (y_true, y_pred)} for a recorded run (test RMSE checked)."""
    env = json.loads((run_dir / "env.json").read_text())
    dc, rc = load_configs(run_dir / "config_data.yaml", run_dir / "config_run.yaml")
    td = load_training_data(fill_data_root(dc))
    if td.source_path != env["dataset"]["path"]:
        raise RuntimeError(f"Dataset mismatch for {run_dir}")
    model = joblib.load(Path(env["roots"]["scratch_root"]) / "models" / run_dir.name / "model.joblib")
    masks = grouped_split(td.sim_index, rc.split)
    out = {}
    for name in ("train", "val", "test"):
        if not np.asarray(masks[name]).any():
            continue
        sub = _subset(td, masks[name])
        y = np.asarray(sub.y, float)
        out[name] = (y, np.asarray(model.predict(sub.X, sub.X_cond, sub.X_params), float).reshape(y.shape))
    y, p = out["test"]
    recorded = json.loads((run_dir / "summary.json").read_text())["test_rmse"]
    if not np.isclose(np.sqrt(np.mean((p - y) ** 2)), recorded, rtol=1e-4):
        raise RuntimeError(f"{run_dir.name}: reloaded test RMSE does not match the record.")
    k = np.asarray(td.k, float)
    if k.max() > kmax or np.any(y <= 0):
        raise RuntimeError("Unexpected k window or non-positive SP(k).")
    return k, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", action="store_true", help="Train + record seeds 0/1/2 first.")
    ap.add_argument("--seed", type=int, default=0, help="Training seed to plot.")
    ap.add_argument("--design", choices=DESIGNS, default="arcmin",
                    help="Radial design of the input profile (default: arcmin).")
    ap.add_argument("--cosmo", action="store_true",
                    help="Condition on the five cosmological parameters (default: profile only).")
    ap.add_argument("--kmax", type=float, default=K_MAX_DEFAULT,
                    help="Upper edge [h/Mpc] of the inclusive target window [0, kmax] (default: 10).")
    ap.add_argument("--snap", type=int, choices=SNAPS, default=74,
                    help="Snapshot: 74 (DESI LRG, z=0.47; default) or 82 (DESI BGS, z=0.21).")
    args = ap.parse_args()
    group = _group(args.design, args.cosmo, args.kmax, args.snap)

    if args.train:
        cfg = _write_data_config(args.design, args.cosmo, args.kmax, args.snap)
        dc, rc = load_configs(cfg, repo_root() / RUN_CFG)
        assert dc.target_mode == "k_range" and tuple(dc.k_range) == (0.0, args.kmax)
        assert dc.snapshot == args.snap and list(dc.number_density_indices) == [2]
        want_r = RADIAL_RANGE_MPCH.get(args.design)
        got_r = None if dc.radial_range_mpch is None else tuple(map(float, dc.radial_range_mpch))
        assert got_r == want_r and dc.include_camels_params == args.cosmo
        for d in _train_group(cfg, repo_root() / RUN_CFG, group):
            print(d)

    run_dir = _run_dir(group, args.seed)
    k, splits = _predict_splits(run_dir, args.kmax)
    out_dir = figure_dir(group)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_dir.name}__rmse_vs_k_pub.png"
    stats = plot_rmse_vs_k_pub(k, splits, out, held="test", snapshot=args.snap, cosmo=args.cosmo)
    print(json.dumps({"run": run_dir.name, **stats}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
