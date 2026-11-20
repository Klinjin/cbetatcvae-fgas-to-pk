"""Joint cVAE without any conditioning: f_gas(R) -> SP(k), no X_cond (n, z) and no cosmology.

The ``nocond`` arm of ``scripts/joint_cvae.py``: the same single conditional VAE
(``scripts/configs/run/cvae_noise_joint.yaml``), the same stacked data (1024 sims x
10 (snapshot, n) rows, snap74 DESI LRG z = 0.47 and snap82 DESI BGS z = 0.21), the
same 9-column f_gas/f_b profile input (1--6 arcmin), outer split (seed 10) and seeds
0/1/2, with both conditioning modalities removed:

* ``X``        -- the stacked f_gas(R)/f_b profile (unchanged);
* ``X_params`` -- none (``include_camels_params = False``);
* ``X_cond``   -- none (no log10 n, no z).

The cVAE's three networks then see the profile only (``n_context = 0``). This is the
expected worst case of the joint comparison (``scripts/compare_joint_models.py``):
the network cannot tell the snapshot, the halo sample's number density or the
cosmology apart except through the profile itself. Note that the input is still
f_gas / f_b with the per-simulation f_b = Omega_b / Omega_m, so some cosmological
information enters through the input normalisation even without ``X_params``.

Records go to ``experiments/runs/joint_cvae_nocond/<run_id>/`` (``x_cond.json``
records no columns; the ledger line states ``include_camels_params: false``) plus
one ledger line; the checkpoint to ``$FGAS_SCRATCH_ROOT/models/<run_id>/model.joblib``.

Usage::

    python scripts/joint_cvae_nocond.py --train          # GPU, seeds 0/1/2
    python scripts/joint_cvae_nocond.py --diagnostics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_cvae as J  # noqa: E402

from fgas_spk.paths import figure_dir  # noqa: E402

GROUP = "joint_cvae_nocond"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", action="store_true", help="Train + record the seeds of --seeds.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(J.SEEDS),
                    help="Seeds to train (default 0 1 2).")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Runner's standard diagnostic figures per seed and tracer.")
    ap.add_argument("--out-dir", default=None, help="Default: today's figures/.../joint_cvae_nocond.")
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
            J.train(seed, group=GROUP, run_cfg=J.RUN_CFG, nocond=True)
    if args.diagnostics:
        J.diagnostics(Path(args.out_dir) if args.out_dir else figure_dir(GROUP),
                      group=GROUP, run_cfg=J.RUN_CFG, nocond=True)


if __name__ == "__main__":
    main()
