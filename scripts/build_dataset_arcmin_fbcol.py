"""Add a derived ``f_b = Omega_b / Omega_m`` column to the arcmin-design datasets' CAMELS params.

The loader selects conditioning parameters by stored column name
(``camels_param_names``) and has no derived-feature mechanism, so conditioning on
the cosmic baryon fraction needs it stored as a column. This writes a copy of the
verified ``20260930-arcmin-all`` dataset (f_gas normalised by f_b; all 18 bins,
all five number densities) whose ``camels_params`` gains one trailing column
``f_b`` = OmegaBaryon / Omega0 (35 SB35 columns -> 36). Every other array (fgas,
fgas_std, k, SP(k), sim_ids, radii) is copied unchanged; the source ``__meta__``
stamps (fgas definition, realized range, radial grid) are carried over.

Verification before saving (per snapshot): the new column equals the ``fb``
saved in every simulation's raw profile file (the f_b the builder divided by);
the first 35 columns and every other array equal the source dataset exactly.

Usage::

    python scripts/build_dataset_arcmin_fbcol.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from fgas_spk.camels_params import param_names_for
from fgas_spk.schema import load_dataset, resolve_dataset_path, save_dataset

BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"
PROJECT = "/pscratch/sd/l/lindajin/fgas-to-pk-ml"
SUITE = "CAMELS-IllustrisTNG-L50n512-SB35"
SRC_TAG, TAG = "20260930-arcmin-all", "20260930-arcmin-all-fbcol"
RANK = "mstar"
SNAPSHOTS = {74: 0.47, 82: 0.21}
FB_NAME = "f_b"
META_CARRY = ("fgas_definition", "fgas_sanity_range", "fgas_realized_range_per_bin", "radial_grid")


def main() -> int:
    status = 0
    for snapshot, redshift in SNAPSHOTS.items():
        src = resolve_dataset_path(PROJECT, SUITE, snapshot, redshift, "lindajin", RANK, SRC_TAG)
        ds = load_dataset(src)
        src_meta = json.loads(str(np.load(src)["__meta__"]))
        names = tuple(ds.camels_param_names or param_names_for(SUITE))
        P = np.asarray(ds.camels_params, float)
        fb_col = P[:, names.index("OmegaBaryon")] / P[:, names.index("Omega0")]
        fname = f"Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma_snap{snapshot}.npz"
        fb_raw = np.array([float(np.load(Path(BASE.format(int(s))) / fname)["fb"]) for s in ds.sim_ids])
        new = dataclasses.replace(ds, camels_params=np.hstack([P, fb_col[:, None]]),
                                  camels_param_names=(*names, FB_NAME))
        checks = {
            "f_b column == raw-profile fb (every sim)": np.allclose(fb_col, fb_raw, rtol=1e-12, atol=0),
            "first 35 columns unchanged": np.array_equal(new.camels_params[:, :-1], ds.camels_params),
            "fgas / SP(k) / sim_ids unchanged": all(np.array_equal(getattr(new, a), getattr(ds, a))
                                                    for a in ("fgas", "fgas_std", "suppression", "k", "sim_ids")),
        }
        print(f"=== snap{snapshot}: params {P.shape} -> {new.camels_params.shape}; "
              f"f_b {fb_col.min():.4f} -- {fb_col.max():.4f}")
        for key, ok in checks.items():
            print(f"  {'OK  ' if ok else 'FAIL'} {key}")
        if not all(checks.values()):
            print(f"snap{snapshot}: verification FAILED; not saved.")
            status = 1
            continue
        out = save_dataset(
            new, project_root=PROJECT, suite=SUITE, snapshot=snapshot, redshift=redshift,
            source="lindajin", tag=TAG,
            provenance={
                "derived_from": str(src),
                "operation": f"camels_params gains a trailing derived column '{FB_NAME}' = "
                             "OmegaBaryon / Omega0; all other arrays copied unchanged",
                "script": "scripts/build_dataset_arcmin_fbcol.py",
                "note": "f_b column verified equal to the raw profile files' saved fb for every simulation.",
            },
            notes=(f"As {SRC_TAG} (f_gas normalised by f_b) with a derived conditioning column "
                   f"'{FB_NAME}' = Omega_b / Omega_m appended to camels_params (36 columns)."),
            extra_meta={k: src_meta[k] for k in META_CARRY if k in src_meta},
        )
        print("saved:", out, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
