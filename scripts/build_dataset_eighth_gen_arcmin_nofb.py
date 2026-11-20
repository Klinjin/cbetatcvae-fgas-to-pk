"""Build the arcmin-design datasets WITHOUT the f_b = Omega_b / Omega_m normalisation.

The standard builds (``scripts/build_dataset_eighth_gen_arcmin.py``,
:data:`fgas_spk.builder.FGAS_DEFINITION`) store

    f_gas(R) = <Delta Sigma_ionized>(R) / <Delta Sigma_total>(R) / f_b,

with ``<.>`` the nanmean over the top-N halos (ranked by saved stellar mass) x 3
projections and f_b the per-simulation cosmic baryon fraction. Here the same
ratio of stacked profiles is stored WITHOUT the final division by f_b:

    f_gas,raw(R) = <Delta Sigma_ionized>(R) / <Delta Sigma_total>(R),

for all five number densities of each snapshot and all 18 arcmin-design bins
(the loader's ``radial_range_arcmin: 6`` selects the 9 bins <= 6 arcmin, as for
the ``20260930-arcmin-all`` build). Construction: the verified
``20260930-arcmin-all`` dataset is multiplied, per simulation, by the ``fb``
saved in that simulation's raw profile file; ``fgas_std`` scales by the same
factor (it is ``|f_gas|`` times a dimensionless relative error). Every other
array (k, SP(k), sim_ids, camels_params, radii) is copied unchanged.

Verification before saving (per snapshot):

* saved ``fb`` equals OmegaBaryon / Omega0 from the CAMELS parameter matrix;
* the new f_gas equals an independent ratio-of-stacks recomputation from the
  raw per-sim ``prof{1,2}_*_DSigma_arcmin2_{xy,xz,yz}`` profiles with no f_b
  division (all 18 bins, every nd);
* new f_gas / f_b reproduces the source dataset exactly, and the relative
  error ``fgas_std / fgas`` is unchanged.

Usage::

    python scripts/build_dataset_eighth_gen_arcmin_nofb.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from fgas_spk.builder import fgas_meta_stamp
from fgas_spk.camels_params import CAMELS_PARAM_NAMES
from fgas_spk.schema import load_dataset, resolve_dataset_path, save_dataset

DATA_DIR = Path("/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data")
BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"
PROJECT = "/pscratch/sd/l/lindajin/fgas-to-pk-ml"
SUITE = "CAMELS-IllustrisTNG-L50n512-SB35"
SRC_TAG, TAG = "20260930-arcmin-all", "20261001-arcmin-all-nofb"
RANK = "mstar"
SNAPSHOTS = {74: 0.47, 82: 0.21}
BOX_SIZE_MPCH = 50.0
PROJECTIONS = ("xy", "xz", "yz")
IDENTITY_ATOL = 1e-10
FGAS_DEFINITION_NOFB = (
    "ratio of stacked profiles WITHOUT f_b normalisation: nanmean of Delta "
    "Sigma_ionized over the selected halos x 3 projections per radial bin, "
    "divided by the same nanmean of Delta Sigma_total. NOT divided by the "
    "per-simulation f_b = Omega_b / Omega_m (cf. fgas_spk.builder.FGAS_DEFINITION, "
    "which is this quantity / f_b)."
)


def _raw(snapshot: int, n_halos: list[int], sim_ids: np.ndarray):
    """Per-sim saved f_b and the un-normalised ratio of stacks from the raw profiles.

    Returns:
        tuple: ``fb`` (n_sims,); ``ratio`` (n_sims, n_nd, 18).
    """
    fname = f"Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma_snap{snapshot}.npz"
    fb, ratio = [], []
    for j, sim_id in enumerate(sim_ids):
        d = np.load(Path(BASE.format(int(sim_id))) / fname)
        order = np.argsort(-np.asarray(d["stellar_masses"]), kind="stable")
        ion = np.stack([d[f"prof2_ionized_gas_DSigma_arcmin2_{p}"] for p in PROJECTIONS], 1)[:, :, order]
        tot = np.stack([d[f"prof1_total_DSigma_arcmin2_{p}"] for p in PROJECTIONS], 1)[:, :, order]
        fb.append(float(d["fb"]))
        ratio.append([np.nanmean(ion[:, :, :n], axis=(1, 2)) / np.nanmean(tot[:, :, :n], axis=(1, 2))
                      for n in n_halos])
        if (j + 1) % 256 == 0:
            print(f"  raw {j + 1}/{len(sim_ids)}", flush=True)
    return np.asarray(fb), np.asarray(ratio)


def main() -> int:
    names = list(CAMELS_PARAM_NAMES[SUITE])
    i_m, i_b = names.index("Omega0"), names.index("OmegaBaryon")
    status = 0
    for snapshot, redshift in SNAPSHOTS.items():
        src = resolve_dataset_path(PROJECT, SUITE, snapshot, redshift, "lindajin", RANK, SRC_TAG)
        ds = load_dataset(src)
        src_meta = json.loads(str(np.load(src)["__meta__"]))
        n_halos = [int(round(nd * BOX_SIZE_MPCH**3)) for nd in ds.number_densities]
        print(f"\n=== snap{snapshot}: {src.name}; fgas {ds.fgas.shape}; n_halos {n_halos}", flush=True)

        fb, ratio = _raw(snapshot, n_halos, ds.sim_ids)
        fgas = ds.fgas * fb[:, None, None]
        fgas_std = ds.fgas_std * fb[:, None, None]
        P = np.asarray(ds.camels_params)
        checks = {
            "fb == OmegaBaryon/Omega0": np.allclose(fb, P[:, i_b] / P[:, i_m], rtol=1e-12, atol=0),
            "fgas == raw ratio of stacks (no f_b)": float(np.nanmax(np.abs(fgas - ratio))) < IDENTITY_ATOL,
            "fgas / fb == source dataset": np.allclose(fgas / fb[:, None, None], ds.fgas, rtol=1e-13, atol=0),
            "fgas_std/fgas unchanged": np.allclose(fgas_std / fgas, ds.fgas_std / ds.fgas, rtol=1e-12),
            "all finite": bool(np.isfinite(fgas).all() and np.isfinite(fgas_std).all()),
        }
        print("  max |fgas - raw| =", float(np.nanmax(np.abs(fgas - ratio))))
        for key, ok in checks.items():
            print(f"  {'OK  ' if ok else 'FAIL'} {key}")
        if not all(checks.values()):
            print(f"snap{snapshot}: verification FAILED; not saved.")
            status = 1
            continue
        print(f"  f_b range {fb.min():.4f} -- {fb.max():.4f}; "
              f"fgas range {fgas.min():.4f} -- {fgas.max():.4f} (was {ds.fgas.min():.4f} -- {ds.fgas.max():.4f})")

        new = dataclasses.replace(ds, fgas=fgas, fgas_std=fgas_std)
        stamp = fgas_meta_stamp(fgas, tuple(src_meta["fgas_sanity_range"]))
        stamp["fgas_definition"] = FGAS_DEFINITION_NOFB
        stamp["fgas_fb_normalised"] = False
        stamp["radial_grid"] = src_meta["radial_grid"]
        out = save_dataset(
            new, project_root=PROJECT, suite=SUITE, snapshot=snapshot, redshift=redshift,
            source="lindajin", tag=TAG,
            provenance={
                "derived_from": str(src),
                "operation": "fgas and fgas_std multiplied per simulation by the raw profile file's fb "
                             "(= OmegaBaryon / Omega0); all other arrays copied unchanged",
                "script": "scripts/build_dataset_eighth_gen_arcmin_nofb.py",
                "note": (f"Verified equal (max abs diff < {IDENTITY_ATOL:g}) to a ratio-of-stacks "
                         "recomputation from the raw arcmin2 profiles without the f_b division."),
            },
            notes=("ARCMIN radial design, all 18 bins (as 20260930-arcmin-all), f_gas NOT normalised by "
                   f"f_b. snap{snapshot} number-density bins {list(map(float, ds.number_densities))} "
                   f"(Mpc/h)^-3, n_halos {n_halos}. mean_halo_mass is NaN."),
            extra_meta=stamp,
        )
        print("saved:", out, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
