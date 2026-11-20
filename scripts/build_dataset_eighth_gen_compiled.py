"""Build the snap74 and snap82 datasets directly from the eighth-gen compiled files.

Reads the producer's compiled f_gas files (``test_CAMELS_eighth_gen.ipynb``),

    DeltaSigma_kSZ_over_DeltaSigma_total_profiles_snap{NN}_nd_{i}_n_{N}_eighth_gen.npz

(comoving Mpc/h radial design, 20 bins 0.1--20 Mpc/h; halos already ranked by
saved stellar mass before the ``[:, :n_halos]`` cut), for all five number
densities of each snapshot, pairs them with ``Ptot_Pdm_ratio_snap{NN}.npz``
(the full 255-bin k grid of ``Pk_512_3Dfield_snap{NN}.npz``, 0.178--32.1 h/Mpc)
and the CAMELS parameter matrix, and saves one dataset per snapshot under tag
``20260930``.

The two snapshots use different number-density bins (the notebook's
``number_densities_to_plot`` and ``number_densities_to_plot_82``); the halo
counts in the filenames follow from ``int(nd * 50**3)``. The snap82 files store
the snap74 list in their ``number_density`` scalar (a producer bug; their
``n_halos`` are correct), which the builder warns about and overrides with the
snap82 list below.

Before saving, each build is verified against an in-memory stellar-ranked raw
rebuild (:func:`build_fgas_spk_dataset`, ``rank_key='stellar'``) from the
per-sim profile files, so the ``fgas_definition`` stamp (ratio of stacked
profiles) and the ``stellar`` rank are checked, not assumed.

    python scripts/build_dataset_eighth_gen_compiled.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fgas_spk.builder import (
    build_fgas_spk_dataset,
    build_fgas_spk_from_compiled,
    fgas_meta_stamp,
)
from fgas_spk.schema import ensure_store_dirs, save_dataset

DATA_DIR = Path("/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data")
BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"
PROJECT = "/pscratch/sd/l/lindajin/fgas-to-pk-ml"
SUITE = "CAMELS-IllustrisTNG-L50n512-SB35"
TAG = "20260930"
BOX_SIZE_MPCH = 50.0

# snapshot -> (redshift, number-density bins [(Mpc/h)^-3]) as defined in the
# producer notebook's configuration cell.
SNAPSHOTS = {
    74: (0.47, (1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3)),  # n = 12 ... 300
    82: (0.21, (1.0e-3, 4.0e-3, 8.0e-3, 2.0e-2, 3.0e-2)),  # n = 125 ... 3750
}

# Whole-grid validation range for the realized-range stamp only (the compiled
# path applies no guard). Matches the maintainer-widened range of tag 20260929:
# sparse-stack outer-radius tails at nd 0 reach -28.4 / +30.5 (snap74).
STAMP_SANITY_RANGE = (-30.0, 31.0)

# Maximum |compiled - raw rebuild| accepted as "the same statistic".
IDENTITY_ATOL = 1e-10


def main() -> int:
    """Build, verify against the raw rebuild, and save both snapshots.

    Returns:
        int: 0 on success; 1 if any verification fails (nothing is saved for
            that snapshot).
    """
    ensure_store_dirs(PROJECT)
    status = 0
    for snapshot, (redshift, nds) in SNAPSHOTS.items():
        sup = DATA_DIR / f"Ptot_Pdm_ratio_snap{snapshot}.npz"
        params = DATA_DIR / "camels_params_matrix.npy"
        n_halos = [int(nd * BOX_SIZE_MPCH**3) for nd in nds]
        print(f"\n=== snap{snapshot} (z={redshift}); n_halos per nd: {n_halos}")

        ds = build_fgas_spk_from_compiled(
            data_dir=DATA_DIR,
            suppression_path=sup,
            snapshot=snapshot,
            number_densities=nds,
            box_size_mpch=BOX_SIZE_MPCH,
            name_extra="",
            name_suffix="_eighth_gen",
            params_path=params,
            rank_key="stellar",
        )
        print("compiled fgas:", ds.fgas.shape, "| k:", ds.k.size,
              f"[{ds.k.min():.3f}, {ds.k.max():.2f}] h/Mpc",
              "| radii:", ds.radii_mpch.size,
              f"[{ds.radii_mpch.min():g}, {ds.radii_mpch.max():g}] Mpc/h")

        # Verification against the stellar-ranked raw rebuild (not saved).
        raw = build_fgas_spk_dataset(
            base_path_template=BASE,
            suppression_path=sup,
            snapshot=snapshot,
            number_densities=nds,
            box_size_mpch=BOX_SIZE_MPCH,
            rank_key="stellar",
            params_path=params,
            fgas_sanity_range=(-np.inf, np.inf),
            progress=lambda i, n: print(f"\rraw rebuild {i}/{n}", end="", flush=True),
        )
        print()
        checks = {
            "sim_ids": np.array_equal(raw.sim_ids, ds.sim_ids),
            "radii_mpch": np.allclose(raw.radii_mpch, ds.radii_mpch),
            "k": np.array_equal(raw.k, ds.k),
            "suppression": np.array_equal(raw.suppression, ds.suppression),
            "camels_params": np.array_equal(raw.camels_params, ds.camels_params),
        }
        dfgas = np.abs(raw.fgas - ds.fgas)
        per_nd = np.nanmax(dfgas, axis=(0, 2))
        print("max |compiled - raw| fgas per nd:", np.array2string(per_nd, precision=2))
        checks["fgas"] = bool(np.all(per_nd < IDENTITY_ATOL))
        for key, ok in checks.items():
            print(f"  {'OK  ' if ok else 'FAIL'} {key}")
        if not all(checks.values()):
            print(f"snap{snapshot}: verification FAILED; not saved.")
            status = 1
            continue

        stamp = fgas_meta_stamp(ds.fgas, STAMP_SANITY_RANGE)
        out = save_dataset(
            ds,
            project_root=PROJECT,
            suite=SUITE,
            snapshot=snapshot,
            redshift=redshift,
            source="lindajin",
            tag=TAG,
            provenance={
                "profiles_repo": "github.com/Klinjin/SimulationStacker (fork)",
                "producer": (
                    "SimulationStacker/tests/test_CAMELS_eighth_gen.ipynb "
                    "(NAME_EXTRA='_eighth_gen')"
                ),
                "compiled_files": (
                    f"{DATA_DIR}/DeltaSigma_kSZ_over_DeltaSigma_total_profiles"
                    f"_snap{snapshot}_nd_{{0..4}}_n_{{{','.join(map(str, n_halos))}}}"
                    "_eighth_gen.npz (comoving Mpc/h design, not the arcmin design)"
                ),
                "suppression_file": str(sup),
                "note": (
                    "fgas read directly from the producer's compiled files "
                    "(build_fgas_spk_from_compiled); halos ranked by saved "
                    "stellar mass at production. Verified equal (max abs diff "
                    f"< {IDENTITY_ATOL:g}) to a stellar-ranked raw rebuild "
                    "from the per-sim profiles."
                ),
            },
            notes=(
                f"snap{snapshot} number-density bins {list(nds)} (Mpc/h)^-3, "
                f"n_halos {n_halos}. mean_halo_mass is NaN (not stored per sim "
                "in the compiled files). "
                + (
                    "The compiled snap82 files' number_density scalar holds the "
                    "snap74 list (producer bug); overridden by the snap82 list, "
                    "consistent with their n_halos. "
                    if snapshot == 82 else ""
                )
                + "Stamp sanity range (-30, 31) is the realized-range record "
                "only; the compiled path applies no guard."
            ),
            extra_meta=stamp,
        )
        print("saved:", out)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
