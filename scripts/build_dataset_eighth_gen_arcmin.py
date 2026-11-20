"""Build snap74 / snap82 datasets from the eighth-gen ARCMIN-design compiled files.

Reads the producer's compiled f_gas files for the angular radial design
(``test_CAMELS_eighth_gen.ipynb``),

    DeltaSigma_kSZ_over_DeltaSigma_total_profiles_arcmin_snap{NN}_nd_{i}_n_{N}_eighth_gen.npz

for all five number densities of each snapshot, and keeps only the radial bins
at or below ``MAX_RADIUS_ARCMIN`` = 6 arcmin. That is the part of the 18-bin
design that sits at identical angular radii in every simulation (1, 1.625, ...,
5.375, 6.0 arcmin; 9 bins; maintainer decision 2026-09-30, 6.0 included). The
outer bins are log-spaced to an edge fixed at 20 comoving Mpc/h, so their
angular radii vary with each simulation's cosmology.

Even the kept bins map to different comoving radii per simulation (the
arcmin -> Mpc/h conversion depends on each sim's cosmology), so the stored
``radii_mpch`` is the per-bin median, a nominal label only. The
``radial_grid`` stamp in ``__meta__`` records the exact angular grid and the
per-bin comoving min / median / max, and makes the loader refuse a
``radial_range_mpch`` crop on these datasets.

Verification before saving (per snapshot): the compiled f_gas (all 18 bins,
every nd) is compared with an independent ratio-of-stacks recomputation from
the raw per-sim ``prof{1,2}_*_DSigma_arcmin2_{xy,xz,yz}`` profiles, halos ranked
by descending saved ``stellar_masses``; the compiled per-sim radii are compared
with the raw ``r12_arcmin`` / ``r12_to_mpch``.

With ``--all-bins`` the angular cut is dropped and all 18 bins are kept, out
to the 20 comoving Mpc/h edge (tag ``20260930-arcmin-all``). The outer nine
bins then vary between simulations in BOTH angle and comoving radius, so the
``radial_grid`` stamp records per-bin min / median / max on both axes.

    python scripts/build_dataset_eighth_gen_arcmin.py             # 9 bins <= 6'
    python scripts/build_dataset_eighth_gen_arcmin.py --all-bins  # all 18 bins
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fgas_spk.builder import build_fgas_spk_from_compiled, fgas_meta_stamp
from fgas_spk.schema import ensure_store_dirs, save_dataset

DATA_DIR = Path("/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data")
BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"
PROJECT = "/pscratch/sd/l/lindajin/fgas-to-pk-ml"
SUITE = "CAMELS-IllustrisTNG-L50n512-SB35"
TAG_LE6, TAG_ALL = "20260930-arcmin-le6", "20260930-arcmin-all"
BOX_SIZE_MPCH = 50.0
MAX_RADIUS_ARCMIN = 6.0  # default angular cut (inclusive); None with --all-bins
PROJECTIONS = ("xy", "xz", "yz")

# snapshot -> (redshift, number-density bins [(Mpc/h)^-3]) as defined in the
# producer notebook's configuration cell.
SNAPSHOTS = {
    74: (0.47, (1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3)),  # n = 12 ... 300
    82: (0.21, (1.0e-3, 4.0e-3, 8.0e-3, 2.0e-2, 3.0e-2)),  # n = 125 ... 3750
}

# Realized-range record only (the compiled path applies no guard); same range
# as the Mpc/h-design builds of 2026-09-29/30.
STAMP_SANITY_RANGE = (-30.0, 31.0)
IDENTITY_ATOL = 1e-10


def _compiled_path(snapshot: int, nd_idx: int, n_halos: int) -> Path:
    return DATA_DIR / (
        "DeltaSigma_kSZ_over_DeltaSigma_total_profiles_arcmin"
        f"_snap{snapshot}_nd_{nd_idx}_n_{n_halos}_eighth_gen.npz"
    )


def _raw_arcmin_fgas(snapshot: int, n_halos: list[int], n_sims: int):
    """Recompute the arcmin-design f_gas from the raw per-sim profiles.

    Returns:
        tuple: ``fgas`` (n_sims, n_nd, 18) ratio of stacked profiles over the
            top-N halos by saved stellar mass x 3 projections, divided by f_b;
            ``r_arcmin`` and ``r_mpch`` (n_sims, 18) per-sim radii.
    """
    fname = f"Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma_snap{snapshot}.npz"
    fgas, r_arcmin, r_mpch = [], [], []
    for sim_id in range(n_sims):
        d = np.load(Path(BASE.format(sim_id)) / fname)
        order = np.argsort(-np.asarray(d["stellar_masses"]), kind="stable")
        ion = np.stack([d[f"prof2_ionized_gas_DSigma_arcmin2_{p}"] for p in PROJECTIONS], 1)
        tot = np.stack([d[f"prof1_total_DSigma_arcmin2_{p}"] for p in PROJECTIONS], 1)
        ion, tot = ion[:, :, order], tot[:, :, order]       # (n_radii, 3, n_halos)
        fb = float(d["fb"])
        fgas.append([
            np.nanmean(ion[:, :, :n].reshape(ion.shape[0], -1), axis=1)
            / np.nanmean(tot[:, :, :n].reshape(tot.shape[0], -1), axis=1) / fb
            for n in n_halos
        ])
        r_arcmin.append(np.asarray(d["r12_arcmin"]))
        r_mpch.append(np.asarray(d["r12_to_mpch"]))
        if (sim_id + 1) % 128 == 0:
            print(f"  raw recompute {sim_id + 1}/{n_sims}", flush=True)
    return np.asarray(fgas), np.asarray(r_arcmin), np.asarray(r_mpch)


def main(argv: list[str] | None = None) -> int:
    """Build, verify, and save both snapshots.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: 0 on success; 1 if any verification fails (that snapshot unsaved).
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--all-bins", action="store_true",
                        help="Keep all 18 arcmin-design bins (to 20 cMpc/h), no angular cut.")
    args = parser.parse_args(argv)
    max_radius = None if args.all_bins else MAX_RADIUS_ARCMIN
    tag = TAG_ALL if args.all_bins else TAG_LE6
    ensure_store_dirs(PROJECT)
    status = 0
    for snapshot, (redshift, nds) in SNAPSHOTS.items():
        sup = DATA_DIR / f"Ptot_Pdm_ratio_snap{snapshot}.npz"
        n_halos = [int(nd * BOX_SIZE_MPCH**3) for nd in nds]
        print(f"\n=== snap{snapshot} (z={redshift}); n_halos per nd: {n_halos}")

        ds = build_fgas_spk_from_compiled(
            data_dir=DATA_DIR, suppression_path=sup, snapshot=snapshot,
            number_densities=nds, box_size_mpch=BOX_SIZE_MPCH, name_extra="",
            name_suffix="_eighth_gen", params_path=DATA_DIR / "camels_params_matrix.npy",
            rank_key="stellar", radial_design="arcmin",
            max_radius_arcmin=max_radius,
        )

        # Full 18-bin compiled arrays, for verification and the grid stamp.
        comp = [np.load(_compiled_path(snapshot, i, n)) for i, n in enumerate(n_halos)]
        comp_fgas = np.stack([c["prof"] for c in comp], axis=1)   # (S, nd, 18)
        comp_ra, comp_rm = comp[0]["radii_arcmin"], comp[0]["radii_mpch"]
        if max_radius is None:
            keep = np.ones(comp_ra.shape[1], dtype=bool)
        else:
            keep = np.all(comp_ra <= max_radius * (1.0 + 1e-9), axis=0)

        raw_fgas, raw_ra, raw_rm = _raw_arcmin_fgas(snapshot, n_halos, comp_fgas.shape[0])
        per_nd = np.nanmax(np.abs(raw_fgas - comp_fgas), axis=(0, 2))
        print("max |compiled - raw| fgas per nd (18 bins):",
              np.array2string(per_nd, precision=2))
        checks = {
            "fgas (all bins) vs raw": bool(np.all(per_nd < IDENTITY_ATOL)),
            "radii_arcmin vs raw r12_arcmin": np.allclose(comp_ra, raw_ra),
            "radii_mpch vs raw r12_to_mpch": np.allclose(comp_rm, raw_rm),
            "dataset fgas = compiled[kept]": np.array_equal(ds.fgas, comp_fgas[:, :, keep]),
            "dataset radii_mpch = median of kept": np.allclose(
                ds.radii_mpch, np.median(comp_rm[:, keep], axis=0)),
        }
        if max_radius is not None:
            checks["kept bins fixed in arcmin"] = np.allclose(
                comp_ra[:, keep], comp_ra[0, keep])
        for key, ok in checks.items():
            print(f"  {'OK  ' if ok else 'FAIL'} {key}")
        if not all(checks.values()):
            print(f"snap{snapshot}: verification FAILED; not saved.")
            status = 1
            continue

        rm_kept, ra_kept = comp_rm[:, keep], comp_ra[:, keep]
        radial_grid = {
            "design": "arcmin",
            "fixed_across_sims": False,
            "fixed_axis": "arcmin" if max_radius is not None else None,
            "max_radius_arcmin_inclusive": max_radius,
            "radii_mpch_is": "per-bin median over simulations (nominal label only)",
            "radii_mpch_min": [float(v) for v in rm_kept.min(axis=0)],
            "radii_mpch_median": [float(v) for v in np.median(rm_kept, axis=0)],
            "radii_mpch_max": [float(v) for v in rm_kept.max(axis=0)],
            "radii_arcmin_min": [float(v) for v in ra_kept.min(axis=0)],
            "radii_arcmin_median": [float(v) for v in np.median(ra_kept, axis=0)],
            "radii_arcmin_max": [float(v) for v in ra_kept.max(axis=0)],
        }
        print(f"kept {int(keep.sum())} bins; arcmin per bin:",
              np.round(ra_kept.min(0), 3), "->", np.round(ra_kept.max(0), 3))
        print("comoving spread per bin [Mpc/h]:",
              np.round(rm_kept.min(0), 3), "->", np.round(rm_kept.max(0), 3))

        stamp = fgas_meta_stamp(ds.fgas, STAMP_SANITY_RANGE)
        stamp["radial_grid"] = radial_grid
        out = save_dataset(
            ds, project_root=PROJECT, suite=SUITE, snapshot=snapshot,
            redshift=redshift, source="lindajin", tag=tag,
            provenance={
                "profiles_repo": "github.com/Klinjin/SimulationStacker (fork)",
                "producer": ("SimulationStacker/tests/test_CAMELS_eighth_gen.ipynb "
                             "(NAME_EXTRA='_eighth_gen')"),
                "compiled_files": (
                    f"{DATA_DIR}/DeltaSigma_kSZ_over_DeltaSigma_total_profiles_arcmin"
                    f"_snap{snapshot}_nd_{{0..4}}_n_{{{','.join(map(str, n_halos))}}}"
                    "_eighth_gen.npz (arcmin design, 18 bins; "
                    + ("all bins kept)" if max_radius is None
                       else f"bins <= {max_radius:g} arcmin kept)")
                ),
                "suppression_file": str(sup),
                "note": (
                    "fgas read directly from the producer's compiled arcmin-design "
                    "files; halos ranked by saved stellar mass at production. "
                    f"Verified equal (max abs diff < {IDENTITY_ATOL:g}) to a "
                    "ratio-of-stacks recomputation from the raw arcmin2 profiles."
                ),
            },
            notes=(
                (f"ARCMIN radial design, all {int(keep.sum())} bins to the 20 "
                 "cMpc/h edge (outer bins vary per sim in angle and radius). "
                 if max_radius is None else
                 f"ARCMIN radial design, {int(keep.sum())} bins <= "
                 f"{max_radius:g} arcmin (fixed in angle across sims). ")
                + "radii_mpch is the per-bin median comoving radius (varies per "
                "sim; see radial_grid); radial_range_mpch crops are refused. "
                f"snap{snapshot} number-density bins {list(nds)} (Mpc/h)^-3, "
                f"n_halos {n_halos}. mean_halo_mass is NaN."
            ),
            extra_meta=stamp,
        )
        print("saved:", out)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
