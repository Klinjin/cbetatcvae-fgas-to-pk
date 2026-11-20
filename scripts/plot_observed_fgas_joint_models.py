"""SP(k) predicted from the observed, beam-corrected f_gas(theta) by six models, at all 5 n.

Reads the predictions made from the beam-corrected DESI x ACT x HSC f_gas (fig16b LRG ->
snap74, z = 0.47; fig16a BGS -> snap82, z = 0.21; series 'DESI x ACT x HSC
(beam-corrected)'; inputs ``fgas_obs_{lrg,bgs}_hscy3_beamcorr.npz``) by

==========================  ===============================================  =====================================
model                       prediction file (``spk_pred_from_fgas_obs_*``)   script
==========================  ===============================================  =====================================
joint cVAE                  ``<obs>_joint_cvae``                             ``predict_observed_fgas_joint.py``
separate per-n cVAEs        ``<obs>_separate_cvae_all_nd``                   ``predict_observed_fgas.py``
separate per-n nofb cVAEs   ``<obs>_separate_cvae_nofb_all_nd``              ``predict_observed_fgas.py``
joint cVAE + log10 sigma    ``<obs>_joint_cvae_fgas_std``                    ``predict_observed_fgas_joint.py``
joint beta-TCVAE            ``<obs>_beta_tcvae``                             ``predict_observed_fgas_joint.py``
joint cond. beta-TCVAE      ``<obs>_cond_beta_tcvae``                        ``predict_observed_fgas_joint.py``
==========================  ===============================================  =====================================

with ``<obs>`` = ``lrg_hscy3_beamcorr`` / ``bgs_hscy3_beamcorr``. Before plotting it
checks that every model was decoded from the identical observed profile and error, on
the same k grid and percentiles, at the Planck 2018 cosmology (the nofb cVAEs get the
profile and error times the sidecar's ``fb_obs``, as ``predict_observed_fgas.py`` feeds
them). The separate cVAE of number-density index ``i`` is the run group
``joint_cvae._separate_group(snap, i)``, the same mapping whose number density and test
fold ``compare_joint_models.py`` checks against the joint data; the nofb cVAE of index
``i`` is ``nd_compare._group(snap, "nofb", i)`` (f_gas not / f_b, no cosmological
parameters; nd 0, 1, 3 trained by ``nd_compare_nofb.sbatch`` with the nd_2 data config
changed only in the number-density index).

Plots (``fgas_spk.pub_style``; conditioning cosmology Planck 2018; SP(k) = P_tot/P_DM
within the hydro run, the proxy target of CLAUDE.md):

* :func:`plot_grid` -- 2 x 5 grid (rows LRG/snap74 and BGS/snap82, columns the 5
  training n of the snapshot), the 5 models overlaid: median at ``fgas_obs``, light
  16--84% band, and thin dash-dotted 16--84% when ``fgas_obs`` is also drawn from
  ``N(fgas_obs, diag fgas_obs_err^2)``;
* :func:`plot_two_panel` -- one panel per snapshot (LRG/snap74, BGS/snap82), all models
  and all 5 training n overlaid: colour is the model, line style the n; median at
  ``fgas_obs`` and a faint 16--84% band (the measurement-error envelope is left to
  :func:`plot_grid`, where it does not compete with the n line styles);
* :func:`plot_single_model` -- one model, one snapshot, the 5 n overlaid (median and
  16--84% at ``fgas_obs``), in the style of the 10-23 ``..._joint_cvae_all_nd_eighth_gen``
  figure;
* :func:`summary_table` -- median and 16--84% at the k bins nearest 1, 3, 10 h/Mpc.

Used by ``SimulationStacker/tests/apply_vae_obs_fgas_beamcorr.ipynb``; the CLI writes the
two-panel figure and the table.

Usage::

    python scripts/plot_observed_fgas_joint_models.py --data-dir <dir with the npz/json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_cvae import _separate_group  # noqa: E402

from fgas_spk.paths import figure_dir  # noqa: E402

SNAPS = (74, 82)
OBS = {74: "lrg_hscy3_beamcorr", 82: "bgs_hscy3_beamcorr"}
TRACER_ROW = {74: "DESI LRG (snap74, $z=0.47$)", 82: "DESI BGS (snap82, $z=0.21$)"}
SURVEY = {74: r"DESI LRG z bin 1 ($z \sim 0.510$)", 82: r"DESI BGS z bin 0 ($z \sim 0.262$)"}
SNAP_Z = {74: 0.47, 82: 0.21}
N_ND = 5
# nd index -> line style for plot_two_panel (nd_2, the reference stack of the per-n cVAEs, solid)
ND_LS = {0: (0, (1, 1.5)), 1: (0, (6, 1.5, 1, 1.5)), 2: "-", 3: (0, (5, 2)), 4: (0, (8, 2, 1, 2, 1, 2))}
K_REPORT = (1.0, 3.0, 10.0)
_IN = r"\Delta\Sigma_{\rm kSZ}/(f_b\,\Delta\Sigma_{\rm GGL})"
_COSMO = r"\Omega_m,\sigma_8,\Omega_b,h,n_s"
# model -> file suffix, key of nd index i at snapshot s, grid label, colour (the same arms'
# colours in compare_joint_models.ARMS, the nofb arm's the nd_compare "cosmo" blue; all six,
# incl. the cond. beta-TCVAE's #b576af (shared with compare_joint_models.ARMS), validated with the dataviz palette script),
# single-panel legend header (format field z).
MODELS = {
    "joint_cvae": dict(
        suffix="joint_cvae", key=lambda s, i: f"joint_cvae/snap{s}_nd{i}",
        label="joint cVAE", colour="#1a9e77",
        header="joint cVAE$\\{" + _IN + ";\\ " + _COSMO + "\\}$ ($\\mathit{Planck}$ 2018)\n"
               "conditioned on $\\{\\log_{10} n,\\ z=%(z)s\\}$ at"),
    "separate_cvae": dict(
        suffix="separate_cvae_all_nd", key=_separate_group,
        label=r"separate per-$n$ cVAEs ($f_\mathrm{gas}/f_b$ + cosmo.)", colour="#eb6834",
        header="separate per-$n$ cVAEs$\\{" + _IN + ";\\ " + _COSMO + "\\}$ ($\\mathit{Planck}$ 2018)\n"
               "one model per $n$, no $X_\\mathrm{cond}$ (conditioned only by training stack) at"),
    "separate_cvae_nofb": dict(
        suffix="separate_cvae_nofb_all_nd",
        key=lambda s, i: (f"arcmin9nofb_noparams_kmax10_snap{s}_nd2" if i == 2
                          else f"nd_compare/arcmin9nofb_noparams_kmax10_snap{s}_nd{i}"),
        label=r"separate per-$n$ cVAEs ($f_\mathrm{gas}$, no $f_b$, no cosmo.)", colour="#2a78d6",
        header="separate per-$n$ cVAEs$\\{\\Delta\\Sigma_{\\rm kSZ}/\\Delta\\Sigma_{\\rm GGL}\\}$ "
               "(no $f_b$, no cosmological parameters)\n"
               "one model per $n$, no $X_\\mathrm{cond}$ (conditioned only by training stack) at"),
    "joint_cvae_fgas_std": dict(
        suffix="joint_cvae_fgas_std", key=lambda s, i: f"joint_cvae_fgas_std/snap{s}_nd{i}",
        label=r"joint cVAE + $\log_{10}\sigma_{f_\mathrm{gas}}$ input", colour="#c2185b",
        header="joint cVAE$\\{" + _IN + ",\\ \\log_{10}\\sigma_{f_{\\rm gas}};\\ " + _COSMO
               + "\\}$ ($\\mathit{Planck}$ 2018)\n"
               "with $\\sigma_{f_{\\rm gas}} = \\sigma_{\\rm obs}$ (measurement error), "
               "conditioned on $\\{\\log_{10} n,\\ z=%(z)s\\}$ at"),
    "beta_tcvae": dict(
        suffix="beta_tcvae", key=lambda s, i: f"beta_tcvae/snap{s}_nd{i}",
        label=r"joint $\beta$-TCVAE", colour="#8a6d00",
        header="joint $\\beta$-TCVAE$\\{" + _IN + ";\\ " + _COSMO + "\\}$ ($\\mathit{Planck}$ 2018)\n"
               "conditioned on $\\{\\log_{10} n,\\ z=%(z)s\\}$ at"),
    "cond_beta_tcvae": dict(
        suffix="cond_beta_tcvae", key=lambda s, i: f"cond_beta_tcvae/snap{s}_nd{i}",
        label=r"joint $\beta$-TCVAE, prior $p(z\,|\,x,\hat{y})$", colour="#b576af",
        header="joint $\\beta$-TCVAE$\\{" + _IN + ";\\ " + _COSMO + "\\}$ ($\\mathit{Planck}$ 2018)\n"
               "conditional prior $p(z\\,|\\,x,\\hat{y})$, conditioned on $\\{\\log_{10} n,\\ z=%(z)s\\}$ at"),
}


def load_predictions(data_dir: Path) -> dict:
    """Load and cross-check the models' (``MODELS``) predictions for both snapshots.

    Args:
        data_dir: Directory holding ``spk_pred_from_fgas_obs_<obs>_<suffix>.{npz,json}``.

    Returns:
        dict: ``preds[snap][model]`` -> {``k``, ``n`` (5,), ``q_obs`` / ``q_err`` (5, n_pct, n_k),
        ``pct`` (list), ``fgas_input``, ``fgas_input_err``, ``side`` (the sidecar)}.

    Raises:
        RuntimeError: If any model's input profile, error (times ``fb_obs`` for a group
            trained without the f_b normalisation), k grid, percentiles or cosmology
            differs from the joint cVAE's, or a number density disagrees.
    """
    preds = {}
    for s in SNAPS:
        preds[s] = {}
        for name, spec in MODELS.items():
            stem = data_dir / f"spk_pred_from_fgas_obs_{OBS[s]}_{spec['suffix']}"
            arr = np.load(stem.with_suffix(".npz"))
            side = json.loads(stem.with_suffix(".json").read_text())
            keys = [spec["key"](s, i) for i in range(N_ND)]
            missing = [kk for kk in keys if f"{kk}__q_obs" not in arr.files]
            if missing:
                raise RuntimeError(f"{stem}: missing groups {missing}")
            n = [side["groups"][kk]["number_density"] if "number_density" in side["groups"][kk] else np.nan
                 for kk in keys]
            fb_norm = {side["groups"][kk].get("fb_normalised", True) for kk in keys}
            if len(fb_norm) != 1:
                raise RuntimeError(f"{stem}: groups disagree on the f_b normalisation")
            preds[s][name] = dict(
                k=arr["k"], pct=[float(p) for p in arr["percentiles"]], side=side, keys=keys,
                n=np.array(n, dtype=float),
                q_obs=np.stack([arr[f"{kk}__q_obs"] for kk in keys]),
                q_err=np.stack([arr[f"{kk}__q_err_diag"] for kk in keys]),
                fgas_input=np.stack([arr[f"{kk}__fgas_input"] for kk in keys]),
                fgas_input_err=np.stack([arr[f"{kk}__fgas_input_err"] for kk in keys]),
                train_rank=np.stack([arr[f"{kk}__train_percentile_rank"] for kk in keys]),
                # predict_observed_fgas.py feeds the nofb groups fgas_obs * fb_obs
                input_scale=1.0 if fb_norm.pop() else float(side["fb_obs"]))
        ref = preds[s]["joint_cvae"]
        if np.isnan(ref["n"]).any():
            raise RuntimeError(f"snap{s}: joint cVAE sidecar lacks the number densities")
        for name, p in preds[s].items():
            same = (np.array_equal(p["k"], ref["k"]) and p["pct"] == ref["pct"]
                    and p["side"]["cosmology"] == ref["side"]["cosmology"]
                    and np.allclose(p["fgas_input"], ref["fgas_input"] * p["input_scale"], rtol=1e-12, atol=0)
                    and np.allclose(p["fgas_input_err"], ref["fgas_input_err"] * p["input_scale"],
                                    rtol=1e-12, atol=0))
            if not same:
                raise RuntimeError(f"snap{s} {name}: input / k / percentiles / cosmology differ from the joint cVAE's")
            if not name.startswith("separate_cvae") and not np.allclose(p["n"], ref["n"]):
                raise RuntimeError(f"snap{s} {name}: number densities {p['n']} != {ref['n']}")
            p["n"] = ref["n"]
    return preds


def _style_k_axis(ax, k) -> None:
    import matplotlib.ticker

    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(k.min(), 10.0)
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, which="both")


def _n_label(n: float, unit: bool = True) -> str:
    e = int(np.floor(np.log10(n)))
    lab = rf"$n = {n / 10**e:.1f}\times10^{{{e}}}$"
    return lab + r" $(\mathrm{Mpc}/h)^{-3}$" if unit else lab


def plot_grid(preds: dict, out_stem: Path, dpi: int = 150) -> list[Path]:
    """2 x 5 grid, the models overlaid per (snapshot, n) panel; saves ``out_stem``.{png,pdf}."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from fgas_spk.pub_style import PLOT_RC

    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, N_ND, figsize=(27, 12), sharex=True, sharey="row", layout="constrained")
        for r, s in enumerate(SNAPS):
            for c in range(N_ND):
                ax = axes[r, c]
                for name, spec in MODELS.items():
                    p = preds[s][name]
                    i16, i50, i84 = (p["pct"].index(q) for q in (16.0, 50.0, 84.0))
                    k, q, qe = p["k"], p["q_obs"][c], p["q_err"][c]
                    ax.fill_between(k, q[i16], q[i84], color=spec["colour"], alpha=0.15, lw=0)
                    ax.plot(k, q[i50], color=spec["colour"], lw=2.4)
                    ax.plot(k, qe[i16], color=spec["colour"], lw=1.3, ls="-.")
                    ax.plot(k, qe[i84], color=spec["colour"], lw=1.3, ls="-.")
                _style_k_axis(ax, k)
                ax.set_title(f"{TRACER_ROW[s]}\n{_n_label(preds[s]['joint_cvae']['n'][c])}", fontsize=15)
                if r == 1:
                    ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
            axes[r, 0].set_ylabel(r"$P_\mathrm{tot}(k)\,/\,P_\mathrm{DM}(k)$")
        model_handles = [(Patch(facecolor=spec["colour"], alpha=0.15, lw=0),
                          Line2D([], [], color=spec["colour"], lw=2.4)) for spec in MODELS.values()]
        style_handles = [Line2D([], [], color="0.3", lw=2.4), Patch(facecolor="0.3", alpha=0.15, lw=0),
                         Line2D([], [], color="0.3", lw=1.3, ls="-.")]
        style_labels = [r"median at $f_\mathrm{gas}^\mathrm{obs}$", r"16–84% at $f_\mathrm{gas}^\mathrm{obs}$",
                        r"16–84%, $f_\mathrm{gas}\sim\mathcal{N}(f_\mathrm{gas}^\mathrm{obs},"
                        r"\,\mathrm{diag}\,\sigma_\mathrm{obs}^2)$"]
        # Column-major legend, one style entry per row in the last column: pad the model
        # entries with blanks to fill whole columns.
        nrow = len(style_handles)
        pad = -len(model_handles) % nrow
        blank = [Line2D([], [], lw=0)] * pad
        fig.legend(model_handles + blank + style_handles,
                   [spec["label"] for spec in MODELS.values()] + [""] * pad + style_labels,
                   loc="outside upper center", ncol=(len(model_handles) + pad) // nrow + 1,
                   frameon=False, fontsize=16,
                   title=r"beam-corrected DESI $\times$ ACT $\times$ HSC $f_\mathrm{gas}$, "
                         r"$\mathit{Planck}$ 2018 cosmology", title_fontsize=17)
        paths = []
        for ext in ("png", "pdf"):
            path = Path(f"{out_stem}.{ext}")
            fig.savefig(path, dpi=dpi)
            paths.append(path)
        plt.close(fig)
    return paths


def plot_two_panel(preds: dict, out_stem: Path, dpi: int = 150, band_alpha: float = 0.06) -> list[Path]:
    """Two panels (LRG/snap74, BGS/snap82), every model at every training n; saves ``out_stem``.{png,pdf}.

    Colour follows the model (``MODELS``), line style the number-density index
    (``ND_LS``). Each (model, n) draws its median at ``fgas_obs`` and a faint 16--84%
    band at ``fgas_obs``; the n legend of each panel gives that snapshot's number densities.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from fgas_spk.pub_style import PLOT_RC

    with matplotlib.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, len(SNAPS), figsize=(20, 9.5), sharey=True, layout="constrained")
        for ax, s in zip(axes, SNAPS):
            for name, spec in MODELS.items():
                p = preds[s][name]
                i16, i50, i84 = (p["pct"].index(q) for q in (16.0, 50.0, 84.0))
                k = p["k"]
                for i in range(N_ND):
                    q = p["q_obs"][i]
                    ax.fill_between(k, q[i16], q[i84], color=spec["colour"], alpha=band_alpha, lw=0)
                    ax.plot(k, q[i50], color=spec["colour"], lw=2.0, ls=ND_LS[i])
            _style_k_axis(ax, k)
            ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
            ax.set_title(f"{SURVEY[s]} $\\leftrightarrow$ snap{s}", fontsize=17)
            nd_handles = [Line2D([], [], color="0.25", lw=2.0, ls=ND_LS[i]) for i in range(N_ND)]
            ax.legend(nd_handles, [f"nd$_{i}$: {_n_label(preds[s]['joint_cvae']['n'][i])}" for i in range(N_ND)],
                      loc="lower left", fontsize=13, frameon=True, framealpha=0.9, edgecolor="0.7",
                      handlelength=3.2, title="training number density", title_fontsize=13)
        axes[0].set_ylabel(r"$P_\mathrm{tot}(k)\,/\,P_\mathrm{DM}(k)$")
        model_handles = [(Patch(facecolor=spec["colour"], alpha=0.25, lw=0),
                          Line2D([], [], color=spec["colour"], lw=2.4)) for spec in MODELS.values()]
        fig.legend(model_handles, [spec["label"] for spec in MODELS.values()],
                   loc="outside upper center", ncol=3, frameon=False, fontsize=15,
                   title=r"beam-corrected DESI $\times$ ACT $\times$ HSC $f_\mathrm{gas}$, "
                         r"$\mathit{Planck}$ 2018 cosmology; median and 16–84% at $f_\mathrm{gas}^\mathrm{obs}$",
                   title_fontsize=16)
        paths = []
        for ext in ("png", "pdf"):
            path = Path(f"{out_stem}.{ext}")
            fig.savefig(path, dpi=dpi)
            paths.append(path)
        plt.close(fig)
    return paths


def plot_single_model(preds: dict, model: str, snap: int, out_path: Path, dpi: int = 200) -> Path:
    """One model at one snapshot, the 5 training n overlaid (median + 16--84% at ``fgas_obs``).

    Reproduces the layout of ``Spk_pred_from_fgas_obs_lrg_hscy3_combined_joint_cvae_all_nd_eighth_gen``
    (2026-10-23): one (11, 8) panel titled with the survey label, one colour per n from
    ``fgas_spk.pub_style.COLOURMAP`` (twilight_r) sampled on [0.2, 0.85] -- the notebook's
    sampling, which reproduces the reference's colours; ``PLOT_RC``'s cycle runs to 0.98,
    where its fifth colour is near-white -- a framed upper-left legend whose title is the
    model description. The measurement-error envelope is left to :func:`plot_grid`
    (ten more dash-dotted lines would hide the five medians).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from fgas_spk.pub_style import COLOURMAP, PLOT_RC

    p, spec = preds[snap][model], MODELS[model]
    i16, i50, i84 = (p["pct"].index(q) for q in (16.0, 50.0, 84.0))
    k = p["k"]
    with matplotlib.rc_context(PLOT_RC):
        colours = COLOURMAP(np.linspace(0.2, 0.85, N_ND))
        fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
        handles, labels = [], []
        for i in range(N_ND):
            col = colours[i]
            q = p["q_obs"][i]
            ax.fill_between(k, q[i16], q[i84], color=col, alpha=0.15, lw=0)
            ax.plot(k, q[i50], color=col, lw=2.4)
            handles.append((Patch(facecolor=col, alpha=0.15, lw=0), Line2D([], [], color=col, lw=2.4)))
            labels.append(_n_label(p["n"][i]))
        _style_k_axis(ax, k)
        ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel(r"$P_\mathrm{tot}(k)\,/\,P_\mathrm{DM}(k)$")
        ax.set_title(SURVEY[snap])
        leg = ax.legend(handles, labels, loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.7",
                        fontsize=15, title=spec["header"] % {"z": f"{SNAP_Z[snap]:g}"}, title_fontsize=14)
        leg.get_title().set_multialignment("left")
        # Raise the y-axis top until no band lies under the legend: the curves are bounded by
        # the 84th percentile over the k range the legend spans (axes x fraction [0, x1]).
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
        x0, x1 = ax.get_xlim()
        k_edge = 10 ** (np.log10(x0) + bb.x1 * (np.log10(x1) - np.log10(x0)))
        under = k <= k_edge
        top_data = float(p["q_obs"][:, i84, under].max())
        ylo, yhi = ax.get_ylim()
        room = bb.y0 - 0.03  # axes fraction left below the legend, with a small gap
        if ylo + room * (yhi - ylo) < top_data:
            ax.set_ylim(ylo, ylo + (top_data - ylo) / room)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return out_path


def summary_table(preds: dict) -> list[dict]:
    """Median and 16--84% (at ``fgas_obs`` and with the diag-error draws) at k nearest 1, 3, 10 h/Mpc."""
    rows = []
    for s in SNAPS:
        for name in MODELS:
            p = preds[s][name]
            i16, i50, i84 = (p["pct"].index(q) for q in (16.0, 50.0, 84.0))
            for i in range(N_ND):
                row = {"snapshot": s, "tracer": "LRG" if s == 74 else "BGS", "model": name,
                       "nd_index": i, "number_density": float(p["n"][i])}
                for kt in K_REPORT:
                    j = int(np.argmin(np.abs(np.log(p["k"]) - np.log(kt))))
                    q, qe = p["q_obs"][i], p["q_err"][i]
                    row[f"k={kt:g}"] = {"k_bin": float(p["k"][j]), "median": float(q[i50, j]),
                                        "p16": float(q[i16, j]), "p84": float(q[i84, j]),
                                        "p16_with_obs_err": float(qe[i16, j]),
                                        "p84_with_obs_err": float(qe[i84, j])}
                rows.append(row)
    return rows


def table_markdown(rows: list[dict]) -> str:
    """Markdown rendering of :func:`summary_table`: median [16, 84] (16, 84 with obs error)."""
    kb = {kt: rows[0][f"k={kt:g}"]["k_bin"] for kt in K_REPORT}
    lines = ["| tracer | n [(Mpc/h)⁻³] | model | " + " | ".join(f"k={kb[kt]:.3g}" for kt in K_REPORT) + " |",
             "|---|---|---|" + "---|" * len(K_REPORT)]
    for r in rows:
        cells = [f"{c['median']:.3f} [{c['p16']:.3f}, {c['p84']:.3f}] "
                 f"({c['p16_with_obs_err']:.3f}, {c['p84_with_obs_err']:.3f})"
                 for c in (r[f"k={kt:g}"] for kt in K_REPORT)]
        lines.append(f"| {r['tracer']} | {r['number_density']:.2g} | {r['model']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, help="Directory holding the prediction npz/json files.")
    ap.add_argument("--out-dir", default=None, help="Default: today's figures/.../observed_pred.")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else figure_dir("observed_pred")
    import matplotlib
    matplotlib.use("Agg")
    preds = load_predictions(Path(args.data_dir))
    for path in plot_two_panel(preds, out_dir / "Spk_pred_6models_beamcorr_all_nd_2panel"):
        print(path)
    rows = summary_table(preds)
    (out_dir / "Spk_pred_6models_beamcorr_summary.json").write_text(json.dumps(rows, indent=2))
    md = table_markdown(rows)
    (out_dir / "Spk_pred_6models_beamcorr_summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
