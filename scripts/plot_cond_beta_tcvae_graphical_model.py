"""Graphical models for the conditional-prior beta-TCVAE (``cond_beta_tcvae``).

Panel (a): the data-generating graph of the CAMELS-IllustrisTNG SB35 training set.

* Per simulation s: cosmology Omega = (Omega_m, sigma_8, Omega_b, h, n_s) and the
  astrophysical / feedback parameters theta of SB35.
* Per snapshot: redshift z (0.47 LRG, snap 74; 0.21 BGS, snap 82) and the box-wide
  power-spectrum suppression SP(k) = P_hydro / P_DMO, which depends on (Omega, theta, z)
  but not on any galaxy selection.
* Per number density n (5 values): the halo sample H is the top-n ranked objects of the
  snapshot, a selection rule with no Omega or theta input (edges n -> H, z -> H only).
  The stacked gas-fraction profile f_gas in arcmin apertures is measured on H in three
  projections (x3). n enters f_gas only through H: f_gas _||_ n | (H, z). Omega, theta
  and z also reach f_gas directly (cosmic baryon fraction and the arcmin-to-comoving
  conversion D_A(z; Omega); gas content at fixed halo mass), which is also the route by
  which the masses of the selected halos respond to Omega and theta.
* Measurement noise (f_gas^obs, sigma_g) is left out of this version of the figure.

Panel (b): the conditional generative model actually trained (Sohn et al. 2015 with the
recurrent connection), in VAE notation: phi = recognition (variational) parameters,
theta = generative parameters (prior network, baseline and decoder). x = f_gas,
c = (log10 n, z, Omega) drawn component by component, y = SP(k) (79 k bins) and the
latent zeta (the z_i of the loss). Feedback is not an input: zeta is the only route for
the feedback information that x and c leave out. The baseline y_hat enters the prior
through .detach() (stop-gradient; ``baseline_detach: true``).

Run:
    python scripts/plot_cond_beta_tcvae_graphical_model.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK = "#1f1f1f"
MUTED = "#6b6b6b"
SHADE = "#d9d9d9"
ACCENT = "#b576af"      # cond_beta_tcvae colour in the comparison figures
TARGET = "#2a6f97"      # the predicted quantity y = SP(k)
R = 0.40                # node radius (data units, equal aspect)


class Graph:
    """Minimal daft-style drawing on a matplotlib axis."""

    def __init__(self, ax):
        self.ax = ax
        self.nodes: dict[str, tuple[float, float, float, str]] = {}

    def node(self, key, xy, label, sub=None, kind="latent", r=R, edge=INK, fs=13,
             sub_dy=None, sub_ha="center", sub_dx=0.0):
        x, y = xy
        if kind == "fixed":   # fixed hyper-parameter / design value: small filled dot
            self.ax.add_patch(Circle(xy, 0.08, fc=INK, ec=INK, zorder=3))
            self.nodes[key] = (x, y, 0.10, kind)
            self.ax.text(x + 0.16, y, label, ha="left", va="center", fontsize=fs, zorder=4)
        elif kind == "det":   # deterministic network output: diamond
            d = r * 1.15
            self.ax.add_patch(Polygon([(x, y + d), (x + d, y), (x, y - d), (x - d, y)],
                                      closed=True, fc="white", ec=edge, lw=1.4, zorder=3))
            self.nodes[key] = (x, y, d * 0.80, kind)
            self.ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)
        else:
            fc = SHADE if kind == "obs" else "white"
            self.ax.add_patch(Circle(xy, r, fc=fc, ec=edge, lw=1.4 if edge == INK else 2.0,
                                     zorder=3))
            self.nodes[key] = (x, y, r, kind)
            self.ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)
        if sub:
            dy = -(self.nodes[key][2] + 0.12) if sub_dy is None else sub_dy
            self.ax.text(x + sub_dx, y + dy, sub, ha=sub_ha, va="top" if dy < 0 else "bottom",
                         fontsize=7.6, color=MUTED, zorder=4, linespacing=1.15)

    def edge(self, a, b, rad=0.0, ls="-", color=INK, lw=1.2, label=None, lpos=0.5,
             loff=(0.0, 0.12), lcolor=None, loc_fs=8, lfamily=None, lha="center"):
        xa, ya, ra, _ = self.nodes[a]
        xb, yb, rb, _ = self.nodes[b]
        pts = 72.0 / self.ax.figure.dpi  # shrink is given in points
        scale = self._pts_per_unit()
        arr = FancyArrowPatch((xa, ya), (xb, yb), connectionstyle=f"arc3,rad={rad}",
                              arrowstyle="-|>,head_length=6,head_width=3", lw=lw,
                              ls=ls, color=color, shrinkA=ra * scale + 1,
                              shrinkB=rb * scale + 2, zorder=2)
        del pts
        self.ax.add_patch(arr)
        if label:
            mx = xa + lpos * (xb - xa) + loff[0]
            my = ya + lpos * (yb - ya) + loff[1]
            self.ax.text(mx, my, label, fontsize=loc_fs, color=lcolor or color, ha=lha,
                         va="center", zorder=5, family=lfamily,
                         bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))

    def plate(self, x0, y0, x1, y1, label, color=MUTED, loc="br"):
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                         boxstyle="round,pad=0,rounding_size=0.12",
                                         fc="none", ec=color, lw=1.0, zorder=1))
        tx = x1 - 0.1 if loc[1] == "r" else x0 + 0.1
        ty = y0 + 0.08 if loc[0] == "b" else y1 - 0.08
        self.ax.text(tx, ty, label, ha="right" if loc[1] == "r" else "left",
                     va="bottom" if loc[0] == "b" else "top", fontsize=8.5, color=color,
                     zorder=4, bbox=dict(fc="white", ec="none", pad=0.5))

    def _pts_per_unit(self):
        fig = self.ax.figure
        fig.canvas.draw()
        bbox = self.ax.get_window_extent()
        x0, x1 = self.ax.get_xlim()
        return bbox.width / (x1 - x0) * 72.0 / fig.dpi


def physical_panel(ax):
    ax.set_xlim(0, 9.4)
    ax.set_ylim(-0.25, 6.6)
    g = Graph(ax)

    g.plate(0.10, 0.10, 9.30, 6.20, r"simulations $s = 1,\dots,1024$",
            loc="tr")
    g.plate(0.30, 0.30, 9.10, 4.40, r"snapshots: $z = 0.47,\ 0.21$", loc="tr")
    g.plate(0.50, 0.50, 6.30, 2.85, r"$n_j,\ j = 1,\dots,5$", loc="tr")

    g.node("Om", (1.3, 5.25), r"$\boldsymbol{\Omega}$", kind="obs",
           sub="cosmology",
           sub_dx=0.55, sub_ha="left", sub_dy=0.05)
    g.node("th", (4.7, 5.25), r"$\boldsymbol{\theta}$",
           sub="feedback /\nastrophysics", sub_dx=0.55, sub_ha="left", sub_dy=0.05)

    g.node("z", (1.3, 3.55), r"$z$", kind="obs", sub="redshift")
    g.node("SP", (8.05, 3.30), r"$\mathrm{SP}(k)$", edge=TARGET, fs=10.5,
           sub=r"$P_{\rm hydro}/P_{\rm DMO}$")

    g.node("n", (1.3, 1.55), r"$n$", kind="obs", sub="number density")
    g.node("H", (3.0, 1.55), r"$\mathcal{H}$", sub="halo sample")
    g.node("f", (4.7, 1.55), r"$f_{\rm gas}$", fs=11,
           sub="stacked profile")

    # Halo sample: the top-n ranked objects of the snapshot -- a selection rule with no
    # cosmology or feedback input.
    g.edge("n", "H")
    g.edge("z", "H")
    g.edge("H", "f", label=r"$\times 3$", loc_fs=10, loff=(0.0, 0.20), lcolor=INK)
    # Direct routes into f_gas: Omega_b/Omega_m and D_A(z; Omega) (apertures in arcmin),
    # theta via the gas content at fixed halo mass, z via evolution and projection.
    g.edge("Om", "f")
    g.edge("th", "f")
    g.edge("z", "f")
    for a in ("Om", "th", "z"):
        g.edge(a, "SP", color=TARGET, lw=1.3)

    ax.text(4.7, -0.02, r"$f_{\rm gas}\ \perp\!\!\!\perp\ n\ \mid\ \mathcal{H},\,z$"
            r"$\qquad \mathrm{SP}(k)\ \perp\!\!\!\perp\ n,\mathcal{H}\ \mid\ "
            r"\boldsymbol{\Omega},\boldsymbol{\theta},z$",
            ha="center", va="top", fontsize=10)
    ax.set_title("(a) data-generating model (simulations)", fontsize=11, loc="left")


def model_panel(ax, title=r"(b) conditional $\beta$-TCVAE", standalone=False):
    """Panel (b). ``standalone``: only the graph (no plate, title or loss text), with the
    c_i nodes stacked vertically to the lower left of x."""
    g = Graph(ax)
    if standalone:
        ax.set_xlim(-0.9, 8.75)
        ax.set_ylim(1.7, 6.0)
    else:
        ax.set_xlim(0, 8.75)
        ax.set_ylim(-0.25, 6.6)
        g.plate(0.15, 1.20, 8.62, 6.15,
                r"rows $i = 1,\dots,N$  ($N = 7170$ train: sim $\times\,z\,\times\,n$)")

    # Layout (Sohn et al. 2015, recurrent CGM):   y_hat --> zeta
    #                                              ^      /     \
    #                                              x  ----------> y
    #                                              c (below x)
    g.node("yh", (1.65, 5.15), r"$\hat{\mathbf{y}}_i$", kind="det", edge=ACCENT)
    g.node("zeta", (4.75, 5.15), r"$\boldsymbol{\zeta}_i$", edge=ACCENT,
           sub="latent (2-D), stands in for feedback", sub_dy=0.50)
    g.node("x", (1.65, 3.20), r"$\mathbf{x}_i$", kind="obs",
           sub=r"$f_{\rm gas}(\vartheta)$, 9 bins")
    g.node("y", (7.55, 3.20), r"$\mathbf{y}_i$", kind="latent", edge=TARGET,
           sub=r"$\mathrm{SP}(k)$, 79 bins")

    # c_i = (log10 n, z, Omega), concatenated and standardised -> one context vector that
    # enters the baseline, the prior, the decoder and the recognition network.
    rc = 0.31
    if standalone:
        # Vertical stack directly left of x. Separate exits keep every edge straight
        # and clear of x: c -> y_hat from the top of the box, c -> zeta from its upper
        # right (passing above x), c -> y from its lower right (passing below x's label).
        ax.add_patch(FancyBboxPatch((-0.10, 2.10), 0.90, 2.20,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="none", ec=INK, lw=1.0, ls=(0, (3, 2)), zorder=1))
        ax.text(-0.22, 3.20, r"$\mathbf{c}_i$", ha="right", va="center", fontsize=12)
        g.node("cOm", (0.35, 3.90), r"$\boldsymbol{\Omega}$", kind="obs", r=rc, fs=11)
        g.node("cz", (0.35, 3.20), r"$z$", kind="obs", r=rc, fs=11)
        g.node("cn", (0.35, 2.50), r"$\log_{10}n$", kind="obs", r=rc, fs=8)
        g.nodes["cL"] = (0.35, 4.30, 0.0, "anchor")   # -> y_hat
        g.nodes["cZ"] = (0.80, 4.05, 0.0, "anchor")   # -> zeta
        g.nodes["c"] = (0.80, 2.15, 0.0, "anchor")    # -> y
    else:
        ax.add_patch(FancyBboxPatch((0.55, 1.45), 2.20, 0.84,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="none", ec=INK, lw=1.0, ls=(0, (3, 2)), zorder=1))
        ax.text(2.85, 1.60, r"$\mathbf{c}_i$", ha="left", va="center", fontsize=12)
        g.node("cOm", (0.95, 1.87), r"$\boldsymbol{\Omega}$", kind="obs", r=rc, fs=11)
        g.node("cz", (1.65, 1.87), r"$z$", kind="obs", r=rc, fs=11)
        g.node("cn", (2.35, 1.87), r"$\log_{10}n$", kind="obs", r=rc, fs=8)
        g.nodes["c"] = (2.75, 1.87, 0.0, "anchor")   # edges leave from the c_i box
        g.nodes["cL"] = (0.55, 1.87, 0.0, "anchor")  # left side, routes c -> y_hat around x
        g.nodes["cZ"] = g.nodes["c"]

    # Baseline y_hat = f_theta(x, c) (source of the recurrent connection).
    g.edge("x", "yh")
    g.edge("cL", "yh")
    # Prior network p_theta(zeta | x, y_hat.detach(), c).
    g.edge("yh", "zeta", color=ACCENT, label=".detach()", lpos=0.5, loff=(0.0, 0.20),
           lfamily="monospace")
    g.edge("x", "zeta", color=ACCENT,
           label="prior network\n" r"$p_\theta(\boldsymbol{\zeta}\,|\,\mathbf{x},\hat{\mathbf{y}},\mathbf{c})$",
           lpos=0.40, loff=(0.50, -0.40) if standalone else (-0.05, 0.45), lcolor=ACCENT)
    g.edge("cZ", "zeta", color=ACCENT)
    # Generation network p_theta(y | x, zeta, c).
    g.edge("zeta", "y", lw=1.4)
    g.edge("x", "y", lw=1.4,
           label="generation network  "
                 r"$p_\theta(\mathbf{y}\,|\,\mathbf{x},\boldsymbol{\zeta},\mathbf{c})$",
           lpos=0.655 if standalone else 0.62, loff=(0.0, 0.24), lcolor=INK)
    g.edge("c", "y")
    # Recognition network q_phi(zeta | x, c, y) (training only; it also reads x and c).
    xz, yz, rz, _ = g.nodes["zeta"]
    xy_, yy_, ry, _ = g.nodes["y"]
    dx, dy = xy_ - xz, yy_ - yz
    norm = (dx * dx + dy * dy) ** 0.5
    off = 0.22                                   # perpendicular offset (data units)
    ox, oy = -dy / norm * off, dx / norm * off   # to the upper right of zeta -> y
    ox, oy = (ox, oy) if oy > 0 else (-ox, -oy)
    g.nodes["zeta_r"] = (xz + ox, yz + oy, (rz * rz - off * off) ** 0.5, "anchor")
    g.nodes["y_r"] = (xy_ + ox, yy_ + oy, (ry * ry - off * off) ** 0.5, "anchor")
    g.edge("y_r", "zeta_r", ls=(0, (4, 3)), color=MUTED,
           label="recognition network\n"
                 r"$q_\phi(\boldsymbol{\zeta}\,|\,\mathbf{x},\mathbf{c},\mathbf{y})$",
           lpos=0.5, loff=(0.80, 0.42), lcolor=MUTED)

    if standalone:
        return

    # Loss as implemented (CondBetaTcVae._objective). The KL split is taken in the
    # prior-whitened u = (zeta - mu_theta)/sigma_theta, where the prior is N(0, I) and the
    # Chen et al. identity holds exactly; i indexes the training row (MSS, M = 128).
    # The baseline's own regression NLL is part of the implemented objective.
    eqs = (r"$\bf{Loss}$:  $\mathcal{L}=\mathcal{L}_{\rm rec}"
           r"+\lambda\,\mathrm{KL}(q(\boldsymbol{\zeta})\,\|\,p(\boldsymbol{\zeta}))"
           r"+\mathcal{L}_{\rm base}$"
           "\n"
           r"  Reconstruction:  $\mathcal{L}_{\rm rec}=-\mathbb{E}_{q_\phi(\boldsymbol{\zeta}|\mathbf{x},\mathbf{c},\mathbf{y})}"
           r"\log p_\theta(\mathbf{y}\,|\,\mathbf{x},\boldsymbol{\zeta},\mathbf{c})$"
           "\n"
           r"  KL penalty:  $\mathrm{KL}(q(\boldsymbol{\zeta})\,\|\,p(\boldsymbol{\zeta}))"
           r"=\mathrm{KL}(q_\phi(\boldsymbol{\zeta}|\mathbf{x},\mathbf{c},\mathbf{y})\,\|\,"
           r"p_\theta(\boldsymbol{\zeta}|\mathbf{x},\hat{\mathbf{y}},\mathbf{c}))"
           r"=\alpha\,\mathrm{MI}+\beta\,\mathrm{TC}+\gamma\,\mathrm{dwKL}$, in "
           r"$\mathbf{u}=(\boldsymbol{\zeta}-\boldsymbol{\mu}_\theta)/\boldsymbol{\sigma}_\theta$"
           "\n"
           r"     $\mathrm{MI}=\mathbb{E}_{q(\mathbf{u},i)}[\log q(\mathbf{u}|i)-\log q(\mathbf{u})]$,"
           r"  $\mathrm{TC}=\mathrm{KL}(q(\mathbf{u})\,\|\,\Pi_j\, q(u_j))$,"
           r"  $\mathrm{dwKL}=\Sigma_j\,\mathrm{KL}(q(u_j)\,\|\,\mathcal{N}(0,1))$"
           "\n"
           r"  Baseline:  $\mathcal{L}_{\rm base}=-\log\mathcal{N}(\mathbf{y};\hat{\mathbf{y}},\sigma_k^2)$")
    ax.text(0.2, 1.02, eqs, ha="left", va="top", fontsize=8.6, linespacing=1.7)
    ax.set_title(title, fontsize=11, loc="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "figures/2026-11/11-18/graphical_model"
                                         / "cond_beta_tcvae_graphical_model"))
    args = ap.parse_args()

    plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})
    fig = plt.figure(figsize=(14.2, 6.3), dpi=150)
    ax_a = fig.add_axes([0.01, 0.07, 0.51, 0.90])
    ax_b = fig.add_axes([0.53, 0.07, 0.465, 0.90])
    for ax in (ax_a, ax_b):
        ax.set_aspect("equal")
        ax.set_axis_off()
    physical_panel(ax_a)
    model_panel(ax_b)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    print(out.with_suffix(".png"))

    # Panel (b) alone, at the same physical scale (axes 0.465 x 14.2 in wide) so node
    # sizes, fonts and arrow shrink match the two-panel figure.
    fig_b = plt.figure(figsize=(6.9, 6.3), dpi=150)
    ax = fig_b.add_axes([0.02, 0.07, 0.465 * 14.2 / 6.9, 0.90])
    ax.set_aspect("equal")
    ax.set_axis_off()
    model_panel(ax, standalone=True)
    out_b = out.with_name(out.name + "_panel_b")
    for ext in ("png", "pdf"):
        fig_b.savefig(out_b.with_suffix(f".{ext}"), bbox_inches="tight")
    print(out_b.with_suffix(".png"))


if __name__ == "__main__":
    main()
