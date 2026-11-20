"""Shared publication plotting convention (serif/mathtext, twilight_r colour cycle).

Used by the runner's publication figures (:mod:`fgas_spk.train`) and by the
publication scripts under ``scripts/``. Importing this module imports
matplotlib, so import it lazily from code that must not require matplotlib.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Effective redshift of each snapshot, and the spectroscopic tracer it represents.
REDSHIFT = {74: 0.47, 82: 0.21}
TRACER = {74: "DESI LRG", 82: "DESI BGS"}

COLOURMAP = matplotlib.cm.twilight_r
PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": False,   # latex not available on compute nodes; use matplotlib mathtext
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    # Wider than the notebook's (0.2, 0.85) so the lines used here separate more;
    # twilight_r is cyclic and washes out to near-white beyond ~0.1 / ~0.92.
    "axes.prop_cycle": plt.cycler("color", COLOURMAP(np.linspace(0.25, 0.98, 5))),
}
