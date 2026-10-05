"""Figure 1: matrix transfer functions G(Omega).

(a) Re G, the in-phase (storage) part; (b) -Im G, the quadrature (loss)
part; (c) the same curves in the Cole-Cole plane.  Curves: slab,
cylinder and sphere blocks, first-order (Warren-Root) exchange, and
slabs with power-law size distributions q = 0.5 and 0.8.  Dashed: the
half-order asymptote (i Omega)^(-1/2) of a slab.  Dotted verticals: the
tidal constituents of the reference aquifer.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from pasang.capacity import (g_cylinder, g_powerlaw_slabs, g_slab,
                             g_sphere, g_warren_root)
from pasang.io_utils import write_csv
from pasang.plotting import (SEQ, handles, mark_constituents, panel_label,
                             save, setup)
from pasang.scenario import CONSTITUENTS, omega_of

setup()
om = np.logspace(-3, 4, 700)
curves = [
    ("slab", g_slab(om), SEQ[0], "-"),
    ("cylinder", g_cylinder(om), SEQ[1], "-"),
    ("sphere", g_sphere(om), SEQ[2], "-"),
    ("first order", g_warren_root(om), SEQ[3], "--"),
    ("power law, $q=0.5$", g_powerlaw_slabs(om, 0.5), SEQ[4], "-."),
    ("power law, $q=0.8$", g_powerlaw_slabs(om, 0.8), SEQ[5], "-."),
]
cons = [omega_of(n) for n in CONSTITUENTS]
asym = (1j * om) ** -0.5

fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.3),
                        gridspec_kw={"width_ratios": [1, 1, 1, 0.62]})
leg_ax = axs[3]
leg_ax.axis("off")
axs = axs[:3]
for lab, G, c, ls in curves:
    axs[0].loglog(om, G.real, color=c, ls=ls)
    axs[1].loglog(om, -G.imag, color=c, ls=ls)
    axs[2].plot(G.real, -G.imag, color=c, ls=ls)
hi = om > 3
axs[0].loglog(om[hi], asym[hi].real, color="#777777", ls="--", lw=0.8)
axs[1].loglog(om[hi], -asym[hi].imag, color="#777777", ls="--", lw=0.8)
for ax in axs[:2]:
    mark_constituents(ax, cons)
    ax.set_xlabel(r"$\Omega = \omega\tau$")
    ax.set_xlim(om[0], om[-1])
axs[0].set_ylim(1e-2, 1.5)
axs[1].set_ylim(1e-3, 1.0)
axs[0].set_ylabel(r"$\mathrm{Re}\,G$")
axs[1].set_ylabel(r"$-\mathrm{Im}\,G$")
axs[2].set_xlabel(r"$\mathrm{Re}\,G$")
axs[2].set_ylabel(r"$-\mathrm{Im}\,G$")
axs[2].set_xlim(0, 1.02)
axs[2].set_ylim(0, 0.52)
for ax, lab in zip(axs, "abc"):
    panel_label(ax, f"({lab})")
hs, ls_ = handles([c[0] for c in curves] + [r"$(i\Omega)^{-1/2}$"],
                  [c[2] for c in curves] + ["#777777"],
                  [c[3] for c in curves] + ["--"])
leg_ax.legend(hs, ls_, loc="center left", frameon=False, fontsize=7,
              handlelength=1.8, labelspacing=0.5, borderaxespad=0.0)
fig.tight_layout(w_pad=1.2)
keys = ["slab", "cylinder", "sphere", "first_order", "powerlaw_q05",
        "powerlaw_q08"]
write_csv("fig01_transfer", {"omega": om,
                             **{k: c[1] for k, c in zip(keys, curves)}})
save(fig, "fig01_capacity")
print("figure 1 written")
