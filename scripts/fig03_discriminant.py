"""Figure 3: what a constituent measures, and how mechanisms differ.

Each constituent gives a and b, equivalent to the complex capacity:
-Im c = (a^2 - b^2)/Omega and Re c = 2ab/Omega (units of S_m when x is
scaled by x_*).  (a) Loss part -Im c: leakage is a zero-frequency pole
lambda/Omega; storage memory is a relaxation peak.  (b) Storage part
Re c: unity for leakage; a step from 1 + beta to 1 for memory.
(c) Apparent diffusivities from the classical formulas,
D_a = Omega/(2a^2) (solid) and D_b = Omega/(2b^2) (dashed), in units of
T/S_m.  Curves: slab memory beta = 20, power-law slabs q = 0.5 with
beta = 20, leakage lambda = 3.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from pasang.capacity import capacity, capacity_leaky
from pasang.io_utils import write_csv
from pasang.plotting import (SEQ, handles, mark_constituents, panel_label,
                             save, setup)
from pasang.propagation import apparent_diffusivities, kappa_squared_parts
from pasang.scenario import CONSTITUENTS, REF, omega_of

setup()
om = np.logspace(-3, 4, 900)
cons = [omega_of(n) for n in CONSTITUENTS]
models = [
    ("slab memory, $\\beta=20$", capacity(om, REF.beta), SEQ[0], "-"),
    ("power-law memory, $q=0.5$",
     capacity(om, REF.beta, "powerlaw", q=0.5), SEQ[4], "-."),
    ("leakage, $\\lambda=3$", capacity_leaky(om, 3.0), SEQ[2], ":"),
]
fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.35),
                        gridspec_kw={"width_ratios": [1, 1, 1, 0.62]})
leg_ax = axs[3]
leg_ax.axis("off")
axs = axs[:3]
cols = {"omega": om}
for i, (lab, c, col, ls) in enumerate(models):
    d, s = kappa_squared_parts(om, c)
    axs[0].loglog(om, d / om, color=col, ls=ls)
    axs[1].loglog(om, s / om, color=col, ls=ls)
    Da, Db = apparent_diffusivities(om, c)
    axs[2].loglog(om, Da, color=col, ls="-", lw=1.1)
    axs[2].loglog(om, Db, color=col, ls="--", lw=1.1)
    cols[f"loss_{i}"] = d / om
    cols[f"storage_{i}"] = s / om
    cols[f"Da_{i}"] = Da
    cols[f"Db_{i}"] = Db
for ax, lab in zip(axs, "abc"):
    mark_constituents(ax, cons)
    ax.set_xlim(om[0], om[-1])
    ax.set_xlabel(r"$\Omega = \omega\tau$")
    panel_label(ax, f"({lab})")
axs[0].set_ylim(1e-3, 1e4)
axs[0].set_ylabel(r"$-\mathrm{Im}\,c = (a^2-b^2)/\Omega$")
axs[1].set_ylim(0.5, 40)
axs[1].set_ylabel(r"$\mathrm{Re}\,c = 2ab/\Omega$")
axs[2].set_ylim(1e-3, 3)
axs[2].set_ylabel(r"$D_a,\ D_b$  ($T/S_m$)")
extra = [r"$D_a$ in (c)", r"$D_b$ in (c)"]
hs, ls_ = handles([m[0] for m in models] + extra,
                  [m[2] for m in models] + ["#555555"] * 2,
                  [m[3] for m in models] + ["-", "--"])
leg_ax.legend(hs, ls_, loc="center left", frameon=False, fontsize=7,
              handlelength=1.8, labelspacing=0.5, borderaxespad=0.0)
fig.tight_layout(w_pad=1.0)
write_csv("fig03_discriminant", cols)
save(fig, "fig03_discriminant")
print("figure 3 written")
