"""Figure 2: the ratio spectrum r(Omega) = b/a.

(a) Single slab, capacity ratios beta = 1, 10, 100, 1000.  Dashed:
tan(pi/8), the half-order plateau.  Dotted: the exact floor for any
mixture of slabs, tan(pi/8 + delta_slab/2).  (b) Mechanisms that can
lower r: reference slab memory (beta = 20), power-law slabs (q = 0.5,
beta = 20), a Weyl time-fractional medium (gamma = 0.75, frequency
independent), and leakage with lambda = 0.3, 3, 30.  Dotted verticals:
constituents of the reference aquifer.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from pasang.bounds import ratio_floor, slab_delta
from pasang.capacity import (capacity, capacity_caputo, capacity_leaky)
from pasang.io_utils import save_cache, write_csv
from pasang.plotting import (SEQ, below_legend, handles, mark_constituents,
                             panel_label, save, setup)
from pasang.propagation import ratio
from pasang.scenario import CONSTITUENTS, REF, omega_of

setup()
om = np.logspace(-3, 7, 1200)
cons = [omega_of(n) for n in CONSTITUENTS]
d_slab, _ = slab_delta()
floor = ratio_floor(d_slab)
plateau = np.tan(np.pi / 8)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3))
cols = {"omega": om}
betas = [1, 10, 100, 1000]
BCOL = [plt.get_cmap("viridis")(v) for v in (0.0, 0.33, 0.62, 0.88)]
for b, c in zip(betas, BCOL):
    r = ratio(capacity(om, b))
    axs[0].semilogx(om, r, color=c)
    cols[f"r_slab_beta{b}"] = r
for ax in axs:
    ax.axhline(plateau, color="#777777", ls="--", lw=0.8)
    ax.axhline(floor, color="#777777", ls=":", lw=0.8)
    ax.axhline(1.0, color="#bbbbbb", lw=0.6)
axs[0].set_xlim(1e-2, 1e7)

om2 = np.logspace(-3, 4, 900)
mech = [
    ("slab, $\\beta=20$", ratio(capacity(om2, REF.beta)), SEQ[0], "-"),
    ("power law $q=0.5$, $\\beta=20$",
     ratio(capacity(om2, REF.beta, "powerlaw", q=0.5)), SEQ[4], "-."),
    ("Weyl fractional, $\\gamma=0.75$",
     ratio(capacity_caputo(om2, 0.75)), SEQ[5], "--"),
]
for lam, c in zip([0.3, 3.0, 30.0], [SEQ[1], SEQ[2], SEQ[3]]):
    mech.append((f"leakage, $\\lambda={lam:g}$",
                 ratio(capacity_leaky(om2, lam)), c, ":"))
for lab, r, c, ls in mech:
    axs[1].semilogx(om2, r, color=c, ls=ls)
mark_constituents(axs[1], cons)
axs[1].set_xlim(om2[0], om2[-1])
for ax, lab in zip(axs, "ab"):
    ax.set_xlabel(r"$\Omega = \omega\tau$")
    ax.set_ylabel(r"$r = b/a$")
    ax.set_ylim(0, 1.05)
    panel_label(ax, f"({lab})")
h1, l1 = handles([f"$\\beta={b}$" for b in betas], BCOL)
h2, l2 = handles([m[0] for m in mech], [m[2] for m in mech],
                 [m[3] for m in mech])
h3, l3 = handles([r"$\tan(\pi/8)$", "slab floor"], ["#777777"] * 2,
                 ["--", ":"])
below_legend(axs[0], h1 + h3, l1 + l3, ncol=3)
below_legend(axs[1], h2, l2, ncol=2)
fig.tight_layout(w_pad=2.0)
write_csv("fig02a_ratio_slab", cols)
write_csv("fig02b_ratio_mechanisms", {"omega": om2, **{
    f"r{i}": m[1] for i, m in enumerate(mech)}})
rows = {n: float(ratio(capacity(omega_of(n), REF.beta)))
        for n in CONSTITUENTS}
save_cache("fig02", names=np.array(list(rows)),
           r_ref=np.array(list(rows.values())),
           omegas=np.array([omega_of(n) for n in rows]))
save(fig, "fig02_ratio")
print("figure 2 written; reference r:", {k: round(v, 4)
                                         for k, v in rows.items()})
