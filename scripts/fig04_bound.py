"""Figure 4: the half-order bound and the fractional plateau.

(a) Phase excess arg G + pi/4 for slab, cylinder and sphere blocks.
Only the slab goes negative, on pi/2 < u < pi, u = sqrt(Omega/2).
(b) Minimum over frequency of r for c = 1 + beta G, versus beta.
Dashed: tan(pi/8); dotted: the slab floor tan(pi/8 + delta/2).
(c) Local fractional order gamma_eff = 4 theta/pi for power-law slab
distributions, beta = 1e6.  Dashed: the predicted plateau 1 - q/2 for
q < 1 and 1/2 for q > 1.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from pasang.bounds import (geometry_delta, min_ratio, ratio_floor,
                           slab_delta, slab_phase_excess)
from pasang.capacity import GEOMETRIES, capacity
from pasang.io_utils import save_cache, write_csv
from pasang.plotting import (SEQ, below_legend, handles, panel_label,
                             save, setup)
from pasang.propagation import local_order

setup()
fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.0))

# (a) phase excess versus u
u = np.linspace(0.05, 6.0, 2000)
om_u = 2.0 * u**2
cols_a = {"u": u}
geos = ["slab", "cylinder", "sphere"]
for g, c in zip(geos, SEQ):
    ex = np.angle(GEOMETRIES[g](om_u)) + np.pi / 4
    axs[0].plot(u, ex, color=c)
    cols_a[g] = ex
assert np.allclose(cols_a["slab"], slab_phase_excess(u), atol=1e-12)
axs[0].axhline(0.0, color="#777777", lw=0.7)
axs[0].set_xlabel(r"$u = \sqrt{\Omega/2}$")
axs[0].set_ylabel(r"$\arg G + \pi/4$  (rad)")
axs[0].set_xlim(0, 6)
axs[0].set_ylim(-0.05, 0.8)

# (b) minimum ratio versus beta
betas = np.logspace(-1, 5, 61)
d_slab, u_star = slab_delta()
cols_b = {"beta": betas}
for g, c in zip(geos, SEQ):
    rm = np.array([min_ratio(b, g)[0] for b in betas])
    axs[1].semilogx(betas, rm, color=c)
    cols_b[g] = rm
axs[1].axhline(np.tan(np.pi / 8), color="#777777", ls="--", lw=0.8)
axs[1].axhline(ratio_floor(d_slab), color="#777777", ls=":", lw=0.8)
axs[1].set_xlabel(r"$\beta = S_{im}/S_m$")
axs[1].set_ylabel(r"$\min_\Omega\, r$")
axs[1].set_ylim(0.35, 1.0)
axs[1].set_xlim(betas[0], betas[-1])

# (c) local order for power-law distributions
om = np.logspace(-2, 9, 450)
qs = [0.2, 0.4, 0.6, 0.8, 1.5]
qcol = [plt.get_cmap("plasma")(v) for v in np.linspace(0.0, 0.85, len(qs))]
cols_c = {"omega": om}
for q, c in zip(qs, qcol):
    ge = local_order(capacity(om, 1e6, "powerlaw", q=q))
    axs[2].semilogx(om, ge, color=c)
    axs[2].axhline(1 - q / 2 if q < 1 else 0.5, color=c, ls="--", lw=0.7)
    cols_c[f"q{q:g}"] = ge
axs[2].set_xlabel(r"$\Omega = \omega\tau$")
axs[2].set_ylabel(r"$\gamma_{\mathrm{eff}} = 4\theta/\pi$")
axs[2].set_ylim(0.4, 1.02)
axs[2].set_xlim(om[0], om[-1])
for ax, lab in zip(axs, "abc"):
    panel_label(ax, f"({lab})")
below_legend(axs[0], *handles(geos, SEQ[:3]), ncol=1, dy=-0.33)
below_legend(axs[1], *handles([r"$\tan(\pi/8)$", "slab floor"],
                              ["#777777"] * 2, ["--", ":"]), ncol=1)
below_legend(axs[2], *handles([rf"$q={q:g}$" for q in qs], qcol), ncol=3)
fig.tight_layout(w_pad=1.0)
write_csv("fig04a_phase_excess", cols_a)
write_csv("fig04b_min_ratio", cols_b)
write_csv("fig04c_local_order", cols_c)
dc, oc = geometry_delta("cylinder")
ds, os_ = geometry_delta("sphere")
save_cache("fig04", d_slab=np.array([d_slab]), u_star=np.array([u_star]),
           d_cyl=np.array([dc]), d_sph=np.array([ds]),
           floor=np.array([ratio_floor(d_slab)]),
           rmin_slab_big=np.array([cols_b["slab"][-1]]),
           rmin_cyl_big=np.array([cols_b["cylinder"][-1]]),
           rmin_sph_big=np.array([cols_b["sphere"][-1]]))
save(fig, "fig04_bound")
print(f"slab delta = {d_slab:.10f} at u = {u_star:.10f};"
      f" floor = {ratio_floor(d_slab):.10f}")
