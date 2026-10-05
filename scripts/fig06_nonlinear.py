"""Figure 6: nonlinear unconfined aquifer with matrix memory.

Boussinesq aquifer forced at the reference M2 frequency with tidal
amplitude eps = A/D.  (a) Water-table envelope (max and min, thin) and
mean (thick) for the memory aquifer (beta = 20, vermilion) and the
classical aquifer (beta = 0, blue), eps = 0.2, against x/x_*.  (b) Mean
rise normalized by eps^2/4 against 2ax, a the linear attenuation rate of
each aquifer, for eps = 0.1, 0.2, 0.3 (memory) and 0.2 (classical);
dashed: the second-order law 1 - exp(-2ax), common to both.  (c) Departure
from that law divided by eps^2; its far-field value -1/8 (dashed) is the
exact expansion of sqrt(1 + eps^2/2).  eps = 0.1 is omitted in (c)
because that term falls to the discretization level.  (d) Departure of
the period mean of H^2 from the exact invariant 1 + eps^2/2, maximum over
x, against the number of periods integrated.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from pasang.boussinesq import Boussinesq1D
from pasang.capacity import capacity
from pasang.io_utils import load_cache, save_cache, write_csv
from pasang.plotting import (SEQ, below_legend, handles,
                             panel_label, save, setup)
from pasang.propagation import rates
from pasang.scenario import omega_of

setup()
om = omega_of("M2")
P = 2 * np.pi / om
spp = 80
nper = 120
runs = [("mem", 20.0, 0.2), ("cls", 0.0, 0.2), ("mem", 20.0, 0.1),
        ("mem", 20.0, 0.3)]
out = {}
for tag, beta, eps in runs:
    key = f"fig06_{tag}_{eps:g}"
    C = load_cache(key)
    a, b = rates(om, capacity(om, beta))
    if C is None:
        L = 7.0 / a
        m = Boussinesq1D(L, 140, 12, beta, eps)
        t, H, U = m.integrate(om, P / spp, nper * spp,
                              u0=m.initial_state(om, a))
        Hp = H.reshape(nper, spp, -1)
        inv = np.max(np.abs(np.mean(Hp**2, axis=1) - (1 + eps**2 / 2)),
                     axis=1)
        C = dict(x=m.x, a=np.array([float(a)]), last=Hp[-1],
                 inv=inv, newton=np.array([m.newton_max]),
                 mlast=U[-spp:, m.nx:] if beta > 0 else np.zeros(1),
                 nz=np.array([m.nz]))
        save_cache(key, **C)
    out[(tag, eps)] = C

fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.4))
axs = axs.ravel()
ax = axs[0]
for (tag, eps), c in ((("mem", 0.2), SEQ[4]), (("cls", 0.2), SEQ[1])):
    C = out[(tag, eps)]
    s = C["x"]
    ax.plot(s, C["last"].max(0) - 1, color=c, lw=0.8)
    ax.plot(s, C["last"].min(0) - 1, color=c, lw=0.8)
    ax.plot(s, C["last"].mean(0) - 1, color=c, lw=1.6)
ax.set_xlim(0, 0.78)
ax.set_xlabel(r"$x/x_*$")
ax.set_ylabel(r"$H - 1$")

ax = axs[1]
s = np.linspace(0, 8, 300)
ax.plot(s, 1 - np.exp(-s), color="black", ls="--", lw=0.9)
cols = {}
for (tag, eps), c, ls in ((("mem", 0.1), SEQ[3], "-"),
                          (("mem", 0.2), SEQ[4], "-"),
                          (("mem", 0.3), SEQ[5], "-"),
                          (("cls", 0.2), SEQ[1], "-.")):
    C = out[(tag, eps)]
    sx = 2 * C["a"][0] * C["x"]
    y = (C["last"].mean(0) - 1) / (eps**2 / 4)
    ax.plot(sx, y, color=c, ls=ls)
    cols[f"s_{tag}_{eps:g}"] = sx
    cols[f"rise_{tag}_{eps:g}"] = y
ax.set_xlim(0, 8)
ax.set_ylim(0, 1.1)
ax.set_xlabel(r"$2ax$")
ax.set_ylabel(r"$(\langle H\rangle - 1)/(\varepsilon^2/4)$")

ax = axs[2]
for (tag, eps), c, ls in ((("mem", 0.2), SEQ[4], "-"),
                          (("mem", 0.3), SEQ[5], "-"),
                          (("cls", 0.2), SEQ[1], "-.")):
    sx = cols[f"s_{tag}_{eps:g}"]
    dev = (cols[f"rise_{tag}_{eps:g}"] - (1 - np.exp(-sx))) / eps**2
    ax.plot(sx, dev, color=c, ls=ls)
    cols[f"dev_{tag}_{eps:g}"] = dev
ax.axhline(0, color="#777777", lw=0.6)
ax.axhline(-0.125, color="black", ls="--", lw=0.9)
ax.set_xlim(0, 8)
ax.set_xlabel(r"$2ax$")
ax.set_ylabel(r"[rise $-(1-e^{-2ax})$]$/\varepsilon^2$")

ax = axs[3]
for (tag, eps), c, ls in ((("mem", 0.2), SEQ[4], "-"),
                          (("cls", 0.2), SEQ[1], "-.")):
    inv = out[(tag, eps)]["inv"]
    ax.semilogy(np.arange(1, len(inv) + 1), inv, color=c, ls=ls)
ax.set_xlabel("periods integrated")
ax.set_ylabel(r"$\max_x|\langle H^2\rangle - 1 - \varepsilon^2/2|$")
ax.set_xlim(1, nper)
for ax, lab in zip(axs, "abcd"):
    panel_label(ax, f"({lab})")
eq = r"$\varepsilon$"
below_legend(axs[0], *handles(
    [r"memory, $\beta=20$", "classical", "period mean", "envelope"],
    [SEQ[4], SEQ[1], "#444444", "#444444"], widths=[1.3, 1.3, 1.6, 0.8]),
    ncol=2, dy=-0.22)
below_legend(axs[1], *handles(
    [f"memory, {eq} = 0.1", f"memory, {eq} = 0.2", f"memory, {eq} = 0.3",
     f"classical, {eq} = 0.2", r"$1-e^{-2ax}$"],
    [SEQ[3], SEQ[4], SEQ[5], SEQ[1], "black"],
    ["-", "-", "-", "-.", "--"]), ncol=2, dy=-0.22)
below_legend(axs[2], *handles(
    [f"memory, {eq} = 0.2", f"memory, {eq} = 0.3",
     f"classical, {eq} = 0.2", "exact far field, $-1/8$"],
    [SEQ[4], SEQ[5], SEQ[1], "black"], ["-", "-", "-.", "--"]),
    ncol=2, dy=-0.22)
below_legend(axs[3], *handles(
    [f"memory, {eq} = 0.2", f"classical, {eq} = 0.2"],
    [SEQ[4], SEQ[1]], ["-", "-."]), ncol=2, dy=-0.22)
fig.tight_layout(w_pad=1.4, h_pad=1.2)
write_csv("fig06b_overheight_mem02",
          {"two_ax": cols["s_mem_0.2"], "rise": cols["rise_mem_0.2"]})
write_csv("fig06b_overheight_cls02",
          {"two_ax": cols["s_cls_0.2"], "rise": cols["rise_cls_0.2"]})
summary = {}
for (tag, eps), C in out.items():
    far = C["last"].mean(0)[-1]
    summary[f"{tag}_{eps:g}"] = (
        float(C["inv"][-1]), float(far - np.sqrt(1 + eps**2 / 2)),
        float(np.max(np.abs((C["last"].mean(0) - 1) / (eps**2 / 4)
                            - (1 - np.exp(-2 * C["a"][0] * C["x"]))))),
        int(C["newton"][0]))
save_cache("fig06", keys=np.array(list(summary)),
           vals=np.array(list(summary.values())))
save(fig, "fig06_nonlinear")
print("figure 6 written")
for k, v in summary.items():
    print(f"  {k}: invariant {v[0]:.2e}, far-field vs sqrt {v[1]:.2e},"
          f" rise vs 2nd order {v[2]:.3e}, newton {v[3]}")
