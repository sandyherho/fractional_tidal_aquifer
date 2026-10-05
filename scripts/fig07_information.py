"""Figure 7: telling storage memory from leakage in a tidal record.

Synthetic truth: slab memory. Observations: log-amplitude ratio and
phase lag per constituent at a well placed where the M2 amplitude has
fallen to exp(-1.5) of its coastal value, noise per constituent as in
:mod:`pasang.information` (10 mm in the tidal bands, 50 mm for periods
above two days, 10-min sampling). A single constituent is always fitted
exactly by a leaky aquifer (two data, two parameters), so detection
needs at least two constituents.

(a) Noncentrality lambda of the leaky-model misfit for M2, S2, K1 and O1
from a 30-day record, over the M2 frequency Omega_M2 = omega tau and the
capacity ratio beta. Contours: power 0.5 and 0.95 of the size-0.05 test
(6 degrees of freedom). Markers: the reference aquifer (circle) and a
weak-memory case (square). (b) lambda against record length for the (b)
Record length needed for power 0.95 against beta at the reference tau,
for the three multi-constituent sets, from noise alone (lambda grows
linearly with the number of samples); dotted horizontals: the Rayleigh
length that each set needs regardless of noise. (c) Cramer-Rao relative
standard deviations of X = x/x_*, beta and tau from M2, S2, K1, O1 and
30 days, against beta. Dotted verticals: reference and weak cases.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2

from pasang.capacity import capacity
from pasang.information import crb, detect_power, fisher, fit_leaky, noise
from pasang.io_utils import load_cache, save_cache, write_csv
from pasang.plotting import (SEQ, below_legend, handles, panel_label,
                             save, setup)
from pasang.propagation import rates
from pasang.scenario import CONSTITUENTS, REF, SETS, omega_of, sigma_of

setup()
TIDAL = ["M2", "S2", "K1", "O1"]
base = {n: omega_of(n) / omega_of("M2") for n in CONSTITUENTS}
amp = {n: CONSTITUENTS[n][1] for n in CONSTITUENTS}
sig = {n: sigma_of(n) for n in CONSTITUENTS}
day = 86400.0


def obs(names, om_m2, beta):
    """Observations and noise for a set at a well with a_M2 x = 1.5."""
    oms = np.array([base[n] * om_m2 for n in names])
    a1, _ = rates(om_m2, capacity(om_m2, beta))
    X = 1.5 / float(a1)
    a, b = rates(oms, capacity(oms, beta))
    return oms, a * X, b * X, X


def lam_of(names, om_m2, beta, days):
    """Noncentrality of the leaky fit for one configuration."""
    oms, aX, bX, _ = obs(names, om_m2, beta)
    n = days * day / REF.dt_log
    s = noise(aX, [amp[k] for k in names], n, [sig[k] for k in names])
    return fit_leaky(oms, aX, bX, s)[0]


def lam_at_power(p, dof):
    """Noncentrality giving power p at size 0.05."""
    crit = chi2.ppf(0.95, dof)
    return brentq(lambda L: ncx2.sf(crit, dof, L) - p, 1e-6, 1e5)


om_ref = omega_of("M2")
weak = (om_ref, 0.2)                     # same tau, much less matrix
lev = [lam_at_power(0.5, 6), lam_at_power(0.95, 6)]
C = load_cache("fig07")
if C is None:
    oms_grid = np.logspace(-2, 4, 49)
    betas = np.logspace(-1, 4, 41)
    lam = np.array([[lam_of(TIDAL, o, b, 30.0) for o in oms_grid]
                    for b in betas])
    # lambda is proportional to the number of samples (uniform rescaling
    # of the weights leaves the minimizer unchanged), so the record length
    # for power 0.95 is 30 d * lambda_95 / lambda(30 d), floored by the
    # Rayleigh length of the set
    bline = np.logspace(-3, 3, 49)
    tmin = {}
    for label, names, tr in SETS[1:]:
        dof = 2 * len(names) - 2
        l95 = lam_at_power(0.95, dof)
        tmin[label] = np.array([30.0 * l95
                                / lam_of(names, om_ref, b, 30.0)
                                for b in bline])
    names = SETS[1][1]
    rows = []
    for b in bline:
        oms, aX, bX, X = obs(names, om_ref, b)
        F = fisher(oms, [amp[k] for k in names], X, b,
                   30.0 * day / REF.dt_log, np.array([sig[k]
                                                      for k in names]))
        rows.append(crb(F))
    C = dict(oms_grid=oms_grid, betas=betas, lam=lam, bline=bline,
             crb30=np.array(rows),
             **{f"tmin|{k}": v for k, v in tmin.items()})
    save_cache("fig07", **C)

oms_grid, betas, lam, bline = (C["oms_grid"], C["betas"], C["lam"],
                               C["bline"])
fig = plt.figure(figsize=(7.0, 3.7))
ax0 = fig.add_axes([0.07, 0.44, 0.24, 0.49])
cax = fig.add_axes([0.32, 0.44, 0.012, 0.49])
ax1 = fig.add_axes([0.49, 0.44, 0.19, 0.49])
ax2 = fig.add_axes([0.79, 0.44, 0.19, 0.49])
lamc = np.clip(lam, 1e-3, 1e6)
pc = ax0.pcolormesh(oms_grid, betas, lamc, norm=LogNorm(1e-2, 1e6),
                    cmap="viridis", shading="auto", rasterized=True)
cb = fig.colorbar(pc, cax=cax)
cb.set_label(r"$\lambda$ (30 days)")
ax0.contour(oms_grid, betas, lam, levels=lev, colors=["white", "white"],
            linestyles=["--", "-"], linewidths=0.9)
ax0.plot(om_ref, REF.beta, "o", mfc="none", mec="white", ms=6, mew=1.2)
ax0.plot(*weak, "s", mfc="none", mec="white", ms=6, mew=1.2)
ax0.set_xscale("log")
ax0.set_yscale("log")
ax0.set_xlabel(r"$\Omega_{M2} = \omega_{M2}\tau$")
ax0.set_ylabel(r"$\beta$")

labels = [s[0] for s in SETS[1:]]
cols = {"beta": bline}
for lab, c in zip(labels, SEQ[1:]):
    y = C[f"tmin|{lab}"]
    ax1.loglog(bline, y, color=c)
    ax1.axhline(dict((q[0], q[2]) for q in SETS)[lab], color=c, ls=":",
                lw=0.8)
    cols[f"tmin_{lab.replace(' ', '_').replace('+', 'p')}"] = y
ax1.axvline(REF.beta, color="#777777", lw=0.7, ls="-.")
ax1.axvline(weak[1], color="#777777", lw=0.7, ls="-.")
ax1.set_xlabel(r"$\beta$")
ax1.set_ylabel("record for power 0.95 (days)")
ax1.set_ylim(1e-2, 1e4)
ax1.set_xlim(bline[0], bline[-1])

v = C["crb30"]
ax2.loglog(bline, v[:, 0], color=SEQ[0], ls=":")
ax2.loglog(bline, v[:, 1], color=SEQ[0], ls="-")
ax2.loglog(bline, v[:, 2], color=SEQ[0], ls="--")
cols["crb_X"] = v[:, 0]
cols["crb_beta"] = v[:, 1]
cols["crb_tau"] = v[:, 2]
ax2.axvline(REF.beta, color="#777777", lw=0.7, ls="-.")
ax2.axvline(weak[1], color="#777777", lw=0.7, ls="-.")
ax2.set_xlabel(r"$\beta$")
ax2.set_ylabel("relative standard deviation")
ax2.set_xlim(bline[0], bline[-1])
for ax, lab in zip((ax0, ax1, ax2), "abc"):
    panel_label(ax, f"({lab})")
below_legend(ax0, *handles(
    ["power 0.5", "power 0.95", "reference", "weak memory"],
    ["#444444"] * 4, ["--", "-", "none", "none"], [None, None, "o", "s"]),
    ncol=1, dy=-0.27)
below_legend(ax1, *handles(
    labels + ["Rayleigh length", "cases in (a)"],
    list(SEQ[1:4]) + ["#777777"] * 2, ["-", "-", "-", ":", "-."]),
    ncol=1, dy=-0.27)
below_legend(ax2, *handles(
    [r"$\sigma_X/X$", r"$\sigma_\beta/\beta$", r"$\sigma_\tau/\tau$",
     "cases in (a)"],
    [SEQ[0]] * 3 + ["#777777"], [":", "-", "--", "-."]), ncol=1, dy=-0.27)
write_csv("fig07bc_beta", cols)
weak_lam30 = lam_of(TIDAL, *weak, 30.0)
ref_lam30 = lam_of(TIDAL, om_ref, REF.beta, 30.0)
oms1, aX1, bX1, _ = obs(["M2"], om_ref, REF.beta)
s1 = noise(aX1, [amp["M2"]], 30 * day / REF.dt_log, [sig["M2"]])
single = fit_leaky(oms1, aX1, bX1, s1)[0]
r_weak = [float(np.tan(np.angle(capacity(base[n] * weak[0], weak[1]))
                       / 2 + np.pi / 4)) for n in TIDAL]
iref = int(np.argmin(np.abs(bline - REF.beta)))
save_cache("fig07_summary", lev=np.array(lev),
           ref_lam30=np.array([ref_lam30]),
           weak_lam30=np.array([weak_lam30]),
           weak_power30=np.array([detect_power(weak_lam30, 6)]),
           single_lam=np.array([single]), r_weak=np.array(r_weak),
           crb_ref=C["crb30"][iref], beta_ref_grid=np.array([bline[iref]]),
           tmin_weak=np.array([np.interp(np.log(weak[1]), np.log(bline),
                                         C[f"tmin|{labels[0]}"])]),
           tmin_ref=np.array([np.interp(np.log(REF.beta), np.log(bline),
                                        C[f"tmin|{labels[0]}"])]),
           beta_noise_limit=np.array([np.exp(np.interp(
               np.log(14.77), np.log(C[f"tmin|{labels[0]}"][::-1]),
               np.log(bline[::-1])))]))
save(fig, "fig07_information")
print(f"figure 7 written; lambda(power .5,.95) = {lev[0]:.2f}, "
      f"{lev[1]:.2f}; ref {ref_lam30:.3e}; weak {weak_lam30:.3e};"
      f" single-constituent {single:.2e}")
