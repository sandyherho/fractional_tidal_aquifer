"""Figure 8: verification.

(a) Dual-porosity solver, periodic orbit of the semi-discrete system
against the exact finite-domain solution, M2 of the reference aquifer,
grid refined in x and zeta together: mobile head (circles) and matrix
head (squares).  (b) BDF2 started on the exact discrete periodic orbit
and integrated six M2 periods with M2 + K1: error against step.
(c) L1 scheme on a manufactured solution of the semi-discrete system
(spatial error removed by construction): error at t = 1 against step
for gamma = 0.25, 0.5, 0.75; dashed: slope 2 - gamma.  (d) Kramers-
Kronig reconstruction of Re G from Im G: absolute error.  (e) Power-law
slab transfer function by quadrature against the two-term closed form.
(f) Nonlinear solver: departure of the first harmonic from the periodic
orbit of the linearized BDF2-discrete system, divided by eps, against
eps; the first nonlinear correction to the fundamental is cubic, so the
slope is two (dotted).
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as Gamma

from pasang.boussinesq import Boussinesq1D
from pasang.capacity import (capacity, g_cylinder, g_powerlaw_asymptote,
                             g_powerlaw_slabs, g_slab, g_sphere,
                             g_warren_root, z_of)
from pasang.caputo import CaputoL1
from pasang.dualporosity import DualPorosity1D
from pasang.harmonic import harmonic_fit
from pasang.io_utils import load_cache, save_cache, write_csv
from pasang.kk import kk_real_from_imag
from pasang.plotting import (SEQ, below_legend, handles, panel_label,
                             save, setup)
from pasang.propagation import head_finite, rates
from pasang.scenario import REF, omega_of

setup()


def slope(h, e):
    """Observed orders between successive refinements."""
    h, e = np.asarray(h), np.asarray(e)
    return np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])


V = load_cache("verification")
if V is None:
    V = {}
    beta, om = REF.beta, omega_of("M2")
    om2 = omega_of("K1")
    L = 2.0
    z = complex(z_of(om))
    # (a) spatial convergence
    nxs = [40, 80, 160, 320, 640]
    eh, em = [], []
    for nx in nxs:
        nz = nx // 8
        m = DualPorosity1D(L, nx, nz, beta)
        U = m.periodic_amplitude(om)
        H = head_finite(m.x, om, capacity(om, beta), L)
        zeta = np.linspace(0, 1, nz + 1)
        mex = H[:, None] * np.cosh(z * zeta)[None, :] / np.cosh(z)
        eh.append(np.max(np.abs(U[:nx] - H)))
        em.append(np.max(np.abs(m.matrix_field(U) - mex)))
    V["dp_dx"] = L / np.array(nxs)
    V["dp_err_h"] = np.array(eh)
    V["dp_err_m"] = np.array(em)
    # (b) temporal convergence
    m = DualPorosity1D(L, 80, 10, beta)
    consts = [(1.0, om, 0.0), (0.6, om2, 0.7)]
    P = 2 * np.pi / om
    spps = [25, 50, 100, 200, 400, 800]
    et = []
    for spp in spps:
        dt = P / spp
        N = 6 * spp
        u0 = m.periodic_state(0.0, consts)
        u1 = m.periodic_state(dt, consts)
        uN, _ = m.integrate(consts, dt, N - 1, u0, u1)
        et.append(np.max(np.abs(uN - m.periodic_state(N * dt, consts))))
    V["bdf_dt"] = P / np.array(spps)
    V["bdf_err"] = np.array(et)
    # (c) L1 manufactured solution on the semi-discrete system
    Ns = [16, 32, 64, 128, 256, 512]
    for g in (0.25, 0.5, 0.75):
        s = CaputoL1(1.0, 100, g)
        e0 = np.zeros(s.nx)
        e0[0] = 1.0 / s.dx**2

        def ue(t):
            """Return the manufactured solution."""
            return t**2 * np.cos(np.pi * s.x / 2)

        def src(t, g=g, s=s, e0=e0):
            """Return the manufactured source term."""
            lead = 2 * t ** (2 - g) / Gamma(3 - g)
            return lead * np.cos(np.pi * s.x / 2) - s.A @ ue(t) - t**2 * e0

        errs = []
        for N in Ns:
            u, _ = s.run(1.0 / N, N, lambda t: t**2, src)
            errs.append(np.max(np.abs(u - ue(1.0))))
        V[f"l1_err_{g:g}"] = np.array(errs)
    V["l1_dt"] = 1.0 / np.array(Ns)
    # (d) Kramers-Kronig
    omk = np.logspace(-3, 4, 29)
    for name, f in (("slab", g_slab), ("cylinder", g_cylinder),
                    ("sphere", g_sphere), ("first_order", g_warren_root),
                    ("powerlaw", lambda w: g_powerlaw_slabs(w, 0.5))):
        rec = kk_real_from_imag(f, omk, n=20001 if name == "powerlaw"
                                else 400001)
        V[f"kk_{name}"] = np.abs(rec - np.real(f(omk)))
    V["kk_omega"] = omk
    # (e) power-law quadrature vs closed form
    ome = np.logspace(0.5, 6, 23)
    for q in (0.3, 0.5, 0.8):
        V[f"pl_err_{q:g}"] = np.abs(g_powerlaw_slabs(ome, q)
                                    / g_powerlaw_asymptote(ome, q) - 1)
    V["pl_omega"] = ome
    # (f) nonlinear solver against linear orbit
    a, _ = rates(om, capacity(om, beta))
    Lb = 7.0 / float(a)
    epss = [0.0125, 0.025, 0.05, 0.1, 0.2]
    dev = []
    lin = DualPorosity1D(Lb, 140, 12, beta)
    Ulin = lin.periodic_amplitude_bdf2(om, P / 80)[:140]
    for eps in epss:
        b = Boussinesq1D(Lb, 140, 12, beta, eps)
        t, H, _ = b.integrate(om, P / 80, 40 * 80,
                              u0=b.initial_state(om, a))
        sel = slice(-20 * 80, None)
        Z, _ = harmonic_fit(t[sel], H[sel], [om], trend=False)
        dev.append(np.max(np.abs(Z[0] / eps - Ulin)))
    V["nl_eps"] = np.array(epss)
    V["nl_dev"] = np.array(dev)
    save_cache("verification", **V)

fig, axs = plt.subplots(2, 3, figsize=(7.0, 6.2))
axs = axs.ravel()
ax = axs[0]
ax.loglog(V["dp_dx"], V["dp_err_h"], "o-", color=SEQ[1], ms=3.5,
          mfc="none")
ax.loglog(V["dp_dx"], V["dp_err_m"], "s--", color=SEQ[4], ms=3.5,
          mfc="none")
ref = V["dp_err_h"][0] * (V["dp_dx"] / V["dp_dx"][0]) ** 2
ax.loglog(V["dp_dx"], ref, color="#777777", lw=0.7, ls=":")
ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel("max error")

ax = axs[1]
ax.loglog(V["bdf_dt"], V["bdf_err"], "o-", color=SEQ[1], ms=3.5,
          mfc="none")
ref = V["bdf_err"][0] * (V["bdf_dt"] / V["bdf_dt"][0]) ** 2
ax.loglog(V["bdf_dt"], ref, color="#777777", lw=0.7, ls=":")
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel("max error")

ax = axs[2]
for g, c in zip((0.25, 0.5, 0.75), SEQ[1:]):
    e = V[f"l1_err_{g:g}"]
    ax.loglog(V["l1_dt"], e, "o-", color=c, ms=3.5, mfc="none")
    ax.loglog(V["l1_dt"], e[0] * (V["l1_dt"] / V["l1_dt"][0]) ** (2 - g),
              color=c, lw=0.7, ls="--")
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel("error at $t=1$")

ax = axs[3]
kk_names = ["slab", "cylinder", "sphere", "first_order", "powerlaw"]
for n, c in zip(kk_names, SEQ):
    ax.loglog(V["kk_omega"], np.maximum(V[f"kk_{n}"], 1e-17), color=c)
ax.set_xlabel(r"$\Omega$")
ax.set_ylabel(r"$|\mathrm{Re}\,G_{KK} - \mathrm{Re}\,G|$")
ax.set_ylim(1e-14, 1e-3)

ax = axs[4]
for q, c in zip((0.3, 0.5, 0.8), SEQ[1:]):
    ax.loglog(V["pl_omega"], np.maximum(V[f"pl_err_{q:g}"], 1e-17),
              "o-", color=c, ms=3, mfc="none")
ax.set_xlabel(r"$\Omega$")
ax.set_ylabel("relative difference")
ax.set_ylim(1e-17, 1e-3)

ax = axs[5]
ax.loglog(V["nl_eps"], V["nl_dev"], "o-", color=SEQ[4], ms=3.5,
          mfc="none")
ax.loglog(V["nl_eps"],
          V["nl_dev"][-1] * (V["nl_eps"] / V["nl_eps"][-1]) ** 2,
          color="#777777", lw=0.7, ls=":")
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel(r"$\max|\hat H_1/\varepsilon - \hat H_{lin}|$")
for ax, lab in zip(axs, "abcdef"):
    panel_label(ax, f"({lab})")
ref_ = "#777777"
below_legend(axs[0], *handles(
    ["mobile head", "matrix head", "slope 2"], [SEQ[1], SEQ[4], ref_],
    ["-", "--", ":"], ["o", "s", None]), ncol=1, dy=-0.30)
below_legend(axs[1], *handles(
    ["BDF2", "slope 2"], [SEQ[1], ref_], ["-", ":"], ["o", None]),
    ncol=1, dy=-0.30)
below_legend(axs[2], *handles(
    [r"$\gamma=0.25$", r"$\gamma=0.5$", r"$\gamma=0.75$",
     r"slope $2-\gamma$"],
    [SEQ[1], SEQ[2], SEQ[3], ref_], ["-", "-", "-", "--"],
    ["o", "o", "o", None]), ncol=2, dy=-0.30)
below_legend(axs[3], *handles(
    ["slab", "cylinder", "sphere", "first order", r"power law $q=0.5$"],
    SEQ[:5]), ncol=2, dy=-0.30)
below_legend(axs[4], *handles(
    [r"$q=0.3$", r"$q=0.5$", r"$q=0.8$"], [SEQ[1], SEQ[2], SEQ[3]],
    markers=["o", "o", "o"]), ncol=3, dy=-0.30)
below_legend(axs[5], *handles(
    ["Newton-BDF2", "slope 2"], [SEQ[4], ref_], ["-", ":"], ["o", None]),
    ncol=1, dy=-0.30)
fig.tight_layout(w_pad=0.9, h_pad=1.0)
write_csv("fig08a_space", {"dx": V["dp_dx"], "err_h": V["dp_err_h"],
                           "err_m": V["dp_err_m"]})
write_csv("fig08b_time", {"dt": V["bdf_dt"], "err": V["bdf_err"]})
write_csv("fig08c_l1", {"dt": V["l1_dt"],
                        **{f"err_g{g:g}": V[f"l1_err_{g:g}"]
                           for g in (0.25, 0.5, 0.75)}})
write_csv("fig08d_kk", {"omega": V["kk_omega"],
                        **{n: V[f"kk_{n}"] for n in kk_names}})
write_csv("fig08f_nonlinear", {"eps": V["nl_eps"], "dev": V["nl_dev"]})
save(fig, "fig08_verification")
print("space orders", slope(V["dp_dx"], V["dp_err_h"]))
print("BDF2 orders", slope(V["bdf_dt"], V["bdf_err"]))
for g in (0.25, 0.5, 0.75):
    print(f"L1 orders g={g}", slope(V["l1_dt"], V[f"l1_err_{g:g}"]))
print("KK max", {n: float(V[f"kk_{n}"].max()) for n in kk_names})
print("nonlinear slope", slope(V["nl_eps"], V["nl_dev"]))
