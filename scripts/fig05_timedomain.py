"""Figure 5: time-domain solutions against the frequency-domain theory.

(a) Dual-porosity aquifer with resolved slab diffusion, reference
parameters, forced by M2 + K1 from rest with a one-tau ramp.  Heads at
three wells (solid) and the exact periodic solution (dashed) over the
last four days.  (b) Attenuation and phase-lag rates recovered by
least-squares harmonic analysis of the last 15 days at every node
(markers, every 6th node) against the exact rates (lines; a solid,
b dashed).  Dotted: classical a = b for storage S_m + S_im (upper) and
S_m alone (lower); memory places a and b between them and apart.
(c) Time-fractional medium, gamma = 0.75, Caputo derivative from rest,
forcing sin(t): L1 solution at x = 1 (solid), exact solution (Weyl
periodic part plus branch-cut transient, dashed), Weyl part alone
(dotted).  (d) Magnitude of the switch-on transient at x = 1: exact
branch-cut integral (solid), its large-time asymptote (dashed), and the
L1 solution minus the Weyl part after Richardson extrapolation in time
(markers); grey: the classical case gamma = 1.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from pasang.capacity import capacity
from pasang.caputo import (CaputoL1, switch_on_transient,
                           transient_asymptote, weyl_sin)
from pasang.dualporosity import DualPorosity1D
from pasang.harmonic import harmonic_fit, transfer_rates
from pasang.io_utils import load_cache, save_cache, write_csv
from pasang.plotting import (OKABE_ITO, SEQ, below_legend, handles,
                             panel_label, save, setup)
from pasang.propagation import head_finite, rates
from pasang.scenario import CONSTITUENTS, REF, omega_of

setup()
beta = REF.beta
om_m2, om_k1 = omega_of("M2"), omega_of("K1")
A_m2, A_k1 = CONSTITUENTS["M2"][1], CONSTITUENTS["K1"][1]
consts = [(A_m2, om_m2, 0.0), (A_k1, om_k1, 0.9)]
a_m2, b_m2 = rates(om_m2, capacity(om_m2, beta))
a_k1, b_k1 = rates(om_k1, capacity(om_k1, beta))
day = 86400.0 / REF.tau          # one day in units of tau

# ------------------------------------------------ (a, b) dual porosity
C = load_cache("fig05_dp")
L, nx, nz = 3.0, 600, 24
if C is None:
    m = DualPorosity1D(L, nx, nz, beta)
    P = 2 * np.pi / om_m2
    dt = P / 100.0
    t_end = 45.0 * day
    nsteps = int(np.ceil(t_end / dt))
    rec_idx = list(range(nx))
    _, rec = m.integrate(consts, dt, nsteps, record=rec_idx, ramp=1.0)
    t = dt * (np.arange(nsteps + 1) + 1)
    keep = t > t_end - 15.0 * day
    C = dict(t=t[keep], bc=rec[keep, 0], h=rec[keep, 1:], x=m.x, dt=dt)
    save_cache("fig05_dp", **C)
t, bc, h, x = C["t"], C["bc"], C["h"], C["x"]
Zc, _ = harmonic_fit(t, bc, [om_m2, om_k1])
Zw, rms = harmonic_fit(t, h, [om_m2, om_k1])
xs = slice(5, int(3.2 / a_m2 * nx / L), 6)
a_rec = {}
b_rec = {}
for k, name in enumerate(["M2", "K1"]):
    a_rec[name], b_rec[name] = transfer_rates(Zw[k, xs], Zc[k], x[xs])
wells = [0.5 / a_m2, 1.0 / a_m2, 2.0 / a_m2]
iw = [int(np.argmin(np.abs(x - w))) for w in wells]
per = sum(A * np.real(head_finite(x[iw][:, None], o, capacity(o, beta), L)
                      * np.exp(1j * (o * t[None, :] - ph)))
          for A, o, ph in consts)

# ------------------------------------------------ (c, d) Caputo
g, om_c, xc = 0.75, 1.0, 1.0
D = load_cache("fig05_caputo")
Pc = 2 * np.pi / om_c
if D is None:
    s = CaputoL1(16.0, 320, g)
    j = [int(np.argmin(np.abs(s.x - xc)))]
    runs = {}
    for spp in (128, 256):
        _, r = s.run(Pc / spp, 20 * spp, lambda tt: np.sin(om_c * tt),
                     record=j)
        runs[spp] = r[spp // 128 - 1::spp // 128, 0]
    tc = (Pc / 128) * np.arange(1, len(runs[128]) + 1)
    p = 2.0 - g
    rich = (2**p * runs[256] - runs[128]) / (2**p - 1)
    D = dict(tc=tc, u128=runs[128], rich=rich, xnode=np.array(s.x[j]))
    save_cache("fig05_caputo", **D)
tc, u128, rich = D["tc"], D["u128"], D["rich"]
X = float(D["xnode"][0])
w = weyl_sin(X, tc, om_c, g)[0]
tr_t = np.logspace(-1, 3, 120)
tr = switch_on_transient([X], tr_t, om_c, g)[0]
tr1 = switch_on_transient([X], tr_t, om_c, 1.0)[0]
asy = transient_asymptote([X], tr_t, om_c, g)[0]
asy1 = transient_asymptote([X], tr_t, om_c, 1.0)[0]
early = tc <= 6 * Pc
ex_early = w[early] + switch_on_transient([X], tc[early], om_c, g)[0]
pts = np.unique(np.round(np.logspace(np.log10(3.0 / tc[0]),
                                     np.log10(60.0 / tc[0]), 22)).astype(int)
                - 1)
num_tr = rich[pts] - w[pts]
ex_pts = switch_on_transient([X], tc[pts], om_c, g)[0]

# ------------------------------------------------ figure
fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.6))
ax = axs[0, 0]
tday = t / day
sel = tday > tday[-1] - 4.0
for k, c in enumerate(SEQ[1:4]):
    ax.plot(tday[sel], h[sel, iw[k]], color=c, lw=1.1)
    ax.plot(tday[sel], per[k, sel], color="black", lw=0.7, ls="--")
ax.plot(tday[sel], bc[sel], color=OKABE_ITO["grey"], lw=0.8)
ax.set_xlabel("time (days)")
ax.set_ylabel("head (m)")
ax.set_xlim(tday[sel][0], tday[sel][-1])

ax = axs[0, 1]
xx = np.linspace(0.02, x[xs][-1], 200)
for name, om, c in (("M2", om_m2, SEQ[1]), ("K1", om_k1, SEQ[4])):
    Hf = head_finite(xx, om, capacity(om, beta), L)
    ax.plot(xx * REF.xstar / 1e3, -np.log(np.abs(Hf)) / xx, color=c)
    ax.plot(xx * REF.xstar / 1e3, -np.angle(Hf) / xx, color=c, ls="--")
    ax.plot(x[xs] * REF.xstar / 1e3, a_rec[name], "o", color=c, ms=3,
            mfc="none")
    ax.plot(x[xs] * REF.xstar / 1e3, b_rec[name], "s", color=c, ms=3,
            mfc="none")
for om, c in ((om_m2, SEQ[1]), (om_k1, SEQ[4])):
    ax.axhline(np.sqrt(om * (1 + beta) / 2), color=c, ls=":", lw=0.8)
    ax.axhline(np.sqrt(om / 2), color=c, ls=":", lw=0.8)
ax.set_xlabel("distance from coast (km)")
ax.set_ylabel(r"$a,\ b$  ($1/x_*$)")
ax.set_ylim(0, 15)

ax = axs[1, 0]
ax.plot(tc[early] / Pc, u128[early], color=SEQ[5], lw=1.2)
ax.plot(tc[early] / Pc, ex_early, color="black", lw=0.8, ls="--")
ax.plot(tc[early] / Pc, w[early], color=OKABE_ITO["grey"], lw=0.8,
        ls=":")
ax.set_xlabel(r"$t\,/\,(2\pi/\Omega)$")
ax.set_ylabel(r"$h(x=1)$")
ax.set_xlim(0, 6)

ax = axs[1, 1]
ax.loglog(tr_t, np.abs(tr), color=SEQ[5])
ax.loglog(tr_t, np.abs(asy), color=SEQ[5], ls="--", lw=0.8)
ax.loglog(tr_t, np.abs(tr1), color=OKABE_ITO["grey"])
ax.loglog(tr_t, np.abs(asy1), color=OKABE_ITO["grey"], ls="--", lw=0.8)
ax.loglog(tc[pts], np.abs(num_tr), "o", color=SEQ[5], ms=3.5, mfc="none")
ax.set_xlabel(r"$t$")
ax.set_ylabel("switch-on transient")
ax.set_xlim(tr_t[0], tr_t[-1])
ax.set_ylim(1e-6, 1)
for ax, lab in zip(axs.ravel(), "abcd"):
    panel_label(ax, f"({lab})")
grey = OKABE_ITO["grey"]
below_legend(axs[0, 0], *handles(
    ["well 1", "well 2", "well 3", "coast", "exact periodic"],
    SEQ[1:4] + [grey, "black"], ["-", "-", "-", "-", "--"]), ncol=3,
    dy=-0.2)
below_legend(axs[0, 1], *handles(
    ["M2", "K1", "$a$ exact", "$b$ exact", "$a$ recovered",
     "$b$ recovered", "classical $a=b$"],
    [SEQ[1], SEQ[4], "#444444", "#444444", "#444444", "#444444", "#444444"],
    ["-", "-", "-", "--", "none", "none", ":"],
    [None, None, None, None, "o", "s", None]), ncol=2, dy=-0.2)
below_legend(axs[1, 0], *handles(
    ["L1, Caputo from rest", "exact (Weyl + transient)", "Weyl part only"],
    [SEQ[5], "black", grey], ["-", "--", ":"]), ncol=2, dy=-0.2)
below_legend(axs[1, 1], *handles(
    [r"exact, $\gamma=0.75$", r"exact, $\gamma=1$", "asymptote",
     r"L1, $\gamma=0.75$"],
    [SEQ[5], grey, "#444444", SEQ[5]], ["-", "-", "--", "none"],
    [None, None, None, "o"]), ncol=2, dy=-0.2)
fig.tight_layout(h_pad=1.2, w_pad=1.4)
write_csv("fig05a_wells", {"t_days": tday[sel],
                           **{f"h_well{k + 1}": h[sel, iw[k]]
                              for k in range(3)},
                           **{f"h_exact{k + 1}": per[k, sel]
                              for k in range(3)}})
write_csv("fig05b_rates", {"x_km": x[xs] * REF.xstar / 1e3,
                           "a_M2": a_rec["M2"], "b_M2": b_rec["M2"],
                           "a_K1": a_rec["K1"], "b_K1": b_rec["K1"]})
write_csv("fig05d_transient", {"t": tr_t, "exact": tr, "asymptote": asy,
                               "exact_gamma1": tr1})
Hx = {n: head_finite(x[xs], o, capacity(o, beta), L)
      for n, o in (("M2", om_m2), ("K1", om_k1))}
err_rates = max(np.max(np.abs(a_rec[n] + np.log(np.abs(Hx[n])) / x[xs]))
                + np.max(np.abs(b_rec[n] + np.angle(Hx[n]) / x[xs]))
                for n in Hx)
save_cache("fig05", well_err=np.array([np.max(np.abs(h[sel][:, iw].T
                                                     - per[:, sel]))]),
           rate_err=np.array([err_rates]), harm_rms=np.array([rms]),
           wells_m=np.array(wells) * REF.xstar,
           caputo_early_err=np.array([np.max(np.abs(u128[early]
                                                    - ex_early))]),
           caputo_tr_relerr=np.array([np.max(np.abs(num_tr / ex_pts - 1))]),
           caputo_tr_tmax=np.array([tc[pts][-1]]),
           caputo_t0=np.array([float(weyl_sin(X, [0.0], om_c, g)[0, 0]
                                     + switch_on_transient(
                                         [X], [0.0], om_c, g)[0, 0])]))
save(fig, "fig05_timedomain")
print("figure 5 written; max rate error", err_rates)
