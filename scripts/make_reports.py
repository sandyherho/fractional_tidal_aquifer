"""Generate the plain-text reports.

Numbers are recomputed here from the package or read back from the
caches written by the figure scripts, so every value quoted in a report
can be traced to a file under outputs/.
"""
import _bootstrap  # noqa: F401

import numpy as np
from scipy.special import gamma as Gamma

from pasang.bounds import ratio_floor, slab_delta
from pasang.capacity import capacity, powerlaw_constant
from pasang.io_utils import load_cache, write_report
from pasang.propagation import apparent_diffusivities, local_order, rates
from pasang.scenario import CONSTITUENTS, REF, SETS, omega_of, sigma_of


def wrap(text, width=68):
    """Wrap prose to report lines indented two spaces."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append("  " + cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append("  " + cur)
    return lines


def slope(h, e):
    """Observed orders between successive refinements."""
    h, e = np.asarray(h), np.asarray(e)
    return np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])


beta = REF.beta
d_slab, u_star = slab_delta()

# ------------------------------------------------------------ closed forms
rows = ["  name | period (h) | Omega   |   a    |   b    |   r    | gamma_eff"
        " | D_a, D_b (T/S_m) | 1/a (m)", "  " + "-" * 92]
for n, (per, _) in CONSTITUENTS.items():
    om = omega_of(n)
    c = capacity(om, beta)
    a, b = rates(om, c)
    Da, Db = apparent_diffusivities(om, c)
    rows.append(f"  {n:4s} | {per:10.4f} | {om:7.4f} | {float(a):6.3f} |"
                f" {float(b):6.3f} | {float(b / a):6.4f} |"
                f" {float(local_order(c)):9.4f} | {float(Da):.4f},"
                f" {float(Db):.4f}  | {REF.xstar / float(a):8.1f}")
iq = ["  q    | I_q          | plateau order 1 - q/2 | ratio tan(gamma pi/4)",
      "  " + "-" * 64]
for q in (0.2, 0.4, 0.5, 0.6, 0.8):
    g = 1 - q / 2
    iq.append(f"  {q:4.2f} | {powerlaw_constant(q):12.9f} | {g:21.3f} |"
              f" {np.tan(g * np.pi / 4):.6f}")
write_report("closed_forms", "Closed forms", [
    ("model", wrap(
        "Confined aquifer, mobile storage S_m, matrix storage S_im,"
        " transmissivity T.  Per constituent, i omega (S_m + S_im G) H ="
        " T H''.  Dimensionless: Omega = omega tau, tau = ell^2/D_im,"
        " c = 1 + beta G, beta = S_im/S_m, x_* = sqrt(T tau/S_m),"
        " kappa^2 = i Omega c, kappa = a + i b.  Slab G = tanh z/z,"
        " cylinder 2 I_1(z)/(z I_0(z)), sphere 3(z coth z - 1)/z^2,"
        " z = sqrt(i Omega).")),
    ("identities", wrap(
        "a^2 - b^2 = -Omega Im c and 2ab = Omega Re c, exactly.  One"
        " constituent therefore measures the complex capacity at its"
        " frequency.  r = b/a = tan(pi/4 + arg(c)/2).  The classical slope"
        " factor D_a/D_b equals r^2.  Leakage gives c = 1 - i lambda/Omega,"
        " so a^2 - b^2 = lambda is the same at every frequency and Re c = 1;"
        " storage memory gives a relaxation peak in -Im c and a step in"
        " Re c from 1 + beta to 1.")),
    ("single-constituent equivalence", wrap(
        "Any pair (a, b) with 0 <= b <= a is reproduced exactly by a leaky"
        " aquifer with (kappa x)^2 = i Omega s + l, l = (ax)^2 - (bx)^2,"
        " s = 2 (ax)(bx)/Omega, and exactly by a Weyl time-fractional"
        " medium of order gamma = 4 arctan(r)/pi.  A single constituent"
        " cannot identify the mechanism; the misfit of the leaky fit to"
        " one synthetic memory constituent is"
        f" {float(load_cache('fig07_summary')['single_lam'][0]):.1e}"
        " (round-off).")),
    ("bounds on r", wrap(
        "Every G here is a positive superposition of Debye terms, so"
        " arg c lies in [-pi/2, 0] and r <= 1; leakage also gives r <= 1."
        "  A ratio above one cannot come from storage exchange or leakage."
        "  For the slab, arg G + pi/4 = arctan(sin 2u / sinh 2u), u ="
        " sqrt(Omega/2); its minimum is"
        f" delta_slab = {d_slab:.12f} at u = {u_star:.12f}"
        f" (Omega = {2 * u_star**2:.10f}).  Hence for any mixture of"
        " slabs, of any sizes and diffusivities, r >="
        f" tan(pi/8 + delta_slab/2) = {ratio_floor(d_slab):.12f}.  For"
        " cylinders and spheres the large-Omega correction to G is"
        " +i/Omega and +3i/Omega, so arg G approaches -pi/4 from above and"
        f" r > tan(pi/8) = {np.tan(np.pi / 8):.12f}.  For a single slab"
        " the half-order band is 1 << Omega << beta^2.")),
    ("power-law block sizes", wrap(
        "Volume fraction density q s^(q-1) over relative half-thickness"
        " s in (0, 1].  For Omega >> 1, G = q I_q (i Omega)^(-q/2) -"
        " q/(1 - q) (i Omega)^(-1/2) up to exponentially small terms,"
        " I_q = int_0^inf w^(q-2) tanh w dw.  In the band where the first"
        " term dominates both the second and the mobile storage, the"
        " aquifer is a time-fractional medium of order 1 - q/2 in"
        " (1/2, 1).  Matrix diffusion alone cannot produce an order below"
        " one half.") + [""] + iq),
    ("reference aquifer, per constituent", rows),
    ("switch-on transient of a time-fractional medium", wrap(
        "For D_t^gamma h = h_xx from rest with h(0,t) = sin(Omega t), the"
        " solution is the Weyl periodic state Im exp(i Omega t - (i Omega)"
        "^(gamma/2) x) minus (1/pi) int_0^inf exp(-rho t) Omega/(rho^2 +"
        " Omega^2) Im exp(-rho^(gamma/2) e^(i pi gamma/2) x) d rho.  The"
        " second term decays as x sin(pi gamma/2) Gamma(1 + gamma/2)/"
        "(pi Omega) t^(-1 - gamma/2).  For gamma = 0.75 the exponent is"
        f" {-1 - 0.375:.3f} against -1.5 for classical diffusion; the"
        f" prefactor ratio is {np.sin(0.375 * np.pi) * Gamma(1.375):.6f}"
        f" against {Gamma(1.5):.6f}.")),
    ("nonlinear unconfined aquifer", wrap(
        "For dH/dt + beta d<m>/dt = (1/2) d^2(H^2)/dx^2 with H(0,t) ="
        " 1 + eps cos(Omega t) and no flux inland, the period mean of H^2"
        " equals 1 + eps^2/2 at every x in any periodic state, for any"
        " beta, geometry and eps.  At second order the mean rise is"
        " (eps^2/4)(1 - exp(-2ax)) with a the linear attenuation rate: the"
        " overheight carries attenuation only, not the phase lag.  The"
        " far-field rise is sqrt(1 + eps^2/2) - 1 = eps^2/4 - eps^4/32 +"
        " ...")),
])

# ------------------------------------------------------------ verification
V = load_cache("verification")
F5 = load_cache("fig05")
F6 = load_cache("fig06")
blocks = [("independent checks", wrap(
    "Frequency-domain closed forms are checked against time-domain"
    " solvers that share no code path with them: a dual-porosity"
    " finite-difference model with resolved matrix diffusion (BDF2), an"
    " L1 solver for the Caputo equation, and a Newton-BDF2 solver for"
    " the nonlinear unconfined problem.  The transfer functions are"
    " checked for causality by Kramers-Kronig reconstruction, and the"
    " power-law quadrature against its closed-form asymptote."))]
if V is not None:
    lines = [
        "  dual porosity, periodic orbit vs exact, dx = "
        + ", ".join(f"{v:.4g}" for v in V["dp_dx"]),
        "      mobile max error: " + ", ".join(f"{v:.3e}"
                                               for v in V["dp_err_h"]),
        "      matrix max error: " + ", ".join(f"{v:.3e}"
                                               for v in V["dp_err_m"]),
        "      observed orders: " + ", ".join(
            f"{v:.4f}" for v in slope(V["dp_dx"], V["dp_err_h"])),
        "  BDF2 from the discrete periodic orbit, six M2 periods:",
        "      max error: " + ", ".join(f"{v:.3e}" for v in V["bdf_err"]),
        "      observed orders: " + ", ".join(
            f"{v:.4f}" for v in slope(V["bdf_dt"], V["bdf_err"])),
    ]
    for g in (0.25, 0.5, 0.75):
        e = V[f"l1_err_{g:g}"]
        lines.append(f"  L1 manufactured solution, gamma = {g}: orders "
                     + ", ".join(f"{v:.4f}" for v in slope(V["l1_dt"], e))
                     + f" (theory {2 - g:.2f})")
    for n in ("slab", "cylinder", "sphere", "first_order", "powerlaw"):
        lines.append(f"  Kramers-Kronig, {n:11s} max abs error"
                     f" {V[f'kk_{n}'].max():.3e}")
    for q in (0.3, 0.5, 0.8):
        e = V[f"pl_err_{q:g}"]
        lines.append(f"  power-law quadrature vs closed form, q = {q}:"
                     f" {e[V['pl_omega'] > 1e3].max():.3e}"
                     " (Omega > 1e3)")
    lines.append("  nonlinear first harmonic vs linear BDF2 orbit, eps = "
                 + ", ".join(f"{v:g}" for v in V["nl_eps"]))
    lines.append("      deviation / eps: " + ", ".join(
        f"{v:.3e}" for v in V["nl_dev"]))
    lines.append("      observed slopes (theory 2): " + ", ".join(
        f"{v:.4f}" for v in slope(V["nl_eps"], V["nl_dev"])))
    blocks.append(("measured residuals", lines))
if F5 is not None:
    blocks.append(("time-domain runs (figure 5)", [
        "  dual porosity from rest, M2 + K1, last 4 days, max |h - exact|"
        f" {F5['well_err'][0]:.3e} m",
        "  harmonic analysis, max error in a plus max error in b"
        f" {F5['rate_err'][0]:.3e} (1/x_*), fit rms {F5['harm_rms'][0]:.3e}"
        " m",
        "  Caputo from rest, first six periods, L1 (128 per period) vs"
        f" exact: {F5['caputo_early_err'][0]:.3e}",
        "  switch-on transient, Richardson L1 vs exact, max relative"
        f" error for 3 <= t <= {F5['caputo_tr_tmax'][0]:.1f}:"
        f" {F5['caputo_tr_relerr'][0]:.3e}",
        "  exact solution at t = 0 (Weyl part + transient) at x = 1:"
        f" {F5['caputo_t0'][0]:.3e}"]))
if F6 is not None:
    lines = []
    for k, v in zip(F6["keys"], F6["vals"]):
        lines.append(f"  {str(k):8s}: invariant residual {v[0]:.2e},"
                     f" far-field mean minus sqrt(1+eps^2/2) {v[1]:.2e},"
                     f" Newton iterations <= {int(v[3])}")
    blocks.append(("nonlinear runs (figure 6)", lines + wrap(
        "The memory runs approach the invariant more slowly than the"
        " classical run because the matrix relaxes on its own time scale;"
        " the classical residual reaches round-off.")))
blocks.append(("reading the numbers", wrap(
    "The L1 orders approach 2 - gamma from below as the step decreases,"
    " as expected for its truncation error.  The Kramers-Kronig errors"
    " of about 1e-11 are set by the finite frequency window; the larger"
    " power-law value reflects the coarser grid used for that more costly"
    " function.  The harmonic-analysis error includes the spatial"
    " discretization of the 600-node run.")))
write_report("verification", "Numerical verification", blocks)

# -------------------------------------------------------------- parameters
write_report("parameters", "Symbols, units, and reference values", [
    ("reference aquifer (illustrative, not calibrated)", [
        f"  T = {REF.T} m^2 s^-1, S_m = {REF.S_m}, S_im = {REF.S_im},"
        f" beta = {REF.beta:g}",
        f"  ell = {REF.ell} m, D_im = {REF.D_im} m^2 s^-1, tau ="
        f" {REF.tau:.1f} s = {REF.tau / 86400:.4f} d",
        f"  x_* = sqrt(T tau / S_m) = {REF.xstar:.1f} m",
        f"  head noise {REF.sigma_h * 1e3:.0f} mm in the tidal bands,"
        f" {REF.sigma_lp * 1e3:.0f} mm above two days; logging every"
        f" {REF.dt_log:.0f} s"]),
    ("constituents (period, illustrative coastal amplitude, noise)", [
        f"  {n:4s} {p:12.6f} h  {A:5.2f} m  {sigma_of(n) * 1e3:4.0f} mm"
        for n, (p, A) in CONSTITUENTS.items()]),
    ("constituent sets and Rayleigh lengths", [
        f"  {lab:12s} {', '.join(names)}  ({t:.2f} d)"
        for lab, names, t in SETS]),
    ("provenance", wrap(
        "The parameter values are round numbers chosen to sit in a"
        " plausible range for a karstic limestone coastal aquifer with"
        " decimetre-to-metre matrix blocks.  They are not measurements and"
        " no site is represented.  Constituent periods are the standard"
        " astronomical values; amplitudes are illustrative.")),
    ("what the results do not depend on", wrap(
        "The identities for a^2 - b^2 and 2ab, the bound r <= 1, the slab"
        " floor, the power-law plateau orders, the H^2 invariant and the"
        " attenuation-only overheight hold for any parameter values.")),
])

# ------------------------------------------------------------ figure notes
F2 = load_cache("fig02")
F4 = load_cache("fig04")
F7 = load_cache("fig07_summary")
notes = [("purpose", wrap(
    "Figures carry no in-panel annotation; values are recorded here and"
    " every panel is available as CSV under outputs/data."))]
if F2 is not None:
    notes.append(("figure 2, reference ratios", [
        f"  {str(n):4s} Omega = {o:9.5f}  r = {r:.5f}"
        for n, o, r in zip(F2["names"], F2["omegas"], F2["r_ref"])]))
if F4 is not None:
    notes.append(("figure 4, bounds", [
        f"  slab phase excess minimum {F4['d_slab'][0]:.10f} at u ="
        f" {F4['u_star'][0]:.10f}",
        f"  cylinder and sphere minima on the grid (large-Omega end):"
        f" {F4['d_cyl'][0]:.3e}, {F4['d_sph'][0]:.3e}",
        f"  min r at beta = 1e5: slab {F4['rmin_slab_big'][0]:.6f},"
        f" cylinder {F4['rmin_cyl_big'][0]:.6f}, sphere"
        f" {F4['rmin_sph_big'][0]:.6f}"]))
if F5 is not None:
    notes.append(("figure 5, wells", [
        "  well distances (m): "
        + ", ".join(f"{v:.1f}" for v in F5["wells_m"])]))
if F7 is not None:
    notes.append(("figure 7, detectability", [
        f"  lambda for power 0.5 and 0.95 (6 dof): {F7['lev'][0]:.3f},"
        f" {F7['lev'][1]:.3f}",
        f"  reference aquifer, M2 S2 K1 O1, 30 d: lambda ="
        f" {F7['ref_lam30'][0]:.4e}",
        f"  weak case beta = 0.2, 30 d: lambda = {F7['weak_lam30'][0]:.4e},"
        f" power {F7['weak_power30'][0]:.4f}",
        "  weak case r at M2, S2, K1, O1: " + ", ".join(
            f"{v:.4f}" for v in F7["r_weak"]),
        f"  record for power 0.95 from noise alone, M2 S2 K1 O1: reference"
        f" {F7['tmin_ref'][0]:.3g} d, weak {F7['tmin_weak'][0]:.3g} d",
        "  beta below which the four-constituent set needs more than its"
        f" Rayleigh length: {F7['beta_noise_limit'][0]:.3f}",
        "  Cramer-Rao relative std at the reference, 30 d (X, beta, tau):"
        " " + ", ".join(f"{v:.4f}" for v in F7["crb_ref"])]))
write_report("figure_notes", "Figure notes and tabulated values", notes)

# -------------------------------------------------------------- open items
write_report("open_items", "Open items and negative results", [
    ("what the ratio alone cannot do", wrap(
        "A single constituent never distinguishes storage memory from"
        " leakage or from a time-fractional medium; each reproduces any"
        " pair with b <= a exactly.  The discriminant is the frequency"
        " dependence: a^2 - b^2 is frequency independent for leakage and a"
        " relaxation for memory, and the two diverge most at long periods."
        "  Long-period constituents are, however, the ones most"
        " contaminated by weather and river stage in real records.")),
    ("a claim that had to be corrected", wrap(
        "The expectation that matrix diffusion is bounded by the half-order"
        " phase, r >= tan(pi/8), holds for cylinders and spheres but not"
        " for slabs: planar blocks undershoot by delta_slab/2 in theta"
        " because of the reflected wave at the block centre.  The bound"
        " reported is therefore tan(pi/8 + delta_slab/2).")),
    ("ratios above one", wrap(
        "Neither storage exchange nor leakage can produce r > 1.  Wedge-"
        "shaped or finite aquifers, sloping beaches and the finite depth of"
        " unconfined aquifers can, and none of these is included here.  A"
        " field ratio above one is evidence for geometry, not memory.")),
    ("assumptions carried by the whole construction", wrap(
        "One horizontal dimension; a vertical coastline with a prescribed"
        " head; uniform matrix geometry along x; linear storage exchange"
        " with the matrix even in the nonlinear runs; no density effects,"
        " no seepage face, no capillary fringe, no aquifer loading by the"
        " tide.  The identifiability study assumes white noise, exact"
        " coastal records and correctly specified constituents, so its"
        " detection thresholds are optimistic.")),
    ("not attempted", wrap(
        "Calibration against field records; two-dimensional or layered"
        " aquifers; leakage and memory acting together; the influence of"
        " pumping; seawater intrusion.")),
])
print("reports written")
