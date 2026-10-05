"""Figure 0: geometry and balance diagrams.

(a) Cross-section of the idealized coastal aquifer.  The sea imposes the
head A cos(omega t) at x = 0; a confined layer between aquitards carries
a conduit network (mobile storage S_m, transmissivity T) that surrounds
matrix blocks (storage S_im, diffusivity D_im, half-thickness ell).  The
dashed curves are the envelope +/- A exp(-a x) of the reference M2
constituent, and the solid curve is the head at one instant, both
computed from the model.  A no-flux boundary closes the layer at x = L.
The shaded slice is the control volume of (b).
(b) Control-volume diagram of a slice of width dx: Darcy fluxes q at
both faces, storage in the conduits, and exchange q_ex dx with the
matrix.  The aquitard leakage L h dx (grey, dashed) belongs to the
comparison model and is absent from the memory model.
(c) One matrix block: the faces follow the conduit head, the centre
plane carries no flux, and the exchange equals the rate of change of
the block-averaged head.  Curves are the computed profiles at eight
phases of the M2 cycle for the reference aquifer.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from pasang.capacity import capacity, z_of
from pasang.io_utils import write_csv
from pasang.plotting import OKABE_ITO, panel_label, save, setup
from pasang.propagation import kappa
from pasang.scenario import REF, omega_of

setup()
om = omega_of("M2")
k = complex(kappa(om, capacity(om, REF.beta)))
z = complex(z_of(om))

INK = "#222222"
SEA = "#cfe6f5"
SEA_EDGE = OKABE_ITO["blue"]
CLAY = "#e4e4e4"
ROCK = "#efe3c8"
ROCK_EDGE = "#b59a62"
EXCH = OKABE_ITO["vermil"]
LEAK = "#8c8c8c"


def arrow(ax, p0, p1, color=INK, lw=1.0, ls="-", ms=8, style="-|>"):
    """Draw a straight arrow from p0 to p1 in data coordinates."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style,
                                 mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def hatch_band(ax, x0, x1, y0, y1, label=None):
    """Aquitard drawn as a grey band with diagonal hatching."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=CLAY, ec="none",
                           hatch="////", lw=0))
    ax.plot([x0, x1], [y0, y0], color=INK, lw=0.6)
    ax.plot([x0, x1], [y1, y1], color=INK, lw=0.6)


fig = plt.figure(figsize=(7.0, 5.0))
axa = fig.add_axes([0.02, 0.50, 0.96, 0.47])
axb = fig.add_axes([0.02, 0.02, 0.50, 0.42])
axc = fig.add_axes([0.58, 0.02, 0.40, 0.42])
for ax in (axa, axb, axc):
    ax.axis("off")

# ================================================================ (a)
ax = axa
ax.set_xlim(-0.24, 1.06)
ax.set_ylim(-0.30, 1.08)
y_bot, y_top = 0.0, 0.46            # confined layer
L_end = 1.0                         # no-flux boundary
xmax = 5.0 / k.real                 # dimensionless extent drawn
# aquitards
hatch_band(ax, 0.0, L_end, -0.10, y_bot)
hatch_band(ax, 0.0, L_end, y_top, y_top + 0.10)
# sea, with mean, high and low water
PH = 0.15 * 2 * np.pi              # instant shown
eta0 = 0.86 + 0.12 * np.cos(PH)
ax.add_patch(Polygon([(-0.24, -0.10), (0.0, -0.10), (0.0, eta0),
                      (-0.24, eta0)], closed=True, fc=SEA, ec="none"))
ax.plot([-0.24, 0.0], [eta0, eta0], color=SEA_EDGE, lw=1.4)
ax.plot([-0.24, 0.0], [0.86, 0.86], color="#bbbbbb", lw=0.5)
for yy in (0.98, 0.74):
    ax.plot([-0.24, 0.0], [yy, yy], color=SEA_EDGE, lw=0.8, ls="--")
arrow(ax, (-0.12, 0.86), (-0.12, 0.98), SEA_EDGE, lw=0.8, ms=6,
      style="<|-|>")
ax.text(-0.13, 1.01, r"$\eta = A\cos\omega t$", ha="center", va="bottom",
        fontsize=8, color=SEA_EDGE)
ax.text(-0.12, 0.40, "sea", ha="center", va="center", fontsize=8,
        color=SEA_EDGE, style="italic")
# matrix blocks and conduits
nb, rows = 12, 2
gap = 0.026
bw = L_end / nb
bh = (y_top - y_bot) / rows
for i in range(nb):
    for r in range(rows):
        ax.add_patch(Rectangle((i * bw + gap / 2, y_bot + r * bh + gap / 2),
                               bw - gap, bh - gap, fc=ROCK, ec=ROCK_EDGE,
                               lw=0.5))
# exchange arrows on one block (into the block, both faces)
ib, rb = 3, 1  # upper-row block; its top face meets the aquitard
xb0 = ib * bw + gap / 2
yb0 = y_bot + rb * bh + gap / 2
for fy in (0.3, 0.7):
    ya = yb0 + fy * (bh - gap)
    arrow(ax, (xb0 - gap / 2, ya), (xb0 + 0.02, ya), EXCH, lw=0.9, ms=6)
    arrow(ax, (xb0 + bw - gap / 2, ya), (xb0 + bw - gap - 0.02, ya),
          EXCH, lw=0.9, ms=6)
xa = xb0 + 0.5 * (bw - gap)
arrow(ax, (xa, yb0 - gap / 2), (xa, yb0 + 0.03), EXCH, lw=0.9, ms=6)
# no-flux boundary
ax.plot([L_end, L_end], [y_bot - 0.10, y_top + 0.10], color=INK, lw=1.4)
for yy in np.linspace(y_bot - 0.08, y_top + 0.08, 9):
    ax.plot([L_end, L_end + 0.025], [yy, yy + 0.025], color=INK, lw=0.6)
ax.text(L_end + 0.035, 0.5 * (y_bot + y_top), r"$\partial_x h = 0$",
        ha="left", va="center", fontsize=8, rotation=90)
# head envelope and snapshot above the layer
xs = np.linspace(0.0, L_end, 300)
xd = xs * xmax
env = np.exp(-k.real * xd)
snap = np.real(np.exp(-k * xd + 1j * PH))
y0h, amp = 0.86, 0.12
ax.plot(xs, y0h + amp * env, color=INK, lw=0.8, ls="--")
ax.plot(xs, y0h - amp * env, color=INK, lw=0.8, ls="--")
ax.plot(xs, y0h + amp * snap, color=SEA_EDGE, lw=1.4)
ax.plot([0, L_end], [y0h, y0h], color="#bbbbbb", lw=0.5)
ax.text(0.20, y0h + amp * np.exp(-k.real * 0.20 * xmax) + 0.02,
        r"$\pm A\,e^{-ax}$", ha="left", va="bottom", fontsize=8)
ax.text(0.45, y0h + 0.03, r"$h(x,t)$", ha="left", va="bottom",
        fontsize=8, color=SEA_EDGE)
# control volume
cv0 = 6 * bw
ax.add_patch(Rectangle((cv0, y_bot), bw, y_top - y_bot, fc="none",
                       ec=INK, lw=1.0, ls=(0, (3, 2))))
ax.text(cv0 + bw / 2, y_top + 0.14, "(b)", ha="center", va="bottom",
        fontsize=8)
# x axis
arrow(ax, (0.0, -0.22), (0.32, -0.22), INK, lw=0.8, ms=7)
ax.text(0.33, -0.22, r"$x$", ha="left", va="center", fontsize=9)
ax.text(0.0, -0.25, r"$0$", ha="center", va="top", fontsize=8)
ax.text(L_end, -0.25, r"$L$", ha="center", va="top", fontsize=8)
# labels
ax.annotate(r"matrix: $S_{im}$, $D_{im}$, $\ell$",
            xy=(8.5 * bw, y_bot + 0.5 * bh), xytext=(0.46, -0.21),
            fontsize=8, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK,
                            shrinkA=2, shrinkB=0))
ax.annotate("conduits: $T$, $S_m$", xy=(10 * bw, y_bot + 0.5 * bh),
            xytext=(0.80, -0.21), fontsize=8, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK,
                            shrinkA=2, shrinkB=0))
ax.text(0.14, y_top + 0.05, "aquitard", ha="center", va="center",
        fontsize=7, color="#444444", style="italic",
        bbox=dict(fc="white", ec="none", pad=0.8))
panel_label(ax, "(a)", dx=0.0, dy=0.97)

# ================================================================ (b)
ax = axb
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.10, 1.05)
X0, X1, Y0, Y1 = 0.30, 0.70, 0.30, 0.72
hatch_band(ax, 0.12, 0.88, Y1, Y1 + 0.06)
ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fc="white", ec=INK,
                       lw=1.1))
ax.add_patch(Rectangle((X0 + 0.07, Y0 - 0.20), X1 - X0 - 0.14, 0.17,
                       fc=ROCK, ec=ROCK_EDGE, lw=0.6))
ax.text(0.5, Y0 - 0.115, "matrix", ha="center", va="center", fontsize=7,
        color="#6b5530", style="italic")
ym = 0.5 * (Y0 + Y1)
arrow(ax, (0.06, ym), (X0, ym), INK, lw=1.2, ms=9)
arrow(ax, (X1, ym), (0.94, ym), INK, lw=1.2, ms=9)
ax.text(0.17, ym + 0.04, r"$q(x)$", ha="center", va="bottom", fontsize=9)
ax.text(0.83, ym + 0.04, r"$q(x+\mathrm{d}x)$", ha="center", va="bottom",
        fontsize=9)
ax.text(0.5, ym + 0.07, r"$S_m\,\partial_t h\,\mathrm{d}x$", ha="center",
        va="center", fontsize=9)
arrow(ax, (0.5, Y0 + 0.06), (0.5, Y0 - 0.04), EXCH, lw=1.2, ms=9)
ax.text(0.53, Y0 + 0.07, r"$q_{ex}\,\mathrm{d}x$", ha="left",
        va="center", fontsize=9, color=EXCH)
arrow(ax, (0.5, Y1 + 0.13), (0.5, Y1 + 0.005), LEAK, lw=1.0, ms=8,
      ls="--")
ax.text(0.53, Y1 + 0.12, r"$L\,h\,\mathrm{d}x$", ha="left", va="center",
        fontsize=9, color=LEAK)
ax.plot([X0, X0], [Y0 - 0.26, Y0 - 0.23], color=INK, lw=0.6)
ax.plot([X1, X1], [Y0 - 0.26, Y0 - 0.23], color=INK, lw=0.6)
arrow(ax, (X0, Y0 - 0.245), (X1, Y0 - 0.245), INK, lw=0.6, ms=5,
      style="<|-|>")
ax.text(0.5, Y0 - 0.27, r"$\mathrm{d}x$", ha="center", va="top",
        fontsize=8)
ax.text(0.5, 1.0, r"$S_m\,\partial_t h + q_{ex} = T\,\partial_x^2 h$,"
        r"$\quad q = -T\,\partial_x h$", ha="center", va="top",
        fontsize=8.5)
panel_label(ax, "(b)", dx=0.0, dy=1.0)

# ================================================================ (c)
ax = axc
ax.set_xlim(-1.75, 1.75)
ax.set_ylim(-0.62, 1.75)
ax.add_patch(Rectangle((-1.0, -0.38), 2.0, 1.76, fc=ROCK, ec=ROCK_EDGE,
                       lw=0.8, alpha=0.55))
ax.plot([0, 0], [-0.38, 1.38], color=INK, lw=0.6, ls=(0, (2, 2)))
zeta = np.linspace(-1.0, 1.0, 201)
prof = np.cosh(z * np.abs(zeta)) / np.cosh(z)
cols = {"zeta": zeta}
base, sc = 0.50, 0.75
for j, ph in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
    y = np.real(prof * np.exp(1j * ph))
    ax.plot(zeta, base + sc * y, color=plt.get_cmap("viridis")(j / 8),
            lw=0.9)
    cols[f"phase{j}"] = y
ax.plot(zeta, base + sc * np.abs(prof), color=INK, lw=0.8, ls="--")
ax.plot(zeta, base - sc * np.abs(prof), color=INK, lw=0.8, ls="--")
cols["envelope"] = np.abs(prof)
for s in (-1, 1):
    arrow(ax, (1.55 * s, base), (1.03 * s, base), EXCH, lw=1.1, ms=8)
    ax.text(1.32 * s, base + 0.10, r"$m=h$", ha="center", va="bottom",
            fontsize=8)
ax.text(0.04, 1.30, r"$\partial_\zeta m = 0$", ha="left", va="top",
        fontsize=8)
ax.text(0.0, 1.74, r"$\partial_t m = D_{im}\,\partial_\zeta^2 m$,"
        r"$\quad q_{ex} = S_{im}\,\partial_t\langle m\rangle$",
        ha="center", va="top", fontsize=8.5)
arrow(ax, (-1.0, -0.47), (1.0, -0.47), INK, lw=0.6, ms=5, style="<|-|>")
ax.text(0.0, -0.50, r"$2\ell$", ha="center", va="top", fontsize=8)
panel_label(ax, "(c)", dx=0.0, dy=1.0)

write_csv("fig00c_block_profile", cols)
save(fig, "fig00_schematic")
print(f"M2 Omega = {om:.4f}, beta = {REF.beta:g}, a = {k.real:.4f},"
      f" b = {k.imag:.4f}")
