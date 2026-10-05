"""Animation 1: matrix blocks breathing under the tide.

One M2 cycle of the reference aquifer from the exact periodic solution.
Top: conduit head h(x, t) (amber) with its envelope; the classical
aquifer with the same total storage (cyan, thin) for comparison.
Bottom: matrix head inside a stack of blocks, cross-section zeta in
[-1, 1] at each x, rendered with a sign-preserving square-root display
map so the decaying signal stays visible inland.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from pasang.anim import FG, MEMORY, SIGNED, TIDE, dark, fig_to_rgb, write_gif
from pasang.capacity import capacity, z_of
from pasang.propagation import kappa
from pasang.scenario import REF, omega_of

dark()
om = omega_of("M2")
k = complex(kappa(om, capacity(om, REF.beta)))
kc = complex(kappa(om, 1.0 + REF.beta))
z = complex(z_of(om))
xmax = 3.0 / k.real
x = np.linspace(0, xmax, 420)
nblk, rows = 12, 3
zeta = np.linspace(-1, 1, 41)
prof = np.cosh(z * np.abs(zeta)) / np.cosh(z)
H = np.exp(-k * x)
Hc = np.exp(-kc * x)
xkm = x * REF.xstar / 1e3

frames = []
nfr = 72
for n in range(nfr):
    ph = 2 * np.pi * n / nfr
    fig = plt.figure(figsize=(6.4, 3.6), dpi=110)
    ax = fig.add_axes([0.09, 0.62, 0.88, 0.33])
    bx = fig.add_axes([0.09, 0.12, 0.88, 0.42])
    ax.fill_between(xkm, -np.abs(H), np.abs(H), color=MEMORY, alpha=0.12,
                    lw=0)
    ax.plot(xkm, np.real(Hc * np.exp(1j * ph)), color=TIDE, lw=0.9,
            alpha=0.8)
    ax.plot(xkm, np.real(H * np.exp(1j * ph)), color=MEMORY, lw=1.8)
    ax.set_xlim(xkm[0], xkm[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(r"$h / A$")
    ax.set_xticklabels([])
    # matrix field: blocks along x, cross-sections stacked in rows
    img = np.zeros((rows * (len(zeta) + 3), len(x)))
    field = np.real(np.outer(prof, H) * np.exp(1j * ph))
    for r in range(rows):
        r0 = r * (len(zeta) + 3)
        img[r0:r0 + len(zeta)] = field
        img[r0 + len(zeta):r0 + len(zeta) + 3] = np.real(
            H * np.exp(1j * ph))[None, :]
    disp = np.sign(img) * np.sqrt(np.abs(img))
    # thin vertical gaps between blocks (conduits along block edges)
    edges = np.searchsorted(x, np.linspace(0, xmax, nblk + 1)[1:-1])
    for e in edges:
        g = np.real(H[e] * np.exp(1j * ph))
        disp[:, max(e - 1, 0):e + 1] = np.sign(g) * np.sqrt(abs(g))
    bx.imshow(disp, aspect="auto", cmap=SIGNED, vmin=-1, vmax=1,
              extent=[xkm[0], xkm[-1], 0, rows], origin="lower",
              interpolation="bilinear")
    bx.set_yticks([])
    bx.set_xlabel("distance from coast (km)")
    bx.set_ylabel("matrix blocks")
    fig.text(0.97, 0.965, f"M2 phase {360 * n / nfr:5.0f}" + r"$^\circ$",
             ha="right", va="top", color=FG, fontsize=8)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
print(write_gif("anim01_breathing", frames, fps=20))
