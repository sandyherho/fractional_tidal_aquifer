"""Animation 2: the ratio spectrum as matrix storage grows.

r(Omega) for slab memory while beta sweeps from 1e-2 to 1e4 and back
(logarithmically).  Dots: the eight constituents of the reference
aquifer.  Dashed: tan(pi/8); dotted: the slab floor.  Faint: leakage
curves for comparison, fixed.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from pasang.anim import FG, MEMORY, TIDE, dark, fig_to_rgb, write_gif
from pasang.bounds import ratio_floor, slab_delta
from pasang.capacity import capacity, capacity_leaky
from pasang.propagation import ratio
from pasang.scenario import CONSTITUENTS, omega_of

dark()
om = np.logspace(-3, 6, 700)
cons = np.array([omega_of(n) for n in CONSTITUENTS])
floor = ratio_floor(slab_delta()[0])
leaks = [ratio(capacity_leaky(om, lam)) for lam in (0.3, 3.0, 30.0)]
nh = 60
lb = np.concatenate([np.linspace(-2, 4, nh), np.linspace(4, -2, nh)[1:-1]])
frames = []
for e in lb:
    beta = 10.0**e
    fig = plt.figure(figsize=(6.0, 3.4), dpi=110)
    ax = fig.add_axes([0.11, 0.16, 0.85, 0.76])
    for r in leaks:
        ax.semilogx(om, r, color=TIDE, lw=0.8, alpha=0.35, ls=":")
    ax.axhline(np.tan(np.pi / 8), color="#8a93a6", ls="--", lw=0.8)
    ax.axhline(floor, color="#8a93a6", ls=":", lw=0.8)
    r = ratio(capacity(om, beta))
    ax.semilogx(om, r, color=MEMORY, lw=2.0)
    ax.plot(cons, ratio(capacity(cons, beta)), "o", color="white", ms=4)
    ax.set_xlim(om[0], om[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\Omega = \omega\tau$")
    ax.set_ylabel(r"$r = b/a$")
    fig.text(0.95, 0.89, rf"$\beta = S_{{im}}/S_m = 10^{{{e:+.1f}}}$",
             ha="right", color=FG, fontsize=9)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
print(write_gif("anim02_ratio", frames, fps=16))
