"""Animation 3: nonlinear water table, classical and with memory.

One forcing period of the converged Boussinesq runs of Figure 6
(eps = 0.2, reference M2 frequency): classical aquifer (top, cyan) and
memory aquifer, beta = 20 (bottom, amber), with the envelope, the period
mean (white) and the exact far-field mean sqrt(1 + eps^2/2) (dashed).
Requires the caches written by fig06_nonlinear.py.
"""
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from pasang.anim import FG, MEMORY, TIDE, dark, fig_to_rgb, write_gif
from pasang.io_utils import load_cache

dark()
mem = load_cache("fig06_mem_0.2")
cls = load_cache("fig06_cls_0.2")
if mem is None or cls is None:
    raise SystemExit("run fig06_nonlinear.py first")
eps = 0.2
far = np.sqrt(1 + eps**2 / 2) - 1
frames = []
nfr = mem["last"].shape[0]
for n in range(0, nfr, 1):
    fig = plt.figure(figsize=(6.0, 3.6), dpi=110)
    axs = [fig.add_axes([0.11, 0.56, 0.85, 0.38]),
           fig.add_axes([0.11, 0.13, 0.85, 0.38])]
    for ax, C, col, lab in ((axs[0], cls, TIDE, "classical"),
                            (axs[1], mem, MEMORY, r"memory, $\beta=20$")):
        x, Hl = C["x"], C["last"] - 1
        ax.fill_between(x, Hl.min(0), Hl.max(0), color=col, alpha=0.13,
                        lw=0)
        ax.fill_between(x, -0.25, Hl[n], color=col, alpha=0.35, lw=0)
        ax.plot(x, Hl[n], color=col, lw=1.8)
        ax.plot(x, Hl.mean(0), color="white", lw=0.9)
        ax.axhline(far, color="#8a93a6", ls="--", lw=0.8)
        ax.set_xlim(0, 0.78)
        ax.set_ylim(-0.22, 0.22)
        ax.set_ylabel(r"$(h - D)/D$")
        ax.text(0.77, 0.17, lab, ha="right", color=FG, fontsize=8)
    axs[0].set_xticklabels([])
    axs[1].set_xlabel(r"$x / x_*$")
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
print(write_gif("anim03_watertable", frames, fps=20))
