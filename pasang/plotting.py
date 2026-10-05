"""Figure style, palette, and multi-format export.

Panel labels sit outside the axes frame, above its top-left corner.
Legends sit below the axes as a single unframed figure-level legend, so
that no curve is obscured.  Figures carry no titles and no in-panel text:
numerical values belong in the reports and the manuscript captions.
Axis quantities are dimensionless unless a unit is given in the label.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

matplotlib.use("Agg")

__all__ = ["OKABE_ITO", "SEQ", "setup", "panel_label", "outside_legend",
           "handles", "save", "sci", "FIGDIR", "mark_constituents",
           "below_legend", "side_legend"]

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermil": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#8C8C8C",
}

SEQ = [OKABE_ITO["black"], OKABE_ITO["blue"], OKABE_ITO["green"],
       OKABE_ITO["orange"], OKABE_ITO["vermil"], OKABE_ITO["purple"],
       OKABE_ITO["skyblue"]]

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(_HERE), "outputs", "figures")


def setup():
    """Apply the figure style used by every script in this repository."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sci(x):
    """Mathtext for a positive number as m x 10^k, dropping m = 1."""
    k = int(np.floor(np.log10(x) + 1e-9))
    m = x / 10.0 ** k
    if abs(m - 1.0) < 1e-9:
        return rf"10^{{{k}}}"
    return rf"{m:g}\times10^{{{k}}}"


def panel_label(ax, text, dx=0.0, dy=1.03):
    """Panel label outside the axes, above its top-left corner."""
    ax.text(dx, dy, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold")


def handles(labels, colors, styles=None, markers=None, widths=None):
    """Legend proxies for a family of curves."""
    styles = styles or ["-"] * len(labels)
    markers = markers or [None] * len(labels)
    widths = widths or [1.3] * len(labels)
    return ([Line2D([], [], color=c, ls=s, lw=w, marker=m, ms=3.8,
                    mfc="none")
             for c, s, m, w in zip(colors, styles, markers, widths)],
            list(labels))


def outside_legend(fig, hs, labels, ncol=None, y=0.0):
    """Single unframed figure-level legend below the axes."""
    ncol = ncol or len(labels)
    return fig.legend(hs, labels, loc="upper center",
                      bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                      handlelength=1.8, columnspacing=1.4, handletextpad=0.5,
                      borderaxespad=0.0)


def below_legend(ax, hs, labels, ncol=1, dy=-0.28):
    """Unframed legend centred below one panel, under its x label.

    Use when the curves differ between panels, so that every legend sits
    with the panel it describes.  ``dy`` is the top of the legend in axes
    fraction and must clear the tick labels and the x label.
    """
    return ax.legend(hs, labels, loc="upper center",
                     bbox_to_anchor=(0.5, dy), ncol=ncol, frameon=False,
                     fontsize=7, handlelength=1.8, columnspacing=1.0,
                     handletextpad=0.4, labelspacing=0.3, borderaxespad=0.0)


def side_legend(ax, hs, labels):
    """Unframed legend to the right of a panel, vertically centred.

    Use when every panel of a row shows the same curves, so one legend
    beside the row serves all of them.
    """
    return ax.legend(hs, labels, loc="center left",
                     bbox_to_anchor=(1.04, 0.5), frameon=False, fontsize=7,
                     handlelength=1.8, handletextpad=0.4, labelspacing=0.45,
                     borderaxespad=0.0)


def mark_constituents(ax, omegas, color="#BBBBBB"):
    """Thin vertical guides at constituent frequencies (no text)."""
    for om in omegas:
        ax.axvline(om, color=color, lw=0.6, ls=":", zorder=0)


def save(fig, stem, dpi=600):
    """Write a vector PDF and a high-resolution PNG."""
    os.makedirs(FIGDIR, exist_ok=True)
    for ax in fig.axes:   # no labels on minor ticks of logarithmic axes
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
        if ax.get_yscale() == "log":
            ax.yaxis.set_minor_formatter(NullFormatter())
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": dpi})):
        path = os.path.join(FIGDIR, f"{stem}.{ext}")
        fig.savefig(path, **kw)
        paths.append(path)
    plt.close(fig)
    return paths[0]
