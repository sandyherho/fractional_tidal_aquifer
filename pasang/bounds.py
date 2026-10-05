r"""Bounds on the tidal ratio imposed by passive storage memory.

Two results are evaluated here.

1. For any capacity of the form :math:`c = 1 + \beta G` with :math:`G` a
   Stieltjes function of :math:`i\Omega`, :math:`\arg c \in
   [-\pi/2, 0]`, hence :math:`\theta \in [0, \pi/4]` and
   :math:`r \le 1`.  The same holds for leakage.  A ratio above one
   cannot come from storage exchange or leakage.

2. For diffusive exchange with matrix blocks, :math:`\arg G` is bounded
   below by a geometry constant :math:`-\pi/4 + \delta_g`.  Because
   :math:`c` is a positive combination of terms whose arguments lie in
   :math:`[-\pi/4 + \delta_{min}, 0]`, the same cone bounds
   :math:`\arg c` for any mixture of block sizes, diffusivities and
   shapes, and

   .. math::

       r \ge \tan\!\left(\frac{\pi}{8} + \frac{\delta_{min}}{2}\right).

   For a slab, :math:`\arg G = -\pi/4 + \arctan(\sin 2u/\sinh 2u)`
   with :math:`u = \sqrt{\Omega/2}`; the bracket is negative on
   :math:`\pi/2 < u < \pi`, so slabs undershoot the half-order phase
   slightly.  For cylinders and spheres the large-:math:`\Omega`
   correction to :math:`G` is :math:`+i/\Omega` times a positive
   constant, and :math:`\arg G` approaches :math:`-\pi/4` from above.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .capacity import GEOMETRIES
from .propagation import ratio

__all__ = ["slab_phase_excess", "slab_delta", "geometry_delta",
           "ratio_floor", "min_ratio", "slab_band"]


def slab_phase_excess(u):
    r"""Return :math:`\arg G_{slab} + \pi/4` as a function of u."""
    u = np.asarray(u, dtype=float)
    return np.arctan2(np.sin(2.0 * u), np.sinh(2.0 * u))


def slab_delta():
    """Minimum of the slab phase excess and the u where it occurs."""
    res = minimize_scalar(lambda u: float(slab_phase_excess(u)),
                          bounds=(np.pi / 2, np.pi), method="bounded",
                          options={"xatol": 1e-14})
    return float(res.fun), float(res.x)


def geometry_delta(geometry, omega=None):
    r"""Numerical minimum of :math:`\arg G + \pi/4` over a frequency grid."""
    if omega is None:
        omega = np.logspace(-4, 9, 260001)
    ph = np.angle(GEOMETRIES[geometry](omega)) + 0.25 * np.pi
    i = int(np.argmin(ph))
    return float(ph[i]), float(omega[i])


def ratio_floor(delta):
    r"""Lower bound :math:`\tan(\pi/8 + \delta/2)` on r."""
    return float(np.tan(np.pi / 8 + 0.5 * delta))


def min_ratio(beta, geometry="slab", omega=None):
    r"""Minimum over frequency of r for :math:`c = 1 + \beta G`."""
    if omega is None:
        omega = np.logspace(-3, 3 + 2.2 * np.log10(max(beta, 1.0) + 1.0),
                            40001)
    r = ratio(1.0 + beta * GEOMETRIES[geometry](omega))
    i = int(np.argmin(r))
    return float(r[i]), float(omega[i])


def slab_band(beta):
    r"""Return the asymptotic half-order band of a single slab.

    The band is :math:`1 \ll \Omega \ll \beta^2`.  In it
    :math:`G \simeq (i\Omega)^{-1/2}` and the matrix term dominates the
    mobile storage, so the aquifer behaves as a time-fractional medium
    of order one half.
    """
    return 1.0, float(beta) ** 2
