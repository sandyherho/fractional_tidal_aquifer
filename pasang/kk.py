r"""Kramers-Kronig consistency of a storage transfer function.

With :math:`G(\Omega) = \int_0^\infty g(t)\,e^{-i\Omega t}\,dt` for
a real causal kernel :math:`g`, and :math:`G(\infty) = 0`,

.. math::

    \mathrm{Re}\,G(\Omega) = -\frac{2}{\pi}\,
    \mathrm{P}\!\int_0^\infty \frac{\Omega'\,\mathrm{Im}\,G(\Omega')}
    {\Omega'^2 - \Omega^2}\,d\Omega' .

The principal value is removed by subtracting :math:`\Omega f(\Omega)`
from the numerator and adding back its integral over the truncated range
in closed form.  A capacity with a zero-frequency loss term, such as
leakage, is not of this form and fails the test by construction.
"""

import numpy as np

__all__ = ["kk_real_from_imag"]


def kk_real_from_imag(func, omega, lo=1e-12, hi=1e14, n=400001):
    r"""Reconstruct Re G at ``omega`` from Im G of ``func`` on [lo, hi].

    The integral is taken in :math:`y = \ln\Omega'` on a uniform grid
    with the composite trapezoidal rule; Im G is evaluated once on that
    grid.  An evaluation frequency that falls within 1e-6 of a node in
    ``y`` uses the limit of the subtracted integrand there, which is
    finite because the singularity is removable; the derivative needed
    for that limit is taken by central differences.  The range above
    ``hi`` is added in closed form assuming Im G decays as a power of
    frequency there, :math:`\int_{hi}^\infty A w^{-1-p}\,dw =
    f(hi)/p`, which removes the dominant truncation error.
    """
    y = np.linspace(np.log(lo), np.log(hi), n)
    dy = y[1] - y[0]
    wp = np.exp(y)
    f = np.imag(func(wp))
    # power-law tail beyond hi: Im G ~ A w^(-p), estimated from the grid
    p = -np.log(f[-1] / f[-2]) / dy
    tail = f[-1] / p if p > 0 else 0.0
    om_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    out = np.empty(len(om_arr))
    for k, om in enumerate(om_arr):
        f0 = float(np.imag(func(np.array([om])))[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = (wp * f - om * f0) / (wp**2 - om**2) * wp
        bad = np.abs(y - np.log(om)) < 1e-6
        if np.any(bad):
            h = 1e-5 * om
            fp = float(np.imag(func(np.array([om + h])))[0]
                       - np.imag(func(np.array([om - h])))[0]) / (2 * h)
            # d/dw (w f) / d/dw (w^2) at w = om, times the Jacobian om
            integrand[bad] = (f0 + om * fp) / (2 * om) * om
        val = np.trapezoid(integrand, dx=dy)
        corr = om * f0 * (np.log(abs((hi - om) / (hi + om)))
                          - np.log(abs((lo - om) / (lo + om)))) / (2 * om)
        out[k] = -2.0 / np.pi * (val + corr + tail)
    return out
