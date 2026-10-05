r"""Fisher information and detectability of storage memory.

Observation model.  At a well at dimensionless distance :math:`X = x/x_*`
the harmonic analysis of a record of :math:`N` samples returns, for each
constituent :math:`k`, the log-amplitude ratio :math:`a_k X` and the phase
lag :math:`b_k X`.  For white head noise of standard deviation
:math:`\sigma_h` and a well amplitude :math:`A_k e^{-a_k X}`, both
estimates are unbiased with standard deviation

.. math::

    \sigma_k = \sqrt{2/N}\;\sigma_h \,/\, (A_k e^{-a_k X}),

asymptotically independent across constituents and between amplitude and
phase.  The coastal record is taken as exact.

Detectability.  If the data come from a memory model and are fitted by a
leaky (or classical) model, the expected likelihood-ratio statistic is the
noncentrality

.. math::

    \lambda = \min_{\vartheta}\sum_k
    \frac{(a_k X - a_k^{\vartheta} X)^2 + (b_k X - b_k^{\vartheta} X)^2}
    {\sigma_k^2},

and the power of the test at size 0.05 follows from the noncentral
chi-square distribution with :math:`2K - p` degrees of freedom, :math:`p`
being the number of fitted parameters.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2, ncx2

from .capacity import capacity
from .propagation import rates

__all__ = ["predict", "noise", "fisher", "crb", "fit_leaky",
           "detect_power"]


def predict(omegas, X, beta, geometry="slab", tau_scale=1.0):
    r"""Return (aX, bX) for the memory model at frequencies ``omegas``.

    ``tau_scale`` multiplies the matrix time (frequencies become
    :math:`\Omega\,\mathrm{tau\_scale}`) while the length scale is
    held fixed by rescaling X, so that the three parameters
    :math:`(X, \beta, \tau)` enter independently.
    """
    om = np.asarray(omegas, dtype=float) * tau_scale
    a, b = rates(om, capacity(om, beta, geometry))
    s = X / np.sqrt(tau_scale)
    return a * s, b * s


def noise(aX, amps, nsamp, sigma_h):
    """Return the standard deviation of log-amplitude and phase estimates.

    ``sigma_h`` may be a scalar or one value per constituent (e.g. larger
    for long-period constituents, which share their band with weather).
    """
    return np.sqrt(2.0 / nsamp) * np.asarray(sigma_h) / (
        np.asarray(amps) * np.exp(-np.asarray(aX)))


def fisher(omegas, amps, X, beta, nsamp, sigma_h, h=1e-6):
    r"""Fisher matrix for :math:`(\ln X, \ln\beta, \ln\tau)`."""
    p0 = np.log([X, beta, 1.0])

    def obs(p):
        aX, bX = predict(omegas, np.exp(p[0]), np.exp(p[1]),
                         tau_scale=np.exp(p[2]))
        return np.concatenate([aX, bX])

    aX, _ = predict(omegas, X, beta)
    s = noise(aX, amps, nsamp, sigma_h)
    sig = np.concatenate([s, s])
    J = np.empty((2 * len(omegas), 3))
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        J[:, i] = (obs(p0 + e) - obs(p0 - e)) / (2 * h)
    Jw = J / sig[:, None]
    return Jw.T @ Jw


def crb(F):
    """Cramer-Rao standard deviations (relative, since log-parameters)."""
    try:
        C = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return np.full(F.shape[0], np.inf)
    d = np.diag(C)
    return np.where(d > 0, np.sqrt(np.abs(d)), np.inf)


def fit_leaky(omegas, aX, bX, sig, classical=False):
    r"""Weighted least-squares fit of the leaky model to (aX, bX).

    Leaky model: :math:`(\kappa X)^2 = i\Omega s + \ell` with
    :math:`s, \ell \ge 0`.  ``classical=True`` fixes :math:`\ell = 0`.
    Returns the minimum weighted misfit (the noncentrality) and the
    parameters.
    """
    om = np.asarray(omegas, dtype=float)
    target = np.concatenate([aX, bX])
    w = 1.0 / np.concatenate([sig, sig])

    def model(p):
        p = np.clip(p, -60.0, 60.0)
        s = np.exp(p[0])
        ell = 0.0 if classical else np.exp(p[1])
        k = np.sqrt(1j * om * s + ell)
        return np.concatenate([k.real, k.imag])

    best = None
    # exact single-constituent inversions give the starting points:
    # l = (aX)^2 - (bX)^2 and s = 2 aX bX / Omega at each constituent
    s_k = 2.0 * aX * bX / om
    l_k = np.maximum(aX**2 - bX**2, 1e-12)
    if classical:
        starts = [[np.log(np.median((aX**2 + bX**2) / om))]]
    else:
        starts = [[np.log(sk), np.log(lk)] for sk, lk in zip(s_k, l_k)]
        starts += [[np.log(np.median(s_k)), np.log(f * np.median(l_k))]
                   for f in (1e-3, 1.0, 1e3)]
    for p0 in starts:
        r = least_squares(lambda p: (model(p) - target) * w, p0,
                          method="lm", xtol=1e-14, ftol=1e-14)
        val = float(np.sum(r.fun**2))
        if best is None or val < best[0]:
            best = (val, np.exp(r.x))
    return best


def detect_power(lam, dof, size=0.05):
    """Power of the chi-square test with noncentrality ``lam``."""
    if dof <= 0:
        return np.nan
    crit = chi2.ppf(1.0 - size, dof)
    return float(ncx2.sf(crit, dof, lam))
