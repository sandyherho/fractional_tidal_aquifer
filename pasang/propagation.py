r"""Inland propagation of one tidal constituent.

With the normalized complex capacity :math:`c(\Omega)` and the length
scale :math:`x_* = \sqrt{T\tau/S_m}`, a constituent of dimensionless
frequency :math:`\Omega` propagates as :math:`e^{i\Omega t - \kappa x}`
with

.. math::

    \kappa^2 = i\Omega\,c(\Omega), \qquad
    \kappa = a + i b, \quad a, b \ge 0 .

The log-amplitude attenuation rate is :math:`a` and the phase-lag rate is
:math:`b`.  Two exact identities follow from :math:`\kappa^2`:

.. math::

    a^2 - b^2 = -\Omega\,\mathrm{Im}\,c, \qquad
    2ab = \Omega\,\mathrm{Re}\,c,

so the pair :math:`(a, b)` measured at one constituent is equivalent to
the complex capacity at that frequency, up to the scale :math:`T`.
The ratio :math:`r = b/a = \tan\theta` with
:math:`\theta = \pi/4 + \tfrac12 \arg c`.
"""

import numpy as np

__all__ = ["kappa", "rates", "ratio", "theta", "local_order",
           "head_semi_infinite", "head_finite", "apparent_diffusivities",
           "kappa_squared_parts"]


def kappa(omega, c):
    r"""Principal square root of :math:`i\Omega c` (positive real part)."""
    k = np.sqrt(1j * np.asarray(omega, dtype=float) * np.asarray(c))
    return np.where(k.real < 0, -k, k)


def rates(omega, c):
    """Return the attenuation rate ``a`` and phase-lag rate ``b``."""
    k = kappa(omega, c)
    return k.real, k.imag


def theta(c):
    r"""Return :math:`\theta = \pi/4 + \arg(c)/2`, the argument of kappa."""
    return 0.25 * np.pi + 0.5 * np.angle(c)


def ratio(c):
    r"""Phase-lag to log-attenuation ratio :math:`r = \tan\theta`."""
    return np.tan(theta(c))


def local_order(c):
    r"""Local fractional order :math:`\gamma = 4\theta/\pi`.

    It equals the order of the time-fractional equation that reproduces
    the measured ratio at that frequency: :math:`r = \tan(\gamma\pi/4)`.
    It is 1 for classical diffusion.
    """
    return 4.0 * theta(c) / np.pi


def kappa_squared_parts(omega, c):
    """Return :math:`a^2 - b^2` and :math:`2ab` computed from ``c``."""
    om = np.asarray(omega, dtype=float)
    c = np.asarray(c)
    return -om * c.imag, om * c.real


def head_semi_infinite(x, omega, c):
    r"""Complex head amplitude :math:`e^{-\kappa x}` per unit coastal tide."""
    return np.exp(-kappa(omega, c) * np.asarray(x))


def head_finite(x, omega, c, length):
    r"""Complex head amplitude with a no-flux landward boundary at ``length``.

    :math:`H(x) = \cosh(\kappa(L - x))/\cosh(\kappa L)`, written with
    decaying exponentials so that it does not overflow.
    """
    k = kappa(omega, c)
    x = np.asarray(x)
    num = np.exp(-k * x) + np.exp(-k * (2.0 * length - x))
    den = 1.0 + np.exp(-2.0 * k * length)
    return num / den


def apparent_diffusivities(omega, c):
    r"""Diffusivities inferred by the classical formula from a and b.

    Classical theory gives :math:`a = b = \sqrt{\Omega/2D}`, so
    :math:`D_a = \Omega/(2a^2)` and :math:`D_b = \Omega/(2b^2)`, both in
    units of :math:`T/S_m`.  Their ratio :math:`D_a/D_b = r^2` is the
    slope factor of the tidal-method literature.
    """
    a, b = rates(omega, c)
    om = np.asarray(omega, dtype=float)
    return om / (2.0 * a**2), om / (2.0 * b**2)
