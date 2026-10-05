r"""Complex storage capacity of a dual-porosity aquifer.

A tidally forced confined aquifer with a mobile (fracture or conduit)
storage :math:`S_m` and an immobile (matrix) storage :math:`S_{im}` obeys,
in the frequency domain with time dependence :math:`e^{i\omega t}`,

.. math::

    i\omega\,[S_m + S_{im}\,G(\omega)]\,H = T\,H'',

where :math:`G` is the ratio of the volume-averaged immobile head to the
mobile head.  All functions here are dimensionless.  Frequency is
:math:`\Omega = \omega\tau` with :math:`\tau = \ell^2/D_{im}` the
diffusion time of the largest matrix block of half-thickness (or radius)
:math:`\ell`, and capacity is normalized by :math:`S_m`, so that

.. math::

    c(\Omega) = 1 + \beta\,G(\Omega), \qquad \beta = S_{im}/S_m .

Every :math:`G` below is a Stieltjes function of :math:`i\Omega`
(a positive superposition of Debye relaxations), hence passive and causal.
The leaky-aquifer and pure time-fractional (Caputo, Weyl-initialized)
capacities are provided for comparison; neither is a relaxation function.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import ive

__all__ = ["z_of", "g_slab", "g_cylinder", "g_sphere", "g_warren_root",
           "g_powerlaw_slabs", "g_powerlaw_asymptote", "powerlaw_constant",
           "capacity",
           "capacity_leaky", "capacity_caputo", "GEOMETRIES"]


def z_of(omega):
    r"""Return :math:`z = \sqrt{i\Omega}` on the principal branch."""
    return np.sqrt(1j * np.asarray(omega, dtype=float))


def _small(z):
    return np.abs(z) < 1e-3


def g_slab(omega):
    r"""Slab of half-thickness 1, :math:`G = \tanh z / z`."""
    z = z_of(omega)
    out = np.empty_like(z)
    s = _small(z)
    zs = z[s]
    out[s] = 1.0 - zs**2 / 3.0 + 2.0 * zs**4 / 15.0
    zl = z[~s]
    out[~s] = np.tanh(zl) / zl
    return out


def g_cylinder(omega):
    """Cylinder of radius 1, :math:`G = 2 I_1(z) / (z I_0(z))`.

    Exponentially scaled Bessel functions avoid overflow; their ratio is
    unaffected by the common scaling.
    """
    z = z_of(omega)
    out = np.empty_like(z)
    s = _small(z)
    zs = z[s]
    out[s] = 1.0 - zs**2 / 8.0 + zs**4 / 48.0
    zl = z[~s]
    out[~s] = 2.0 * ive(1, zl) / (zl * ive(0, zl))
    return out


def g_sphere(omega):
    r"""Sphere of radius 1, :math:`G = 3(z\coth z - 1)/z^2`."""
    z = z_of(omega)
    out = np.empty_like(z)
    s = _small(z)
    zs = z[s]
    out[s] = 1.0 - zs**2 / 15.0 + 2.0 * zs**4 / 315.0
    zl = z[~s]
    out[~s] = 3.0 * (zl / np.tanh(zl) - 1.0) / zl**2
    return out


GEOMETRIES = {"slab": g_slab, "cylinder": g_cylinder, "sphere": g_sphere}


def g_warren_root(omega):
    r"""First-order (Warren-Root) exchange, :math:`G = 1/(1 + i\Omega)`."""
    return 1.0 / (1.0 + 1j * np.asarray(omega, dtype=float))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def g_powerlaw_slabs(omega, q, panels=400):
    r"""Slabs with a power-law distribution of half-thickness.

    The matrix volume fraction carried by slabs of relative half-thickness
    :math:`s = \ell/\ell_{max} \in (0, 1]` has density
    :math:`q s^{q-1}`, :math:`q > 0`.  Then

    .. math::

        G(\Omega) = \int_0^1 q\,s^{q-1}\,g_{slab}(\Omega s^2)\,ds
                   = \int_{-\infty}^0 q\,e^{qy}\,g_{slab}(\Omega e^{2y})
                     \,dy,

    evaluated by composite 64-point Gauss-Legendre quadrature in
    :math:`y`.  The lower limit is cut where :math:`e^{qy} < 10^{-17}`.
    """
    om = np.atleast_1d(np.asarray(omega, dtype=float))
    ymin = np.log(1e-17) / q
    edges = np.linspace(ymin, 0.0, panels + 1)
    a, b = edges[:-1], edges[1:]
    y = (0.5 * (b - a)[:, None] * _GL_X[None, :]
         + 0.5 * (a + b)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * _GL_W[None, :]).ravel()
    wy = w * q * np.exp(q * y)
    out = np.empty(om.shape, dtype=complex)
    for k, o in enumerate(om):
        out[k] = np.sum(wy * g_slab(o * np.exp(2.0 * y)))
    return out.reshape(np.shape(omega)) if np.ndim(omega) else out[0]


def powerlaw_constant(q):
    r"""Return :math:`I_q = \int_0^\infty w^{q-2}\tanh w\,dw`, 0<q<1.

    For :math:`\Omega \gg 1` the distributed-slab transfer function
    approaches :math:`q I_q (i\Omega)^{-q/2}`; the contour from the ray
    :math:`\arg w = \pi/4` is rotated onto the real axis, which encloses
    no pole of :math:`\tanh` and loses no arc contribution for
    :math:`q < 1`.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")

    def f(w):
        return w ** (q - 2.0) * np.tanh(w)

    a, _ = quad(f, 0.0, 1.0, limit=200)
    b, _ = quad(lambda w: w ** (q - 2.0) * (np.tanh(w) - 1.0), 1.0, np.inf,
                limit=200)
    c = 1.0 / (1.0 - q)  # int_1^inf w^(q-2) dw
    return a + b + c


def g_powerlaw_asymptote(omega, q):
    r"""Two-term large-:math:`\Omega` form of :func:`g_powerlaw_slabs`.

    With :math:`z = \sqrt{i\Omega}`,

    .. math::

        G = q z^{-q}\int_0^{z} w^{q-2}\tanh w\,dw
          \simeq q I_q (i\Omega)^{-q/2}
          - \frac{q}{1-q}\,(i\Omega)^{-1/2},

    the second term being the cutoff at the largest block, where
    :math:`\tanh w \to 1` up to exponentially small corrections.  The
    first term alone is a constant-phase (time-fractional) capacity of
    order :math:`\gamma = 1 - q/2`.
    """
    s = 1j * np.asarray(omega, dtype=float)
    return (q * powerlaw_constant(q) * s ** (-0.5 * q)
            - q / (1.0 - q) * s ** (-0.5))


def capacity(omega, beta, geometry="slab", q=None):
    r"""Return the normalized complex capacity :math:`c = 1 + \beta G`.

    ``geometry`` is ``"slab"``, ``"cylinder"``, ``"sphere"``,
    ``"warren_root"``, or ``"powerlaw"`` (distributed slabs, needs ``q``).
    """
    if geometry == "powerlaw":
        g = g_powerlaw_slabs(omega, q)
    elif geometry == "warren_root":
        g = g_warren_root(omega)
    else:
        g = GEOMETRIES[geometry](omega)
    return 1.0 + beta * g


def capacity_leaky(omega, lam):
    r"""Leaky confined aquifer, :math:`c = 1 - i\lambda/\Omega`.

    :math:`\lambda = L\tau/S_m` with :math:`L` the aquitard leakance.
    The term is a zero-frequency loss, the analogue of a DC conductivity,
    not a storage relaxation.
    """
    om = np.asarray(omega, dtype=float)
    return 1.0 - 1j * lam / om


def capacity_caputo(omega, gamma, coef=1.0):
    r"""Constant-phase capacity of a time-fractional equation.

    The Weyl (periodic-steady-state) form of
    :math:`\mathrm{coef}\,D_t^{\gamma} h = h_{xx}` gives
    :math:`i\Omega\,c = \mathrm{coef}\,(i\Omega)^{\gamma}`, so
    :math:`c = \mathrm{coef}\,(i\Omega)^{\gamma - 1}`.
    """
    om = np.asarray(omega, dtype=float)
    return coef * (1j * om) ** (gamma - 1.0)
