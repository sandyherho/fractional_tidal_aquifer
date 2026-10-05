r"""Time-fractional tidal diffusion: L1 solver and exact solutions.

The dimensionless equation is

.. math::

    {}^{C}\!D_t^{\gamma} h = \partial_x^2 h, \qquad 0 < \gamma \le 1,

with the Caputo derivative taken from :math:`t = 0`, zero initial head,
and coastal forcing :math:`h(0, t) = \sin\Omega t`.  On the half-line
the Laplace transform gives :math:`\hat h = \hat F(s)\,e^{-s^{\gamma/2}x}`
with :math:`\hat F = \Omega/(s^2 + \Omega^2)`.  Deforming the Bromwich
contour around the poles :math:`s = \pm i\Omega` and the branch cut on
the negative real axis splits the solution exactly into

.. math::

    h = \underbrace{\mathrm{Im}\,e^{i\Omega t - (i\Omega)^{\gamma/2}x}}
        _{\text{Weyl periodic state}}
      \; - \; \frac{1}{\pi}\int_0^\infty e^{-\rho t}\,
        \frac{\Omega}{\rho^2 + \Omega^2}\,
        \mathrm{Im}\,e^{-\rho^{\gamma/2} e^{i\pi\gamma/2} x}\,d\rho .

The second term is the memory of the switch-on.  For large t it behaves
as

.. math::

    \frac{x\,\sin(\pi\gamma/2)\,\Gamma(1 + \gamma/2)}{\pi\Omega}\,
    t^{-1-\gamma/2},

slower than the classical :math:`t^{-3/2}` whenever :math:`\gamma < 1`.

The numerical solver uses the L1 quadrature on a uniform step,

.. math::

    {}^{C}\!D_t^{\gamma} u(t_n) \approx
    \frac{\Delta t^{-\gamma}}{\Gamma(2-\gamma)}
    \sum_{k=0}^{n-1} b_k\,(u^{n-k} - u^{n-k-1}),
    \quad b_k = (k+1)^{1-\gamma} - k^{1-\gamma},

implicit in space with second-order finite differences.  Its local
truncation error is :math:`O(\Delta t^{2-\gamma})` for smooth solutions.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.special import gamma as Gamma

__all__ = ["CaputoL1", "weyl_sin", "switch_on_transient",
           "transient_asymptote", "laplacian_1d"]


def laplacian_1d(nx, dx):
    """Dirichlet-left, no-flux-right second-difference matrix."""
    main = -2.0 * np.ones(nx)
    up = np.ones(nx - 1)
    lo = np.ones(nx - 1)
    lo[-1] = 2.0  # ghost node at the no-flux boundary
    return sp.diags([lo, main, up], [-1, 0, 1]).tocsc() / dx**2


class CaputoL1:
    r"""L1 scheme for :math:`D^\gamma u = u_{xx} + s(x, t)` on (0, L]."""

    def __init__(self, length, nx, gamma):
        """Build the grids and the sparse operators."""
        self.L, self.nx, self.g = float(length), nx, float(gamma)
        self.dx = self.L / nx
        self.x = self.dx * np.arange(1, nx + 1)
        self.A = laplacian_1d(nx, self.dx)

    def run(self, dt, nsteps, bc, source=None, record=None, u0=None):
        """Advance ``nsteps`` steps from ``u0`` (zero by default).

        ``bc(t)`` is the coastal head; ``source(t)`` an optional vector.
        Returns the final state and, if ``record`` lists node indices,
        their heads at every step.
        """
        g = self.g
        a0 = dt ** (-g) / Gamma(2.0 - g)
        k = np.arange(nsteps + 1, dtype=float)
        b = (k + 1.0) ** (1.0 - g) - k ** (1.0 - g)
        lu = splu((a0 * sp.identity(self.nx) - self.A).tocsc())
        u = np.zeros(self.nx) if u0 is None else u0.copy()
        diffs = np.zeros((nsteps, self.nx))
        rec = None if record is None else np.empty((nsteps, len(record)))
        for n in range(1, nsteps + 1):
            t = n * dt
            hist = np.zeros(self.nx)
            if n > 1:
                hist = b[1:n] @ diffs[n - 2::-1][:n - 1]
            rhs = a0 * u - a0 * hist
            rhs[0] += bc(t) / self.dx**2
            if source is not None:
                rhs += source(t)
            un = lu.solve(rhs)
            diffs[n - 1] = un - u
            u = un
            if rec is not None:
                rec[n - 1] = u[record]
        return u, rec


def weyl_sin(x, t, omega, gamma):
    r"""Periodic (Weyl) response to :math:`h(0,t) = \sin\Omega t`."""
    k = (1j * omega) ** (0.5 * gamma)
    x = np.atleast_1d(np.asarray(x, dtype=float))[:, None]
    t = np.atleast_1d(np.asarray(t, dtype=float))[None, :]
    return np.imag(np.exp(1j * omega * t - k * x))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def switch_on_transient(x, t, omega, gamma, ymin=-60.0, ymax=40.0,
                        panels=1200):
    r"""Branch-cut integral (memory of the switch-on), exact to quadrature.

    Evaluated in :math:`\rho = e^y` with composite Gauss-Legendre.
    ``x`` and ``t`` are 1-D arrays; the result has shape (len(x), len(t)).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    edges = np.linspace(ymin, ymax, panels + 1)
    a, bb = edges[:-1], edges[1:]
    y = (0.5 * (bb - a)[:, None] * _GL_X + 0.5 * (a + bb)[:, None]).ravel()
    w = (0.5 * (bb - a)[:, None] * _GL_W).ravel()
    rho = np.exp(y)
    base = w * rho * omega / (rho**2 + omega**2)
    ph = np.exp(1j * np.pi * gamma / 2.0)
    out = np.empty((len(x), len(t)))
    for i, xi in enumerate(x):
        im = np.imag(np.exp(-rho ** (0.5 * gamma) * ph * xi))
        f = base * im
        for j, tj in enumerate(t):
            out[i, j] = -np.sum(f * np.exp(-rho * tj)) / np.pi
    return out


def transient_asymptote(x, t, omega, gamma):
    """Large-time form of :func:`switch_on_transient`."""
    return (np.asarray(x)[:, None] * np.sin(np.pi * gamma / 2.0)
            * Gamma(1.0 + gamma / 2.0) / (np.pi * omega)
            * np.asarray(t)[None, :] ** (-1.0 - gamma / 2.0))
