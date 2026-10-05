r"""Unconfined (Boussinesq) aquifer with resolved matrix diffusion.

Dimensionless form, heads scaled by the mean saturated thickness
:math:`D`, length by :math:`x_* = \sqrt{K D\tau/S_y}`, time by
:math:`\tau`:

.. math::

    \partial_t H + \beta\,\partial_t\langle m\rangle
      = \tfrac12\,\partial_x^2 (H^2),
    \qquad \partial_t m = \partial_\zeta^2 m,

with :math:`H(0,t) = 1 + \varepsilon\cos\Omega t`, no flux at
:math:`x = L`, and the matrix closure of :mod:`pasang.dualporosity`.

Exact invariant.  In a time-periodic state the time means of
:math:`\partial_t H` and :math:`\partial_t\langle m\rangle` vanish, so
:math:`\partial_x^2\langle H^2\rangle = 0`.  No flux at :math:`x = L`
then forces

.. math::

    \langle H^2 \rangle(x) = 1 + \varepsilon^2/2
    \quad\text{for every } x,

independently of :math:`\beta`, the matrix geometry and the
nonlinearity.  The discretization below conserves this property exactly:
the flux is the discrete Laplacian of :math:`H^2`, the exchange telescopes,
and the BDF2 difference sums to zero over an exactly periodic sequence.

Second-order overheight.  Writing :math:`H = 1 + \varepsilon H_1 +
\varepsilon^2 H_2` and averaging the second-order equation,

.. math::

    \langle H \rangle - 1 \simeq \frac{\varepsilon^2}{4}
    \left(1 - e^{-2 a x}\right),

where :math:`a` is the linear attenuation rate.  The mean water-table
rise depends on attenuation only and carries no information on the phase
lag.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

__all__ = ["Boussinesq1D"]


class Boussinesq1D:
    """Nonlinear unconfined dual-porosity model, BDF2 with Newton."""

    def __init__(self, length, nx, nz, beta, eps):
        """Build the grids and the sparse operators."""
        self.L, self.nx, self.nz = float(length), nx, nz
        self.beta, self.eps = float(beta), float(eps)
        self.dx = self.L / nx
        self.dz = 1.0 / nz
        self.x = self.dx * np.arange(1, nx + 1)
        self.n = nx + (nx * nz if beta > 0 else 0)
        self._build()

    def _build(self):
        nx, nz, dx, dz, b = self.nx, self.nz, self.dx, self.dz, self.beta
        main = -2.0 * np.ones(nx)
        up = np.ones(nx - 1)
        lo = np.ones(nx - 1)
        lo[-1] = 2.0
        self.Lap = sp.diags([lo, main, up], [-1, 0, 1]).tocsc() / dx**2
        mdiag = np.ones(self.n)
        if b > 0:
            mdiag[:nx] = 1.0 + b * dz / 2.0
            r, c, v = [], [], []
            for i in range(nx):
                r += [i, i]
                c += [i, nx + i * nz + nz - 1]
                v += [-b / dz, b / dz]
                for j in range(nz):
                    k = nx + i * nz + j
                    if j == 0:
                        nb = nx + i * nz + 1 if nz > 1 else i
                        r += [k, k]
                        c += [k, nb]
                        v += [-2.0 / dz**2, 2.0 / dz**2]
                        continue
                    nb = nx + i * nz + j + 1 if j < nz - 1 else i
                    r += [k, k, k]
                    c += [k, k - 1, nb]
                    v += [-2.0 / dz**2, 1.0 / dz**2, 1.0 / dz**2]
            self.E = sp.csc_matrix((v, (r, c)), shape=(self.n, self.n))
        else:
            self.E = sp.csc_matrix((self.n, self.n))
        self.Mdiag = mdiag

    def bc(self, t, omega):
        """Coastal head."""
        return 1.0 + self.eps * np.cos(omega * t)

    def rhs(self, u, t, omega):
        """Right-hand side N(u, t)."""
        nx = self.nx
        H = u[:nx]
        f = self.E @ u
        w = H**2
        lap = self.Lap @ w
        lap[0] += self.bc(t, omega) ** 2 / self.dx**2
        f[:nx] += 0.5 * lap
        return f

    def jac(self, u):
        """Jacobian of :meth:`rhs`."""
        nx = self.nx
        if not hasattr(self, "_pad"):
            self._pad = sp.eye(self.n, nx, format="csc")
        return (self.E + self._pad @ (self.Lap @ sp.diags(u[:nx]))
                @ self._pad.T).tocsc()

    def initial_state(self, omega, a):
        r"""Near-periodic start: linear orbit plus second-order mean rise.

        The linear part is the exact periodic orbit of the linearized
        semi-discrete system (identical operators), so the remaining
        transient is of order :math:`\varepsilon^3`.
        """
        from .dualporosity import DualPorosity1D
        lin = DualPorosity1D(self.L, self.nx, self.nz if self.beta > 0
                             else 1, self.beta)
        U = lin.periodic_amplitude(omega)
        rise = 0.25 * self.eps**2 * (1.0 - np.exp(-2.0 * a * self.x))
        u = np.ones(self.n)
        u[:self.nx] += self.eps * U[:self.nx].real + rise
        if self.beta > 0:
            u[self.nx:] += (self.eps * U[self.nx:].real
                            + np.repeat(rise, self.nz))
        return u

    def integrate(self, omega, dt, nsteps, record_every=1, tol=1e-12,
                  u0=None):
        """Integrate with BDF2 and Newton from ``u0`` (rest by default).

        The first step uses backward Euler.  The Jacobian is formed and
        factorized once per step (simplified Newton).  Returns times,
        mobile heads and full states at the recorded steps.
        """
        u = np.ones(self.n) if u0 is None else u0.copy()
        um = None
        out_t, out_h, out_u = [], [], []
        md = self.Mdiag
        newton_max = 0
        for n in range(1, nsteps + 1):
            t = n * dt
            if um is None:
                c0, c1, cm = 1.0 / dt, -1.0 / dt, 0.0
                v = u.copy()
            else:
                c0, c1, cm = 1.5 / dt, -2.0 / dt, 0.5 / dt
                v = 2.0 * u - um
            lu = splu((sp.diags(c0 * md) - self.jac(v)).tocsc())
            for it in range(50):
                prev = cm * um if um is not None else 0.0
                F = (md * (c0 * v + c1 * u + prev)
                     - self.rhs(v, t, omega))
                dv = lu.solve(-F)
                v += dv
                if np.max(np.abs(dv)) < tol:
                    break
            newton_max = max(newton_max, it + 1)
            um, u = u, v
            if n % record_every == 0:
                out_t.append(t)
                out_h.append(u[:self.nx].copy())
                out_u.append(u.copy())
        self.newton_max = newton_max
        return np.array(out_t), np.array(out_h), np.array(out_u)
