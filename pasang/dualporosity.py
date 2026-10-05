r"""Time-domain dual-porosity aquifer with resolved matrix diffusion.

Dimensionless equations (length :math:`x_*`, time :math:`\tau`, matrix
coordinate :math:`\zeta \in [0, 1]` across a slab half-thickness):

.. math::

    \partial_t h + \beta\,\partial_t\langle m\rangle = \partial_x^2 h,
    \qquad
    \partial_t m = \partial_\zeta^2 m,

with :math:`m(x, 1, t) = h(x, t)`, :math:`\partial_\zeta m(x, 0, t) = 0`,
:math:`\langle m \rangle = \int_0^1 m\,d\zeta`, a prescribed coastal
head :math:`h(0, t)`, and no flux at :math:`x = L`.

Space is discretized with second-order finite differences on uniform
grids.  The matrix average uses the trapezoidal rule including the
interface node, which, combined with the discrete matrix equations,
reduces exactly to

.. math::

    (1 + \beta\Delta\zeta/2)\,\dot h_i = (\delta_x^2 h)_i
      - \beta\,(h_i - m_{i,N-1})/\Delta\zeta ,

so the exchange is conservative at the discrete level.  The resulting
linear system :math:`M\dot u = A u + f(t)` is advanced with BDF2 using a
single sparse LU factorization, and its periodic orbit is obtained
exactly by solving :math:`(i\Omega M - A)U = F` per constituent.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

__all__ = ["DualPorosity1D"]


class DualPorosity1D:
    r"""Semi-discrete dual-porosity model on :math:`x \in (0, L]`."""

    def __init__(self, length, nx, nz, beta):
        """Build the grids and the sparse operators."""
        self.L, self.nx, self.nz, self.beta = float(length), nx, nz, beta
        self.dx = self.L / nx
        self.dz = 1.0 / nz
        self.x = self.dx * np.arange(1, nx + 1)
        self.n = nx * (1 + nz)
        self._build()

    # unknown layout: h_0..h_{nx-1}, then m_{i,j} at nx + i*nz + j
    def _mi(self, i, j):
        return self.nx + i * self.nz + j

    def _build(self):
        nx, nz, dx, dz, b = self.nx, self.nz, self.dx, self.dz, self.beta
        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(r)
            cols.append(c)
            vals.append(v)

        for i in range(nx):
            # mobile Laplacian, Dirichlet at x=0 enters through forcing
            add(i, i, -2.0 / dx**2)
            if i > 0:
                add(i, i - 1, 1.0 / dx**2)
            if i < nx - 1:
                add(i, i + 1, 1.0 / dx**2)
            else:
                add(i, i - 1, 1.0 / dx**2)  # ghost node, no flux
            if b > 0:
                add(i, i, -b / dz)
                add(i, self._mi(i, nz - 1), b / dz)
            # matrix nodes
            for j in range(nz):
                r = self._mi(i, j)
                if j == 0:
                    if nz == 1:
                        add(r, r, -2.0 / dz**2)
                        add(r, i, 2.0 / dz**2)
                    else:
                        add(r, r, -2.0 / dz**2)
                        add(r, self._mi(i, 1), 2.0 / dz**2)
                    continue
                add(r, r, -2.0 / dz**2)
                add(r, self._mi(i, j - 1), 1.0 / dz**2)
                if j < nz - 1:
                    add(r, self._mi(i, j + 1), 1.0 / dz**2)
                else:
                    add(r, i, 1.0 / dz**2)  # interface node equals h_i
        self.A = sp.csc_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        mdiag = np.ones(self.n)
        mdiag[:nx] = 1.0 + b * dz / 2.0
        self.Mdiag = mdiag
        self.M = sp.diags(mdiag).tocsc()
        self.fvec = np.zeros(self.n)
        self.fvec[0] = 1.0 / dx**2  # times boundary head

    # ------------------------------------------------------------ frequency
    def periodic_amplitude(self, omega):
        """Complex state amplitude for a unit coastal tide at ``omega``."""
        K = 1j * omega * self.M - self.A
        return spsolve(K.tocsc(), self.fvec.astype(complex))

    def periodic_amplitude_bdf2(self, omega, dt):
        r"""Periodic amplitude of the BDF2-discretized system.

        For forcing :math:`e^{i\Omega t_n}` the BDF2 recursion has the
        exact periodic solution :math:`U e^{i\Omega t_n}` with
        :math:`[M(3 - 4\zeta^{-1} + \zeta^{-2})/(2\Delta t) - A]U = F`,
        :math:`\zeta = e^{i\Omega\Delta t}`.  It isolates spatial and
        nonlinear effects from the time-stepping error.
        """
        zi = np.exp(-1j * omega * dt)
        s = (3.0 - 4.0 * zi + zi**2) / (2.0 * dt)
        K = s * self.M - self.A
        return spsolve(K.tocsc(), self.fvec.astype(complex))

    def periodic_state(self, t, consts):
        r"""Exact periodic orbit of the semi-discrete system.

        ``consts`` is a list of ``(amplitude, omega, phase)`` with coastal
        head :math:`\sum A\cos(\Omega t - \phi)`.
        """
        u = np.zeros(self.n)
        for amp, om, ph in consts:
            U = self.periodic_amplitude(om)
            u += amp * np.real(U * np.exp(1j * (om * t - ph)))
        return u

    # ------------------------------------------------------------ time
    def integrate(self, consts, dt, nsteps, u0=None, u1=None, t0=0.0,
                  record=None, ramp=0.0):
        r"""BDF2 integration.

        Starts from ``u0`` at ``t0`` and ``u1`` at ``t0 + dt`` (both zero
        when omitted).  ``record`` is a list of mobile-node indices whose
        heads are returned at every step; the final state is also
        returned.  ``ramp`` > 0 multiplies the forcing by
        :math:`1 - e^{-t/\mathrm{ramp}}`.
        """
        u0 = np.zeros(self.n) if u0 is None else u0.copy()
        u1 = np.zeros(self.n) if u1 is None else u1.copy()
        lu = splu((1.5 / dt * self.M - self.A).tocsc())
        md = self.Mdiag / (2.0 * dt)

        def bc(t):
            s = sum(a * np.cos(o * t - p) for a, o, p in consts)
            if ramp > 0:
                s *= 1.0 - np.exp(-t / ramp)
            return s

        rec = None
        if record is not None:
            rec = np.empty((nsteps + 1, len(record) + 1))
            rec[0, 0] = bc(t0 + dt)
            rec[0, 1:] = u1[record]
        um, uc = u0, u1
        for n in range(1, nsteps + 1):
            t = t0 + (n + 1) * dt
            rhs = md * (4.0 * uc - um) + self.fvec * bc(t)
            un = lu.solve(rhs)
            um, uc = uc, un
            if rec is not None:
                rec[n, 0] = bc(t)
                rec[n, 1:] = un[record]
        return uc, rec

    def matrix_field(self, u):
        """Return the matrix head as an (nx, nz+1) array including h."""
        m = u[self.nx:].reshape(self.nx, self.nz)
        return np.hstack([m, u[:self.nx, None]])
