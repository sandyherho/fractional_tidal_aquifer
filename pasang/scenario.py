r"""Reference parameters and tidal constituents.

Values are illustrative, chosen to sit in a plausible range for a karstic
limestone coastal aquifer of the kind common on uplifted islands of the
Indonesian Maritime Continent.  They are not calibrated to any site, and
every dimensional statement scales with them.  The dimensionless results
depend only on :math:`\beta`, :math:`\Omega` and :math:`x/x_*`.
"""

import numpy as np

__all__ = ["REF", "CONSTITUENTS", "omega_of", "SETS", "Reference",
           "sigma_of"]


class Reference:
    """Dimensional reference aquifer (SI units)."""

    T = 5.0e-2        # transmissivity of the conduit network, m^2 s^-1
    S_m = 5.0e-5      # storativity of the conduit network
    S_im = 1.0e-3     # storativity of the matrix (beta = 20)
    ell = 0.5         # largest matrix block half-thickness, m
    D_im = 1.93e-6    # matrix hydraulic diffusivity, m^2 s^-1
    sigma_h = 1.0e-2  # head noise in the tidal bands, m
    sigma_lp = 5.0e-2  # noise for periods above two days (weather), m
    dt_log = 600.0    # logger sampling interval, s

    @property
    def beta(self):
        """Capacity ratio :math:`S_{im}/S_m`."""
        return self.S_im / self.S_m

    @property
    def tau(self):
        r"""Matrix diffusion time :math:`\ell^2/D_{im}`, s."""
        return self.ell**2 / self.D_im

    @property
    def xstar(self):
        r"""Length scale :math:`\sqrt{T\tau/S_m}`, m."""
        return np.sqrt(self.T * self.tau / self.S_m)


REF = Reference()

# name: (period in hours, illustrative coastal amplitude in m)
CONSTITUENTS = {
    "M2": (12.4206012, 0.50),
    "S2": (12.0000000, 0.20),
    "K1": (23.9344696, 0.30),
    "O1": (25.8193417, 0.20),
    "MSf": (354.3670666, 0.02),
    "Mm": (661.3111655, 0.02),
    "Ssa": (4382.905209, 0.03),
    "Sa": (8765.812770, 0.08),
}

# constituent sets used in the identifiability study, with the record
# length (days) at which each set is resolvable by the Rayleigh criterion
SETS = [
    ("M2", ["M2"], 1.0),
    ("+S2 K1 O1", ["M2", "S2", "K1", "O1"], 14.77),
    ("+MSf Mm", ["M2", "S2", "K1", "O1", "MSf", "Mm"], 27.55),
    ("+Ssa Sa", ["M2", "S2", "K1", "O1", "MSf", "Mm", "Ssa", "Sa"],
     365.26),
]


def omega_of(name, tau=None):
    r"""Dimensionless frequency :math:`2\pi\tau/P` of a constituent."""
    tau = REF.tau if tau is None else tau
    return 2.0 * np.pi * tau / (CONSTITUENTS[name][0] * 3600.0)


def sigma_of(name):
    """Head noise assigned to a constituent's frequency band, m."""
    return REF.sigma_lp if CONSTITUENTS[name][0] > 48.0 else REF.sigma_h
