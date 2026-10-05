"""pasang: tidal propagation through coastal aquifers with storage memory.

Modules
-------
capacity      matrix transfer functions and complex storage capacity
propagation   attenuation and phase-lag rates, ratio, local order
bounds        bounds on the tidal ratio from passive storage memory
kk            Kramers-Kronig consistency check
dualporosity  time-domain dual-porosity solver with resolved matrix
caputo        L1 solver and exact solutions of the time-fractional problem
boussinesq    nonlinear unconfined aquifer with matrix memory
harmonic      least-squares harmonic analysis
information   Fisher information and detectability of memory
scenario      reference parameters and tidal constituents
plotting, anim, io_utils   figure style, GIF export, CSV and reports
"""

__version__ = "0.1.0"
