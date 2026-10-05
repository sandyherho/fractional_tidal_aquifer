"""Least-squares harmonic analysis of head records."""

import numpy as np

__all__ = ["harmonic_fit", "transfer_rates"]


def harmonic_fit(t, y, omegas, trend=True):
    r"""Fit a mean, a trend and harmonics to a record.

    The model is :math:`y \approx c_0 + c_1 t + \mathrm{Re}\sum_k Z_k
    e^{i\omega_k t}`.

    Returns the complex amplitudes :math:`Z_k` (one per frequency) and the
    root-mean-square residual.  ``y`` may be two-dimensional with time
    along the first axis.
    """
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t)]
    if trend:
        cols.append(t - t.mean())
    for om in omegas:
        cols += [np.cos(om * t), np.sin(om * t)]
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    off = 2 if trend else 1
    A = coef[off::2]
    B = coef[off + 1::2]
    Z = A - 1j * B
    resid = y - X @ coef
    return Z, float(np.sqrt(np.mean(resid**2)))


def transfer_rates(z_well, z_coast, x):
    r"""Attenuation and phase-lag rates from complex amplitudes at distance x.

    :math:`Z_{well}/Z_{coast} = e^{-(a + ib)x}`, so
    :math:`a = -\ln|\cdot|/x` and :math:`b = -\arg(\cdot)/x`
    (the phase is assumed not to wrap, i.e. :math:`bx < \pi`).
    """
    q = np.asarray(z_well) / np.asarray(z_coast)
    return -np.log(np.abs(q)) / x, -np.angle(q) / x
