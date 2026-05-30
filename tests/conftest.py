import numpy as np
import pytest

from cusp_halo_relation.eps_growth import EPSGrowth


def make_spectrum(n=-2.0, kcut=1.0, sig0=5.0, nk=2000):
    '''Scale-free matter spectrum with a Gaussian cutoff, in the dimensionless
    Delta^2 convention used by CuspHalo (sigma_j^2 = integral P k^(2j) dlnk).
    Amplitude is scaled so the unsmoothed sigma_0 at a=1 equals sig0.'''
    k = np.geomspace(1e-3, 1e2, nk)
    P = k ** (3 + n) * np.exp(-(k / kcut) ** 2)
    P *= (sig0 / np.sqrt(np.trapezoid(P, np.log(k)))) ** 2
    return k, P


@pytest.fixture(scope="session")
def spectrum():
    return make_spectrum()


@pytest.fixture(scope="session")
def eps(spectrum):
    k, P = spectrum
    return EPSGrowth(k, P, 1.0, delta_c=1.686)
