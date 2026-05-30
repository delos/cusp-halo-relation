import numpy as np
from scipy.integrate import simpson, solve_ivp

from cusp_halo_relation.mpk_helpers.perturbations import W


def test_main_progenitor_identity(eps):
    # zero time offset returns the starting mass
    for M0 in [1e2, 1e3, 1e4]:
        assert np.isclose(eps.main_progenitor(M0, 0.0), M0, rtol=1e-5)


def test_main_progenitor_monotonic(eps):
    M0 = 1e3
    dw = np.linspace(0.0, 5.0, 25)
    Mp = np.array([eps.main_progenitor(M0, w) for w in dw])
    # progenitor mass strictly decreases going back in time
    assert np.all(np.diff(Mp) < 0)
    assert np.all(Mp <= M0 * (1 + 1e-6))


def test_main_progenitor_freezes(eps):
    # at large enough time offset the progenitor freezes at the resolved floor
    M0 = 1e3
    big = (eps._Omega_hi - eps._Omega_lo) + 100.0
    a = eps.main_progenitor(M0, big)
    b = eps.main_progenitor(M0, 2 * big)
    assert np.isclose(a, b, rtol=1e-9)
    assert np.isclose(a, eps._M_lo, rtol=1e-6)


def test_autonomous_flow_matches_direct_ode(eps):
    # the Omega-flow must reproduce a direct integration of dlnM/domega=-rate(M)
    M0 = 1e3
    w = np.linspace(0.0, 4.0, 9)
    sol = solve_ivp(lambda t, y: [-eps._rate_of_M(np.exp(y[0]))],
                    [0.0, 4.0], [np.log(M0)], t_eval=w,
                    rtol=1e-9, atol=1e-12, max_step=0.05)
    M_ode = np.exp(sol.y[0])
    M_flow = np.array([eps.main_progenitor(M0, wi) for wi in w])
    assert np.allclose(M_ode, M_flow, rtol=1e-3)


def test_sigma_decreasing(eps):
    M = np.geomspace(1e1, 1e5, 50)
    s = eps.sigma(M)
    assert np.all(np.diff(s) < 0)


def test_sigma_matches_quadrature(eps, spectrum):
    # sigma(M) must match a direct top-hat variance quadrature
    k, P = spectrum
    rho = 1.0
    for M in [1e2, 1e3, 1e4]:
        R = (3 * M / (4 * np.pi * rho)) ** (1.0 / 3.0)
        s_direct = np.sqrt(simpson(P * W(k * R) ** 2, x=np.log(k)))
        assert np.isclose(eps.sigma(M), s_direct, rtol=2e-3)


def test_M_nonlinear_inverts_sigma(eps):
    # M_nonlinear(omega) solves sigma(M*) = omega
    for omega in [1.0, 1.686, 2.5]:
        Mstar = eps.M_nonlinear(omega)
        assert np.isclose(eps.sigma(Mstar), omega, rtol=2e-3)
