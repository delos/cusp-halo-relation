'''
Physical-consistency tests for cusp_halo_relation.cuspEinasto.

These check the mathematical relationships that must hold between the profile
quantities (density, mass, potential, velocity dispersion, distribution
function) by comparing the module's output against independent high-accuracy
quadrature of the (exact, closed-form) density. Parameters span the
dimensionless cusp parameter y = A/(rho_s*r_s**1.5) from 0 (pure Einasto) to
its largest sensible value y_max(alpha), a range of Einasto shape parameters
alpha, and small phase-space cores z = r_c/r_s (we assume r_c << r_s).
'''

import numpy as np
import pytest
from scipy.integrate import quad, simpson
from scipy.special import gamma, gammainc

from cusp_halo_relation import cuspEinasto, cuspNFW

# the reference quad() calls cannot reach their default ~1e-8 tolerance on the
# cusped mass integrals, but stay far more accurate than the tolerances asserted
# below, so the resulting roundoff warnings are expected and silenced.
pytestmark = pytest.mark.filterwarnings(
    "ignore::scipy.integrate.IntegrationWarning")


# --- parameter grids --------------------------------------------------------

ALPHAS = [0.12, 0.17, 0.3]


def _ys(alpha):
  '''y values from pure Einasto (0) to the largest sensible cusp (y_max).'''
  ym = cuspEinasto.y_max(alpha)
  return [0.0, 1e-3, 0.5, 2.0, ym]


# core radii: r_c << r_s only (the profile is not built for large cores)
ZS = [0.0, 1e-3, 1e-2, 3e-2]

# (r_s, rho_s, G) sets that exercise the dimensional wrappers
SCALES = [(1.0, 1.0, 1.0), (2.5, 0.4, 4.301e-6)]

# radii in units of r_s
X_RADII = np.array([0.03, 0.3, 1.0, 3.0, 30.0])
X_RADII_DF = np.array([0.1, 1.0, 10.0])


def _A(y, r_s, rho_s):
  '''Cusp coefficient A reproducing dimensionless cusp parameter y.'''
  return y * rho_s * r_s ** 1.5


def _params_z0():
  '''(alpha, y, r_s, rho_s, G, A) cases with no core.'''
  for r_s, rho_s, G in SCALES:
    for alpha in ALPHAS:
      for y in _ys(alpha):
        yield alpha, y, r_s, rho_s, G, _A(y, r_s, rho_s)


def _params_zc():
  '''(alpha, y, z, r_s, rho_s, G, A) cases with and without a small core.'''
  for r_s, rho_s, G in SCALES:
    for alpha in ALPHAS:
      for z in ZS:
        ys = _ys(alpha)
        for y in (ys if z == 0.0 else [ys[0], ys[2], ys[-1]]):
          yield alpha, y, z, r_s, rho_s, G, _A(y, r_s, rho_s)


# --- 1. closed-form limits --------------------------------------------------

@pytest.mark.parametrize("alpha", ALPHAS)
def test_einasto_and_cusp_limits(alpha):
  r_s, rho_s = 2.5, 0.4
  r = r_s * np.geomspace(1e-2, 1e2, 20)
  # A = 0 is the Einasto profile, with its closed-form enclosed mass
  rho = cuspEinasto.density(r, r_s, rho_s, 0., alpha)
  rho_E = rho_s * np.exp(-2. / alpha * ((r / r_s) ** alpha - 1.))
  assert np.allclose(rho, rho_E, rtol=1e-12)
  n = 3. / alpha
  M_E = (4 * np.pi * rho_s * r_s ** 3 * np.exp(2. / alpha) / alpha
         * (alpha / 2.) ** n * gamma(n)
         * gammainc(n, 2. / alpha * (r / r_s) ** alpha))
  M = cuspEinasto.mass(r, r_s, rho_s, 0., alpha)
  assert np.allclose(M, M_E, rtol=1e-10)
  # r -> 0 is the bare A r^-1.5 cusp
  A = 1.3 * rho_s * r_s ** 1.5
  r0 = r_s * 1e-12
  assert np.isclose(cuspEinasto.density(r0, r_s, rho_s, A, alpha) * r0 ** 1.5,
                    A, rtol=1e-6)
  # at y_max the cusp and Einasto terms are equal at r_s
  ym = cuspEinasto.y_max(alpha)
  rho_rs = cuspEinasto.density(r_s, r_s, rho_s, _A(ym, r_s, rho_s), alpha)
  assert np.isclose(rho_rs, np.sqrt(2.) * rho_s, rtol=1e-12)


# --- 2. sanity --------------------------------------------------------------

@pytest.mark.parametrize("alpha,y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_density_sanity(alpha, y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  r = r_s * np.geomspace(1e-3, 1e3, 200)
  rho = cuspEinasto.density(r, r_s, rho_s, A, alpha, r_c=r_c)
  assert np.all(np.isfinite(rho)) and np.all(rho > 0.)
  assert np.all(np.diff(rho) < 0.)  # strictly decreasing
  # zero points / positivity of the derived quantities
  assert cuspEinasto.mass(0., r_s, rho_s, A, alpha, r_c=r_c) == 0.
  assert cuspEinasto.potential(0., r_s, rho_s, A, alpha, r_c=r_c, G=G) == 0.
  s2 = cuspEinasto.veldisp2_r(r, r_s, rho_s, A, alpha, r_c=r_c, G=G)
  assert np.all(np.isfinite(s2)) and np.all(s2 > 0.)
  # mass strictly increasing
  M = cuspEinasto.mass(r, r_s, rho_s, A, alpha, r_c=r_c)
  assert np.all(np.diff(M) > 0.)


# --- 3. density slope -------------------------------------------------------

@pytest.mark.parametrize("alpha,y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_density_slope(alpha, y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  r = r_s * np.geomspace(1e-4, 1e2, 50)
  h = 1e-5
  rp = cuspEinasto.density(r * np.exp(h), r_s, rho_s, A, alpha, r_c=r_c)
  rm = cuspEinasto.density(r * np.exp(-h), r_s, rho_s, A, alpha, r_c=r_c)
  slope = (np.log(rp) - np.log(rm)) / (2. * h)
  s = cuspEinasto.density_slope(r, r_s, rho_s, A, alpha, r_c=r_c)
  assert np.allclose(s, slope, atol=1e-6)


# --- 4. integral of density is mass -----------------------------------------

@pytest.mark.parametrize("alpha,y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_mass_is_density_integral(alpha, y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  for r in X_RADII * r_s:
    M_ref = quad(lambda rr: 4 * np.pi * rr ** 2
                 * cuspEinasto.density(rr, r_s, rho_s, A, alpha, r_c=r_c),
                 0., r, limit=200)[0]
    M = cuspEinasto.mass(r, r_s, rho_s, A, alpha, r_c=r_c)
    assert np.isclose(M, M_ref, rtol=1e-5), (alpha, y, z, r / r_s, M, M_ref)


# --- 5. integral of mass is potential ---------------------------------------

@pytest.mark.parametrize("alpha,y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_potential_is_mass_integral(alpha, y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  for r in X_RADII * r_s:
    Phi_ref = quad(lambda rr: G
                   * cuspEinasto.mass(rr, r_s, rho_s, A, alpha, r_c=r_c)
                   / rr ** 2, 0., r, limit=200)[0]
    Phi = cuspEinasto.potential(r, r_s, rho_s, A, alpha, r_c=r_c, G=G)
    assert np.isclose(Phi, Phi_ref, rtol=1e-5), (alpha, y, z, r / r_s,
                                                 Phi, Phi_ref)


# --- 6. potential at infinity / zero_at_inf ---------------------------------

@pytest.mark.parametrize("alpha,y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_potential_inf_and_zero_at_inf(alpha, y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  Phi_inf_ref = _phi_inf(r_s, rho_s, A, alpha, r_c, G)
  for r in X_RADII * r_s:
    Phi0 = cuspEinasto.potential(r, r_s, rho_s, A, alpha, r_c=r_c, G=G)
    Phii = cuspEinasto.potential(r, r_s, rho_s, A, alpha, r_c=r_c, G=G,
                                 zero_at_inf=True)
    # the offset between the two conventions is the potential at infinity
    assert np.isclose(Phi0 - Phii, Phi_inf_ref, rtol=1e-5), (alpha, y, z,
                                                             r / r_s)
  # potential -> 0 at large radius when zero_at_inf=True
  big = cuspEinasto.potential(1e6 * r_s, r_s, rho_s, A, alpha, r_c=r_c, G=G,
                              zero_at_inf=True)
  assert abs(big) < 1e-5 * abs(Phi_inf_ref)


# --- 7. Jeans equation (isotropic) ------------------------------------------

@pytest.mark.parametrize("alpha,y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_jeans_dispersion(alpha, y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  for r in X_RADII * r_s:
    lhs = (cuspEinasto.density(r, r_s, rho_s, A, alpha, r_c=r_c)
           * cuspEinasto.veldisp2_r(r, r_s, rho_s, A, alpha, r_c=r_c, G=G))
    rhs = quad(lambda rr: cuspEinasto.density(rr, r_s, rho_s, A, alpha,
                                              r_c=r_c)
               * G * cuspEinasto.mass(rr, r_s, rho_s, A, alpha, r_c=r_c)
               / rr ** 2, r, np.inf, limit=400)[0]
    assert np.isclose(lhs, rhs, rtol=1e-4), (alpha, y, z, r / r_s, lhs, rhs)
    # veldisp_r is its square root
    s = cuspEinasto.veldisp_r(r, r_s, rho_s, A, alpha, r_c=r_c, G=G)
    rho = cuspEinasto.density(r, r_s, rho_s, A, alpha, r_c=r_c)
    assert np.isclose(s ** 2, lhs / rho, rtol=1e-12)


# --- distribution function helpers ------------------------------------------

def _phi_inf(r_s, rho_s, A, alpha, r_c, G):
  return quad(lambda rr: G
              * cuspEinasto.mass(rr, r_s, rho_s, A, alpha, r_c=r_c)
              / rr ** 2, 0., np.inf, limit=400)[0]


def _df_velocity_moment(r, power, r_s, rho_s, A, alpha, r_c, G, nv=600):
  '''
  4*pi * integral_0^{v_esc} f(Phi(r)+v^2/2) v^power dv, with G=1 so the speed
  v = sqrt(2*(E-Phi)) directly. power=2 recovers density; power=4 recovers
  3*rho*sigma_r^2 (=rho*<v^2>).
  '''
  Phi = cuspEinasto.potential(r, r_s, rho_s, A, alpha, r_c=r_c, G=G)
  Phi_inf = _phi_inf(r_s, rho_s, A, alpha, r_c, G)
  v_esc = np.sqrt(2. * (Phi_inf - Phi))
  v = np.linspace(0., v_esc, nv)
  E = Phi + 0.5 * v ** 2
  f = cuspEinasto.df(E, r_s, rho_s, A, alpha, r_c=r_c, G=G)
  f = np.where(np.isfinite(f), f, 0.)
  return 4. * np.pi * simpson(f * v ** power, x=v)


def _df_ys():
  for alpha in [0.12, 0.3]:
    for y in [0.0, 0.5, cuspEinasto.y_max(alpha)]:
      yield alpha, y


# --- 8. integral of df over velocities is density ---------------------------

@pytest.mark.parametrize("z", [0.0, 1e-2])
@pytest.mark.parametrize("alpha,y", list(_df_ys()))
@pytest.mark.parametrize("r_s,rho_s,G", [(1.0, 1.0, 1.0)])
def test_df_recovers_density(alpha, y, z, r_s, rho_s, G):
  A = _A(y, r_s, rho_s)
  r_c = z * r_s
  for r in X_RADII_DF * r_s:
    rho_ref = cuspEinasto.density(r, r_s, rho_s, A, alpha, r_c=r_c)
    rho_df = _df_velocity_moment(r, 2, r_s, rho_s, A, alpha, r_c, G)
    assert np.isclose(rho_df, rho_ref, rtol=1e-3), (alpha, y, z, r / r_s,
                                                    rho_df, rho_ref)


# --- 9. integral of df * v^2 is density * dispersion^2 ----------------------

@pytest.mark.parametrize("z", [0.0, 1e-2])
@pytest.mark.parametrize("alpha,y", list(_df_ys()))
@pytest.mark.parametrize("r_s,rho_s,G", [(1.0, 1.0, 1.0)])
def test_df_recovers_dispersion(alpha, y, z, r_s, rho_s, G):
  A = _A(y, r_s, rho_s)
  r_c = z * r_s
  for r in X_RADII_DF * r_s:
    lhs = (cuspEinasto.density(r, r_s, rho_s, A, alpha, r_c=r_c)
           * cuspEinasto.veldisp2_r(r, r_s, rho_s, A, alpha, r_c=r_c, G=G))
    # rho*<v^2> = 4pi integral f v^4 dv, and <v^2> = 3 sigma_r^2 (isotropic)
    rhs = _df_velocity_moment(r, 4, r_s, rho_s, A, alpha, r_c, G) / 3.
    assert np.isclose(lhs, rhs, rtol=1e-3), (alpha, y, z, r / r_s, lhs, rhs)


# --- 10. df positivity, monotonicity and energy zero points -----------------

@pytest.mark.parametrize("alpha,y", list(_df_ys()))
def test_df_positive_decreasing_and_zero_points(alpha, y):
  r_s, rho_s, G = 2.5, 0.4, 4.301e-6
  A = _A(y, r_s, rho_s)
  Phi_inf = (cuspEinasto.potential(1., r_s, rho_s, A, alpha, G=G)
             - cuspEinasto.potential(1., r_s, rho_s, A, alpha, G=G,
                                     zero_at_inf=True))
  # log-spaced toward both the centre and Phi_inf
  t = np.concatenate([np.geomspace(1e-6, 0.5, 100),
                      1 - np.geomspace(0.5, 1e-6, 100)[1:]])
  E = t * Phi_inf
  f0 = cuspEinasto.df(E, r_s, rho_s, A, alpha, G=G)
  assert np.all(f0 > 0.)
  assert np.all(np.diff(f0) < 0.)
  # f(E) and f(E-Phi_inf, zero_at_inf=True) are the same function
  fi = cuspEinasto.df(E - Phi_inf, r_s, rho_s, A, alpha, G=G, zero_at_inf=True)
  assert fi == pytest.approx(f0, rel=1e-6, abs=0)
  # unbound energies have f = 0
  assert np.all(cuspEinasto.df(np.array([-0.1, 1.1]) * Phi_inf, r_s, rho_s, A,
                               alpha, G=G) == 0.)


@pytest.mark.parametrize("z", [1e-3, 1e-1])
@pytest.mark.parametrize("alpha,y", list(_df_ys()))
def test_df_smooth_in_core(alpha, y, z):
  r_s, rho_s, G = 2.5, 0.4, 4.301e-6
  A, r_c = _A(y, r_s, rho_s), z * r_s
  # in a core f(E) levels off at the bottom of the potential; it must stay
  # positive and smooth there, down to E - Phi(0) = 1e-8 of the depth
  depth = -cuspEinasto.potential(0., r_s, rho_s, A, alpha, r_c=r_c, G=G,
                                 zero_at_inf=True)
  E = depth * np.geomspace(1e-8, 1e-2, 200)
  f = cuspEinasto.df(E, r_s, rho_s, A, alpha, r_c=r_c, G=G)
  assert np.all(f > 0.)
  assert np.max(np.abs(np.diff(np.log(f), 2))) < 1e-2


# --- 11. r_{-2} round-trip and density slope --------------------------------

@pytest.mark.parametrize("alpha,y,r_s,rho_s,G,A", list(_params_z0()))
def test_r2_roundtrip_and_slope(alpha, y, r_s, rho_s, G, A):
  r_2 = cuspEinasto.r2_from_rs(r_s, rho_s, A, alpha)
  assert r_2 <= r_s * (1 + 1e-12)
  r_s_back = cuspEinasto.rs_from_r2(r_2, rho_s, A, alpha)
  assert np.isclose(r_s_back, r_s, rtol=1e-10)
  # dln(rho)/dln(r) = -2 at r_2
  h = 1e-4
  rp = cuspEinasto.density(r_2 * np.exp(h), r_s, rho_s, A, alpha)
  rm = cuspEinasto.density(r_2 * np.exp(-h), r_s, rho_s, A, alpha)
  slope = (np.log(rp) - np.log(rm)) / (2. * h)
  assert np.isclose(slope, -2.0, atol=1e-4)


# --- 12. scale_from_c reproduces mass and concentration ---------------------

@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("c", [4.0, 10.0, 25.0])
@pytest.mark.parametrize("M,rho_vir", [(1e12, 200.0), (1.0, 1.0)])
@pytest.mark.parametrize("Afrac", [0.0, 0.3, 0.9, 0.999])
def test_scale_from_c(alpha, c, M, rho_vir, Afrac):
  A = Afrac * cuspEinasto.A_max(c, M, rho_vir, alpha)
  r_s, rho_s = cuspEinasto.scale_from_c(c, M, A, rho_vir, alpha)
  R = cuspEinasto.R_from_M(M, rho_vir)
  assert np.isclose(cuspEinasto.mass(R, r_s, rho_s, A, alpha), M, rtol=1e-6)
  assert np.isclose(cuspEinasto.r2_from_rs(r_s, rho_s, A, alpha), R / c,
                    rtol=1e-6)
  assert A / (rho_s * r_s ** 1.5) <= cuspEinasto.y_max(alpha) * (1 + 1e-9)


# --- 13. minimum-parameter helper consistency -------------------------------

@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("c", [0.5, 4.0, 25.0])
@pytest.mark.parametrize("M,rho_vir", [(1e12, 200.0), (1.0, 1.0)])
def test_min_param_consistency(alpha, c, M, rho_vir):
  Amax = cuspEinasto.A_max(c, M, rho_vir, alpha)
  # A_max <-> M_min are mutual inverses
  assert np.isclose(cuspEinasto.M_min(c, Amax, rho_vir, alpha), M, rtol=1e-12)
  # A_max is the largest A on the curve of halos with this (c, M): a scan of
  # y = A/(rho_s*r_s**1.5) at fixed (c, M) never exceeds it
  R = cuspEinasto.R_from_M(M, rho_vir)
  for y in cuspEinasto.y_max(alpha) * np.geomspace(1e-2, 1., 15):
    # unit-y halo with this concentration, rescaled to mass M
    u = cuspEinasto.r2_from_rs(1., 1., y, alpha)
    m = cuspEinasto.mass(c * u, 1., 1., y, alpha)
    r_s = R / (c * u)
    rho_s = M / (m * r_s ** 3)
    assert y * rho_s * r_s ** 1.5 <= Amax * (1 + 1e-7)
  # c_min inverts A_max: for A just below A_max(c), A_max(c_min) == A
  A = 0.95 * Amax
  cm = cuspEinasto.c_min(M, A, rho_vir, alpha)
  assert 0. < cm < c
  assert np.isclose(cuspEinasto.A_max(cm, M, rho_vir, alpha), A, rtol=1e-6)
  # above A_max the profile is invalid: raises by default, and with
  # cmin_error=False it falls back to the halo at c_min
  A_bad = 1.01 * Amax
  with pytest.raises(Exception):
    cuspEinasto.scale_from_c(c, M, A_bad, rho_vir, alpha)
  r_s, rho_s = cuspEinasto.scale_from_c(c, M, A_bad, rho_vir, alpha,
                                        cmin_error=False)
  cm = cuspEinasto.c_min(M, A_bad, rho_vir, alpha)
  assert np.isclose(cuspEinasto.mass(R, r_s, rho_s, A_bad, alpha), M,
                    rtol=1e-6)
  assert np.isclose(cuspEinasto.r2_from_rs(r_s, rho_s, A_bad, alpha), R / cm,
                    rtol=1e-6)


# --- 14. phase-space core radius --------------------------------------------

@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("y", [0.0, 0.5, 2.0])
def test_rc_from_fmax(alpha, y):
  r_s, rho_s, G = 2.5, 0.4, 4.301e-6
  A = _A(y, r_s, rho_s)
  prefactor = 3e-5
  for rc_target in [1e-4, 1e-2]:
    # f_max for which the phase-space barrier rho r^6 = prefactor/(G^3 f^2)
    # sits at rc_target, then invert
    rho = cuspEinasto.density(rc_target * r_s, r_s, rho_s, A, alpha)
    fmax = np.sqrt(prefactor / (G ** 3 * rho * (rc_target * r_s) ** 6))
    r_c = cuspEinasto.rc_from_fmax(fmax, r_s, rho_s, A, alpha, G=G,
                                   prefactor=prefactor)
    assert np.isclose(r_c, rc_target * r_s, rtol=1e-8)


# --- 15. phase-space core ---------------------------------------------------

@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("y", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("z", [1e-3, 1e-2])
def test_core(alpha, y, z):
  r_s, rho_s = 2.5, 0.4
  A, r_c = _A(y, r_s, rho_s), z * r_s
  # the core leaves the profile intact outside r_c and flattens it inside, to
  # the uncored density at radius r_c
  rho0 = cuspEinasto.density(0., r_s, rho_s, A, alpha, r_c=r_c)
  assert np.isclose(rho0, cuspEinasto.density(r_c, r_s, rho_s, A, alpha),
                    rtol=1e-12)
  r0 = 1e-6 * r_c
  assert np.isclose(cuspEinasto.density(r0, r_s, rho_s, A, alpha, r_c=r_c),
                    rho0, rtol=1e-9)
  assert abs(cuspEinasto.density_slope(r0, r_s, rho_s, A, alpha,
                                       r_c=r_c)) < 1e-9
  r = r_c * np.array([1e2, 1e3])
  assert np.allclose(cuspEinasto.density(r, r_s, rho_s, A, alpha, r_c=r_c),
                     cuspEinasto.density(r, r_s, rho_s, A, alpha), rtol=1e-3)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("z", [1e-6, 1e-5])
def test_core_matches_cuspNFW(alpha, z):
  # the core enters exactly as in cusp-NFW, so where both profiles are the
  # bare cusp A r^-1.5 (r, r_c << r_s), the cored-to-uncored ratios agree
  r_s, rho_s, y = 2.5, 0.4, 2.0
  A, r_c = _A(y, r_s, rho_s), z * r_s
  rhosN = A / (0.5 * r_s ** 1.5)  # a cusp-NFW halo with the same A, y = 0.5
  r = r_c * np.geomspace(1e-3, 10., 30)
  # the cusp dominates both profiles here, by a factor of 25 or more
  assert np.all(_A(y, r_s, rho_s) * r ** -1.5
                > 25 * cuspEinasto.density(r, r_s, rho_s, 0., alpha))
  assert np.all(A * r ** -1.5 > 25 * cuspNFW.density(r, r_s, rhosN, 0.))
  qE = (cuspEinasto.density(r, r_s, rho_s, A, alpha, r_c=r_c)
        / cuspEinasto.density(r, r_s, rho_s, A, alpha))
  qN = (cuspNFW.density(r, r_s, rhosN, A, r_c=r_c)
        / cuspNFW.density(r, r_s, rhosN, A))
  assert np.allclose(qE, qN, rtol=2e-3)


# --- 16. scale_from_c branches ----------------------------------------------

def _A_on_curve(c, M, rho_vir, alpha, y):
  '''Cusp coefficient A of the halo with concentration c, mass M and this y.'''
  R = cuspEinasto.R_from_M(M, rho_vir)
  u = cuspEinasto.r2_from_rs(1., 1., y, alpha)
  r_s = R / (c * u)
  rho_s = M / (cuspEinasto.mass(c * u, 1., 1., y, alpha) * r_s ** 3)
  return y * rho_s * r_s ** 1.5


@pytest.mark.parametrize("alpha,c", [(0.3, 25.0), (0.3, 0.5), (0.17, 300.0),
                                     (0.17, 60.0)])
def test_scale_from_c_branches(alpha, c):
  # both branches solve the same two conditions and the high one has the
  # larger y. A second solution exists only for A between the y = y_max value
  # and A_max, a range that is empty if A_max is reached at y_max (as for
  # alpha = 0.17, c = 60); where there is one, the two settings must agree.
  M, rho_vir = 1.0, 1.0
  R = cuspEinasto.R_from_M(M, rho_vir)
  Amax = cuspEinasto.A_max(c, M, rho_vir, alpha)
  ym = cuspEinasto.y_max(alpha)
  A_ym = _A_on_curve(c, M, rho_vir, alpha, ym)
  for f in (0.5, 0.9, 0.99, 0.995, 0.999):
    A = f * Amax
    lo = cuspEinasto.scale_from_c(c, M, A, rho_vir, alpha)
    hi = cuspEinasto.scale_from_c(c, M, A, rho_vir, alpha, branch="high")
    for r_s, rho_s in (lo, hi):
      assert np.isclose(cuspEinasto.mass(R, r_s, rho_s, A, alpha), M,
                        rtol=1e-8)
      assert np.isclose(cuspEinasto.r2_from_rs(r_s, rho_s, A, alpha), R / c,
                        rtol=1e-8)
    y_lo, y_hi = A / (lo[1] * lo[0] ** 1.5), A / (hi[1] * hi[0] ** 1.5)
    assert y_hi <= ym * (1 + 1e-9)
    if A > A_ym * (1 + 1e-6):  # two halos
      assert y_hi > y_lo * (1 + 1e-3)
    else:
      assert lo == pytest.approx(hi, rel=1e-8)
  # the two merge as A -> A_max
  A = (1 - 1e-9) * Amax
  lo = cuspEinasto.scale_from_c(c, M, A, rho_vir, alpha)
  hi = cuspEinasto.scale_from_c(c, M, A, rho_vir, alpha, branch="high")
  assert lo == pytest.approx(hi, rel=1e-3)


# --- 17. broadcasting -------------------------------------------------------

@pytest.mark.parametrize("fn", ["density", "density_slope", "mass",
                                "potential", "veldisp2_r", "veldisp_r", "df"])
def test_broadcasting(fn):
  # r (or E) broadcasts against r_s, rho_s and A by the ufunc rules: equal
  # shapes pair elementwise, r[:,None] with A[None,:] gives the grid, and
  # scalars in give a scalar out
  f = getattr(cuspEinasto, fn)
  alpha, r_s, rho_s, G = 0.17, 2.5, 0.4, 4.301e-6
  A = np.array([0., 0.5, 2., cuspEinasto.y_max(alpha)]) * rho_s * r_s ** 1.5
  kw = {"G": G} if fn in ("potential", "veldisp2_r", "veldisp_r", "df") else {}
  if fn == "df":
    # an energy bound for every A, since Phi_inf grows with A
    Phi_inf = (cuspEinasto.potential(1., r_s, rho_s, 0., alpha, G=G)
               - cuspEinasto.potential(1., r_s, rho_s, 0., alpha, G=G,
                                       zero_at_inf=True))
    first = np.array([0.05, 0.2, 0.5, 0.9]) * Phi_inf
  else:
    first = r_s * np.array([0.03, 0.3, 3., 30.])
  assert np.shape(f(first, r_s, rho_s, A[0], alpha, **kw)) == (4,)
  assert np.shape(f(first[0], r_s, rho_s, A, alpha, **kw)) == (4,)
  assert np.shape(f(first, r_s, rho_s, A, alpha, **kw)) == (4,)
  assert np.ndim(f(first[0], r_s, rho_s, A[0], alpha, **kw)) == 0
  grid = f(first[:, None], r_s, rho_s, A[None, :], alpha, **kw)
  assert np.shape(grid) == (4, 4)
  loop = np.array([[f(t, r_s, rho_s, a, alpha, **kw) for a in A]
                   for t in first])
  assert grid == pytest.approx(loop, rel=1e-9, abs=0)


# --- 18. weak cusps, weakly bound energies, pure-Einasto potential ----------

@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("c", [0.5, 10.0, 300.0])
@pytest.mark.parametrize("Afrac", [1e-6, 1e-13, 1e-20])
def test_scale_from_c_weak_cusp(alpha, c, Afrac):
  # a vanishing cusp solves the same conditions and tends to the A = 0 halo
  M, rho_vir = 1e15, 200.0
  R = cuspEinasto.R_from_M(M, rho_vir)
  A = Afrac * cuspEinasto.A_max(c, M, rho_vir, alpha)
  r_s, rho_s = cuspEinasto.scale_from_c(c, M, A, rho_vir, alpha)
  assert np.isclose(cuspEinasto.mass(R, r_s, rho_s, A, alpha), M, rtol=1e-8)
  assert np.isclose(cuspEinasto.r2_from_rs(r_s, rho_s, A, alpha), R / c,
                    rtol=1e-8)
  # the y = 0 halo uses the closed-form mass, so agreement floors at the
  # accuracy of the tabulated one
  r_s0, rho_s0 = cuspEinasto.scale_from_c(c, M, 0., rho_vir, alpha)
  assert (r_s, rho_s) == pytest.approx((r_s0, rho_s0),
                                       rel=max(10 * Afrac ** 0.5, 1e-7))


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("y", [0.0, 0.5])
def test_df_weakly_bound(alpha, y):
  # with the zero point at infinity, small binding energies are resolved: f
  # matches the zero-at-centre value where both resolve it, and keeps falling
  # (rather than freezing at a rounding error) beyond that
  A = _A(y, 1., 1.)
  Phi_inf = (cuspEinasto.potential(1., 1., 1., A, alpha)
             - cuspEinasto.potential(1., 1., 1., A, alpha, zero_at_inf=True))
  ecal = Phi_inf * np.array([1e-2, 1e-4, 1e-6])
  fi = cuspEinasto.df(-ecal, 1., 1., A, alpha, zero_at_inf=True)
  f0 = cuspEinasto.df(Phi_inf - ecal, 1., 1., A, alpha)
  assert fi == pytest.approx(f0, rel=1e-3)
  # f falls off as exp(-C Ecal**-alpha) (the profile is exponentially
  # truncated), so dln f/dln Ecal keeps steepening down to underflow
  ecal = Phi_inf * np.geomspace(1e-3, 1e-9, 13)
  fi = cuspEinasto.df(-ecal, 1., 1., A, alpha, zero_at_inf=True)
  lf = np.log(fi[fi > 1e-290])
  assert len(lf) >= 6
  assert np.all(np.diff(np.diff(lf)) < 0.)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_pure_einasto_potential_closed_form(alpha):
  # the y = 0 closed form agrees with the numerical profile as y -> 0, in both
  # zero points, and its two zero points differ by exactly Phi_inf
  x = np.geomspace(1e-6, 1e4, 41)
  for zi in (False, True):
    p0 = cuspEinasto.potential(x, 1., 1., 0., alpha, zero_at_inf=zi)
    pn = cuspEinasto.potential(x, 1., 1., 1e-12, alpha, zero_at_inf=zi)
    assert p0 == pytest.approx(pn, rel=1e-6)  # accuracy of the table
  Phi_inf = (cuspEinasto.potential(x, 1., 1., 0., alpha)
             - cuspEinasto.potential(x, 1., 1., 0., alpha, zero_at_inf=True))
  assert Phi_inf == pytest.approx(Phi_inf[0], rel=1e-12)
