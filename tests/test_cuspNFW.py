'''
Physical-consistency tests for cusp_halo_relation.cuspNFW.

These check the mathematical relationships that must hold between the profile
quantities (density, mass, potential, velocity dispersion, distribution
function) by comparing the module's output against independent high-accuracy
quadrature of the (exact, closed-form) density. Parameters span the validity
edges of the dimensionless cusp parameter y = A/(rho_s*r_s**1.5) in [0,1] and
straddle every internal branch boundary, plus small phase-space cores
z = r_c/r_s (we assume r_c << r_s).
'''

import numpy as np
import pytest
from scipy.integrate import quad, simpson

from cusp_halo_relation import cuspNFW

# the reference quad() calls cannot reach their default ~1e-8 tolerance on the
# cusped mass integrals, but stay far more accurate than the tolerances asserted
# below, so the resulting roundoff warnings are expected and silenced.
pytestmark = pytest.mark.filterwarnings(
    "ignore::scipy.integrate.IntegrationWarning")


# --- parameter grids --------------------------------------------------------

# y values straddling the internal branch boundaries 0.0001, 0.001, 0.999,
# 0.9999 and including the edges 0 (pure NFW) and 1 (steepest allowed cusp).
YS_FULL = [0.0, 8e-5, 1.2e-4, 8e-4, 1.2e-3, 0.5,
           0.9988, 0.9992, 0.99988, 0.99992, 1.0]

# core radii: r_c << r_s only (the profile is not built for large cores)
ZS = [0.0, 1e-3, 1e-2, 3e-2]

# for z>0 the closed-form y-branches are bypassed (mass/potential go numerical),
# so a coarser y grid suffices
YS_CORE = [0.0, 0.5, 0.99, 1.0]

# (r_s, rho_s, G) sets that exercise the dimensional wrappers
SCALES = [(1.0, 1.0, 1.0), (2.5, 0.4, 4.301e-6)]

# radii in units of r_s
X_RADII = np.array([0.03, 0.3, 1.0, 3.0, 30.0])
X_RADII_DF = np.array([0.1, 1.0, 10.0])


def _A(y, r_s, rho_s):
  '''Cusp coefficient A reproducing dimensionless cusp parameter y.'''
  return y * rho_s * r_s ** 1.5


def _params_z0():
  '''(y, r_s, rho_s, G, A) cases with no core.'''
  for r_s, rho_s, G in SCALES:
    for y in YS_FULL:
      yield y, r_s, rho_s, G, _A(y, r_s, rho_s)


def _params_zc():
  '''(y, z, r_s, rho_s, G, A) cases with a small core.'''
  for r_s, rho_s, G in SCALES:
    for z in ZS:
      for y in (YS_FULL if z == 0.0 else YS_CORE):
        yield y, z, r_s, rho_s, G, _A(y, r_s, rho_s)


# --- 1. sanity --------------------------------------------------------------

@pytest.mark.parametrize("y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_density_sanity(y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  r = r_s * np.geomspace(1e-3, 1e3, 200)
  rho = cuspNFW.density(r, r_s, rho_s, A, r_c=r_c)
  assert np.all(np.isfinite(rho)) and np.all(rho > 0.)
  assert np.all(np.diff(rho) < 0.)  # strictly decreasing
  # zero points / positivity of the derived quantities
  assert cuspNFW.mass(0., r_s, rho_s, A, r_c=r_c) == 0.
  assert cuspNFW.potential(0., r_s, rho_s, A, r_c=r_c, G=G) == 0.
  s2 = cuspNFW.veldisp2_r(r, r_s, rho_s, A, r_c=r_c, G=G)
  assert np.all(np.isfinite(s2)) and np.all(s2 > 0.)
  # mass strictly increasing
  M = cuspNFW.mass(r, r_s, rho_s, A, r_c=r_c)
  assert np.all(np.diff(M) > 0.)


# --- 2. integral of density is mass -----------------------------------------

@pytest.mark.parametrize("y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_mass_is_density_integral(y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  rtol = 1e-4 if z == 0.0 else 2e-3
  for r in X_RADII * r_s:
    M_ref = quad(lambda rr: 4 * np.pi * rr ** 2
                 * cuspNFW.density(rr, r_s, rho_s, A, r_c=r_c),
                 0., r, limit=200)[0]
    M = cuspNFW.mass(r, r_s, rho_s, A, r_c=r_c)
    assert np.isclose(M, M_ref, rtol=rtol), (y, z, r / r_s, M, M_ref)


# --- 3. integral of mass is potential ---------------------------------------

@pytest.mark.parametrize("y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_potential_is_mass_integral(y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  for r in X_RADII * r_s:
    Phi_ref = quad(lambda rr: G * cuspNFW.mass(rr, r_s, rho_s, A, r_c=r_c)
                   / rr ** 2, 0., r, limit=200)[0]
    Phi = cuspNFW.potential(r, r_s, rho_s, A, r_c=r_c, G=G)
    assert np.isclose(Phi, Phi_ref, rtol=2e-3), (y, z, r / r_s, Phi, Phi_ref)


# --- 4. potential at infinity / zero_at_inf ---------------------------------

@pytest.mark.parametrize("y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_potential_inf_and_zero_at_inf(y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  Phi_inf_ref = quad(lambda rr: G * cuspNFW.mass(rr, r_s, rho_s, A, r_c=r_c)
                     / rr ** 2, 0., np.inf, limit=400)[0]
  for r in X_RADII * r_s:
    Phi0 = cuspNFW.potential(r, r_s, rho_s, A, r_c=r_c, G=G)
    Phii = cuspNFW.potential(r, r_s, rho_s, A, r_c=r_c, G=G, zero_at_inf=True)
    # the offset between the two conventions is the potential at infinity
    assert np.isclose(Phi0 - Phii, Phi_inf_ref, rtol=2e-3), (y, z, r / r_s)
  # potential -> 0 at large radius when zero_at_inf=True
  big = cuspNFW.potential(1e6 * r_s, r_s, rho_s, A, r_c=r_c, G=G,
                          zero_at_inf=True)
  assert abs(big) < 1e-3 * abs(Phi_inf_ref)


# --- 5. Jeans equation (isotropic) ------------------------------------------

@pytest.mark.parametrize("y,z,r_s,rho_s,G,A", list(_params_zc()))
def test_jeans_dispersion(y, z, r_s, rho_s, G, A):
  r_c = z * r_s
  for r in X_RADII * r_s:
    lhs = (cuspNFW.density(r, r_s, rho_s, A, r_c=r_c)
           * cuspNFW.veldisp2_r(r, r_s, rho_s, A, r_c=r_c, G=G))
    rhs = quad(lambda rr: cuspNFW.density(rr, r_s, rho_s, A, r_c=r_c)
               * G * cuspNFW.mass(rr, r_s, rho_s, A, r_c=r_c) / rr ** 2,
               r, np.inf, limit=400)[0]
    assert np.isclose(lhs, rhs, rtol=3e-3), (y, z, r / r_s, lhs, rhs)


# --- distribution function helpers ------------------------------------------

def _phi_inf(r_s, rho_s, A, r_c, G):
  return quad(lambda rr: G * cuspNFW.mass(rr, r_s, rho_s, A, r_c=r_c)
              / rr ** 2, 0., np.inf, limit=400)[0]


def _df_velocity_moment(r, power, r_s, rho_s, A, r_c, G, nv=600):
  '''
  4*pi * integral_0^{v_esc} f(Phi(r)+v^2/2) v^power dv, with G=1 so the speed
  v = sqrt(2*(E-Phi)) directly. power=2 recovers density; power=4 recovers
  3*rho*sigma_r^2 (=rho*<v^2>).
  '''
  Phi = cuspNFW.potential(r, r_s, rho_s, A, r_c=r_c, G=G)
  Phi_inf = _phi_inf(r_s, rho_s, A, r_c, G)
  v_esc = np.sqrt(2. * (Phi_inf - Phi))
  v = np.linspace(0., v_esc, nv)
  E = Phi + 0.5 * v ** 2
  f = cuspNFW.df(E, r_s, rho_s, A, r_c=r_c, G=G)
  f = np.where(np.isfinite(f), f, 0.)
  return 4. * np.pi * simpson(f * v ** power, x=v)


# --- 6. integral of df over velocities is density ---------------------------

@pytest.mark.parametrize("z", [0.0, 1e-2])
@pytest.mark.parametrize("y", [0.0, 0.5, 0.99, 1.0])
@pytest.mark.parametrize("r_s,rho_s,G", [(1.0, 1.0, 1.0)])
def test_df_recovers_density(y, z, r_s, rho_s, G):
  A = _A(y, r_s, rho_s)
  r_c = z * r_s
  for r in X_RADII_DF * r_s:
    rho_ref = cuspNFW.density(r, r_s, rho_s, A, r_c=r_c)
    rho_df = _df_velocity_moment(r, 2, r_s, rho_s, A, r_c, G)
    assert np.isclose(rho_df, rho_ref, rtol=1e-2), (y, z, r / r_s,
                                                    rho_df, rho_ref)


# --- 7. integral of df * v^2 is density * dispersion^2 ----------------------

@pytest.mark.parametrize("z", [0.0, 1e-2])
@pytest.mark.parametrize("y", [0.0, 0.5, 0.99, 1.0])
@pytest.mark.parametrize("r_s,rho_s,G", [(1.0, 1.0, 1.0)])
def test_df_recovers_dispersion(y, z, r_s, rho_s, G):
  A = _A(y, r_s, rho_s)
  r_c = z * r_s
  for r in X_RADII_DF * r_s:
    lhs = (cuspNFW.density(r, r_s, rho_s, A, r_c=r_c)
           * cuspNFW.veldisp2_r(r, r_s, rho_s, A, r_c=r_c, G=G))
    # rho*<v^2> = 4pi integral f v^4 dv, and <v^2> = 3 sigma_r^2 (isotropic)
    rhs = _df_velocity_moment(r, 4, r_s, rho_s, A, r_c, G) / 3.
    assert np.isclose(lhs, rhs, rtol=1e-2), (y, z, r / r_s, lhs, rhs)


# --- 8. r_{-2} round-trip and density slope ---------------------------------

@pytest.mark.parametrize("y,r_s,rho_s,G,A", list(_params_z0()))
def test_r2_roundtrip_and_slope(y, r_s, rho_s, G, A):
  r_2 = cuspNFW.r2_from_rs(r_s, rho_s, A)
  r_s_back = cuspNFW.rs_from_r2(r_2, rho_s, A)
  assert np.isclose(r_s_back, r_s, rtol=1e-10)
  # dln(rho)/dln(r) = -2 at r_2
  h = 1e-4
  rp = cuspNFW.density(r_2 * np.exp(h), r_s, rho_s, A)
  rm = cuspNFW.density(r_2 * np.exp(-h), r_s, rho_s, A)
  slope = (np.log(rp) - np.log(rm)) / (2. * h)
  assert np.isclose(slope, -2.0, atol=1e-4)


# --- 9. scale_from_c reproduces mass and concentration ----------------------

@pytest.mark.parametrize("c", [4.0, 10.0, 25.0])
@pytest.mark.parametrize("M,rho_vir", [(1e12, 200.0), (1.0, 1.0)])
@pytest.mark.parametrize("Afrac", [0.0, 0.3, 0.9])
def test_scale_from_c(c, M, rho_vir, Afrac):
  A = Afrac * cuspNFW.A_max(c, M, rho_vir)
  r_s, rho_s = cuspNFW.scale_from_c(c, M, A, rho_vir)
  R = cuspNFW.R_from_M(M, rho_vir)
  assert np.isclose(cuspNFW.mass(R, r_s, rho_s, A), M, rtol=1e-6)
  assert np.isclose(cuspNFW.r2_from_rs(r_s, rho_s, A), R / c, rtol=1e-6)


# --- 10. minimum-parameter helper consistency -------------------------------

@pytest.mark.parametrize("c", [0.5, 4.0, 25.0])
@pytest.mark.parametrize("M,rho_vir", [(1e12, 200.0), (1.0, 1.0)])
def test_min_param_consistency(c, M, rho_vir):
  Amax = cuspNFW.A_max(c, M, rho_vir)
  # A_max <-> M_min are mutual inverses
  assert np.isclose(cuspNFW.M_min(c, Amax, rho_vir), M, rtol=1e-12)
  # c_min inverts A_max. A just below A_max(c) keeps the required A above the
  # global minimum A_max(0), so c_min > 0 and A_max(c_min) == A.
  A = 0.95 * Amax
  cm = cuspNFW.c_min(M, A, rho_vir)
  assert cm > 0.
  assert np.isclose(cuspNFW.A_max(cm, M, rho_vir), A, rtol=1e-6)
  # above A_max the profile is invalid: raises by default, and with
  # cmin_error=False it falls back to the y==1 (rho_s*r_s**1.5 == A) boundary
  A_bad = 1.01 * Amax
  with pytest.raises(Exception):
    cuspNFW.scale_from_c(c, M, A_bad, rho_vir)
  r_s, rho_s = cuspNFW.scale_from_c(c, M, A_bad, rho_vir, cmin_error=False)
  assert np.isclose(A_bad / (rho_s * r_s ** 1.5), 1.0, rtol=1e-9)
