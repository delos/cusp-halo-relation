"""self-consistency of the cusp-NFW profile

every check ties two independently implemented quantities together, so none of
it depends on a stored reference:

  density  -> mass         by integrating 4 pi r^2 rho
  mass     -> potential    by integrating G M/r^2, including out to Phi_inf
  mass     -> veldisp      by the isotropic Jeans equation
  df       -> density      by integrating f over velocity
  df       -> veldisp      by the second velocity moment of f
  r_2, scale_from_c, A_max, rc_from_fmax   by their defining equations

everything is in physical units with G != 1, so the r_s, rho_s and G scalings
are exercised too, and every relation is checked across 0 <= y <= 1,
y = A/(rho_s r_s**1.5)

tolerances follow the least accurate ingredient of each relation, never the
quadrature, which is good to ~1e-12: the df fit (0.9%) sets 1.5% on the two
moment tests, the veldisp2_r fit (0.35%) sets 1% on Jeans, which differentiates
it, and closed-form mass and potential (~1e-9) set 1e-7 on the exact integrals

scaling one quantity by a constant and rerunning shows what is pinned down:
1+1e-5 on density, mass or potential each break ten or more tests, while the
fits take 1+5e-3 (veldisp2_r) and 1+2e-2 (df). veldisp2_r cannot be pinned
tighter here: the only checks on it are Jeans, which differentiates its own fit
error, and a moment of the df fit
"""
import numpy as np
import pytest

import os,sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
from cusp_halo_relation import cuspNFW as cn

G = 4.30091e-6 # kpc (km/s)^2/Msun
RS, RHOS = 2.3, 1.7e7 # kpc, Msun/kpc^3
RHO_VIR, MVIR = 200*2.775e2*0.3, 1e12
YS = [0.,0.003,0.03,0.1,0.35,0.7,1.]
RADII = RS*np.array([1e-3,1e-2,0.1,0.5,1.,3.,10.,50.])

def A_of(y):
  return y*RHOS*RS**1.5

def quad_log(f,a,b,npan=40,ngl=16):
  # int_a^b f(t) dt by Gauss-Legendre on panels geometric in ln t
  xg,wg = np.polynomial.legendre.leggauss(ngl)
  e = np.log(np.geomspace(a,b,npan+1))
  lo,hi = e[:-1,None],e[1:,None]
  t = np.exp(0.5*(hi-lo)*xg + 0.5*(hi+lo))
  return float(np.sum(f(t)*t*(0.5*(hi-lo)*wg)))

def phi_inf(A):
  # Phi(inf) - Phi(0), from the difference of the two zero points
  return cn.potential(RS,RS,RHOS,A,G=G) - cn.potential(RS,RS,RHOS,A,G=G,zero_at_inf=True)

def df_moment(r,A,k,**kw):
  # 4 pi int_0^vesc f(E) v^(2+k) dv, in whichever energy convention resolves the
  # integrand: deeply bound orbits need the zero point at the centre, weakly
  # bound ones at infinity
  psi = -cn.potential(r,RS,RHOS,A,G=G,zero_at_inf=True) # Phi_inf - Phi
  vesc = np.sqrt(2*psi)
  if psi < 0.5*phi_inf(A): # weakly bound
    f = lambda v: cn.df(-psi+0.5*v**2,RS,RHOS,A,G=G,zero_at_inf=True)
  else:
    phi = cn.potential(r,RS,RHOS,A,G=G)
    f = lambda v: cn.df(phi+0.5*v**2,RS,RHOS,A,G=G)
  return 4*np.pi*quad_log(lambda v: f(v)*v**(2+k),vesc*1e-9,vesc,**kw)

# the profile chain

@pytest.mark.parametrize('y',YS)
def test_density_integrates_to_mass(y):
  """M(r2) - M(r1) = int 4 pi r^2 rho dr"""
  A = A_of(y)
  for r1,r2 in zip(RADII[:-1],RADII[1:]):
    got = cn.mass(r2,RS,RHOS,A) - cn.mass(r1,RS,RHOS,A)
    want = quad_log(lambda s: 4*np.pi*s**2*cn.density(s,RS,RHOS,A),r1,r2)
    assert got == pytest.approx(want,rel=1e-7)

@pytest.mark.parametrize('y',YS)
def test_mass_integrates_to_potential(y):
  """Phi(r2) - Phi(r1) = int G M/r^2 dr"""
  A = A_of(y)
  for r1,r2 in zip(RADII[:-1],RADII[1:]):
    got = cn.potential(r2,RS,RHOS,A,G=G) - cn.potential(r1,RS,RHOS,A,G=G)
    want = quad_log(lambda s: G*cn.mass(s,RS,RHOS,A)/s**2,r1,r2)
    assert got == pytest.approx(want,rel=1e-7)

@pytest.mark.parametrize('y',YS)
def test_potential_reaches_phi_inf(y):
  """Phi(inf) - Phi(0) = int_0^inf G M/r^2 dr, the two ends coming from the
  potential itself: exact as r -> 0, and zero_at_inf gives Phi_inf - Phi(r)"""
  A = A_of(y)
  r1,r2 = 1e-4*RS,1e12*RS
  inner = cn.potential(r1,RS,RHOS,A,G=G)
  outer = -cn.potential(r2,RS,RHOS,A,G=G,zero_at_inf=True)
  mid = quad_log(lambda s: G*cn.mass(s,RS,RHOS,A)/s**2,r1,r2,npan=400)
  assert inner + mid + outer == pytest.approx(phi_inf(A),rel=1e-9)

@pytest.mark.parametrize('y',YS)
def test_jeans_equation(y):
  """d(rho sigma_r^2)/dln r = -rho G M/r, for an isotropic system"""
  A = A_of(y)
  h = 1e-3
  P = lambda s: cn.density(s,RS,RHOS,A)*cn.veldisp2_r(s,RS,RHOS,A,G=G)
  for r in RADII:
    lhs = (P(r*np.exp(h)) - P(r*np.exp(-h)))/(2*h)
    rhs = -cn.density(r,RS,RHOS,A)*G*cn.mass(r,RS,RHOS,A)/r
    assert lhs == pytest.approx(rhs,rel=1e-2)

# the distribution function

@pytest.mark.parametrize('y',YS)
def test_df_integrates_to_density(y):
  """rho(r) = 4 pi int_0^vesc f(Phi + v^2/2) v^2 dv"""
  A = A_of(y)
  for r in RADII:
    assert df_moment(r,A,0) == pytest.approx(cn.density(r,RS,RHOS,A),rel=1.5e-2)

@pytest.mark.parametrize('y',YS)
def test_df_second_moment_is_veldisp(y):
  """sigma_r^2 = <v^2>/3 = int f v^4 dv/(3 int f v^2 dv)"""
  A = A_of(y)
  for r in RADII:
    got = df_moment(r,A,2)/(3*df_moment(r,A,0))
    assert got == pytest.approx(cn.veldisp2_r(r,RS,RHOS,A,G=G),rel=1.5e-2)

@pytest.mark.parametrize('y',YS)
def test_df_positive_and_decreasing(y):
  """f > 0 and df/dE < 0 at every bound energy"""
  A = A_of(y)
  f = cn.df(np.geomspace(1e-8,1-1e-8,400)*phi_inf(A),RS,RHOS,A,G=G)
  assert np.all(f > 0.)
  assert np.all(np.diff(f) < 0.)

@pytest.mark.parametrize('y',YS)
def test_df_energy_zero_points_agree(y):
  """f(E) and f(E-Phi_inf,zero_at_inf=True) are the same function"""
  A = A_of(y)
  p0 = phi_inf(A)
  E = np.geomspace(1e-6,1-1e-6,200)*p0
  # abs=0: f reaches ~1e-15, under approx's default 1e-12 absolute floor
  assert cn.df(E,RS,RHOS,A,G=G) == pytest.approx(cn.df(E-p0,RS,RHOS,A,G=G,zero_at_inf=True),rel=1e-6,abs=0)

# the scale relations

@pytest.mark.parametrize('y',YS)
def test_r2_is_where_the_slope_is_minus_two(y):
  """dln rho/dln r = -2 at r_2, and rs_from_r2 inverts r2_from_rs"""
  A = A_of(y)
  r2 = cn.r2_from_rs(RS,RHOS,A)
  h = 1e-5
  slope = (np.log(cn.density(r2*np.exp(h),RS,RHOS,A)) - np.log(cn.density(r2*np.exp(-h),RS,RHOS,A)))/(2*h)
  assert slope == pytest.approx(-2.,abs=1e-8)
  assert cn.rs_from_r2(r2,RHOS,A) == pytest.approx(RS,rel=1e-10)

@pytest.mark.parametrize('c',[4.,10.,25.])
@pytest.mark.parametrize('frac',[0.,0.3,0.9,1-1e-9])
def test_scale_from_c_round_trip(c,frac):
  """scale_from_c returns an (r_s,rho_s) whose mass inside R is M and whose r_2
  is R/c. only that is asserted: above c = 10.146 the pair need not be unique,
  which test_scale_from_c_branches covers"""
  A = frac*cn.A_max(c,MVIR,RHO_VIR)
  R = cn.R_from_M(MVIR,RHO_VIR)
  r_s,rho_s = cn.scale_from_c(c,MVIR,A,RHO_VIR)
  assert cn.mass(R,r_s,rho_s,A) == pytest.approx(MVIR,rel=1e-8)
  assert cn.r2_from_rs(r_s,rho_s,A) == pytest.approx(R/c,rel=1e-8)

def A_of_y_on_curve(c,y):
  """the cusp coefficient of the halo of mass MVIR and concentration c whose
  cusp parameter is y. r_2 = R/c fixes r_s, and the mass then fixes rho_s"""
  R = cn.R_from_M(MVIR,RHO_VIR)
  r_s = R/(c*cn.r2_from_rs(1.,1.,y))
  return y*MVIR/(r_s**1.5*cn.mass(R/r_s,1.,1.,y))

@pytest.mark.parametrize('c',[4.,10.,25.])
def test_scale_from_c_branches(c):
  """both branches solve the same two conditions and the high one has the
  larger y. a second solution exists only for A between the y = 1 value and
  A_max, a range that is empty below c = 10.146"""
  R = cn.R_from_M(MVIR,RHO_VIR)
  A1 = A_of_y_on_curve(c,1.)
  for f in (0.5,0.999,1.,1.0002,1.001):
    A = f*A1
    try:
      rs,rhos = cn.scale_from_c(c,MVIR,A,RHO_VIR,branch='high')
    except Exception:
      continue # A past the turning point, so no halo at all
    assert cn.mass(R,rs,rhos,A) == pytest.approx(MVIR,rel=1e-8)
    assert cn.r2_from_rs(rs,rhos,A) == pytest.approx(R/c,rel=1e-8)
    assert A/(rhos*rs**1.5) <= 1.+1e-9
    rl,pl = cn.scale_from_c(c,MVIR,A,RHO_VIR) # the default branch
    assert A/(pl*rl**1.5) <= A/(rhos*rs**1.5)
    if f < 1.: # only one halo there, so the two settings must agree
      assert (rl,pl) == pytest.approx((rs,rhos),rel=1e-8)

@pytest.mark.parametrize('c',[2.,8.,10.,25.,200.])
def test_scale_from_c_raises_the_concentration(c):
  """with cmin_error off and an A too large for c, the halo returned is the one
  at c_min, where A is exactly that concentration's largest"""
  R = cn.R_from_M(MVIR,RHO_VIR)
  A = 1.3*cn.A_max(c,MVIR,RHO_VIR)
  r_s,rho_s = cn.scale_from_c(c,MVIR,A,RHO_VIR,cmin_error=False)
  cmin = cn.c_min(MVIR,A,RHO_VIR)
  assert cmin > c
  assert cn.mass(R,r_s,rho_s,A) == pytest.approx(MVIR,rel=1e-8)
  assert cn.r2_from_rs(r_s,rho_s,A) == pytest.approx(R/cmin,rel=1e-10)
  assert A == pytest.approx(cn.A_max(cmin,MVIR,RHO_VIR),rel=1e-10)
  with pytest.raises(Exception):
    cn.scale_from_c(c,MVIR,A,RHO_VIR)

@pytest.mark.parametrize('c',[4.,10.,25.,200.])
def test_A_max_is_the_largest_A_on_the_curve(c):
  """A_max is the peak of A(y): nothing on the curve exceeds it, a halo exists
  just inside it and none just outside, and M_min and c_min invert it. below
  c = 10.146 the peak is the y = 1 halo, whose scale radius is 2R/c"""
  R = cn.R_from_M(MVIR,RHO_VIR)
  A = cn.A_max(c,MVIR,RHO_VIR)
  peak = A_of_y_on_curve(c,np.linspace(1e-3,1.,400)).max()
  assert peak == pytest.approx(A,rel=1e-6)
  assert peak <= A*(1+1e-9)
  assert cn.M_min(c,A,RHO_VIR) == pytest.approx(MVIR,rel=1e-10)
  assert cn.c_min(MVIR,A,RHO_VIR) == pytest.approx(c,rel=1e-6)
  for branch in ('low','high'):
    rs,rhos = cn.scale_from_c(c,MVIR,A*(1-1e-9),RHO_VIR,branch=branch)
    assert cn.mass(R,rs,rhos,A*(1-1e-9)) == pytest.approx(MVIR,rel=1e-8)
    assert cn.r2_from_rs(rs,rhos,A*(1-1e-9)) == pytest.approx(R/c,rel=1e-8)
  with pytest.raises(Exception):
    cn.scale_from_c(c,MVIR,A*(1+1e-6),RHO_VIR)
  if c < 10.146:
    r_s = 2*R/c
    assert cn.mass(R,r_s,A/r_s**1.5,A) == pytest.approx(MVIR,rel=1e-9)

@pytest.mark.parametrize('y',[0.03,0.35,1.])
def test_rc_from_fmax(y):
  """the core radius solves rho(r_c) r_c^6 = prefactor G^-3 fmax^-2"""
  A,fmax,pre = A_of(y),1e-4,3e-5
  rc = cn.rc_from_fmax(fmax,RS,RHOS,A,G=G,prefactor=pre)
  assert cn.density(rc,RS,RHOS,A)*rc**6 == pytest.approx(pre*G**-3*fmax**-2,rel=1e-8)

# the interfaces

@pytest.mark.parametrize('fn',['density','mass','potential','veldisp2_r','veldisp_r','df'])
def test_broadcasting(fn):
  """r (or E) broadcasts against r_s, rho_s and A by the ufunc rules: equal
  shapes pair elementwise, r[:,None] with A[None,:] gives the grid, and scalars
  in give a scalar out"""
  f = getattr(cn,fn)
  A = np.array([0.,0.2,0.6,1.])*RHOS*RS**1.5
  kw = {} if fn in ('density','mass') else {'G':G}
  # df takes an energy; use one that is bound for every A, since Phi_inf grows
  first = np.array([0.05,0.2,0.5,0.9])*phi_inf(0.) if fn == 'df' else RS*np.array([0.03,0.3,3.,30.])
  assert np.shape(f(first,RS,RHOS,A[0],**kw)) == (4,)
  assert np.shape(f(first[0],RS,RHOS,A,**kw)) == (4,)
  assert np.shape(f(first,RS,RHOS,A,**kw)) == (4,)
  assert np.shape(f(first[:,None],RS,RHOS,A[None,:],**kw)) == (4,4)
  assert np.ndim(f(first[0],RS,RHOS,A[0],**kw)) == 0
  grid = f(first[:,None],RS,RHOS,A[None,:],**kw)
  loop = np.array([[f(t,RS,RHOS,a,**kw) for a in A] for t in first])
  assert grid == pytest.approx(loop,rel=1e-9,abs=0)
