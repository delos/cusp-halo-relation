import numpy as np
from scipy.optimize import root_scalar, minimize_scalar
from scipy.integrate import cumulative_simpson, simpson
from scipy.interpolate import CubicSpline
from scipy.special import gammaln, gammainc, gammaincc, expit

# rho^2 = rho_E^2 + [A r^-1.5 T(x)]^2, x = r/r_s, with Einasto profile
# rho_E = rho_s exp{-(2/alpha)[x^alpha-1]} and T = exp{-(2/alpha)[(1+x)^alpha-1]},
# which tends to the cusp-NFW factor (1+x)^-2 as alpha -> 0. The core enters as
# in cuspNFW, through xt.

# dimensionless radial profiles

def __xt(x,z):
  return (x**2+4*z**2)**1.5 / (x**2+8*z**2+1e-300) # guards against 0/0
def __lnterms(xt,y,alpha):
  # logs of the Einasto and cusp terms (their squares underflow before rho does)
  lh = -2./alpha*(xt**alpha-1.)
  with np.errstate(divide='ignore',invalid='ignore'):
    lc = np.where(y > 0., np.log(y) - 1.5*np.log(xt) - 2./alpha*np.expm1(alpha*np.log1p(xt)), -np.inf)
  return lh,lc
def __lndensity(x,y,alpha,z=0.):
  lh,lc = __lnterms(__xt(x,z),y,alpha)
  return 0.5*np.logaddexp(2*lh,2*lc)
def __density(x,y,alpha,z=0.):
  return np.exp(__lndensity(x,y,alpha,z))
def __density_slope(x,y,alpha,z=0.):
  xt = __xt(x,z)
  lh,lc = __lnterms(xt,y,alpha)
  w = expit(2*(lh-lc)) # share of rho^2 in the Einasto term
  s = -2*xt**alpha*w + (-1.5-2*xt*(1+xt)**(alpha-1))*(1-w)
  dxt = x**2*(x**2 + 16*z**2) / ((x**2 + 4*z**2)*(x**2 + 8*z**2) + 1e-300) # d ln xt / d ln x
  return s * dxt
def __xfar(alpha):
  # rho < exp(-80) rho_s beyond this radius
  return (1.+40*alpha)**(1./alpha)
def __xcap(alpha):
  # rho < exp(-700) rho_s beyond this radius
  return min((1.+350*alpha)**(1./alpha),1e30)
def __table(xmin,xmax,y,alpha,z=0.,step=1.02):
  # density, mass and potential on a fixed lattice in ln x (so results do not
  # depend on the other radii requested); power law below xmin
  h = np.log(step)
  n0,n1 = int(np.floor(np.log(xmin)/h)),int(np.ceil(np.log(xmax)/h))
  _lx = h*np.arange(n0,n1+1)
  _x = np.exp(_lx)
  xmin = _x[0]
  _lp = __lndensity(_x,y,alpha,z)
  _p = np.exp(_lp)
  g0 = __density_slope(xmin,y,alpha,z)
  _m = cumulative_simpson(4*np.pi*np.exp(_lp+3*_lx),x=_lx,initial=0) + 4*np.pi*_p[0]*xmin**3/(3+g0)
  _pot = cumulative_simpson(_m/_x,x=_lx,initial=0) + _m[0]/(xmin*(2+g0))
  return _x,_p,_m,_pot
def __xmin(x,z):
  xmin = min(float(np.min(x))/1e6, 1e-8)
  if z > 0.:
    xmin = min(xmin, float(z)/100.)
  return max(xmin, 1e-16)
def __interp_log(x,_x,_f):
  return np.exp(CubicSpline(np.log(_x),np.log(_f),extrapolate=False)(np.log(x)))
def __einasto_moment(n,alpha):
  # int_0^inf rho_E x**(n-1) dx / rho_s, in logs (the factors overflow)
  return np.exp(2./alpha + n*np.log(alpha/2.) + gammaln(n))/alpha
def __mass_einasto(x,alpha):
  # closed form for y = 0, z = 0
  n = 3./alpha
  return 4*np.pi*__einasto_moment(n,alpha)*gammainc(n,2./alpha*x**alpha)
def __potential_einasto(x,alpha,zero_at_inf=False):
  # closed form for y = 0, z = 0; Phi - Phi(0) = 4 pi int_0^x rho x dx - M/x and
  # Phi - Phi_inf = -4 pi int_x^inf rho x dx - M/x, so neither cancels
  n,t = 2./alpha,2./alpha*x**alpha
  if zero_at_inf:
    return -4*np.pi*__einasto_moment(n,alpha)*gammaincc(n,t) - __mass_einasto(x,alpha)/x
  return 4*np.pi*__einasto_moment(n,alpha)*gammainc(n,t) - __mass_einasto(x,alpha)/x
def __mass(x,y,alpha,z=0.):
  _shape = np.shape(x)
  x = np.atleast_1d(np.array(x,dtype=float))
  _M = np.zeros_like(x)
  pos = x > 0
  if not np.any(pos):
    return _M.reshape(_shape)
  if y == 0. and z == 0.:
    _M[pos] = __mass_einasto(x[pos],alpha)
    return _M.reshape(_shape)
  _x,_p,_m,_pot = __table(__xmin(x[pos],z),1.1*max(float(np.max(x)),__xfar(alpha)),y,alpha,z)
  _M[pos] = __interp_log(x[pos],_x,_m)
  return _M.reshape(_shape)
def __potential(x,y,alpha,z=0.,zero_at_inf=False):
  _shape = np.shape(x)
  x = np.atleast_1d(np.array(x,dtype=float))
  pot = np.zeros_like(x)
  pos = x > 0
  if y == 0. and z == 0.:
    pot[pos] = __potential_einasto(x[pos],alpha,zero_at_inf)
    if zero_at_inf:
      pot[~pos] = -__potential_inf(y,alpha,z)
    return pot.reshape(_shape)
  if np.any(pos):
    _x,_p,_m,_pot = __table(__xmin(x[pos],z),1.1*max(float(np.max(x)),__xfar(alpha)),y,alpha,z)
    pot[pos] = __interp_log(x[pos],_x,_pot)
  if zero_at_inf:
    pot = pot - __potential_inf(y,alpha,z)
  return pot.reshape(_shape)
def __potential_inf(y,alpha,z=0.):
  if y == 0. and z == 0.:
    return 4*np.pi*__einasto_moment(2./alpha,alpha)
  _x,_p,_m,_pot = __table(__xmin(1.,z),__xfar(alpha),y,alpha,z)
  return _pot[-1] + _m[-1]/_x[-1]
def __veldisp2_r(x,y,alpha,z=0.):
  # rho sigma_r^2 = int_x^inf rho M/x^2 dx, with a power-law tail beyond the table
  x = np.array(x,dtype=float)
  _x,_p,_m,_pot = __table(__xmin(x,z),max(__xfar(alpha),30.*float(np.max(x))),y,alpha,z)
  g = __density_slope(_x[-1],y,alpha,z)
  _ps2_0 = _p[-1]*_m[-1]/(_x[-1]*(1.-g))
  _ps2 = cumulative_simpson((_p*_m/_x)[::-1],x=-np.log(_x)[::-1],initial=0)[::-1] + _ps2_0
  return __interp_log(x,_x,_ps2/_p)

def __broadcast(kernel,first,*params):
  # ufunc-style broadcasting of first (r or E) against the profile parameters;
  # the kernel is called once per distinct parameter set
  arrs = np.broadcast_arrays(*[np.asarray(a,dtype=float) for a in (first,)+params])
  first,params = arrs[0].ravel(),np.stack([a.ravel() for a in arrs[1:]],axis=1)
  out = np.empty(first.shape)
  uniq,inv = np.unique(params,axis=0,return_inverse=True)
  inv = inv.ravel()
  for k,p in enumerate(uniq):
    m = inv == k
    out[m] = kernel(first[m],*p)
  return out.reshape(arrs[0].shape)[()]

# radial profiles

def density(r,r_s,rho_s,A,alpha,r_c=0.):
  '''
  Evaluate density at radius r for a cusp-Einasto density profile with scale
  radius r_s, scale density rho_s, cusp coefficient A, and Einasto shape
  parameter alpha.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
    
    r_s: float or array
      Scale radius of halo: the radius r_{-2} of the Einasto component. The
      whole profile's r_{-2} is smaller if A > 0 (see r2_from_rs).
    
    rho_s: float or array
      Scale density of halo: the density of the Einasto component at r_s.
    
    A: float or array
      Cusp coefficient. This profile only makes sense if
      A <= y_max(alpha) * rho_s * r_s**1.5 (see y_max).
    
    alpha: float or array
      Einasto shape parameter.
    
    r_c: float or array
      Radius of the phase-space core.
  
  Returns:
    
    rho: float or array
      Density at radius r.
  '''
  return __density(r/r_s,A/(rho_s*r_s**1.5),alpha,r_c/r_s) * rho_s

def density_slope(r,r_s,rho_s,A,alpha,r_c=0.):
  '''
  Evaluate the logarithmic density slope dln(rho)/dln(r) at radius r for a
  cusp-Einasto density profile. Parameters are as in density.
  '''
  return __density_slope(r/r_s,A/(rho_s*r_s**1.5),alpha,r_c/r_s)

def mass(r,r_s,rho_s,A,alpha,r_c=0.):
  '''
  Evaluate mass enclosed within radius r for a cusp-Einasto density profile
  with scale radius r_s, scale density rho_s, cusp coefficient A, and Einasto
  shape parameter alpha.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
    
    r_s: float or array
      Scale radius of halo (r_{-2} of the Einasto component).
    
    rho_s: float or array
      Scale density of halo (density of the Einasto component at r_s).
    
    A: float or array
      Cusp coefficient. This profile only makes sense if
      A <= y_max(alpha) * rho_s * r_s**1.5 (see y_max).
    
    alpha: float or array
      Einasto shape parameter.
    
    r_c: float or array
      Radius of the phase-space core.
  
  Returns:
    
    M: float or array
      Mass enclosed within radius r.
  '''
  return __broadcast(lambda r,r_s,rho_s,A,alpha,r_c: r_s**3*rho_s * __mass(r/r_s,A/(rho_s*r_s**1.5),alpha,r_c/r_s),
                     r,r_s,rho_s,A,alpha,r_c)

def veldisp2_r(r,r_s,rho_s,A,alpha,r_c=0.,G=1.):
  '''
  Evaluate squared radial velocity dispersion, sigma_r^2, at radius r for a
  cusp-Einasto density profile with scale radius r_s, scale density rho_s, cusp
  coefficient A, and Einasto shape parameter alpha. We assume an isotropic
  velocity distribution.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
    
    r_s: float or array
      Scale radius of halo (r_{-2} of the Einasto component).
    
    rho_s: float or array
      Scale density of halo (density of the Einasto component at r_s).
    
    A: float or array
      Cusp coefficient. This profile only makes sense if
      A <= y_max(alpha) * rho_s * r_s**1.5 (see y_max).
    
    alpha: float or array
      Einasto shape parameter.
    
    r_c: float or array
      Radius of the phase-space core.
    
    G: float
      Gravitational constant. If not specified, we return sigma_r^2/G, which
      has dimensions of mass/length.
  
  Returns:
    
    sigma_r^2: float or array
      Squared radial velocity dispersion at radius r. If G was not specified,
      this is sigma_r^2/G instead.
  '''
  return G*__broadcast(lambda r,r_s,rho_s,A,alpha,r_c: rho_s*r_s**2 * __veldisp2_r(r/r_s,A/(rho_s*r_s**1.5),alpha,r_c/r_s),
                       r,r_s,rho_s,A,alpha,r_c)

def veldisp_r(r,r_s,rho_s,A,alpha,r_c=0.,G=1.):
  '''
  Evaluate radial velocity dispersion sigma_r at radius r for a cusp-Einasto
  density profile with scale radius r_s, scale density rho_s, cusp coefficient
  A, and Einasto shape parameter alpha. We assume an isotropic velocity
  distribution.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
    
    r_s: float or array
      Scale radius of halo (r_{-2} of the Einasto component).
    
    rho_s: float or array
      Scale density of halo (density of the Einasto component at r_s).
    
    A: float or array
      Cusp coefficient. This profile only makes sense if
      A <= y_max(alpha) * rho_s * r_s**1.5 (see y_max).
    
    alpha: float or array
      Einasto shape parameter.
    
    r_c: float or array
      Radius of the phase-space core.
    
    G: float
      Gravitational constant. If not specified, we return sigma_r/sqrt(G),
      which has dimensions of sqrt(mass/length).
  
  Returns:
    
    sigma_r: float or array
      Radial velocity dispersion at radius r. If G was not specified, this is
      sigma_r/sqrt(G) instead.
  '''
  return np.sqrt(veldisp2_r(r,r_s,rho_s,A,alpha,r_c=r_c,G=G))

def potential(r,r_s,rho_s,A,alpha,r_c=0.,G=1.,zero_at_inf=False):
  '''
  Evaluate gravitational potential Phi at radius r for a cusp-Einasto density
  profile with scale radius r_s, scale density rho_s, cusp coefficient A, and
  Einasto shape parameter alpha. By default, we take the potential to be zero
  at r=0.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
    
    r_s: float or array
      Scale radius of halo (r_{-2} of the Einasto component).
    
    rho_s: float or array
      Scale density of halo (density of the Einasto component at r_s).
    
    A: float or array
      Cusp coefficient. This profile only makes sense if
      A <= y_max(alpha) * rho_s * r_s**1.5 (see y_max).
    
    alpha: float or array
      Einasto shape parameter.
    
    r_c: float or array
      Radius of the phase-space core.
    
    G: float
      Gravitational constant. If not specified, we return Phi/G, which has
      dimensions of mass/length.
    
    zero_at_inf: bool
      If True, the zero point of energy is set so that Phi=0 at r=inf.
      Default is zero_at_inf=False, in which case Phi=0 at r=0.
  
  Returns:
    
    Phi: float or array
      Gravitational potential at radius r. If G was not specified, this is
      Phi/G instead.
  '''
  return G*__broadcast(lambda r,r_s,rho_s,A,alpha,r_c: rho_s*r_s**2 * __potential(r/r_s,A/(rho_s*r_s**1.5),alpha,z=r_c/r_s,zero_at_inf=zero_at_inf),
                       r,r_s,rho_s,A,alpha,r_c)

# distribution function

def __d2rho_dPhi2(_x,_p,_m,g):
  # d2rho/dPhi2 on a radial table, g = dln(rho)/dln(x)
  # (differencing ln(-drho/dPhi) whole; expanding it cancels inside a core)
  _dpdP = _p*g*_x/_m # dPhi/dx = M/x^2
  return _dpdP * np.gradient(np.log(-_dpdP),np.log(_x)) * _x/_m

def __df(e,y,alpha,z=0.,zero_at_inf=False):
  # energy lattice in s(e) = ln e below Phi_inf/2 and -ln(Phi_inf-e) above it
  _shape = np.shape(e)
  e = np.atleast_1d(np.array(e,dtype=float))
  Pinf = __potential_inf(y,alpha,z)
  # ecal = Pinf - e, kept exact if given
  if zero_at_inf:
    e,ecal = e + Pinf,-e
  else:
    ecal = Pinf - e
  out = np.zeros(e.shape)
  bound = (e > 0.)&(ecal > 0.)
  if not np.any(bound):
    return out.reshape(_shape)
  step = 1.01 # step in factors of 1.01
  h = np.log(step)
  ln_half = np.log(0.5*Pinf)
  s_of = lambda e,ecal: np.where(e <= 0.5*Pinf, np.log(np.minimum(e,0.5*Pinf)), 2*ln_half - np.log(np.minimum(ecal,0.5*Pinf)))
  e_of = lambda s: np.where(s <= ln_half, np.exp(np.minimum(s,ln_half)), Pinf - np.exp(2*ln_half - np.maximum(s,ln_half)))
  ec_of = lambda s: np.where(s <= ln_half, Pinf - np.exp(np.minimum(s,ln_half)), np.exp(2*ln_half - np.maximum(s,ln_half)))
  sq = s_of(e[bound],ecal[bound])
  _s = h*np.arange(int(np.floor(np.min(sq)/h))-2,int(np.ceil(np.max(sq)/h))+3)
  _e,_ec = e_of(_s),ec_of(_s)
  # extend the radial table to span these energies' potentials
  xmin = __xmin(1.,z)
  while True:
    _x,_p,_m,_P = __table(xmin,__xfar(alpha),y,alpha,z,step=step)
    if _P[0] <= _e[0]/step**2 or xmin <= 1e-30:
      break
    g0 = __density_slope(xmin,y,alpha,z)
    xmin = max(xmin * 0.5*(_e[0]/step**2/_P[0])**(1./(2+g0)), 1e-30)
  xmax = min(max(__xfar(alpha),step**2*_m[-1]/_ec[-1]),__xcap(alpha))
  if xmax > _x[-1]:
    _x,_p,_m,_P = __table(xmin,xmax,y,alpha,z,step=step)
  # Psi = Phi_inf - Phi, integrated in from outside (precise where Phi -> Phi_inf)
  _Psi = cumulative_simpson((_m/_x)[::-1],x=-np.log(_x)[::-1],initial=0)[::-1] + _m[-1]/_x[-1]
  # Eddington's formula in its integrated-by-parts form,
  # f = 1/(sqrt(8) pi^2) int_e d2rho/dPhi2 dPhi/sqrt(Phi-e), which avoids
  # differencing in e (inaccurate near the bottom of a cored potential)
  _d2pdP2 = __d2rho_dPhi2(_x,_p,_m,__density_slope(_x,y,alpha,z))
  # Phi - e from Phi below Phi_inf/2 and from Psi above it (no cancellation);
  # drop energies beyond the table
  deep = _s <= ln_half
  ok = np.where(deep,_e < _P[-1],_ec > _Psi[-1])
  _s,_e,_ec,deep = _s[ok],_e[ok],_ec[ok],deep[ok]
  _f = np.zeros_like(_e)
  for i,(e1,ec1) in enumerate(zip(_e,_ec)):
    if deep[i]:
      above = _P > e1
      dPhi = _P[above] - e1
      d2pdP2_1 = np.interp(np.log(e1),np.log(_P),_d2pdP2)
    else:
      above = _Psi < ec1
      dPhi = ec1 - _Psi[above]
      d2pdP2_1 = np.interp(np.log(ec1),np.log(_Psi[::-1]),_d2pdP2[::-1])
    integrand = np.insert(_d2pdP2[above],0,d2pdP2_1)
    measure = np.insert(np.sqrt(dPhi),0,0.)
    _f[i] = 1./(np.sqrt(2)*np.pi**2) * simpson(integrand,x=measure)
  with np.errstate(divide='ignore',invalid='ignore'):
    out[bound] = np.exp(np.interp(sq,_s,np.log(_f),left=-np.inf,right=-np.inf))
  return out.reshape(_shape)

def df(E,r_s,rho_s,A,alpha,r_c=0.,G=1.,zero_at_inf=False):
  '''
  Evaluate the distribution function f(E) for a cusp-Einasto density profile
  with scale radius r_s, scale density rho_s, cusp coefficient A, and Einasto
  shape parameter alpha. We assume an isotropic velocity distribution.
  
  Parameters:
    
    E: float or array
      Energy E. If G is not specified, this is assumed to be E/G instead.
    
    r_s: float or array
      Scale radius of halo (r_{-2} of the Einasto component).
    
    rho_s: float or array
      Scale density of halo (density of the Einasto component at r_s).
    
    A: float or array
      Cusp coefficient. This profile only makes sense if
      A <= y_max(alpha) * rho_s * r_s**1.5 (see y_max).
    
    alpha: float or array
      Einasto shape parameter.
    
    r_c: float or array
      Radius of the phase-space core.
    
    G: float
      Gravitational constant. If not specified, then we assume the first
      argument is E/G, which has dimensions of mass/length, and we return
      G^1.5 f(E).
    
    zero_at_inf: bool
      If True, the zero point of energy is set so that the potential is zero at
      r=inf. Default is zero_at_inf=False, in which case the potential is zero
      at r=0.
  
  Returns:
    
    f: float or array
      Distribution function f(E). If G was not specified, this is G^1.5 f(E)
      instead.
  '''
  return __broadcast(lambda E,r_s,rho_s,A,alpha,r_c: __df(E/(G*rho_s*r_s**2),A/(rho_s*r_s**1.5),alpha,z=r_c/r_s,zero_at_inf=zero_at_inf) / (G**1.5*r_s**3*rho_s**0.5),
                     E,r_s,rho_s,A,alpha,r_c)

# conversions from halo concentration

def y_max(alpha):
  '''
  Largest sensible value of y = A/(rho_s*r_s**1.5), at which the cusp term
  equals the Einasto term at r_s. This is the analogue of the cusp-NFW limit
  rho_s*r_s**1.5 >= A, which is where the same happens there. As alpha -> 0,
  y_max -> 4: the cusp factor T then tends to the cusp-NFW factor (1+x)**-2,
  which is 1/4 at r_s. (The Einasto component itself tends to r**-2, not NFW.)
  '''
  return np.exp(2./alpha*(2.**alpha-1.))

def __u2(y,alpha):
  # r_{-2}/r_s
  if y == 0.:
    return 1.
  return np.exp(root_scalar(lambda lx: __density_slope(np.exp(lx),y,alpha)+2.,bracket=[np.log(1e-8),0.]).root)

def r2_from_rs(r_s,rho_s,A,alpha):
  '''
  Find the radius r_{-2} at which dlog(rho)/dlog(r) = -2 for a cusp-Einasto
  density profile with scale radius r_s, scale density rho_s, cusp coefficient
  A, and Einasto shape parameter alpha.
  
  r_{-2} = r_s if A=0 and decreases as A increases.
  '''
  return __u2(A/(rho_s*r_s**1.5),alpha) * r_s

def rs_from_r2(r_2,rho_s,A,alpha):
  '''
  Find the cusp-Einasto scale radius r_s, given r_2 (the radius r_{-2} at which
  dlog(rho)/dlog(r) = -2), the scale density rho_s, the cusp coefficient A, and
  the Einasto shape parameter alpha.
  '''
  if A == 0.:
    return r_2
  w = A / (rho_s * r_2**1.5) # y = w u^1.5 with u = r_2/r_s
  lu = root_scalar(lambda lu: __density_slope(np.exp(lu),w*np.exp(1.5*lu),alpha)+2.,bracket=[np.log(1e-8),0.]).root
  return r_2*np.exp(-lu)

def R_from_M(M,rho_vir):
  '''Virial radius R given halo mass M and virial density rho_vir.'''
  return (3*M/(4*np.pi*rho_vir))**(1./3)

def __rhos_from_c_Einasto(c,rho_vir,alpha):
  return c**3/__mass_einasto(c,alpha)*4*np.pi*rho_vir/3.

def __params(c,y,alpha):
  # M rho_vir / A^2 for the halo with concentration c and this y
  u = __u2(y,alpha)
  x = c*u
  return 3*__mass(x,y,alpha)**2/(4*np.pi*x**3*y**2)
def __min_params(c,alpha,return_y=False):
  # minimum of M rho_vir/A^2 over 0 < y <= y_max; at y_max for intermediate c,
  # but at y < y_max for large or small c
  ym = y_max(alpha)
  P = lambda ly: __params(c,np.exp(ly),alpha)
  res = minimize_scalar(P,bounds=(np.log(ym)-6.,np.log(ym)),method='bounded',options={'xatol':1e-10})
  ly,Pm = (res.x,res.fun) if res.fun < P(np.log(ym)) else (np.log(ym),P(np.log(ym)))
  return (Pm,np.exp(ly)) if return_y else Pm

def scale_from_c(c,M,A,rho_vir,alpha,cmin_error=True,branch='low'):
  '''
  Find the cusp-Einasto profile scale parameters, given halo concentration c
  and mass M.
  
  Parameters:
    
    c, M: floats
      Halo parameters. The concentration is c = R/r_{-2} with r_{-2} that of
      the whole profile.
    
    A: float
      Cusp coefficient parameter.
    
    rho_vir: float
      The virial density (e.g., 200 times the cosmological mean value).
    
    alpha: float
      Einasto shape parameter.
    
    cmin_error: boolean
      If True, we raise an exception if the concentration is too low and would
      violate A <= y_max(alpha) * rho_s * r_s**1.5. If False, we increase the
      concentration to its minimum value and continue. Default is True.
    
    branch: 'low' or 'high'
      Which halo to return when (c,M,A) admits more than one. The two differ in
      y = A/(rho_s * r_s**1.5), and this selects the smaller or the larger of
      them. A second halo exists only if the largest A for this (c,M) is
      reached at y < y_max (see A_max), and only for A above the y = y_max
      value; where there is only one, both settings return it.
  
  Returns:
    
    r_s, rho_s: floats
      The scale radius and scale density, respectively.
  '''
  R = R_from_M(M,rho_vir)
  r_2 = R / c # r_{-2}
  if A == 0.:
    return r_2,__rhos_from_c_Einasto(c,rho_vir,alpha)
  params = M*rho_vir/A**2
  Pm,ys = __min_params(c,alpha,return_y=True)
  if params < Pm:
    if cmin_error:
      raise Exception('M*rho_vir/A**2=%.3e must be >%.3e for c=%.2f.'%(params,Pm,c))
    c = c_min(M,A,rho_vir,alpha)
    ys = __min_params(c,alpha,return_y=True)[1]
    r_s = R/(c*__u2(ys,alpha))
    rho_s = A/(ys*r_s**1.5)
  else:
    # solve for y; M rho_vir/A^2 is minimal at ys, so there is one solution
    # with y <= ys and possibly another with ys < y <= y_max
    def scale(ly):
      r_s = r_2/__u2(np.exp(ly),alpha)
      return r_s, A/(np.exp(ly)*r_s**1.5)
    g = lambda ly: np.log(mass(R,*scale(ly),A,alpha)/M)
    lys,lym = np.log(ys),np.log(y_max(alpha))
    if g(lys) >= 0.: # A is exactly the largest, so the two halos coincide
      ly = lys
    elif branch == 'high' and lys < lym and g(lym) >= 0.:
      ly = root_scalar(g,bracket=[lys,lym],xtol=1e-14).root
    else:
      # small y: the halo tends to the A = 0 one, so start the bracket below
      # that halo's y and step down until g > 0
      rho_s0 = __rhos_from_c_Einasto(c,rho_vir,alpha)
      lo = min(np.log(A/(rho_s0*r_2**1.5)),lys) - 1.
      while g(lo) < 0.:
        lo -= 1.
      ly = root_scalar(g,bracket=[lo,lys],xtol=1e-14).root
    r_s, rho_s = scale(ly)
  return r_s, rho_s

def A_max(c,M,rho_vir,alpha):
  '''
  Maximum sensible cusp coefficient A in a halo of mass M and concentration c.
  Here rho_vir is the virial density and alpha the Einasto shape parameter.
  
  This is the largest A anywhere on the curve of halos with this mass and
  concentration and y = A/(rho_s * r_s**1.5) <= y_max. For intermediate c it
  is the A of the y = y_max halo, but for large (or very small) c the curve
  turns over before y reaches y_max and the maximum, slightly larger, sits at
  y < y_max. In between the two, (c,M,A) admits two halos; see scale_from_c.
  '''
  return np.sqrt(M*rho_vir/__min_params(c,alpha))

def M_min(c,A,rho_vir,alpha):
  '''
  Minimum sensible halo mass M given concentration c and cusp coefficient A.
  Here rho_vir is the virial density and alpha the Einasto shape parameter.
  '''
  return __min_params(c,alpha) * A**2 / rho_vir

def c_min(M,A,rho_vir,alpha):
  '''
  Minimum sensible concentration c given halo mass M and cusp coefficient A.
  Here rho_vir is the virial density and alpha the Einasto shape parameter.
  '''
  params = M*rho_vir/A**2
  # not monotonic in c at c << 1, so take the largest root
  lncs = np.linspace(np.log(1e-4),np.log(1e4),33)
  d = np.array([np.log(__min_params(np.exp(l),alpha)/params) for l in lncs])
  if np.all(d < 0.):
    return 0.
  i = np.nonzero(d >= 0.)[0][-1]
  if i == len(lncs)-1:
    raise ValueError('c_min > 1e4')
  lnc = root_scalar(lambda lnc: np.log(__min_params(np.exp(lnc),alpha)/params),bracket=[lncs[i],lncs[i+1]]).root
  return np.exp(lnc)

# phase-space core

def __f0(y,alpha,z):
  # f at the bottom of the cored potential (e = 0); xmin << z because the
  # power law below the table is only O(xmin^2/z^2) accurate
  _x,_p,_m,_P = __table(z/1e4,max(__xfar(alpha),1e3*z),y,alpha,z,step=1.01)
  _d2pdP2 = __d2rho_dPhi2(_x,_p,_m,__density_slope(_x,y,alpha,z))
  # d2rho/dPhi2 is constant below the second point (the first is one-sided)
  _u = np.sqrt(_P[1:])
  return 1./(np.sqrt(2)*np.pi**2) * (_d2pdP2[1]*_u[0] + simpson(_d2pdP2[1:],x=_u))

def rc_from_fmax(fmax,r_s,rho_s,A,alpha,G=1.):
  '''
  Evaluate the core radius from phase-space conservation: the maximum of the
  cored distribution function (at the bottom of the potential) is f_max.
  
  Parameters:
    
    fmax: float
      Maximum phase-space density f_max. If G is not specified, this is assumed
      to be G^1.5 f_max.
    
    r_s: float
      Scale radius of halo (r_{-2} of the Einasto component).
    
    rho_s: float
      Scale density of halo (density of the Einasto component at r_s).
    
    A: float
      Cusp coefficient. This profile only makes sense if
      A <= y_max(alpha) * rho_s * r_s**1.5 (see y_max).
    
    alpha: float
      Einasto shape parameter.
    
    G: float
      Gravitational constant. If not specified, then we assume the first
      argument is G^1.5 f_max, which has dimensions of
      1/sqrt(mass*length^3). The result does not depend on G otherwise.
  
  Returns:
    
    r_c: float
      Core radius (0 or inf if it would lie outside 1e-60 r_s to 10 r_s).
  '''
  target = np.log(G**1.5 * fmax * r_s**3 * rho_s**0.5)
  y = A / (rho_s * r_s**1.5)
  fun = lambda lnz: np.log(__f0(y, alpha, np.exp(lnz))) - target
  # f0 falls monotonically with the core radius
  try:
    lnz = root_scalar(fun, bracket=[np.log(1e-60), np.log(10.)], xtol=1e-6).root
  except ValueError:
    return 0. if fun(np.log(1e-60)) < 0. else np.inf
  return np.exp(lnz) * r_s
