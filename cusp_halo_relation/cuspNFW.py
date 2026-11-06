import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import cumulative_trapezoid, cumulative_simpson, simpson
from scipy.interpolate import CubicSpline

# dimensionless radial profiles

def __density(x,y,z=0.):
  xt = (x**2+4*z**2)**1.5 / (x**2+8*z**2+1e-300) # guards against 0/0
  return np.sqrt(xt+y**2)/(xt**1.5*(1+xt)**2)
def __density_slope(x,y,z=0.):
  xt = (x**2+4*z**2)**1.5 / (x**2+8*z**2+1e-300)
  s = 0.5*(-7. + 4./(1. + xt) + xt/(xt + y**2 + 1e-300))
  dxt = x**2*(x**2 + 16*z**2) / ((x**2 + 4*z**2)*(x**2 + 8*z**2) + 1e-300) # d ln xt / d ln x
  return s * dxt
def __mass_midy(y,x):
  return 2*np.arcsinh(np.sqrt(x)/y) - (2 - y**2)*np.arctanh(np.sqrt(x*(1 - y**2)/(x + y**2)))/np.sqrt(1 - y**2) - np.sqrt(x*(x + y**2))/(1 + x)
def __mass_smally(y,x):
  return np.log(1+x) - x*(1-0.5*y**2)/(1.+x)
def __mass_largey(y,x):
  return np.sqrt(x)*(-30 - 10*x*(7 - y) + x**2*(-37 + y*(4 + 3*y)))/(15.*(1 + x)**2.5) + 2*np.arcsinh(np.sqrt(x))
def __mass(x,y,z=0.):
  if z == 0.:
    x,y = np.broadcast_arrays(x,y)
    y = y.astype(float)
    return 4*np.pi*np.piecewise(y,[(0.<=y)&(y<0.001),(0.001<=y)&(y<=0.999),(0.999<y)&(y<1.001)],[__mass_smally,__mass_midy,__mass_largey,np.nan],x)
  # numerical for z > 0
  _shape = np.shape(x)
  x = np.atleast_1d(np.array(x,dtype=float))
  _M = np.zeros_like(x)
  pos = x > 0
  if not np.any(pos):
    return _M.reshape(_shape)
  xmin = min(float(z)/100., float(np.min(x[pos]))/2., 1e-4)
  xmin = max(xmin, 1e-10)
  xmax = float(np.max(x))
  Nx = 1+int(np.ceil(np.log(xmax/xmin)/np.log(1.02))) # step in factors of 1.02
  _x = np.geomspace(xmin,xmax,Nx)
  _p = __density(_x,y,z)
  _m = cumulative_trapezoid(4*np.pi*_p*_x**3,x=np.log(_x),initial=0)
  _M[pos] = np.interp(np.log(x[pos]),np.log(_x),_m)
  return _M.reshape(_shape)
def __veldisp2_r_NFW_largex(x):
  logx = np.log(x)
  return (-3./16+logx/4)/x + (69./200+logx/10)/x**2 + (-97./1200-logx/20)/x**3 + (71./3675+logx/35)/x**4 + (-1./3136-logx/56)/x**5 + (-1271./211680+logx/84)/x**6
def __veldisp2_r(x,y,z=0.):
  xmin, xmax = np.min(x), 30.*max(np.max(x),1.,float(z))
  Nx = 1+int(np.ceil(np.log(xmax/xmin)/np.log(1.02))) # step in factors of 1.02
  _x = np.geomspace(xmin,xmax,Nx)
  _p, _m = __density(_x,y,z), __mass(_x,y,z)
  _ps2_0 = 4*np.pi*__veldisp2_r_NFW_largex(_x[-1]) * _p[-1]
  _ps2 = cumulative_trapezoid(-(_p*_m/_x)[::-1],x=np.log(_x)[::-1],initial=0)[::-1] + _ps2_0
  return np.exp(np.interp(np.log(x),np.log(_x),np.log(_ps2/_p),left=np.nan,right=np.nan))
def __potential_NFW_smallx(x):
  return x/2. - x**2/3. + x**3/4. - x**4/5. + x**5/6. - x**6/7.
def __potential_NFW_largex(x):
  return 1. - np.log(1.+x)/x
def __potential_NFW(x):
  return 4*np.pi*np.piecewise(x,[x<=1e-3],[__potential_NFW_smallx,__potential_NFW_largex])
def __potential_smallx(x,y):
  return np.pi/x * ((2*x-y**2)*np.sqrt(x*(x+y**2)) + (4*x+y**2)*y**2*np.log((np.sqrt(x)+np.sqrt(x+y**2))/y))
def __potential(x,y,z=0.,zero_at_inf=False):
  x = np.array(x).astype(float)
  if z == 0.:
    if y == 0.:
      pot = __potential_NFW(x)
    else:
      pot = np.zeros(np.shape(x))
      small = x <= 1e-3
      pot[small&(x>0.)] = __potential_smallx(x[small&(x>0.)],y)
      xmin, xmax = 1e-3, np.max(x)
      if xmax > xmin:
        Nx = 1+int(np.ceil(np.log(xmax/xmin)/np.log(1.02))) # step in factors of 1.02
        _x = np.geomspace(xmin,xmax,Nx)
        _m = __mass(_x,y)
        _pot_0 = __potential_smallx(_x[0],y)
        _pot = cumulative_trapezoid(_m/_x,x=np.log(_x),initial=0) + _pot_0
        pot[~small] = np.exp(CubicSpline(np.log(_x),np.log(_pot),extrapolate=False)(np.log(x[~small])))
  else:
    # z > 0: numerical, anchored by Phi ~ (2pi/3) rho_0 x^2 at small x
    pot = np.zeros(np.shape(x))
    pos = x > 0
    if np.any(pos):
      _p0 = __density(0.,y,z)
      xmin = min(float(z)/100., 1e-4, float(np.min(x[pos]))/2.)
      xmin = max(xmin, 1e-10)
      xmax = max(float(np.max(x)), 10.)
      Nx = 1+int(np.ceil(np.log(xmax/xmin)/np.log(1.02))) # step in factors of 1.02
      _x = np.geomspace(xmin,xmax,Nx)
      _m = __mass(_x,y,z)
      _pot_0 = (2*np.pi/3.)*_p0*xmin**2
      _pot = cumulative_trapezoid(_m/_x,x=np.log(_x),initial=0) + _pot_0
      pot[pos] = np.exp(CubicSpline(np.log(_x),np.log(_pot),extrapolate=False)(np.log(x[pos])))
  if zero_at_inf:
    return pot - __potential_inf(y,z)
  return pot
def __potential_inf_midy(y):
  return 1. - 0.5*y**2/np.sqrt(1.-y**2) * np.log(2.*(1.-np.sqrt(1.-y**2))/y**2 - 1.)
def __potential_inf_largey(y):
  return 2./3 * (1.+2*y)
def __potential_inf(y,z=0.):
  if z == 0.:
    y = np.array(y).astype(float)
    return 4*np.pi*np.piecewise(y,[(0.<=y)&(y<0.0001),(0.0001<=y)&(y<=0.9999),(0.9999<y)&(y<1.0001)],[1.,__potential_inf_midy,__potential_inf_largey,np.nan])
  # numerical for z > 0
  xmin = min(float(z)/100., 1e-4)
  xmin = max(xmin, 1e-10)
  xmax = 1e5
  Nx = 1+int(np.ceil(np.log(xmax/xmin)/np.log(1.02))) # step in factors of 1.02
  _x = np.geomspace(xmin,xmax,Nx)
  _m = __mass(_x,y,z)
  _p0 = __density(0.,y,z)
  _pot_0 = (2*np.pi/3.)*_p0*xmin**2
  # tail beyond xmax, where rho ~ x^g
  _tail = _m[-1]/xmax + 4*np.pi*__density(xmax,y,z)*xmax**2/(-__density_slope(xmax,y,z)-2.)
  return cumulative_trapezoid(_m/_x,x=np.log(_x),initial=0)[-1] + _pot_0 + _tail
def __table(xmin,xmax,y,z=0.,step=1.01):
  # density, mass and potential on a fixed lattice in ln x; power law below xmin
  h = np.log(step)
  n0,n1 = int(np.floor(np.log(xmin)/h)),int(np.ceil(np.log(xmax)/h))
  _lx = h*np.arange(n0,n1+1)
  _x = np.exp(_lx)
  _p = __density(_x,y,z)
  g0 = __density_slope(_x[0],y,z)
  _m = cumulative_simpson(4*np.pi*_p*_x**3,x=_lx,initial=0) + 4*np.pi*_p[0]*_x[0]**3/(3+g0)
  _pot = cumulative_simpson(_m/_x,x=_lx,initial=0) + _m[0]/(_x[0]*(2+g0))
  return _x,_p,_m,_pot

# radial profiles

def density(r,r_s,rho_s,A,r_c=0.):
  '''
  Evaluate density at radius r for a cusp-NFW density profile with scale radius
  r_s, scale density rho_s, and cusp coefficient A.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
      
    r_s: float or array
      Scale radius of halo.
    
    rho_s: float or array
      Scale density of halo.
      
    A: float or array
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
      
    r_c: float or array
      Radius of the phase-space core.
    
  Returns:
    
    rho: float or array
      Density at radius r.
  '''
  return __density(r/r_s,A/(rho_s*r_s**1.5),r_c/r_s) * rho_s

def mass(r,r_s,rho_s,A,r_c=0.):
  '''
  Evaluate mass enclosed within radius r for a cusp-NFW density profile with
  scale radius r_s, scale density rho_s, and cusp coefficient A.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
      
    r_s: float or array
      Scale radius of halo.
    
    rho_s: float or array
      Scale density of halo.
      
    A: float or array
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
    
    r_c: float or array
      Radius of the phase-space core.
    
  Returns:
    
    M: float or array
      Mass enclosed within radius r.
  '''
  return r_s**3*rho_s * __mass(r/r_s,A/(rho_s*r_s**1.5),r_c/r_s)

def veldisp2_r(r,r_s,rho_s,A,r_c=0.,G=1.):
  '''
  Evaluate squared radial velocity dispersion, sigma_r^2, at radius r for a
  cusp-NFW density profile with scale radius r_s, scale density rho_s, and cusp
  coefficient A. We assume an isotropic velocity distribution.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
      
    r_s: float
      Scale radius of halo.
    
    rho_s: float
      Scale density of halo.
      
    A: float
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
    
    r_c: float
      Radius of the phase-space core.
    
    G: float
      Gravitational constant. If not specified, we return sigma_r^2/G, which
      has dimensions of mass/length.
    
  Returns:
    
    sigma_r^2: float or array
      Squared radial velocity dispersion at radius r. If G was not specified,
      this is sigma_r^2/G instead.
  '''
  return G*rho_s*r_s**2 * __veldisp2_r(r/r_s,A/(rho_s*r_s**1.5),r_c/r_s)

def veldisp_r(r,r_s,rho_s,A,r_c=0.,G=1.):
  '''
  Evaluate radial velocity dispersion sigma_r at radius r for a cusp-NFW
  density profile with scale radius r_s, scale density rho_s, and cusp
  coefficient A. We assume an isotropic velocity distribution.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
      
    r_s: float
      Scale radius of halo.
    
    rho_s: float
      Scale density of halo.
      
    A: float
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
    
    r_c: float
      Radius of the phase-space core.
    
    G: float
      Gravitational constant. If not specified, we return sigma_r/sqrt(G),
      which has dimensions of sqrt(mass/length).
    
  Returns:
    
    sigma_r: float or array
      Radial velocity dispersion at radius r. If G was not specified, this is
      sigma_r/sqrt(G) instead.
  '''
  return np.sqrt(veldisp2_r(r,r_s,rho_s,A,r_c=r_c,G=G))

def potential(r,r_s,rho_s,A,r_c=0.,G=1.,zero_at_inf=False):
  '''
  Evaluate gravitational potential Phi at radius r for a cusp-NFW density
  profile with scale radius r_s, scale density rho_s, and cusp coefficient A.
  By default, we take the potential to be zero at r=0.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
      
    r_s: float
      Scale radius of halo.
    
    rho_s: float
      Scale density of halo.
      
    A: float
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
    
    r_c: float
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
  return G*rho_s*r_s**2 * __potential(r/r_s,A/(rho_s*r_s**1.5),z=r_c/r_s,zero_at_inf=zero_at_inf)

# distribution function

def __df(e,y,z=0.,zero_at_inf=False):
  # energy lattice in s(e) = ln e below Phi_inf/2 and -ln(Phi_inf-e) above it
  _shape = np.shape(e)
  e = np.atleast_1d(np.array(e,dtype=float))
  Pinf = __potential_inf(y,z)
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
  # radial table spanning these energies' potentials (Psi ~ 4 pi ln(x)/x at large x)
  xmin = 1e-4 if z == 0. else min(1e-4,float(z)/100.)
  while True:
    _x,_p,_m,_P = __table(xmin,10.,y,z,step=step)
    if _P[0] <= _e[0]/step**2 or xmin <= 1e-300:
      break
    g0 = __density_slope(xmin,y,z)
    xmin = max(xmin * 0.5*(_e[0]/step**2/_P[0])**(1./(2+g0)), 1e-300)
  xmax = 1e5
  while 4*np.pi*np.log(xmax)/xmax > _ec[-1]/step**2 and xmax < 1e300:
    xmax *= 1e3
  _x,_p,_m,_P = __table(xmin,xmax,y,z,step=step)
  # Psi = Phi_inf - Phi, integrated in from outside (precise where Phi -> Phi_inf),
  # plus the tail beyond the table
  g = __density_slope(_x,y,z)
  _Psi_0 = _m[-1]/_x[-1] + 4*np.pi*_p[-1]*_x[-1]**2/(-g[-1]-2.)
  _Psi = cumulative_simpson((_m/_x)[::-1],x=-np.log(_x)[::-1],initial=0)[::-1] + _Psi_0
  # Eddington's formula in its integrated-by-parts form,
  # f = 1/(sqrt(8) pi^2) int_e d2rho/dPhi2 dPhi/sqrt(Phi-e), which avoids
  # differencing in e (inaccurate near the bottom of a cored potential)
  _dpdP = _p*g*_x/_m # dPhi/dx = M/x^2
  _d2pdP2 = _dpdP * (g + np.gradient(np.log(-g),np.log(_x)) + 1. - 4*np.pi*_p*_x**3/_m) * _x/_m
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

def df(E,r_s,rho_s,A,r_c=0.,G=1.,zero_at_inf=False):
  '''
  Evaluate the distribution function f(E) for a cusp-NFW density profile with
  scale radius r_s, scale density rho_s, and cusp coefficient A. We assume an
  isotropic velocity distribution.
  
  Parameters:
    
    E: float or array
      Energy E. If G is not specified, this is assumed to be E/G instead.
      
    r_s: float
      Scale radius of halo.
    
    rho_s: float
      Scale density of halo.
      
    A: float
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
    
    r_c: float
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
  return __df(E/(G*rho_s*r_s**2),A/(rho_s*r_s**1.5),z=r_c/r_s,zero_at_inf=zero_at_inf) / (G**1.5*r_s**3*rho_s**0.5)

# conversions from halo concentration

def r2_from_rs(r_s,rho_s,A):
  '''
  Find the radius r_{-2} at which dlog(rho)/dlog(r) = -2 for a cusp-NFW density
  profile with scale radius r_s, scale density rho_s, and cusp coefficient A.
  
  r_{-2} ranges from r_s/2 (if A=rho_s*r_s**1.5) to r_s (if A=0).
  '''
  y = A / (rho_s * r_s**1.5) # <= 1
  return 0.5 * (1. - 1.5*y**2 + np.sqrt(1.-y**2+2.25*y**4)) * r_s

def rs_from_r2(r_2,rho_s,A):
  '''
  Find the cusp-NFW scale radius r_s, given r_2 (the radius r_{-2} at which
  dlog(rho)/dlog(r) = -2), the scale density rho_s, and the cusp coefficient A.
  '''
  z = A / (rho_s * r_2**1.5) # <= 2**1.5
  f = (1. + 18.*z**2 + 0.75*z*np.sqrt(72 + 564*z**2 + 6*z**4))**(1./3)
  return ((1/3.-z**2/2.)/f + (1+f)/3.) * r_2
  
def R_from_M(M,rho_vir):
  '''Virial radius R given halo mass M and virial density rho_vir.'''
  return (3*M/(4*np.pi*rho_vir))**(1./3)
  
def __rhos_from_c_NFW(c,rho_vir):
  return c**3/(np.log(1+c)-c/(1+c))*rho_vir/3.

def __min_params_largec(c):
  return 384*np.pi*(np.sqrt(c/(2 + c)) - np.arcsinh(np.sqrt(c/2)))**2/c**3
def __min_params_smallc(c):
  return 16*np.pi/3. - 24*np.pi/5.*c + 564*np.pi/175.*c**2
def __min_params(c):
  return np.piecewise(c,[c<0.001],[__min_params_smallc,__min_params_largec])

def scale_from_c(c,M,A,rho_vir,cmin_error=True):
  '''
  Find the cusp-NFW profile scale parameters, given halo concentration c and
  mass M.
  
  Parameters:
    
    c, M: floats
      Halo parameters.
    
    A: float
      Cusp coefficient parameter.
    
    rho_vir: float
      The virial density (e.g., 200 times the cosmological mean value).
    
    cmin_error: boolean
      If True, we raise an exception if the concentration is too low and would
      violate rho_s * r_s**1.5 >= A. If False, we increase the concentration to
      its minimum value and continue. Default is True.
    
  Returns:
    
    r_s, rho_s: floats
      The scale radius and scale density, respectively.
  '''
  R = R_from_M(M,rho_vir)
  r_2 = R / c # r_{-2}
  if A == 0.:
    return r_2,__rhos_from_c_NFW(c,rho_vir)
  if M*rho_vir/A**2 < __min_params(c):
    if cmin_error:
      raise Exception('M*rho_vir/A**2=%.3e must be >%.3e for c=%.2f.'%(M*rho_vir/A**2,__min_params(c),c))
    r_s = 2*R/c_min(M,A,rho_vir)
    rho_s = A/r_s**1.5
  else:
    rho_s = root_scalar(lambda rho_s: np.log(mass(R,rs_from_r2(r_2,rho_s,A),rho_s,A)/M),
                       bracket=[A/(2*r_2)**1.5,__rhos_from_c_NFW(c,rho_vir)*(1+1e-12)]).root
    r_s = rs_from_r2(r_2,rho_s,A)
  return r_s, rho_s

def A_max(c,M,rho_vir):
  '''
  Maximum sensible cusp coefficient A in a halo of mass M and concentration c.
  Here rho_vir is the virial density.
  '''
  return np.sqrt(M*rho_vir/__min_params(c))

def M_min(c,A,rho_vir):
  '''
  Minimum sensible halo mass M given concentration c and cusp coefficient A.
  Here rho_vir is the virial density.
  '''
  return __min_params(c) * A**2 / rho_vir

def c_min(M,A,rho_vir):
  '''
  Minimum sensible concentration c given halo mass M and cusp coefficient A.
  Here rho_vir is the virial density.
  '''
  params = M*rho_vir/A**2
  if params > 16*np.pi/3.:
    return 0.
  lnc = root_scalar(lambda lnc: np.log(__min_params(np.exp(lnc))/params),x0=0.,x1=1.).root
  return np.exp(lnc)

# phase-space core

def rc_from_fmax(fmax,r_s,rho_s,A,G=1.,prefactor=3e-5):
  '''
  Evaluate the core radius from phase-space conservation.
  
  Parameters:
    
    fmax: float
      Maximum phase-space density f_max. If G is not specified, this is assumed
      to be G^1.5 f_max.
      
    r_s: float
      Scale radius of halo.
    
    rho_s: float
      Scale density of halo.
      
    A: float
      Cusp coefficient. The profile only makes sense if rho_s * r_s**1.5 >= A.
    
    G: float
      Gravitational constant. If not specified, then we assume the first
      argument is G^1.5 f_max, which has dimensions of mass/length, and we return
      G^1.5 f(E).
    
    prefactor: float
      Prefactor in the "phase-space barrier" from arXiv:2207.05082. Default is
      prefactor=3e-5 (see appendix C of arXiv:2207.05082).
    
  Returns:
    
    r_c: float
      Core radius.
  '''
  target = prefactor * G**-3 * fmax**-2 / (rho_s * r_s**6)
  y = A / (rho_s * r_s**1.5)
  fun = lambda lnx: np.log(__density(np.exp(lnx), y) * np.exp(6*lnx) / target)
  try:
    lnx = root_scalar(fun, bracket=[-100., 100.]).root
  except ValueError:
    return 0. if fun(0.) > 0. else np.inf
  return np.exp(lnx) * r_s
