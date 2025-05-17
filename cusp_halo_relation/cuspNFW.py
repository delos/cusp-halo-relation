import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import cumtrapz

# dimensionless radial profiles

def __density(x,y):
  return np.sqrt(x+y**2)/(x**1.5*(1+x)**2)
def __mass_midy(y,x):
  return 2*np.arcsinh(np.sqrt(x)/y) - (2 - y**2)*np.arctanh(np.sqrt(x*(1 - y**2)/(x + y**2)))/np.sqrt(1 - y**2) - np.sqrt(x*(x + y**2))/(1 + x)
def __mass_smally(y,x):
  return np.log(1+x) - x*(1-0.5*y**2)/(1.+x)
def __mass_largey(y,x):
  return np.sqrt(x)*(-30 - 10*x*(7 - y) + x**2*(-37 + y*(4 + 3*y)))/(15.*(1 + x)**2.5) + 2*np.arcsinh(np.sqrt(x))
def __mass(x,y):
  x,y = np.broadcast_arrays(x,y)
  return 4*np.pi*np.piecewise(y,[(0.<=y)&(y<0.001),(0.001<=y)&(y<=0.999),(0.999<y)&(y<1.001)],[__mass_smally,__mass_midy,__mass_largey,np.nan],x)
def __veldisp2_r_NFW_largex(x):
  logx = np.log(x)
  return (-3./16+logx/4)/x + (69./200+logx/10)/x**2 + (-97./1200-logx/20)/x**3 + (71./3675+logx/35)/x**4 + (-1./3136-logx/56)/x**5 + (-1271./211680+logx/84)/x**6
def __veldisp2_r(x,y):
  xmin, xmax = np.min(x), 30.*max(np.max(x),1.)
  Nx = int(np.round(np.log(xmax/xmin)/np.log(1.02))) # step in factors of 1.02
  _x = np.geomspace(xmin,xmax,Nx)
  _p, _m = __density(_x,y), __mass(_x,y)
  _ps2_0 = 4*np.pi*__veldisp2_r_NFW_largex(_x[-1]) * _p[-1]
  _ps2 = cumtrapz(-(_p*_m/_x)[::-1],x=np.log(_x)[::-1],initial=0)[::-1] + _ps2_0
  return np.exp(np.interp(np.log(x),np.log(_x),np.log(_ps2/_p),left=np.nan,right=np.nan))
def __potential_smallx(x,y):
  return np.pi/x * ((2*x-y**2)*np.sqrt(x*(x+y**2)) + (4*x+y**2)*y**2*np.log((np.sqrt(x)+np.sqrt(x+y**2))/y))
def __potential(x,y):
  x = np.array(x)
  pot = np.zeros(np.shape(x))
  small = x <= 1e-3
  pot[small] = __potential_smallx(x[small],y)
  xmin, xmax = 1e-3, np.max(x)
  if xmax > xmin:
    Nx = int(np.round(np.log(xmax/xmin)/np.log(1.02))) # step in factors of 1.02
    _x = np.geomspace(xmin,xmax,Nx)
    _m = __mass(_x,y)
    _pot_0 = __potential_smallx(_x[0],y)
    _pot = cumtrapz(_m/_x,x=np.log(_x),initial=0) + _pot_0
    pot[~small] = np.exp(np.interp(np.log(x[~small]),np.log(_x),np.log(_pot),left=np.nan,right=np.nan))
  return pot

# radial profiles

def density(r,r_s,rho_s,A):
  '''
  Evaluate density at radius r for a cusp-NFW density profile with scale radius
  r_s, scale density rho_s, and cusp coefficient A.
  
  This profile only makes sense if rho_s * r_s**1.5 >= A.
  '''
  return __density(r/r_s,A/(rho_s*r_s**1.5)) * rho_s

def mass(r,r_s,rho_s,A):
  '''
  Evaluate mass enclosed within radius r for a cusp-NFW density profile with
  scale radius r_s, scale density rho_s, and cusp coefficient A.
  
  This profile only makes sense if rho_s * r_s**1.5 >= A.
  '''
  return r_s**3*rho_s * __mass(r/r_s,A/(rho_s*r_s**1.5))

def veldisp2_r(r,r_s,rho_s,A,G=1.):
  '''
  Evaluate squared radial velocity dispersion, sigma_r^2, at radius r for a
  cusp-NFW density profile with scale radius r_s, scale density rho_s, and cusp
  coefficient A. We assume an isotropic velocity distribution.
  
  This profile only makes sense if rho_s * r_s**1.5 >= A.
  
  If G is not specified, we return sigma_r^2/G, which has dimensions of
  mass/length.
  '''
  return G*rho_s*r_s**2 * __veldisp2_r(r/r_s,A/(rho_s*r_s**1.5))

def veldisp_r(r,r_s,rho_s,A,G=1.):
  '''
  Evaluate radial velocity dispersion sigma_r at radius r for a cusp-NFW
  density profile with scale radius r_s, scale density rho_s, and cusp
  coefficient A. We assume an isotropic velocity distribution.
  
  This profile only makes sense if rho_s * r_s**1.5 >= A.
  
  If G is not specified, we return sigma_r/sqrt(G), which has dimensions of
  sqrt(mass/length).
  '''
  return np.sqrt(veldisp2_r(r,r_s,rho_s,A,G=G))

def potential(r,r_s,rho_s,A,G=1.):
  '''
  Evaluate radial velocity dispersion at radius r for a cusp-NFW density
  profile with scale radius r_s, scale density rho_s, and cusp coefficient A.
  We assume an isotropic velocity distribution.
  
  This profile only makes sense if rho_s * r_s**1.5 >= A.
  
  If G is not specified, we return sigma_r/sqrt(G), which has dimensions of
  sqrt(mass/length).
  '''
  return G*rho_s*r_s**2 * __potential(r/r_s,A/(rho_s*r_s**1.5))

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
    else:
      c = c_min(M,A,rho_vir)
      r_2 = R / c # r_{-2}
  rho_s = root_scalar(lambda rho_s: np.log(mass(R,rs_from_r2(r_2,rho_s,A),rho_s,A)/M),
                     bracket=[A/(2*r_2)**1.5,__rhos_from_c_NFW(c,rho_vir)]).root
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
  if params >= 16*np.pi/3.:
    return 0.
  lnc = root_scalar(lambda lnc: np.log(__min_params(np.exp(lnc))/params),x0=0.,x1=1.).root
  return np.exp(lnc)
