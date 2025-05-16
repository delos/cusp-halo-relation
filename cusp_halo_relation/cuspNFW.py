import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import cumtrapz

def density(r,r_s,rho_s,A):
  '''
  Evaluate density at radius r for a cusp-NFW density profile with scale radius
  r_s, scale density rho_s, and cusp amplitude A.
  
  This profile only makes sense if rho_s * r_s**1.5 >= A.
  '''
  x = r/r_s
  y = A / (rho_s * r_s**1.5) # <= 1
  return np.sqrt(x+y**2)/(x**1.5*(1+x)**2) * rho_s

def __mass(y,x):
  return 2*np.arcsinh(np.sqrt(x)/y) - (2 - y**2)*np.arctanh(np.sqrt(x*(1 - y**2)/(x + y**2)))/np.sqrt(1 - y**2) - np.sqrt(x*(x + y**2))/(1 + x)
def __mass_smallA(y,x):
  return np.log(1+x) - x*(1-0.5*y**2)/(1.+x)
def __mass_largeA(y,x):
  return np.sqrt(x)*(-30 - 10*x*(7 - y) + x**2*(-37 + y*(4 + 3*y)))/(15.*(1 + x)**2.5) + 2*np.arcsinh(np.sqrt(x))

def mass(r,r_s,rho_s,A):
  '''
  Evaluate mass enclosed within radius r for a cusp-NFW density profile with
  scale radius r_s, scale density rho_s, and cusp amplitude A.
  
  This profile only makes sense if rho_s * r_s**1.5 >= A.
  '''
  x,y = np.broadcast_arrays(r/r_s,A/(rho_s * r_s**1.5))
  return 4*np.pi*r_s**3*rho_s*np.piecewise(y,[(0.<=y)&(y<0.001),(0.001<=y)&(y<=0.999),(0.999<y)&(y<1.001)],[__mass_smallA,__mass,__mass_largeA,np.nan],x)

def r2_from_rs(r_s,rho_s,A):
  '''
  Find the radius r_{-2} at which dlog(rho)/dlog(r) = -2 for a cusp-NFW density
  profile with scale radius r_s, scale density rho_s, and cusp amplitude A.
  
  r_{-2} ranges from r_s/2 (if A=rho_s*r_s**1.5) to r_s (if A=0).
  '''
  y = A / (rho_s * r_s**1.5) # <= 1
  return 0.5 * (1. - 1.5*y**2 + np.sqrt(1.-y**2+2.25*y**4)) * r_s

def rs_from_r2(r_2,rho_s,A):
  '''
  Find the cusp-NFW scale radius r_s, given r_2 (the radius r_{-2} at which
  dlog(rho)/dlog(r) = -2), the scale density rho_s, and the cusp amplitude A.
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
      Cusp amplitude parameter.
    
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
  Maximum sensible cusp amplitude A in a halo of mass M and concentration c.
  Here rho_vir is the virial density.
  '''
  return np.sqrt(M*rho_vir/__min_params(c))

def M_min(c,A,rho_vir):
  '''
  Minimum sensible halo mass M given concentration c and cusp amplitude A.
  Here rho_vir is the virial density.
  '''
  return __min_params(c) * A**2 / rho_vir

def c_min(M,A,rho_vir):
  '''
  Minimum sensible concentration c given halo mass M and cusp amplitude A.
  Here rho_vir is the virial density.
  '''
  params = M*rho_vir/A**2
  if params >= 16*np.pi/3.:
    return 0.
  lnc = root_scalar(lambda lnc: np.log(__min_params(np.exp(lnc))/params),x0=0.,x1=1.).root
  return np.exp(lnc)

def __sigma2_r_large(x):
  logx = np.log(x)
  return (-3./16+logx/4)/x + (69./200+logx/10)/x**2 + (-97./1200-logx/20)/x**3 + (71./3675+logx/35)/x**4 + (-1./3136-logx/56)/x**5 + (-1271./211680+logx/84)/x**6

def sigma_r(r,r_s,rho_s,A,G=1.):
  '''
  Evaluate radial velocity dispersion at radius r for a cusp-NFW density
  profile with scale radius r_s, scale density rho_s, and cusp amplitude A.
  We assume an isotropic velocity distribution.
  
  This profile only makes sense if rho_s * r_s**1.5 >= A.
  
  If G is not specified, we return sigma_r/sqrt(G), which has dimensions of
  sqrt(mass/length).
  '''
  rmin = np.min(r)
  rmax = 30.*max(np.max(r),r_s)
  Nr = int(np.round(np.log(rmax/rmin)/np.log(1.02))) # step in factors of 1.02
  _r = np.geomspace(rmin,rmax,Nr)
  _p = density(_r,r_s,rho_s,A)
  _m = mass(_r,r_s,rho_s,A)
  _ps2_0 = 4*np.pi*G*rho_s*r_s**2*__sigma2_r_large(_r[-1]/r_s) * _p[-1]
  _ps2 = cumtrapz(-(_p*G*_m/_r)[::-1],x=np.log(_r)[::-1],initial=0)[::-1] + _ps2_0
  return np.interp(np.log(r),np.log(_r),np.sqrt(_ps2/_p),left=np.nan,right=np.nan)