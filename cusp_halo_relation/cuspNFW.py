import numpy as np
from scipy.optimize import root_scalar

def density(r,rs,rhos,A):
  '''
  Evaluate density at radius r for a cusp-NFW density profile with scale radius
  rs, scale density rhos, and cusp amplitude A.
  
  This profile only makes sense if rhos * rs**1.5 >= A.
  '''
  x = r/rs
  y = A / (rhos * rs**1.5) # <= 1
  return np.sqrt(x+y**2)/(x**1.5*(1+x)**2) * rhos

def __mass(y,x):
  return 2*np.arcsinh(np.sqrt(x)/y) - (2 - y**2)*np.arctanh(np.sqrt(x*(1 - y**2)/(x + y**2)))/np.sqrt(1 - y**2) - np.sqrt(x*(x + y**2))/(1 + x)
def __mass_smallA(y,x):
  return np.log(1+x) - x*(1-0.5*y**2)/(1.+x)
def __mass_largeA(y,x):
  return np.sqrt(x)*(-30 - 10*x*(7 - y) + x**2*(-37 + y*(4 + 3*y)))/(15.*(1 + x)**2.5) + 2*np.arcsinh(np.sqrt(x))

def mass(r,rs,rhos,A):
  '''
  Evaluate mass enclosed within radius r for a cusp-NFW density profile with
  scale radius rs, scale density rhos, and cusp amplitude A.
  
  This profile only makes sense if rhos * rs**1.5 >= A.
  '''
  x,y = np.broadcast_arrays(r/rs,A/(rhos * rs**1.5))
  return 4*np.pi*rs**3*rhos*np.piecewise(y,[y<0.001,y>0.999],[__mass_smallA,__mass_largeA,__mass],x)

def r2_from_rs(rs,rhos,A):
  '''
  Find the radius r_{-2} at which dlog(rho)/dlog(r) = -2 for a cusp-NFW density
  profile with scale radius rs, scale density rhos, and cusp amplitude A.
  
  r2 ranges from rs/2 (if A=rhos*rs**1.5) to rs (if A=0).
  '''
  y = A / (rhos * rs**1.5) # <= 1
  return (1/2. - 3*y**2/4. + np.sqrt(4 - 4*y**2 + 9*y**4)/4.) * rs

def rs_from_r2(r2,rhos,A):
  '''
  Find the cusp-NFW scale radius rs, given r2 (the radius r_{-2} at which
  dlog(rho)/dlog(r) = -2), the scale density rhos, and the cusp amplitude A.
  '''
  z = A / (rhos * r2**1.5) # <= 2**1.5
  f = (8 + 6*z*(24*z + np.sqrt(72 + 564*z**2 + 6*z**4)))**(1./3)
  return (4 + f*(2 + f) - 6*z**2)/(6.*f) * r2
  
def R_from_M(M,rho_vir):
  '''Virial radius R given halo mass M and virial density rho_vir.'''
  return (3*M/(4*np.pi*rho_vir))**(1./3)
  
def __rhos_from_c_NFW(c,rho_vir):
  return c**3/(np.log(1+c)-c/(1+c))*rho_vir/3.

def __min_params_largec(c):
  return 384*np.pi*(np.sqrt(c/(2 + c)) - np.arcsinh(np.sqrt(c/2)))**2/c**3
def __min_params_smallc(c):
  return 16.755160819145562 - 15.079644737231007*c + 10.12490432356939*c**2
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
      violate rhos * rs**1.5 >= A. If False, we increase the concentration to
      its minimum value and continue. Default is True.
    
  Returns:
    
    rs, rhos: floats
      The scale radius and scale density, respectively.
  
  '''
  R = R_from_M(M,rho_vir)
  r2 = R / c # r_{-2}
  if A == 0.:
    return r2,__rhos_from_c_NFW(c,rho_vir)
  if M*rho_vir/A**2 < __min_params(c):
    if cmin_error:
      raise Exception('M*rho_vir/A**2=%.3e must be >%.3e for c=%.2f.'%(M*rho_vir/A**2,__min_params(c),c))
    else:
      c = c_min(M,A,rho_vir)
      r2 = R / c # r_{-2}
  rhos = root_scalar(lambda rhos: np.log(mass(R,rs_from_r2(r2,rhos,A),rhos,A)/M),
                     bracket=[A/(2*r2)**1.5,__rhos_from_c_NFW(c,rho_vir)]).root
  rs = rs_from_r2(r2,rhos,A)
  return rs, rhos

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
  lnc = root_scalar(lambda lnc: np.log(__min_params(np.exp(lnc))*A**2/(M*rho_vir)),x0=0.,x1=1.).root
  return np.exp(lnc)