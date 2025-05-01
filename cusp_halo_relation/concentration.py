import numpy as np
from scipy.optimize import root_scalar

__tab_c = np.geomspace(1,1e3,1000)
__tab_m2_NFW = (np.log(4)-1)/2 / (np.log(1+__tab_c)-__tab_c/(1+__tab_c))
__tab_p2_NFW = __tab_c**3 * __tab_m2_NFW
__tab_logc = np.log(__tab_c)
__tab_logm2_NFW = np.log(__tab_m2_NFW)
__tab_logp2_NFW = np.log(__tab_p2_NFW)

def __m2_from_p2_NFW(p2,Dvir,C):
  return np.exp(np.interp(np.log(p2*C/Dvir),__tab_logp2_NFW,__tab_logm2_NFW,left=np.nan,right=np.nan))
def __c_from_p2_NFW(p2,Dvir,C):
  return np.exp(np.interp(np.log(p2*C/Dvir),__tab_logp2_NFW,__tab_logc,left=np.nan,right=np.nan))

def c_L13_NFW(rho,M,Dvir=200.,C=776.,tmin=None):
  '''
  Estimate halo concentration c=R_vir/r_-2 from mass accretion history. Here
  R_vir is the virial radius and r_-2 is the radius at which dlnrho/dlnr=-2.
  We use the result from Ludlow et al. (2013) [arXiv:1302.0288] that the
  average density within r_-2 is C times the density of the universe when the
  halo virial mass was M_-2, where M_-2 is the mass enclosed in r_-2.
  
  Here we approximate that the halo has an NFW profile for radii r>r_-2.
  
  Parameters:
    
    rho: callable
      Density of the universe as a function of normalized time, where the
      normalization is such that the time is 1 at the "current" time (when we
      want to evaluate the halo concentration). The time parameter can be
      anything that is always positive (like cosmic time or scale factor).
      
    M: callable
      Halo mass as a function of the same normalized time.
    
    Dvir: float
      Virial overdensity factor. Default is 200.
      
    C: float
      Model parameter. Default is 776.
    
    tmin: float
      Minimum time parameter to consider. Default is None (no minimum).
      
  Returns:
    
    c: float
      Halo concentration parameter, c=R_vir/r_-2. 
  '''
  
  rho1 = rho(1.)
  M1 = M(1.)
  
  # identify minimum t that we can use
  if tmin is None:
    rootparams = dict(x0=0.,x1=-1.)
  else:
    rootparams = dict(bracket=[np.log(tmin),0.])
  try:
    logtmin_rho = root_scalar(lambda logt: np.log(rho(np.exp(logt))/rho1*C/Dvir)-__tab_logp2_NFW[-1],**rootparams).root
  except ValueError:
    logtmin_rho = np.log(tmin)
  try:
    logtmin_M = root_scalar(lambda logt: np.log(M(np.exp(logt))/M1)-__tab_logm2_NFW[-1],**rootparams).root
  except ValueError:
    logtmin_M = np.log(tmin)
  logtmin = max(logtmin_rho,logtmin_M)
  
  # evaluate concentration
  logt2 = root_scalar(lambda logt: np.log(__m2_from_p2_NFW(rho(np.exp(logt))/rho1,Dvir=Dvir,C=C)/(M(np.exp(logt))/M1)),bracket=(logtmin,0.)).root
  return __c_from_p2_NFW(rho(np.exp(logt2))/rho1,Dvir=Dvir,C=C)
