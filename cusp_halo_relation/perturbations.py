import numpy as np
import os
from scipy.special import hyp2f1
from scipy.integrate import simpson
from .transfer_EHfit import transferfunction_EisensteinHu

def W(x):
  '''Top-hat window function in Fourier space.'''
  return np.piecewise(x,[x<=0.16],[lambda x: 1 - x**2/10. + x**4/280. - x**6/15120. + x**8/1330560. - x**10/172972800.,lambda x: 3/x**3 * (np.sin(x)-x*np.cos(x))])

def growth_late(a,OmegaM,f_nc):
  '''
  
  Linear-order growth function in a spatially flat matter/dark energy-dominated
  universe.
  
  Parameters:
    
    a: float or array
      The scale factor.
      
    OmegaM: float
      The matter density parameter. We assume OmegaLambda = 1-OmegaM.
      
    f_nc: float
      Fraction of the matter that does not contribute to clustering (e.g.,
      baryons below the Jeans length scale).
      
  Returns:
    
    D(a): float or array
  
  '''
  g = 1.25 * np.sqrt(1.-0.96*f_nc) - 0.25
  x = (1.-OmegaM)/OmegaM * a**3
  return a**g * hyp2f1(g/3.,(g+2.)/3.,(7.+4.*g)/6.,-x)

def __power_table(k,species,f_nc,h,OmegaM,n_s,A_s=None,sigma8=None):
  
  # read file
  current_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
  zstr = '37'
  a = 1./(1+float(zstr))
  table = np.loadtxt(current_path + '/data/T2_%s.txt'%(zstr)).T
  k0 = table[0]
  P0 = table[1+species]
  
  # apply spectral tilt
  P0 *= (k0/0.05)**(n_s-1)
  
  # scale appropriately
  if (A_s is None and sigma8 is None) or (A_s is not None and sigma8 is not None):
    raise Exception('must specify exactly one of A_s and sigma8')
  if A_s is not None:
    P0 *= A_s
  if sigma8 is not None:
    sigma8_a = sigma8 * growth_late(a,OmegaM=OmegaM,f_nc=0.) / growth_late(1.,OmegaM=OmegaM,f_nc=0.)
    P0 *= sigma8_a**2 / simpson(P0*W(k0*8./h)**2,x=np.log(k0))
  
  # evolve to a=1
  P0 *= (growth_late(1.,OmegaM=OmegaM,f_nc=f_nc)/growth_late(a,OmegaM=OmegaM,f_nc=f_nc))**2
  
  # return interpolated P
  return np.exp(np.interp(np.log(k),np.log(k0),np.log(P0),left=0.,right=0.))

def __power_EisensteinHu(k,species,h,OmegaM,OmegaB,n_s,sigma8):
  tf = transferfunction_EisensteinHu(k,OmegaM*h**2,OmegaB/OmegaM,0.)[species]
  P0 = k**(3+n_s) * tf**2
  P0 *= sigma8**2 / simpson(P0*W(k*8./h)**2,x=np.log(k))
  return P0
  
def prepare_power(k,species,f_nc,h,OmegaM,OmegaB,n_s,A_s=None,sigma8=None,method='table'):
  '''
  
  Read pre-generated power spectrum, evolve it to z=0 with linear theory, and
  interpolate it over the provided k.
  
  Parameters:
    
    k: float or array
      Wavenumbers in Mpc^-1.
    
    species: 'm' (matter) or 'X' (dark matter)
    
    f_nc: float
      Fraction of the matter that does not contribute to clustering (e.g.,
      baryons below the Jeans length scale).
    
    h, OmegaM, OmegaB, n_s: floats
      Cosmological parameters. We assume OmegaLambda = 1-OmegaM.
      
    A_s, sigma8: floats
      Exactly one must be specified.
    
    method: 'table' or 'EH'
      - With 'table', we read the supplied CLASS-generated power spectrum.
      - With 'EH', we use the fitting function from Eisenstein & Hu
        [arXiv:astro-ph/9710252].
      
  Returns:
    
    P(k): array
  
  '''
    
  # sanitize species selection input
  if species.lower() == 'x' or 'dm' in species.lower():
    ispec = 0
  elif species[0].lower() == 'm':
    ispec = 2
  else:
    raise Exception('Invalid species selection: %s'%species)
  
  if method.lower() == 'table':
    return __power_table(k=k,species=ispec,f_nc=f_nc,h=h,OmegaM=OmegaM,n_s=n_s,A_s=A_s,sigma8=sigma8)
  elif method.lower() == 'eh':
    if f_nc > 0.:
      print('Warning: Eisenstein & Hu power spectrum does not account for f_nc>0 on small scales.')
    if (A_s is not None) or (sigma8 is None):
      raise Exception('A_s is currently not supported for Eisenstein & Hu method. Use sigma8.')
    return __power_EisensteinHu(k=k,species=ispec,h=h,OmegaM=OmegaM,OmegaB=OmegaB,n_s=n_s,sigma8=sigma8)
  else:
    raise Exception('Invalid power spectrum method: %s'%method)