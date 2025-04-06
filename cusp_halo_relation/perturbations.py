import numpy as np
import os
from scipy.special import hyp2f1

def growth(a,OmegaM,fb):
  '''
  
  Linear-order growth function in a spatially flat matter/dark energy-dominated
  universe.
  
  Parameters:
    
    a: float or array
      The scale factor.
      
    OmegaM: float
      The matter density parameter. We assume OmegaLambda = 1-OmegaM.
      
    fb: float
      Fraction of the matter that does not contribute to clustering (e.g.,
      baryons below the Jeans length scale).
      
  Returns:
    
    D(a): float or array
  
  '''
  g = 1.25 * np.sqrt(1.-0.96*fb) - 0.25
  x = (1.-OmegaM)/OmegaM * a**3
  return a**g * hyp2f1(g/3.,(g+2.)/3.,(7.+4.*g)/6.,-x)

def load_power(k,which,h,OmegaM,fb):
  '''
  
  Read pre-generated power spectrum, evolve it to z=0 with linear theory, and
  interpolate it over the provided k.
  
  Parameters:
    
    k: float or array
    
    which: 'm' (matter) or 'X' (dark matter'
    
    h, OmegaM: floats
      Cosmological parameters. We assume OmegaLambda = 1-OmegaM.
      
    fb: float
      Fraction of the matter that does not contribute to clustering (e.g.,
      baryons below the Jeans length scale).
      
  Returns:
    
    P(k): array
  
  '''
  
  # sanitize species selection input
  if which.lower() == 'x' or 'dm' in which.lower():
    which = 'X'
  elif which[0].lower() == 'm':
    which = 'm'
  else:
    raise ValueError('invalid species selection')
  
  # read file
  current_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
  zstr = '37'
  k0,P0 = np.loadtxt(current_path + '/data/power_%s_%s.txt'%(which,zstr)).T
  
  # evolve to a=1
  a = 1./(1+float(zstr))
  P0 *= (growth(1.,OmegaM,0.)/growth(a,OmegaM,0.))**2
  
  # return interpolated P
  return np.exp(np.interp(np.log(k),np.log(k0),np.log(P0),left=0.,right=0.))