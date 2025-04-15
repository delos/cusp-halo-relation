import numpy as np
import os
from scipy.special import hyp2f1
from scipy.integrate import simpson

def W(x):
  '''Top-hat window function in Fourier space.'''
  return np.piecewise(x,[x<=0.16],[lambda x: 1 - x**2/10. + x**4/280. - x**6/15120. + x**8/1330560. - x**10/172972800.,lambda x: 3/x**3 * (np.sin(x)-x*np.cos(x))])

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

def load_power(k,which,fb,h,OmegaM,n_s,A_s=None,sigma8=None):
  '''
  
  Read pre-generated power spectrum, evolve it to z=0 with linear theory, and
  interpolate it over the provided k.
  
  Parameters:
    
    k: float or array
    
    which: 'm' (matter) or 'X' (dark matter)
    
    fb: float
      Fraction of the matter that does not contribute to clustering (e.g.,
      baryons below the Jeans length scale).
    
    h, OmegaM, n_s: floats
      Cosmological parameters. We assume OmegaLambda = 1-OmegaM.
      
    A_s, sigma8: floats
      Exactly one must be specified.
      
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
  k0,P0 = np.loadtxt(current_path + '/data/T2_%s_%s.txt'%(which,zstr)).T
  
  # apply spectral tilt
  P0 *= (k0/0.05)**(n_s-1)
  
  # evolve to a=1
  a = 1./(1+float(zstr))
  P0 *= (growth(1.,OmegaM=OmegaM,fb=fb)/growth(a,OmegaM=OmegaM,fb=fb))**2
  
  # scale appropriately
  if (A_s is None and sigma8 is None) or (A_s is not None and sigma8 is not None):
    raise ValueError('must specify exactly one of A_s and sigma8')
  if A_s is not None:
    P0 *= A_s
  if sigma8 is not None:
    P0 *= sigma8**2 / simpson(P0*W(k0*8.)**2,x=np.log(k0))
  
  # h/Mpc to Mpc
  k0 *= h
  
  # return interpolated P
  return np.exp(np.interp(np.log(k),np.log(k0),np.log(P0),left=0.,right=0.))