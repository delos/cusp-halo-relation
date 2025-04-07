import numpy as np
from scipy.optimize import root_scalar

from . import main
from . import perturbations

# constants
rhoCrit_h2 = 2.7744948e11 # Msol/Mpc^3

# free-streaming models
def free_streaming_length(model,mX,omegaX,h,spin=0.5):
  '''
  
  Warm dark matter free-streaming length, alpha.
  
  Parameters:
      
    model: 'VA23' or 'V05'
      - 'VA23': Vogel & Abazajian (2023), arXiv:2210.10753
      - 'V05': Viel et al. (2005), arXiv:astro-ph/0501562
    
    mX: float
      Dark matter mass in keV.
      
    omegaX: float
      Dark matter density parameter, OmegaX * h^2
      
    h: float
    
    spin: float
      1/2 or 3/2. Only relevant if model=='VA23'. Default is 1/2.
  
  Returns:
    
    alpha: float
      Free-streaming length in Mpc.
  
  '''
  if model == 'VA23':
    p = {
      0.5:[0.0437,-1.188,1.049,2.012,0.2463],
      1.5:[0.0345,-1.195,1.025,2.012,0.2463],
      }[spin]
    alpha = p[0] * mX**p[1] * (omegaX/0.12)**p[4] * (h/0.6736)**p[3] * h**-1
  elif model == 'V05':
    alpha = 0.070 * mX**-1.11 * (omegaX/0.1225)**0.11
  else:
    raise ValueError('invalid free-streaming model')
  return alpha

def free_streaming_T(x,model,spin=0.5):
  '''
  
  Warm dark matter transfer function T(k).
  
  Parameters:
    
    x: array or float
      alpha*k, where alpha is the free-streaming length.
      
    model: 'VA23' or 'V05'
      - 'VA23': Vogel & Abazajian (2023), arXiv:2210.10753
      - 'V05': Viel et al. (2005), arXiv:astro-ph/0501562
    
    spin: float
      1/2 or 3/2. Only relevant if model=='VA23'. Default is 1/2.
  
  Returns:
    
    T(k): array or float
  
  '''
  if model == 'VA23':
    nu = {0.5:1.049,1.5:1.025}[spin]
  elif model == 'V05':
    nu = 1.12
  else:
    raise ValueError('invalid free-streaming model')
  
  return (1. + x**(2.*nu))**(-5./nu)

class CuspHaloModelWDM(main.CuspHaloModel):
  '''
  
  Class for evaluating the cusp-halo relation for a cosmology with warm dark
  matter.
  
  Parameters:
    
    mX: float
      Dark matter particle mass in keV. Either mX or Mhm must be specified, but
      not both.
    
    Mhm: float
      Half-mode mass scale.
    
    h, OmegaM, OmegaB: floats
      Cosmological parameters.
      Defaults are h=0.6774, OmegaM=0.3089, OmegaB=0.04886.
      Note: we assume baryons contribute to the structure growth rate and halo
      masses but not to cusp m and A.
      
    fs_model: 'VA23' or 'V05'
      Warm dark matter power spectrum model to use. Default is 'VA23'.
      - 'VA23': Vogel & Abazajian (2023), arXiv:2210.10753
      - 'V05': Viel et al. (2005), arXiv:astro-ph/0501562
      
    spin: 1/2 or 3/2
      Dark matter spin, only relevant if transfer=='VA23'. Default is 1/2.
  
  Methods:
    
    m(M,z):
      Cusp mass m, given halo mass M and redshift z.
    
    A(M,D): 
      Cusp coefficient A, given halo mass M and redshift z.
  
  '''
  
  def __init__(self,mX=None,Mhm=None,h=0.6736,OmegaM=0.3089,OmegaB=0.04886,fs_model='VA23',spin=0.5):
    
    # cosmology
    OmegaX = OmegaM - OmegaB
    rhoCrit = rhoCrit_h2 * h**2
    rhoM = rhoCrit * OmegaM
    
    # warm DM
    xhm = np.real(root_scalar(lambda x: free_streaming_T(x,fs_model,spin)-0.5,bracket=(1e-3,1e3),).root)
    if (mX is None and Mhm is None) or (mX is not None and Mhm is not None):
      raise ValueError('must specify exactly one of mX and Mhm')
    if mX is not None:
      lfs = free_streaming_length(fs_model,mX,OmegaX*h**2,h,spin)
      khm = xhm / lfs
      Mhm = 4*np.pi/3 * rhoM * (np.pi/khm)**3
    if Mhm is not None: # = 4*np.pi/3 * rhoM * (lhm/2)**3
      khm = np.pi * (4*np.pi/3 * rhoM/Mhm)**(1./3)
      lfs = xhm / khm
      mX = np.real(root_scalar(lambda m: free_streaming_length(fs_model,m,OmegaX*h**2,h,spin)/lfs-1.,x0=1.,x1=10).root)
    
    print('mX = %f keV'%float(mX))
    print('Mhm = %e Msol'%Mhm)
    
    # load power spectrum and apply transfer function
    k = np.geomspace(1e-5,1e3/lfs,1000)
    T = free_streaming_T(lfs*k,fs_model,spin)
    P = perturbations.load_power(k,'m',h,OmegaM,0.) * T**2
    
    # initialize parent
    super().__init__(k,P,growth=lambda a: perturbations.growth(a,OmegaM,0.),
                     rho=rhoM,fDM=OmegaX/OmegaM,amax=1.)
    
  def m_at_z(self,M,z=0.):
    '''
    
    Evaluate the predicted cusp mass m for a halo of mass M at redshift z.
    
    Parameters:
      
      M: float or array
      
      z: float or array
        Redshift, default is z=0.
    
    Returns:
      
      m: float or array
        
    '''
    
    return self.model_m(M,1./(1.+z))
  
  def A_at_z(self,M,z=0.):
    '''
    
    Evaluate the predicted cusp A for a halo of mass M at redshift z.
    
    Parameters:
      
      M: float or array
      
      a: float or array
        Redshift, default is z=0.
    
    Returns:
      
      A: float or array
        
    '''
    
    return self.model_A(M,1./(1.+z))
