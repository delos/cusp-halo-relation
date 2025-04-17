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
  
    cutoff: string or tuple
      Warm dark matter power spectrum model to use. Default is 'VA23'.
      - 'VA23': Vogel & Abazajian (2023), arXiv:2210.10753
      - 'V05': Viel et al. (2005), arXiv:astro-ph/0501562
      Alternatively, pass a custom transfer function in the form (k/Mpc^-1, T),
      where T(k) is the transfer function multiplying the Fourier-space density
      contrast \delta(k).
    
    mX: float
      Dark matter particle mass in keV. Either mX or Mhm may be specified, but
      not both. Alternatively, if a custom transfer function is used above,
      then neither mX nor Mhm should be specified.
    
    Mhm: float
      Half-mode mass scale.
    
    h, OmegaM, OmegaB: floats
      Cosmological parameters.
      Defaults are h=0.6774, OmegaM=0.3089, OmegaB=0.04886.
      Note: we assume baryons contribute to the structure growth rate and halo
      masses but not to cusp m and A.
      
    spin: 1/2 or 3/2
      Dark matter spin, only relevant if transfer=='VA23'. Default is 1/2.
      
    n_s: float
      Primordial spectral index, default 0.9649.
      
    A_s, sigma8:
      Primordial spectral amplitude or sigma_8. If specified, sigma8 supersedes
      A_s. Default is A_s=2.100e-9.
  
  Methods:
    
    m_at_z(M,z):
      Cusp mass m, given halo mass M and redshift z.
    
    A_at_z(M,z): 
      Cusp coefficient A, given halo mass M and redshift z.
      
    c_at_z(z):
      Typical concentration parameter for a small halo at redshift z.
  
  '''
  
  def __init__(self,cutoff='VA23',mX=None,Mhm=None,h=0.6736,OmegaM=0.3089,OmegaB=0.04886,spin=0.5,n_s=0.9649,A_s=2.100e-9,sigma8=None):
    
    # cosmology
    OmegaX = OmegaM - OmegaB
    rhoCrit = rhoCrit_h2 * h**2
    rhoM = rhoCrit * OmegaM
    
    # warm DM
    if isinstance(cutoff,str):
      xhm = np.exp(root_scalar(lambda logx: free_streaming_T(np.exp(logx),cutoff,spin)-0.5,bracket=(-7.,7.),).root)
      if (mX is None and Mhm is None) or (mX is not None and Mhm is not None):
        raise ValueError('must specify exactly one of mX and Mhm')
      if mX is not None:
        lfs = free_streaming_length(cutoff,mX,OmegaX*h**2,h,spin)
        khm = xhm / lfs
      if Mhm is not None: # = 4*np.pi/3 * rhoM * (lhm/2)**3
        khm = np.pi * (4*np.pi/3 * rhoM/Mhm)**(1./3)
        lfs = xhm / khm
        mX = np.exp(root_scalar(lambda logm: free_streaming_length(cutoff,np.exp(logm),OmegaX*h**2,h,spin)/lfs-1.,x0=0.,x1=1.).root)
      print('CuspHaloModelWDM: Using model %s with mX = %f keV'%(cutoff,mX))
      T = lambda k: free_streaming_T(lfs*k,cutoff,spin)
    else:
      if not (mX is None and Mhm is None):
        raise ValueError('if custom T(k) is passed, neither mX nor Mhm should be')
      cut_k, cut_T = cutoff
      cut_sort = np.argsort(cut_k)
      cut_k, cut_T = cut_k[cut_sort], cut_T[cut_sort]
      cut_ihm = np.where(cut_T<0.5)[0][0]
      khm = np.exp(np.interp(0.5,cut_T[cut_ihm:cut_ihm-2:-1],np.log(cut_k[cut_ihm:cut_ihm-2:-1]),left=np.nan,right=np.nan))
      T = lambda k: np.interp(np.log(k),np.log(cut_k),cut_T,left=1.,right=0.)
    self.khm = khm
    self.Mhm = 4*np.pi/3 * rhoM * (np.pi/khm)**3
    print('CuspHaloModelWDM: khm = %e Mpc^-1, Mhm = %e Msol'%(self.khm,self.Mhm))
    
    # load power spectrum and apply transfer function
    k = np.geomspace(1e-5,1e3*khm,1000)
    P = perturbations.load_power(k,which='m',fb=0.,h=h,OmegaM=OmegaM,n_s=n_s,A_s=(A_s if sigma8 is None else None),sigma8=sigma8) * T(k)**2
    
    # initialize parent
    super().__init__(k,P,growth=lambda a: perturbations.growth(a,OmegaM=OmegaM,fb=0.),
                     rho=rhoM,fDM=OmegaX/OmegaM,amax=1.)
    
  def m_at_z(self,M,z):
    '''Predicted cusp mass m for a halo of mass M at redshift z.'''
    return self.m(M,1./(1.+z))
  
  def A_at_z(self,M,z):
    '''Predicted cusp A for a halo of mass M at redshift z.'''
    return self.A(M,1./(1.+z))
  
  def c_at_z(self,z):
    '''Estimated typical concentration parameter of small halos at redshift z.'''
    return self.characteristic_c(1./(1.+z))
