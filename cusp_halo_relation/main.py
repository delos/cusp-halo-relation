import numpy as np
from scipy.integrate import simpson

# moments of the power spectrum
def sigmaj2(j,k,P):
  integrand = P*k**(2*j)
  return simpson(integrand,x=np.log(k),axis=0)
def sigmaj(j,k,P):
  return np.sqrt(sigmaj2(j,k,P))

class CuspHaloModel(object):
  '''
  
  Class for evaluating the cusp-halo relation for a general cosmology.
  
  Parameters:
    
    k, P: arrays
      A table of the dimensionless matter power spectrum P(k), evaluated in
      linear theory when a=1.
    
    growth: float or function with 1 argument
      The linear growth function, D(a), as a function of the scale factor a.
      If a float g, we will assume D(a) = a^g. Default is g=1.
    
    rho: float
      The cosmological mean density of matter that contributes to gravitational
      clustering. This is used for halo masses. The density should be specified
      when a=1. Default is rho=1, so results are in units of the mean density.
    
    fDM: float
      The fraction of rho that is dark matter. This is a scaling factor for the
      prompt cusps, since they are assumed to be dark matter only. Default is
      fDM=1.
      
    amax: float
      Largest scale factor we are interested in. Default is 1.
  
  Methods:

    model_m(M,a): 
      Median cusp mass m at halo mass M and scale factor a.
    
    model_A(M,a):
      Median cusp coefficient A at halo mass M and scale factor a.
      
    characteristic_m(a):
      Characteristic cusp mass at scale factor a.
      
    characteristic_A(a):
      Characteristic cusp coefficient at scale factor a.
    
    collapse_a(M):
      Estimated collapse scale factor for a halo of mass M.
      
    growth(a):
      Linear growth function.
  
  '''
  
  def __init__(self,k,P,growth=1.,rho=1.,fDM=1.,amax=1.):
    
    # evaluate the characteristic scales
    self.sigma0 = sigmaj(0,k,P) # evolves as D
    self.sigma1 = sigmaj(1,k,P) # evolves as D/a
    self.sigma2 = sigmaj(2,k,P) # evolves as D/a^2
    
    # record the density parameters
    self.rho = rho # evolves as a^-3
    self.fDM = fDM
    
    # parameters of the cusp-peak relation
    self.A_coef = 24.
    self.m_coef = 7.3
    
    # parameters of the cusp-halo relation
    self.A_m_index = 2.
    self.A_m_coef  = 0.7
    self.massfac_fun = lambda sigma0: np.exp(-5./sigma0)
    
    # set up the growth function
    amin = 0.1/self.sigma0 # this should be long before any peaks collapse
    self.__prepare_growth(growth,amin,amax)
    
    # prefactors for cusp-halo results
    self.A_pre = self.A_coef*(self.A_coef/(self.A_m_coef*self.m_coef**self.A_m_index))**(1./(2.*self.A_m_index-1.))
    self.m_pre = self.m_coef*(self.A_coef/(self.A_m_coef*self.m_coef**self.A_m_index))**(2./(2.*self.A_m_index-1.))
    
    # scale factor when \sigma_0=1
    self.a0 = np.exp(np.interp(np.log(1./self.sigma0),self.__tab_lnD,self.__tab_lna,
                             left=np.nan,right=np.nan))
    
  def __prepare_growth(self,growth,amin,amax):
    
    # tabulate in intervals of about 5% in a
    tab_N = 1+int(np.round(np.log(amax/amin)/np.log(1.05)))
    self.__tab_a = np.geomspace(amin,amax,tab_N)
    if callable(growth):
      self.__tab_D = growth(self.__tab_a) / growth(1.)
    else:
      self.__tab_D = self.__tab_a**growth
    self.__tab_lnD = np.log(self.__tab_D)
    
    self.__tab_F = (
      self.__tab_a**(-3.+9./(2.*self.A_m_index-1.))
      * self.__tab_D**(1.5-(9.-6.*self.A_m_index)/(2.-4.*self.A_m_index))
      * self.mass_factor(self.__tab_a)
      )
    
    self.__tab_lnF = np.log(self.__tab_F)
    self.__tab_lna = np.log(self.__tab_a)
  
  def mass_factor(self,a):
    
    return self.massfac_fun(a*self.sigma0)
    
  def growth(self,a):
    '''
    
    Evaluate the tabulated growth function.
    
    Parameters:
      
      a: float or array
        The cosmic expansion factor.
        
    Returns:
      
      D(a): float or array
        The linear growth function.
    
    '''
    return np.exp(np.interp(np.log(a),self.__tab_lna,self.__tab_lnD,
                             left=np.nan,right=np.nan))
  
  def A_from_m(self,m):
    '''
    
    Typical cusp coefficient A, given cusp mass m.
    
    Parameters:
      
      m: float or array
      
    Returns:
      
      A: float or array
    
    '''
    return self.fDM * (self.A_m_coef * self.rho**-1
                       * self.sigma0**(2.25-1.5*self.A_m_index)
                       * self.sigma2**(1.5*self.A_m_index-0.75)
                       * (m/self.fDM)**self.A_m_index)

  def m_from_A(self,A):
    '''
    
    Typical cusp mass m, given cusp coefficient A.
    
    Parameters:
      
      A: float or array
      
    Returns:
      
      m: float or array
    
    '''
    return self.fDM * (self.A_m_coef**(-1./self.A_m_index)
                       * self.rho**(1./self.A_m_index)
                       * self.sigma0**(1.5-2.25/self.A_m_index)
                       * self.sigma2**(-1.5+0.75/self.A_m_index)
                       * (A/self.fDM)**(1./self.A_m_index))
    
  def collapse_a(self,M,a=1.):
    '''
    
    Evaluate the collapse time for the central cusp of a halo of mass M at the
    scale factor a.
    
    Parameters:
      
      M: float or array
        The halo mass.
      
      a: float or array
        Scale factor at which we are considering the halo. Default is a=1.
        
    Returns:
      
      a_coll: float or array
        Scale factor at which the central cusp formed.
    
    '''
    F = (
      self.m_pre * self.rho**(3./(2*self.A_m_index-1.))
      * self.sigma0**((9.-6.*self.A_m_index)/(2.-4.*self.A_m_index))
      * self.mass_factor(a)
      ) / (M * self.sigma2**1.5)
    
    a_out = np.exp(np.interp(np.log(F),self.__tab_lnF,self.__tab_lna,
                             left=-np.inf,right=np.inf))
    
    return np.where(a_out<=a,a_out,np.nan)
  
  def characteristic_m(self,a_coll):
    '''
    
    Evaluate the characteristic cusp mass m given the collapse time.
    
    Parameters:
      
      a_coll: float or array
        The collapse scale factor.
        
    Returns:
      
      m: float or array
        Characteristic cusp mass.
    
    '''
    return self.fDM * self.m_pre * (
      self.rho**(3./(2.*self.A_m_index-1.))
      * self.sigma0**((9.-6.*self.A_m_index)/(2.-4.*self.A_m_index))
      * self.growth(a_coll)**(3./(1.-2.*self.A_m_index))
      * a_coll**(3.+9./(1.-2.*self.A_m_index))
      ) / self.sigma2**1.5
  
  def characteristic_A(self,a_coll):
    '''
    
    Evaluate the characteristic cusp coefficient A given the collapse time.
    
    Parameters:
      
      a_coll: float or array
        The collapse scale factor.
        
    Returns:
      
      A: float or array
        Characteristic cusp coefficient.
    
    '''
    return self.fDM * self.A_pre * (
      self.rho**((1.+self.A_m_index)/(2.*self.A_m_index-1.))
      * self.sigma0**((9.-6.*self.A_m_index)/(4.-8.*self.A_m_index))
      * self.growth(a_coll)**(3./(2.-4.*self.A_m_index))
      * a_coll**(9./(2.-4.*self.A_m_index))
      ) / self.sigma2**0.75
  
  def model_m(self,M,a=1.):
    '''
    
    Evaluate the predicted cusp mass m for a halo of mass M at the scale factor
    a.
    
    Parameters:
      
      M: float or array
        The halo mass.
      
      a: float or array
        Scale factor at which we are considering the halo. Default is a=1.
    
    Returns:
      
      m: float or array
        Characteristic cusp mass.
        
    '''
    
    return self.fDM * self.characteristic_m(self.collapse_a(M,a))
  
  def model_A(self,M,a=1.):
    '''
    
    Evaluate the predicted cusp coefficient A for a halo of mass M at the scale
    factor a.
    
    Parameters:
      
      M: float or array
        The halo mass.
      
      a: float or array
        Scale factor at which we are considering the halo. Default is a=1.
    
    Returns:
      
      A: float or array
        Characteristic cusp coefficient.
        
    '''
    
    return self.fDM * self.characteristic_A(self.collapse_a(M,a))

