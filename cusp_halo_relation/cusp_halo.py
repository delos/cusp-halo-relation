import numpy as np
from scipy.integrate import simpson
from . import concentration

# moments of the power spectrum
def sigmaj2(j,k,P):
  '''Evaluate \sigma_j^2 given tabulated power spectrum P(k)'''
  integrand = P*k**(2*j)
  return simpson(integrand,x=np.log(k),axis=0)
def sigmaj(j,k,P):
  '''Evaluate \sigma_j given tabulated power spectrum P(k)'''
  return np.sqrt(sigmaj2(j,k,P))

class CuspHalo(object):
  '''
  
  Class for evaluating the cusp-halo relation for a general cosmology.
  
  Parameters:
    
    k, P: arrays
      A table of the dimensionless matter power spectrum P(k), evaluated in
      linear theory when a=1.
    
    growth: float or callable
      The linear growth function, D(a), as a function of the scale factor a.
      If a float g, we will assume D(a) = a^g. Default is g=1.
    
    rho: float
      The cosmological mean density of matter that contributes to gravitational
      clustering. This is used for halo masses. The density should be specified
      when a=1. Default is rho=1, so results are in units of the mean density.
      
    amax: float
      Largest scale factor we are interested in. Default is 1.
  
  Methods:
    
    m(M,a): 
      Median cusp mass m at halo mass M and scale factor a.
    
    A(M,a):
      Median cusp coefficient A at halo mass M and scale factor a.
      
    characteristic_c(a):
      Estimated halo concentration for small halos at scale factor a.

    characteristic_m(a):
      Characteristic cusp mass for young cusps at scale factor a.
      
    characteristic_A(a):
      Characteristic cusp coefficient for young cusps at scale factor a.
    
    collapse_a(M):
      Estimated scale factor of cusp formation for a halo of mass M.
      
    growth(a):
      Linear growth function.
  
  '''
  
  def __init__(self,k,P,growth=1.,rho=1.,amax=1.):
    
    # record the power spectrum
    self.k = k
    self.P = P
    
    # evaluate the characteristic scales
    self.sigma0 = sigmaj(0,k,P) # evolves as D
    self.sigma1 = sigmaj(1,k,P) # evolves as D/a
    self.sigma2 = sigmaj(2,k,P) # evolves as D/a^2
    self.gamma = self.sigma1**2/(self.sigma0*self.sigma2)
    
    # record the density parameters
    self.rho = rho # evolves as a^-3
    
    # parameters of the cusp-peak relation
    self.A_coef = 24.
    self.m_coef = 7.3364
    
    # parameters of the cusp-halo relation
    self.A_m_index = 1.9
    self.A_m_coef  = 0.8
    self.mass_growth_fun = lambda sigma0: np.exp(-4.5/sigma0)
    
    # parameters of the concentration model
    self.mass_growth_param = 4.5
    
    # set up the growth function
    amin = 0.1/self.sigma0 # this should be long before any peaks collapse
    self.__prepare_growth(growth,amin,amax)
    
    # scale factor when \sigma_0=1
    self.a0 = np.exp(np.interp(np.log(1./self.sigma0),self.__tab_lnD,self.__tab_lna,left=np.nan,right=np.nan))
    
    # prefactors for cusp-halo results
    self.A_pre = self.A_coef*(self.A_coef/(self.A_m_coef*self.m_coef**self.A_m_index))**(1./(2.*self.A_m_index-1.))
    self.m_pre = self.m_coef*(self.A_coef/(self.A_m_coef*self.m_coef**self.A_m_index))**(2./(2.*self.A_m_index-1.))
    
    # set up interpolation table for cusp-halo model
    self.__prepare_cusps(growth,amin,amax)
    
  def __prepare_growth(self,growth,amin,amax):
    
    # tabulate in intervals of about 1% in a (this is overkill)
    tab_N = 1+int(np.round(np.log(amax/amin)/np.log(1.01)))
    self.__tab_a = np.geomspace(amin,amax,tab_N)
    self.__tab_lna = np.log(self.__tab_a)
    if callable(growth):
      self.__tab_D = growth(self.__tab_a) / growth(1.)
    else:
      self.__tab_D = self.__tab_a**growth
    self.__tab_lnD = np.log(self.__tab_D)
  
  def __prepare_cusps(self,growth,amin,amax):
    
    self.__tab_F = self.__tab_D**(3./(2*self.A_m_index-1.)) * self.mass_growth(self.__tab_a)
    self.__tab_lnF = np.log(self.__tab_F)
  
  def mass_growth(self,a):
    '''Evaluate the mass growth factor \chi(a) at scale factor a.'''
    return self.mass_growth_fun(self.growth(a)*self.sigma0)
    
  def growth(self,a):
    '''Evaluate the growth function D(a) at scale factor a.'''
    return np.exp(np.interp(np.log(a),self.__tab_lna,self.__tab_lnD,left=np.nan,right=np.nan))
    
  def inverse_growth(self,D):
    '''Evaluate the scale factor a given the growth function D(a).'''
    return np.exp(np.interp(np.log(D),self.__tab_lnD,self.__tab_lna,left=np.nan,right=np.nan))
  
  def A_from_m(self,m):
    '''Typical cusp coefficient A, given cusp mass m.'''
    return self.A_m_coef * self.rho**(1.-self.A_m_index) * self.sigma0**(2.25-1.5*self.A_m_index) * self.sigma2**(1.5*self.A_m_index-0.75) * m**self.A_m_index

  def m_from_A(self,A):
    '''Typical cusp mass m, given cusp coefficient A.'''
    return self.A_m_coef**(-1./self.A_m_index) * self.rho**(1.-1./self.A_m_index) * self.sigma0**(1.5-2.25/self.A_m_index) * self.sigma2**(-1.5+0.75/self.A_m_index) * A**(1./self.A_m_index)

  def collapse_a(self,M,a):
    '''
    Estimate a cusp's formation scale factor a_coll, given that it is the
    central cusp of a halo of mass M at the scale factor a.
    '''
    F = self.m_pre * self.rho * self.sigma0**((9.-6.*self.A_m_index)/(2.-4.*self.A_m_index)) * self.mass_growth(a) / (M * self.sigma2**1.5)
    a_out = np.exp(np.interp(np.log(F),self.__tab_lnF,self.__tab_lna,left=-np.inf,right=np.inf))
    return np.where(a_out<=a,a_out,np.nan)
  
  def characteristic_m(self,a_coll):
    '''
    Evaluate the characteristic cusp mass m given the formation scale factor
    a_coll.
    '''
    return self.m_pre * self.rho * self.sigma0**((9.-6.*self.A_m_index)/(2.-4.*self.A_m_index)) * self.growth(a_coll)**(3./(1.-2.*self.A_m_index)) / self.sigma2**1.5
  
  def characteristic_A(self,a_coll):
    '''
    Evaluate the characteristic cusp coefficient A given the formation scale
    factor a_coll.
    '''
    return self.A_pre * self.rho * self.sigma0**((9.-6.*self.A_m_index)/(4.-8.*self.A_m_index)) * self.growth(a_coll)**(3./(2.-4.*self.A_m_index)) / a_coll**1.5 / self.sigma2**0.75
  
  def m(self,M,a):
    '''
    Evaluate the predicted cusp mass m for a halo of mass M at the scale factor
    a.
    '''
    return self.characteristic_m(self.collapse_a(M,a))
  
  def A(self,M,a):
    '''
    Evaluate the predicted cusp coefficient A for a halo of mass M at the scale
    factor a.
    '''
    return self.characteristic_A(self.collapse_a(M,a))

  def characteristic_c(self,a):
    '''
    Estimate the concentration parameter c at scale factor a for halos close to
    the cutoff scale.
    '''
    return concentration.characteristic_c(self.growth(a)*self.sigma0,self.mass_growth_param)
