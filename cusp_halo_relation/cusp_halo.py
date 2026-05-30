from inspect import signature
import numpy as np
from scipy.integrate import simpson
from . import concentration
from .eps_growth import EPSGrowth, DELTA_C

default_params = {
  'A_coef':24., # coefficient of A in cusp-peak connection
  'm_coef':7.3364, # coefficient of m in cusp-peak connection
  'A_m_index':1.9, # index of cusp A-m relation
  'A_m_coef':0.8, # coefficient of cusp A-m relation
  'mass_growth_fun':lambda s0: np.exp(-4.5/s0), # mass growth factor, function of sigma0
  'c_param':776., # parameter in concentration model
  'growth_history':'exp', # main-progenitor growth-history model: 'exp' or 'eps'
  'delta_c':DELTA_C, # critical collapse overdensity (only used by 'eps')
  }

# moments of the power spectrum
def sigmaj2(j,k,P,axis=0):
  '''Evaluate sigma_j^2 given tabulated power spectrum P(k)'''
  return simpson(P*k**(2*j),x=np.log(k),axis=axis)
def sigmaj(j,k,P,axis=0):
  '''Evaluate sigma_j given tabulated power spectrum P(k)'''
  return np.sqrt(sigmaj2(j,k,P,axis=axis))

class CuspHalo(object):
  '''
  
  Class for evaluating the cusp-halo relation for a general cosmology.
  
  Parameters:
    
    k, P: arrays
      A table of the dimensionless matter power spectrum P(k), evaluated in
      linear theory when a=1.
    
    growth: float or callable
      The linear growth function, D(a), as a function of the scale factor a,
      or D(a,k) to allow for dependence on the wavenumber.
      If a float g, we will assume D(a) = a^g. Default is g=1.
    
    rho: float
      The cosmological mean density of matter that contributes to gravitational
      clustering. This is used for halo masses. The density should be specified
      when a=1. Default is rho=1, so results are in units of the mean density.
      
    amin, amax: floats
      Smallest and largest scale factors we are interested in. Default is to
      select amin automatically and amax=1.
      
    model_params: dict
      Parameters of the model, including:
      - A_coef: coefficient of A in cusp-peak connection (default 24).
      - m_coef: coefficient of m in cusp-peak connection (default 7.3364).
      - A_m_index: index of cusp A-m relation (default 1.9).
      - A_m_coef: coefficient of cusp A-m relation (default 0.8).
      - mass_growth_fun: mass growth factor, function of sigma0. Only used
          when growth_history=='exp'. Default is exp(-4.5/sigma0).
      - c_param: parameter in Ludlow et al. (2013) concentration model.
          Default is 776.
      - growth_history: model for the main-progenitor mass accretion history.
          'exp' (default) uses the crude universal closed form mass_growth_fun;
          'eps' uses an Extended Press-Schechter calculation (see eps_growth).
      - delta_c: critical linear overdensity for collapse, only used by the
          'eps' growth history. Default is 1.686.
          
    verbose: boolean
      Default True. Change to False to suppress messages.
  
  '''
  
  def __init__(self,k,P,growth=1.,rho=1.,amin=None,amax=1.,model_params=default_params,verbose=True):
    self.k1 = k
    self.P1 = P
    self.rho1 = rho
    self.model_params = model_params
    self.verbose = verbose
    
    # set up the growth function
    self.a_min = amin or 0.05/sigmaj(0,k,P) # this should be long before any peaks collapse
    self.a_max = amax
    self.__prepare_growth(growth,self.a_min,self.a_max)

    # EPS main-progenitor growth history (optional)
    self.growth_history = self.model_params.get('growth_history','exp')
    self.delta_c = self.model_params.get('delta_c',DELTA_C)
    if self.growth_history == 'eps':
      if self.verbose:
        print('CuspHalo: using EPS main-progenitor growth history')
      self._sigma0_1 = sigmaj(0,self.k1,self.P1) # sigma_0 at a=1 -> D(a)=sigma0(a)/sigma0_1
      self._eps = EPSGrowth(self.k1,self.P1,self.rho1,delta_c=self.delta_c)
    elif self.growth_history != 'exp':
      raise Exception("growth_history must be 'exp' or 'eps'")

    # prefactors for cusp-halo results
    self.A_pre = self.model_params['A_coef']*(self.model_params['A_coef']/(self.model_params['A_m_coef']*self.model_params['m_coef']**self.model_params['A_m_index']))**(1./(2.*self.model_params['A_m_index']-1.))
    self.m_pre = self.model_params['m_coef']*(self.model_params['A_coef']/(self.model_params['A_m_coef']*self.model_params['m_coef']**self.model_params['A_m_index']))**(2./(2.*self.model_params['A_m_index']-1.))
    
    # set up interpolation table for cusp-halo model
    self.__prepare_cusps(growth,self.a_min,self.a_max)
    
  def __prepare_growth(self,growth,amin,amax):
    # tabulate in intervals of about 1% in a (this is overkill)
    tab_N = 1+int(np.round(np.log(amax/amin)/np.log(1.01)))
    self.__tab_a = np.geomspace(amin,amax,tab_N)
    self.__tab_lna = np.log(self.__tab_a)
    if callable(growth):
      if len(signature(growth).parameters) == 2:
        if self.verbose:
          print('CuspHalo: using growth function D(a,k)')
        Da = growth(self.__tab_a[None],self.k1[:,None])/growth(1.,self.k1[:,None])
        Pka = Da**2 * self.P1[:,None]
        ka = self.k1[:,None]/self.__tab_a[None]
        self.__tab_s0 = sigmaj(0,ka,Pka,axis=0)
        self.__tab_s1 = sigmaj(1,ka,Pka,axis=0)
        self.__tab_s2 = sigmaj(2,ka,Pka,axis=0)
        
      elif len(signature(growth).parameters) == 1:
        if self.verbose:
          print('CuspHalo: using growth function D(a)')
        Da = growth(self.__tab_a)/growth(1.)
        self.__tab_s0 = sigmaj(0,self.k1,self.P1) * Da
        self.__tab_s1 = sigmaj(1,self.k1,self.P1) * Da / self.__tab_a
        self.__tab_s2 = sigmaj(2,self.k1,self.P1) * Da / self.__tab_a**2
      else:
        raise Exception('Growth function must take 1 or 2 arguments')
    else:
      if self.verbose:
        print('CuspHalo: using growth function D(a)=a^%.3f'%growth)
      Da = self.__tab_a**growth
      self.__tab_s0 = sigmaj(0,self.k1,self.P1) * Da
      self.__tab_s1 = sigmaj(1,self.k1,self.P1) * Da / self.__tab_a
      self.__tab_s2 = sigmaj(2,self.k1,self.P1) * Da / self.__tab_a**2
    self.__tab_lns0 = np.log(self.__tab_s0)
    self.__tab_lns1 = np.log(self.__tab_s1)
    self.__tab_lns2 = np.log(self.__tab_s2)
    
    # scale factor when sigma_0=1
    self.a0 = np.exp(np.interp(0.,self.__tab_lns0,self.__tab_lna,left=np.nan,right=np.nan))
    
    # power when sigma_0=1
    self.k0 = self.k1 / self.a0
    if callable(growth):
      if len(signature(growth).parameters) == 2:
        self.P0 = (growth(self.a0,self.k1)/growth(1.,self.k1))**2 * self.P1
      elif len(signature(growth).parameters) == 1:
        self.P0 = (growth(self.a0)/growth(1.))**2 * self.P1
    else:
      self.P0 = self.a0**2 * self.P1
  
  def __prepare_cusps(self,growth,amin,amax):
    # the 'exp' growth history factorizes into a universal mass-growth table;
    # the 'eps' history is halo-mass dependent and root-finds in collapse_a, so
    # this table is unused there.
    if self.growth_history == 'eps':
      return
    self.__tab_invMreduced = self.__tab_s0**(3./(2*self.model_params['A_m_index']-1.)) * self.mass_growth(self.__tab_a)
    self.__tab_lninvMreduced = np.log(self.__tab_invMreduced)

  def _omega(self,a):
    '''EPS time variable omega(a) = delta_c / D(a) = delta_c sigma0(1) / sigma0(a).'''
    return self.delta_c * self._sigma0_1 / self.sigma0(a)
  
  def rho(self,a=1.):
    '''Density as a function of scale factor a. Default is a=1.'''
    return self.rho1 * a**-3
  
  def sigma0(self,a=1.):
    '''sigma_0 as a fuction of scale factor a. Default is a=1.'''
    return np.exp(np.interp(np.log(a),self.__tab_lna,self.__tab_lns0,left=np.nan,right=np.nan))
  
  def sigma1(self,a=1.):
    '''sigma_1 as a fuction of scale factor a. Default is a=1.'''
    return np.exp(np.interp(np.log(a),self.__tab_lna,self.__tab_lns1,left=np.nan,right=np.nan))
  
  def sigma2(self,a=1.):
    '''sigma_2 as a fuction of scale factor a. Default is a=1.'''
    return np.exp(np.interp(np.log(a),self.__tab_lna,self.__tab_lns2,left=np.nan,right=np.nan))
  
  def gamma(self,a=1.):
    '''sigma_1^2/(sigma_0 sigma_2) as a function of a. Default is a=1.'''
    return self.sigma1(a)**2/(self.sigma0(a)*self.sigma2(a))
  
  def a_from_sigma0(self,sigma0):
    '''Scale factor a at given sigma_0.'''
    return np.exp(np.interp(np.log(sigma0),self.__tab_lns0,self.__tab_lna,left=np.nan,right=np.nan))
  
  def mass_growth(self,a):
    '''Evaluate the mass growth factor chi(a) at scale factor a.'''
    return self.model_params['mass_growth_fun'](self.sigma0(a))
  
  def A_from_m(self,m,a=None):
    '''Typical cusp coefficient A, given cusp mass m.'''
    if a is None:
      a = self.a0
    return self.model_params['A_m_coef'] * (self.rho(a))**(1.-self.model_params['A_m_index']) * self.sigma0(a)**(2.25-1.5*self.model_params['A_m_index']) * self.sigma2(a)**(1.5*self.model_params['A_m_index']-0.75) * m**self.model_params['A_m_index']

  def m_from_A(self,A,a=None):
    '''Typical cusp mass m, given cusp coefficient A.'''
    if a is None:
      a = self.a0
    return self.model_params['A_m_coef']**(-1./self.model_params['A_m_index']) * self.rho(a)**(1.-1./self.model_params['A_m_index']) * self.sigma0(a)**(1.5-2.25/self.model_params['A_m_index']) * self.sigma2(a)**(-1.5+0.75/self.model_params['A_m_index']) * A**(1./self.model_params['A_m_index'])

  def collapse_a(self,M,a):
    '''
    Estimate a cusp's formation scale factor a_coll, given that it is the
    central cusp of a halo of mass M at the scale factor a.
    '''
    if self.growth_history == 'eps':
      return self._collapse_a_eps(M,a)
    reducedM = M / (self.m_pre * self.rho(a)*(self.sigma0(a)/self.sigma2(a))**1.5 * self.mass_growth(a) )
    a_out = np.exp(np.interp(-np.log(reducedM),self.__tab_lninvMreduced,self.__tab_lna,left=-np.inf,right=np.inf))
    return np.where(a_out<=a,a_out,np.nan)

  def _collapse_a_eps(self,M,a):
    '''
    collapse_a using the EPS main-progenitor history. The cusp forms when the
    main progenitor first reaches the characteristic threshold,

      M_prog(a_coll) sigma0(a_coll)^p = m_pre rho(a) (sigma0(a)/sigma2(a))^1.5,

    with p = 3/(2 A_m_index - 1) and M_prog from the EPS mass flow. This is the
    same condition the 'exp' interpolation table encodes, with the universal
    mass-growth shape replaced by the (halo-mass-dependent) EPS history.
    '''
    M = np.asarray(M,dtype=float)
    a = np.asarray(a,dtype=float)
    scalar = (M.ndim==0 and a.ndim==0)
    Mb,ab = np.broadcast_arrays(M,a)
    p = 3./(2.*self.model_params['A_m_index']-1.)
    lna_grid = self.__tab_lna
    lns0_grid = self.__tab_lns0
    omega_grid = self._omega(self.__tab_a)
    out = np.empty(Mb.size)
    for i,(Mi,ai) in enumerate(zip(Mb.ravel(),ab.ravel())):
      domega = omega_grid - self._omega(ai)
      Mprog = self._eps.main_progenitor(Mi,domega)
      lhs = np.log(Mprog) + p*lns0_grid # increasing in a'
      rhs = np.log(self.m_pre * self.rho(ai) * (self.sigma0(ai)/self.sigma2(ai))**1.5)
      a_coll = np.exp(np.interp(rhs,lhs,lna_grid,left=-np.inf,right=np.inf))
      out[i] = a_coll if a_coll<=ai else np.nan
    out = out.reshape(Mb.shape)
    return float(out) if scalar else out
  
  def characteristic_m(self,a_coll):
    '''
    Evaluate the characteristic cusp mass m given the formation scale factor
    a_coll.
    '''
    return self.m_pre * self.rho(a_coll) * self.sigma0(a_coll)**((9.-6.*self.model_params['A_m_index'])/(2.-4.*self.model_params['A_m_index'])) / self.sigma2(a_coll)**1.5
  
  def characteristic_A(self,a_coll):
    '''
    Evaluate the characteristic cusp coefficient A given the formation scale
    factor a_coll.
    '''
    return self.A_pre * self.rho(a_coll) * self.sigma0(a_coll)**((9.-6.*self.model_params['A_m_index'])/(4.-8.*self.model_params['A_m_index'])) / self.sigma2(a_coll)**0.75
  
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
    the cutoff scale. If the class has a method rhoCrit(a) that returns the
    critical density as a function of scale factor, then we use it. Otherwise
    we use the matter density rho(a), which gives less accurate results.
    '''
    if self.growth_history == 'eps':
      return self._characteristic_c_eps(a)
    mass = self.mass_growth(self.__tab_a)/self.mass_growth(a)
    if 'rhoCrit' in dir(self):
      density = self.rhoCrit(self.__tab_a)/self.rhoCrit(a)
    else:
      if self.verbose:
        print('CuspHalo: Warning: using matter density rho(a) for concentrations because rhoCrit(a) is not available.')
      density = self.rho(self.__tab_a)/self.rho(a)
    return concentration.concentration_L13_NFW(density,mass,Dvir=200.,C=self.model_params['c_param'])

  def _characteristic_c_eps(self,a):
    '''characteristic_c using the EPS main-progenitor history of a fixed
    cutoff-scale halo (the half-mode mass Mhm if available, else the spectrum's
    cutoff mass M_cut). Holding the reference mass fixed across epochs is what
    makes the concentration of these small halos rise toward late times, as in
    the 'exp' model.'''
    a = np.asarray(a,dtype=float)
    if a.ndim:
      return np.array([self.__characteristic_c_eps_scalar(float(ai)) for ai in a.ravel()]).reshape(a.shape)
    return self.__characteristic_c_eps_scalar(float(a))

  def __characteristic_c_eps_scalar(self,a):
    M_ref = getattr(self,'Mhm',None) or self._eps.M_cut
    mass = self._eps.main_progenitor(M_ref,self._omega(self.__tab_a)-self._omega(a))/M_ref
    if 'rhoCrit' in dir(self):
      density = self.rhoCrit(self.__tab_a)/self.rhoCrit(a)
    else:
      if self.verbose:
        print('CuspHalo: Warning: using matter density rho(a) for concentrations because rhoCrit(a) is not available.')
      density = self.rho(self.__tab_a)/self.rho(a)
    return concentration.concentration_L13_NFW(density,mass,Dvir=200.,C=self.model_params['c_param'])
