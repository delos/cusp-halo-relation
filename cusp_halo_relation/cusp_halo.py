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
  'c_param':776., # parameter C in the Ludlow et al. (2013) concentration model
  'growth_model':'exp', # main-progenitor growth-history model: 'exp' or 'eps'
  'delta_c':DELTA_C, # critical collapse overdensity (only used by 'eps')
  'concentration_model':'L13', # concentration model: 'L13' or 'L16'
  'c_param_L16':650., # parameter C in the Ludlow et al. (2016) concentration model
  'collapsed_fraction_f':0.02, # progenitor mass fraction f in the L16 model
  'eps_nM':600, # EPSGrowth mass-grid points (sigma(M)/Omega(M) tables)
  'eps_nq':600, # EPSGrowth q-grid points (rate integral, cusp A only)
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
          when growth_model=='exp'. Default is exp(-4.5/sigma0).
      - c_param: parameter C in the Ludlow et al. (2013) concentration model.
          Default is 776.
      - growth_model: default model for the main-progenitor mass accretion
          history. 'exp' (default) uses the crude universal closed form
          mass_growth_fun; 'eps' uses an Extended Press-Schechter calculation
          (see eps_growth). Individual calls to collapse_a/m/A/c may override
          this with a growth_model argument.
      - delta_c: critical linear overdensity for collapse, only used by the
          'eps' growth model and the 'L16' concentration model. Default 1.686.
      - concentration_model: default concentration-mass relation to use; calls
          to c may override it with a concentration_model argument.
          'L13' (default) is Ludlow et al. (2013) [arXiv:1302.0288], driven by
          the main-progenitor mass accretion history. 'L16' is Ludlow et al.
          (2016) [arXiv:1601.02624], driven by the EPS collapsed-mass history;
          it is more robust for truncated (warm dark matter) power spectra.
      - c_param_L16: parameter C in the L16 concentration model. Default is 650,
          the value for the analytic spherical-collapse model implemented here
          (C=400 applies only to Monte Carlo merger trees matched to
          simulations).
      - collapsed_fraction_f: progenitor mass fraction f used by the L16 model.
          Default is 0.02.
          
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

    # default growth and concentration models (either may be overridden per call)
    self.growth_model = self.model_params.get('growth_model','exp')
    self.concentration_model = self.model_params.get('concentration_model','L13')
    self.delta_c = self.model_params.get('delta_c',DELTA_C)
    if self.growth_model not in ('exp','eps'):
      raise Exception("growth_model must be 'exp' or 'eps'")
    if self.concentration_model not in ('L13','L16'):
      raise Exception("concentration_model must be 'L13' or 'L16'")
    # sigma_0 at a=1 -> D(a)=sigma0(a)/sigma0_1, used by _omega for 'eps'/'L16'
    self._sigma0_1 = sigmaj(0,self.k1,self.P1)
    # EPS helper: built eagerly for the 'eps' growth model, lazily otherwise
    # (the 'L16' concentration model and a per-call 'eps' growth model need
    # sigma(M) even when the default growth model is 'exp')
    self._eps = None
    if self.growth_model == 'eps':
      if self.verbose:
        print('CuspHalo: using EPS main-progenitor growth model')
      self._ensure_eps()

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
    # The 'exp' growth model factorizes into this universal mass-growth table.
    # The 'eps' model is halo-mass dependent and root-finds in collapse_a, so it
    # does not use this table; the table is still built unconditionally so that
    # 'exp' remains available as a per-call override under any default model.
    self.__tab_invMreduced = self.__tab_s0**(3./(2*self.model_params['A_m_index']-1.)) * self.mass_growth(self.__tab_a)
    self.__tab_lninvMreduced = np.log(self.__tab_invMreduced)

  def _ensure_eps(self):
    '''Lazily construct the EPSGrowth helper, used for the EPS main-progenitor
    history and for sigma(M) in the L16 concentration model. Built eagerly when
    growth_model=='eps'; built on first use otherwise.'''
    if self._eps is None:
      self._eps = EPSGrowth(self.k1,self.P1,self.rho1,delta_c=self.delta_c,
                            nM=self.model_params.get('eps_nM',600),
                            nq=self.model_params.get('eps_nq',600))
    return self._eps

  def _resolve_growth_model(self,growth_model):
    '''Return a validated growth model, defaulting to the configured one.'''
    growth_model = self.growth_model if growth_model is None else growth_model
    if growth_model not in ('exp','eps'):
      raise Exception("growth_model must be 'exp' or 'eps'")
    return growth_model

  def _resolve_concentration_model(self,concentration_model):
    '''Return a validated concentration model, defaulting to the configured one.'''
    concentration_model = self.concentration_model if concentration_model is None else concentration_model
    if concentration_model not in ('L13','L16'):
      raise Exception("concentration_model must be 'L13' or 'L16'")
    return concentration_model

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

  def collapse_a(self,M,a,growth_model=None):
    '''
    Estimate a cusp's formation scale factor a_coll, given that it is the
    central cusp of a halo of mass M at the scale factor a. growth_model
    overrides the configured main-progenitor growth model ('exp' or 'eps').
    '''
    if self._resolve_growth_model(growth_model) == 'eps':
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
    eps = self._ensure_eps()
    p = 3./(2.*self.model_params['A_m_index']-1.)
    lna_grid = self.__tab_lna
    lns0_grid = self.__tab_lns0
    omega_grid = self._omega(self.__tab_a)
    out = np.empty(Mb.size)
    for i,(Mi,ai) in enumerate(zip(Mb.ravel(),ab.ravel())):
      domega = omega_grid - self._omega(ai)
      Mprog = eps.main_progenitor(Mi,domega)
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
  
  def m(self,M,a,growth_model=None):
    '''
    Evaluate the predicted cusp mass m for a halo of mass M at the scale factor
    a. growth_model overrides the configured main-progenitor growth model.
    '''
    return self.characteristic_m(self.collapse_a(M,a,growth_model))

  def A(self,M,a,growth_model=None):
    '''
    Evaluate the predicted cusp coefficient A for a halo of mass M at the scale
    factor a. growth_model overrides the configured main-progenitor growth model.
    '''
    return self.characteristic_A(self.collapse_a(M,a,growth_model))

  def characteristic_c(self,a):
    '''
    Estimate the concentration parameter c at scale factor a for halos close to
    the cutoff scale, using the mass-independent exp+L13 model (the universal
    closed-form mass-growth history). If the class has a method rhoCrit(a) that
    returns the critical density as a function of scale factor, then we use it.
    Otherwise we use the matter density rho(a), which gives less accurate
    results.

    This estimate is mass-independent and ignores the configured growth_model
    and concentration_model; for the configured model use c(M,a) instead.
    '''
    if self.verbose and (self.concentration_model != 'L13' or self.growth_model != 'exp'):
      print('CuspHalo: Warning: characteristic_c returns the mass-independent '
            'exp+L13 estimate; use c(M,a) for the configured model.')
    mass = self.mass_growth(self.__tab_a)/self.mass_growth(a)
    if 'rhoCrit' in dir(self):
      density = self.rhoCrit(self.__tab_a)/self.rhoCrit(a)
    else:
      if self.verbose:
        print('CuspHalo: Warning: using matter density rho(a) for concentrations because rhoCrit(a) is not available.')
      density = self.rho(self.__tab_a)/self.rho(a)
    return concentration.concentration_L13_NFW(density,mass,Dvir=200.,C=self.model_params['c_param'])

  def c(self,M,a,growth_model=None,concentration_model=None):
    '''
    Predicted concentration c=R_vir/r_-2 for a halo of mass M at scale factor a.
    growth_model and concentration_model override the configured defaults.

    The mass dependence follows the chosen models:
    - L13 + 'exp' growth: mass-independent (the universal mass-growth history;
      equivalent to characteristic_c).
    - L13 + 'eps' growth: mass-dependent via the EPS main-progenitor history.
    - L16 (either growth model): mass-dependent via the EPS collapsed-mass
      history (eqs 6 and 7 of Ludlow et al. 2016).
    '''
    growth_model = self._resolve_growth_model(growth_model)
    concentration_model = self._resolve_concentration_model(concentration_model)
    M = np.asarray(M,dtype=float)
    a = np.asarray(a,dtype=float)
    scalar = (M.ndim==0 and a.ndim==0)
    Mb,ab = np.broadcast_arrays(M,a)
    out = np.array([self.__c_scalar(float(Mi),float(ai),growth_model,concentration_model) for Mi,ai in zip(Mb.ravel(),ab.ravel())])
    out = out.reshape(Mb.shape)
    return float(out) if scalar else out

  def __c_scalar(self,M,a,growth_model,concentration_model):
    if concentration_model == 'L13' and growth_model == 'exp':
      return self.characteristic_c(a) # mass-independent
    if 'rhoCrit' in dir(self):
      density = self.rhoCrit(self.__tab_a)/self.rhoCrit(a)
    else:
      if self.verbose:
        print('CuspHalo: Warning: using matter density rho(a) for concentrations because rhoCrit(a) is not available.')
      density = self.rho(self.__tab_a)/self.rho(a)
    domega = self._omega(self.__tab_a)-self._omega(a)
    if concentration_model == 'L16':
      f = self.model_params.get('collapsed_fraction_f',0.02)
      mass = self._ensure_eps().collapsed_fraction(M,domega,f=f)
      mass = np.maximum(mass,1e-300) # the erfc underflows to 0 at very early times
      C = self.model_params.get('c_param_L16',650.)
    else: # L13 + 'eps'
      mass = self._ensure_eps().main_progenitor(M,domega)/M
      C = self.model_params['c_param']
    return concentration.concentration_L13_NFW(density,mass,Dvir=200.,C=C)
