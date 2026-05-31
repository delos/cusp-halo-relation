import numpy as np

# Concentration lookup range. The cusp-halo model can imply very high
# concentrations for strongly bumped (small-scale-enhanced) spectra, and the
# L16 solver indexes this table at intermediate densities along the accretion
# history; too narrow a range returns spurious NaN even when the final c is
# moderate. 1e6 is far above any physical halo concentration, so genuinely
# runaway cases still fall off the end and are flagged downstream. The
# per-decade sampling matches the original [1,1e3] table (in-range results
# unchanged to ~1e-6).
__tab_c = np.geomspace(1,1e6,2000)
__tab_m2_NFW = (np.log(4)-1)/2 / (np.log(1+__tab_c)-__tab_c/(1+__tab_c))
__tab_p2_NFW = __tab_c**3 * __tab_m2_NFW
__tab_logc = np.log(__tab_c)
__tab_logm2_NFW = np.log(__tab_m2_NFW)
__tab_logp2_NFW = np.log(__tab_p2_NFW)

def __m2_from_p2_NFW(p2,Dvir,C):
  return np.exp(np.interp(np.log(p2*C/Dvir),__tab_logp2_NFW,__tab_logm2_NFW,left=np.nan,right=np.nan))
def __c_from_p2_NFW(p2,Dvir,C):
  return np.exp(np.interp(np.log(p2*C/Dvir),__tab_logp2_NFW,__tab_logc,left=np.nan,right=np.nan))

def concentration_L13_NFW(rho_over_rho0,M_over_M0,Dvir=200.,C=776.):
  '''
  Estimate halo concentration c=R_vir/r_-2 from mass accretion history. Here
  R_vir is the virial radius and r_-2 is the radius at which dlnrho/dlnr=-2.
  We use the result from Ludlow et al. (2013) [arXiv:1302.0288] that the
  average density within r_-2 is C times the density of the universe when the
  halo virial mass was M_-2, where M_-2 is the mass enclosed in r_-2.

  Here we approximate that the halo has an NFW profile for radii r>r_-2.

  This same NFW solver also serves the Ludlow et al. (2016) [arXiv:1601.02624]
  model (eqs 6 and 7): pass the EPS collapsed-mass history (see
  eps_growth.EPSGrowth.collapsed_fraction) as M and use C=650.
  
  Parameters:
    
    rho: array
      Density of the universe over a range of times, specified in units of the
      density at the "current" time (when we want to evaluate the halo
      concentration). Must be in decreasing order.
      
    M: array
      Halo mass over the same range of times, specified in units of the halo
      mass at the "current" time. Must be in increasing order.
    
    Dvir: float
      Virial overdensity factor. Default is 200.
      
    C: float
      Model parameter. Default is 776.
      
  Returns:
    
    c: float
      Halo concentration parameter, c=R_vir/r_-2. 
  '''
  m2arr = __m2_from_p2_NFW(rho_over_rho0,Dvir=Dvir,C=C)
  rho2 = np.exp(np.interp(0,np.log(M_over_M0/m2arr),np.log(rho_over_rho0)))
  return __c_from_p2_NFW(rho2,Dvir=Dvir,C=C)
