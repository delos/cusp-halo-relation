import numpy as np

# fitting parameters from Ludlow et al. (2013) [arXiv:1302.0288]
L13_params = np.array([[0.10,4.124,0.849,0.833],
                       [0.15,3.365,0.692,0.899],
                       [0.20,2.946,0.614,0.953],
                       [0.25,2.697,0.557,1.003],
                       [0.30,2.504,0.530,1.042],
                       [0.35,2.322,0.528,1.068],
                       [0.40,2.154,0.543,1.084],])

def L13_c_prof(c_MAH,alpha_MAH):
  '''
  
  Use fit from Ludlow et al. (2013) [arXiv:1302.0288] to evaluate halo
  concentration c, given c_MAH and alpha_MAH parametrizing the mass accretion
  history.
  
  '''
  p = [np.interp(alpha_MAH,L13_params[:,0],L13_params[:,1+i]) for i in range(3)]
  return p[0] * (1. + p[1] * c_MAH)**p[2]

def characteristic_c(sigma0,mass_growth_param):
  '''
  
  Estimate concentration parameter c for halos close to the cutoff scale.
  
  Parameters:
    
    sigma0: float or array
      The rms density variance in linear theory, a time parameter.
      
    mass_growth_param: float
      Parameter k of the mass accretion history, M \propto e^{-k/\sigma_0}.
  
  Returns:
    
    c: float or array
  
  '''
  c_MAH = (3*(-1 + np.sqrt(17))*sigma0*np.exp((-1 + np.sqrt(17))/4. - mass_growth_param/(3.*sigma0)))/(4.*mass_growth_param)
  alpha_MAH = (-119 + 29*np.sqrt(17))/2.
  return L13_c_prof(c_MAH,alpha_MAH)