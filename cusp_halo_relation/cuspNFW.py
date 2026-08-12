import numpy as np
from scipy.optimize import root_scalar
from scipy.special import spence

# dimensionless radial profiles

def __density(x,y,z=0):
  return np.sqrt(x+y**2)/(x**1.5*(1+x)**2)
def __density_slope(x,y):
  return 0.5*(-7. + 4./(1. + x) + x/(x + y**2))
__zsw = 0.93 # below this z, and
__xysw = 1e-4 # below this x y**2, sum the series rather than cancel
__ysw = 2.5e-3 # below this y, use the matched form instead
def __take(v,m):
  # v[m], unless v is a scalar and so already stands for every element
  return v if np.ndim(v) == 0 else v[m]
def __atanh_over_c_series(c,z):
  # series in c**2, stable as c -> 0
  s,t,q = np.zeros_like(z),z.copy(),(c*z)**2
  for n in range(60):
    s = s + t/(2*n+1)
    t = t*q
    if np.all(np.abs(t) < 1e-19*np.abs(s)+1e-300):
      break
  return s
def __atanh_over_c(c,z):
  z = np.array(z).astype(float)
  if np.ndim(c) == 0: # one c for many z, the usual case
    return np.arctanh(c*z)/c if c >= 0.05 else __atanh_over_c_series(c,z)
  c,z = np.broadcast_arrays(c,z)
  out = np.empty(c.shape)
  direct = c >= 0.05
  if np.any(direct):
    out[direct] = np.arctanh(c[direct]*z[direct])/c[direct]
  if np.any(~direct):
    out[~direct] = __atanh_over_c_series(c[~direct],z[~direct])
  return out
def __mass_NFW(x):
  # ln(1+x) - x/(1+x), which cancels to x**2/2 at small x
  out = np.empty_like(x)
  big = x >= 0.15
  if np.any(big):
    _x = x[big]
    out[big] = np.log1p(_x) - _x/(1.+_x)
  if np.any(~big):
    _x,acc,pw = x[~big],np.zeros_like(x[~big]),x[~big]**2
    for k in range(2,26):
      acc = acc + (-1.)**k*pw*(k-1.)/k
      pw = pw*_x
    out[~big] = acc
  return out
def __mass(x,y):
  # M/(4 pi) = 2 asinh(sqrt(x)/y) - (2-y**2) atanh(c z)/c - z (x+y**2)/(1+x),
  # z = sqrt(x/(x+y**2)) and c = sqrt(1-y**2). Expanding the two inverse
  # hyperbolics in z collects that into
  #   -c**2 x z/(1+x) + y**2 sum over n >= 1 of beta_n z**(2n+1)/(2n+1)
  # which is summed where the closed form would cancel; below y = 1/400 the
  # expansion in y**2, carried to O(y**4), is used instead. y outside [0,1]
  # gives nan, as the profile is only defined for rho_s r_s**1.5 >= A
  uniform = np.ndim(y) == 0 # then every y-dependent quantity below stays scalar
  x,y = np.broadcast_arrays(np.array(x).astype(float),np.array(y).astype(float))
  shape = x.shape
  x,y = np.atleast_1d(x),np.atleast_1d(y)
  out = np.full(x.shape,np.nan)
  nfw = y == 0.
  if np.any(nfw):
    out[nfw] = __mass_NFW(x[nfw])
  cusp = (y > 0.)&(y <= 1.0001)
  if np.any(cusp):
    _x,_y = x[cusp],(y.flat[0] if uniform else y[cusp])
    c = np.sqrt(np.maximum(1.-_y*_y,0.))
    z = np.sqrt(_x/(_x+_y*_y))
    o = np.empty(_x.shape)
    inner = (z < __zsw)&(_x*_y*_y < __xysw)
    if np.any(inner):
      # beta_n = [2-(2-y**2)(1-y**2)**n]/y**2 = 1 + (2-y**2) g_n/y**2, where
      # g_n = 1-(1-y**2)**n comes from the recurrence g_n = y**2 + (1-y**2)
      # g_(n-1) so that it never cancels; the order is set by the largest z
      _z,z2,_xi = z[inner],z[inner]**2,_x[inner]
      y2,c2 = __take(_y,inner)**2,__take(c,inner)**2
      order = max(int(np.ceil(np.log(1e-18)/np.log(max(z2.max(),1e-300))))+4,3)
      n = np.arange(1,order+1)
      if uniform: # one scalar coefficient per order, so the sum is a Horner pass
        l1 = np.log1p(-y2) if y2 < 1. else -np.inf
        co = (1.+(2.-y2)*(-np.expm1(n*l1))/y2)/(2*n+1.)
        s = np.full(_z.shape,co[-1])
        for ck in co[-2::-1]:
          s = s*z2 + ck
        s = s*_z*z2
      else:
        s,pw,g = np.zeros(_z.shape),_z*z2,0.
        for k in n:
          g = y2 + c2*g
          s = s + (1.+(2.-y2)*g/y2)*pw/(2*k+1.)
          pw = pw*z2
      o[inner] = y2*s - c2*_xi*_z/(1.+_xi)
    matched = ~inner&(_y <= __ysw)
    if np.any(matched):
      # m_NFW + y**2 x/(2(1+x)) + y**4 nu(q), the expansion in y**2 with its
      # O(y**4) term supplied by the cusp itself: nu = mu - q**4/2 - q**2/2
      # with 4 pi y**4 mu(q) the mass of the profile without (1+x)**-2, and
      # the leading terms of that difference taken out by hand
      _xm,_ym = _x[matched],__take(_y,matched)
      q = np.sqrt(_xm)/_ym
      r = np.sqrt(1.+q*q)
      nu = q*r/(8.*(q*q+0.5+q*r)) - np.arcsinh(q)/4.
      o[matched] = __mass_NFW(_xm) + _ym*_ym*_xm/(2.*(1.+_xm)) + _ym**4*nu
    closed = ~inner&(_y > __ysw)
    if np.any(closed):
      _xc,_yc,_zc = _x[closed],__take(_y,closed),z[closed]
      o[closed] = 2.*np.arcsinh(np.sqrt(_xc)/_yc) - (2.-_yc*_yc)*__atanh_over_c(__take(c,closed),_zc) - _zc*(_xc+_yc*_yc)/(1.+_xc)
    out[cusp] = o
  return 4*np.pi*out.reshape(shape)

# the potential is closed form: with z = sqrt(x/(x+y**2)) and c = sqrt(1-y**2),
#   Phi/(4 pi) = sqrt(1+y**2/x) - 2 asinh(sqrt(x)/y)/x + [(2-y**2)/x + y**2] atanh(c z)/c
# the three terms cancel to O(y**2 z) for x << y**2, so where the closed form
# would lose that we sum the same expression as a series in z instead, and for
# y below 1/400 we use the matched form
#   Phi/(4 pi) = Phi_NFW/(4 pi) + y**2 [psi(q) - ln(1+x)/2],  q = sqrt(x)/y
# with psi = phi - q**2/2, where 4 pi y**2 phi(q) is the exact potential of the
# profile without its (1+x)**-2 factor; that form is correct through O(y**2) at
# every radius
def __potential_inf_w(y):
  # Phi_inf/(4 pi) - 1 = y**2 atanh(c)/c
  shape = np.shape(y)
  y = np.atleast_1d(np.array(y).astype(float))
  w = np.full(y.shape,np.nan)
  w[y == 0.] = 0.
  ok = (y > 0.)&(y <= 1.0001)
  if np.any(ok):
    _y = y[ok]
    w[ok] = _y*_y*__atanh_over_c(np.sqrt(np.maximum(1.-_y*_y,0.)),np.ones(_y.shape))
  return w.reshape(shape)
def __potential_inf(y):
  return 4*np.pi*(1.+__potential_inf_w(y))
def __potential_NFW(x):
  out = np.empty_like(x)
  small = x < 1e-3
  if np.any(~small):
    _x = x[~small]
    out[~small] = 1. - np.log1p(_x)/_x
  if np.any(small): # 1 - ln(1+x)/x cancels here, so sum its series
    _x,acc,pw = x[small],np.zeros_like(x[small]),x[small].copy()
    for k in range(1,20):
      acc = acc + (-1.)**(k+1)*pw/(k+1.)
      pw = pw*_x
    out[small] = acc
  return out
def __potential(x,y,zero_at_inf=False):
  uniform = np.ndim(y) == 0 # then every y-dependent quantity below stays scalar
  x,y = np.broadcast_arrays(np.array(x).astype(float),np.array(y).astype(float))
  shape = x.shape
  x,y = np.atleast_1d(x),np.atleast_1d(y)
  out = np.empty(x.shape)
  nfw = y == 0.
  if np.any(nfw):
    out[nfw] = __potential_NFW(x[nfw])
  cusp = ~nfw
  if np.any(cusp):
    _x,_y = x[cusp],(y.flat[0] if uniform else y[cusp])
    c = np.sqrt(np.maximum(1.-_y*_y,0.))
    z = np.sqrt(_x/(_x+_y*_y))
    o = np.empty(_x.shape)
    inner = (z < __zsw)&(_x*_y*_y < __xysw)
    if np.any(inner):
      # sum over n >= 2 of beta_n z**(2n-1)/(2n+1), beta_n and its recurrence
      # as in __mass; the order is set by the largest z present
      _z,z2 = z[inner],z[inner]**2
      y2,c2 = __take(_y,inner)**2,__take(c,inner)**2
      order = max(int(np.ceil(np.log(1e-18)/np.log(max(z2.max(),1e-300))))+4,3)
      n = np.arange(2,order+1)
      if uniform: # one scalar coefficient per order, so the sum is a Horner pass
        l1 = np.log1p(-y2) if y2 < 1. else -np.inf
        co = (1.+(2.-y2)*(-np.expm1(n*l1))/y2)/(2*n+1.)
        s = np.full(_z.shape,co[-1])
        for ck in co[-2::-1]:
          s = s*z2 + ck
        s = s*_z*z2
      else:
        s,pw,g = np.zeros(_z.shape),_z*z2,y2
        for k in n:
          g = y2 + c2*g
          s = s + (1.+(2.-y2)*g/y2)*pw/(2*k+1.)
          pw = pw*z2
      o[inner] = _z*(y2+(3.-y2)*z2)/3. - (1.-z2)*s + y2*__atanh_over_c(__take(c,inner),_z)
    closed = ~inner&(_y > __ysw)
    if np.any(closed):
      _c,_z,_xc,_yc = __take(c,closed),z[closed],_x[closed],__take(_y,closed)
      o[closed] = np.sqrt(1.+_yc*_yc/_xc) - 2.*np.arcsinh(np.sqrt(_xc)/_yc)/_xc + ((2.-_yc*_yc)/_xc + _yc*_yc)*__atanh_over_c(_c,_z)
    matched = ~inner&(_y <= __ysw)
    if np.any(matched): # psi is grouped so that the y**2 q**2/2 = x/2 piece never appears
      _xm,_ym = _x[matched],__take(_y,matched)
      q = np.sqrt(_xm)/_ym
      r,a = np.sqrt(1.+q*q),np.arcsinh(q)
      o[matched] = __potential_NFW(_xm) + _ym*_ym*(q/(2.*(r+q)) + a + (a-q*r)/(4.*q*q) - np.log1p(_xm)/2.)
    out[cusp] = o
  pot = 4*np.pi*out
  if zero_at_inf:
    pot = pot - __potential_inf(y.flat[0] if uniform else y)
  return pot.reshape(shape)

# rho sigma_r**2 = int_x^inf rho M/s**2 ds has no elementary closed form, so we
# evaluate it from an analytic fit; with
# q = sqrt(x)/y, c = sqrt(1-y**2), a = tanh(ln x/Wx) and b = tanh(ln(x/y**2)/We)
# the fit is
#   rho sigma_r**2 = [P_NFW(x) + Delta(x,y) u(x,y)] exp[E(x,y)]
#   Delta = (2 pi/15)[(3q+22q**3+(2q**2-3)(1+q**2)**1.5 asinh q)/q**5 - 2 ln(2q)]
#   u     = 1/(1+kappa x)**3, kappa = (3 y**2/dm)**(1/3),
#           dm = 2 ln(2/y) - (2-y**2) atanh(c)/c
#   E     = (1+a)(1-b) sum C[n,m] T_n(a) T_m(b)
# P_NFW is the exact NFW member (Lokas & Mamon 2001) and Delta the exact cusp
# to NFW transition in the y -> 0 limit at fixed q, which the whole transition
# depends on x and y through; u is fixed by the outer limit. E carries the
# factors (1+a) and (1-b), so it vanishes as x -> 0 and as x/y**2 -> inf, and
# the fit is therefore exact on the whole y = 0 slice and in all three limits
#   sigma_r**2 -> (8 pi/3) y sqrt(x) (x->0, y>0), 4 pi x [C0 - ln(x)/2] (x->0,
#   y=0) with C0 = pi**2/2-23/4, and pi (ln x + m_inf + 1/4)/x (x->inf)
# the six coefficients and the two widths are fitted, to 0.35% over
# 0 <= y <= 1 and 1e-12 <= x <= 1e8
__vd_Wx,__vd_We = 5.0,2.0
__vd_C10,__vd_C30,__vd_C50,__vd_C21,__vd_C31,__vd_C52 = 0.0540697378,-0.109475137,0.038015262,0.0765614167,0.0546313524,-0.00778056345
# the Taylor coefficients of [ln(1+x)-x-4x ln(1+x)]/x**2, used below x = 0.05
__vd_W = np.array([-9./2,7./3,-19./12,6./5,-29./30,17./21,-39./56,11./18,-49./90,27./55,-59./132,16./39])
def __veldisp2_r_NFW_largex(x):
  logx = np.log(x)
  return (-3./16+logx/4)/x + (69./200+logx/10)/x**2 + (-97./1200-logx/20)/x**3 + (71./3675+logx/35)/x**4 + (-1./3136-logx/56)/x**5 + (-1271./211680+logx/84)/x**6
def __veldisp2_r_NFW(x):
  # P_NFW/(4 pi) = B(x)/2; the group ln(1+x)/x**2 - 1/x - 4 ln(1+x)/x cancels
  # to O(1) at small x, so it is collected into W and summed as a series there,
  # and the whole bracket cancels to O(ln x/x**3) at large x, so beyond x = 20
  # the large-x expansion of sigma_r**2 is used instead
  out = np.empty_like(x)
  small = x <= 20.
  if np.any(small):
    _x = x[small]
    L = np.log1p(_x)
    W = np.empty_like(_x)
    tiny = _x < 0.05
    if np.any(~tiny):
      u,Lu = _x[~tiny],L[~tiny]
      W[~tiny] = (Lu-u-4.*u*Lu)/(u*u)
    if np.any(tiny):
      u = _x[tiny]
      s = np.full(u.shape,__vd_W[-1])
      for ck in __vd_W[-2::-1]:
        s = s*u + ck
      W[tiny] = s
    v = 1./(1.+_x)
    out[small] = 0.5*(np.pi**2 - np.log(_x) + W - v*v - 6.*v + L - 2.*L*v + 3.*L*L + 6.*spence(1.+_x))
  large = ~small
  if np.any(large):
    _x = x[large]
    out[large] = __veldisp2_r_NFW_largex(_x)/(_x*(1.+_x)**2)
  return out
def __veldisp2_r_cusp(x,y):
  # Delta/(2 pi/15); the bracket cancels to 20/q**2 at small q and to
  # 45/(2 q**2) at large q, so the closed form is used only in between
  q = np.sqrt(x)/y
  out = np.empty_like(q)
  mid = (q >= 0.3)&(q <= 30.)
  if np.any(mid):
    t = q[mid]
    t2 = t*t
    r = 1.+t2
    out[mid] = (3.*t+22.*t*t2+(2.*t2-3.)*r*np.sqrt(r)*np.arcsinh(t))/(t*t2*t2) - 2.*np.log(2.*t)
  small = q < 0.3
  if np.any(small):
    t2 = q[small]**2
    out[small] = 60.*(1./(3.*t2)+31./900+t2/105.-t2*t2/315.+16.*t2**3/10395.) - 2.*np.log(2.*q[small])
  large = q > 30.
  if np.any(large):
    u,L = 1./q[large]**2,np.log(2.*q[large])
    out[large] = u*(45./2 + u*(45./16-15./4*L + u*(-5./6-5./4*L)))
  return 2*np.pi/15.*out
def __veldisp2_r(x,y):
  uniform = np.ndim(y) == 0 # then every y-dependent quantity below stays scalar
  x,y = np.broadcast_arrays(np.array(x).astype(float),np.array(y).astype(float))
  shape = x.shape
  x,y = np.atleast_1d(x),np.atleast_1d(y)
  ps2 = 4*np.pi*__veldisp2_r_NFW(x)
  cusp = y > 0.
  if np.any(cusp):
    _x,_y = x[cusp],(y.flat[0] if uniform else y[cusp])
    dm = 2.*np.log(2./_y) - (2.-_y*_y)*__potential_inf_w(_y)/(_y*_y)
    kappa = (3.*_y*_y/dm)**(1./3)
    p = ps2[cusp] + __veldisp2_r_cusp(_x,_y)/(1.+kappa*_x)**3
    a = np.tanh(np.log(_x)/__vd_Wx)
    b = np.tanh(np.log(_x/(_y*_y))/__vd_We)
    a2 = a*a
    T2,T3,T5 = 2.*a2-1.,a*(4.*a2-3.),a*(a2*(16.*a2-20.)+5.)
    E = __vd_C10*a + __vd_C30*T3 + __vd_C50*T5 + b*(__vd_C21*T2 + __vd_C31*T3) + (2.*b*b-1.)*__vd_C52*T5
    ps2[cusp] = p*np.exp((1.+a)*(1.-b)*E)
  return (ps2/__density(x,y)).reshape(shape)

# radial profiles

def density(r,r_s,rho_s,A,r_c=0.):
  '''
  Evaluate density at radius r for a cusp-NFW density profile with scale radius
  r_s, scale density rho_s, and cusp coefficient A.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
      
    r_s: float or array
      Scale radius of halo.
    
    rho_s: float or array
      Scale density of halo.
      
    A: float or array
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
      
    r_c: float or array
      Radius of the phase-space core.
    
  Returns:
    
    rho: float or array
      Density at radius r.
  '''
  return __density(r/r_s,A/(rho_s*r_s**1.5),r_c/r_s) * rho_s

def mass(r,r_s,rho_s,A):
  '''
  Evaluate mass enclosed within radius r for a cusp-NFW density profile with
  scale radius r_s, scale density rho_s, and cusp coefficient A.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
      
    r_s: float or array
      Scale radius of halo.
    
    rho_s: float or array
      Scale density of halo.
      
    A: float or array
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
    
  Returns:
    
    M: float or array
      Mass enclosed within radius r.
  '''
  return r_s**3*rho_s * __mass(r/r_s,A/(rho_s*r_s**1.5))

def veldisp2_r(r,r_s,rho_s,A,G=1.):
  '''
  Evaluate squared radial velocity dispersion, sigma_r^2, at radius r for a
  cusp-NFW density profile with scale radius r_s, scale density rho_s, and cusp
  coefficient A. We assume an isotropic velocity distribution.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
      
    r_s: float or array
      Scale radius of halo.
    
    rho_s: float or array
      Scale density of halo.
      
    A: float or array
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
    
    G: float
      Gravitational constant. If not specified, we return sigma_r^2/G, which
      has dimensions of mass/length.
    
  Returns:
    
    sigma_r^2: float or array
      Squared radial velocity dispersion at radius r. If G was not specified,
      this is sigma_r^2/G instead.
  '''
  return G*rho_s*r_s**2 * __veldisp2_r(r/r_s,A/(rho_s*r_s**1.5))

def veldisp_r(r,r_s,rho_s,A,G=1.):
  '''
  Evaluate radial velocity dispersion sigma_r at radius r for a cusp-NFW
  density profile with scale radius r_s, scale density rho_s, and cusp
  coefficient A. We assume an isotropic velocity distribution.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
      
    r_s: float or array
      Scale radius of halo.
    
    rho_s: float or array
      Scale density of halo.
      
    A: float or array
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
    
    G: float
      Gravitational constant. If not specified, we return sigma_r/sqrt(G),
      which has dimensions of sqrt(mass/length).
    
  Returns:
    
    sigma_r: float or array
      Radial velocity dispersion at radius r. If G was not specified, this is
      sigma_r/sqrt(G) instead.
  '''
  return np.sqrt(veldisp2_r(r,r_s,rho_s,A,G=G))

def potential(r,r_s,rho_s,A,G=1.,zero_at_inf=False):
  '''
  Evaluate gravitational potential Phi at radius r for a cusp-NFW density
  profile with scale radius r_s, scale density rho_s, and cusp coefficient A.
  By default, we take the potential to be zero at r=0.
  
  Parameters:
    
    r: float or array
      Radius from center of halo.
      
    r_s: float or array
      Scale radius of halo.
    
    rho_s: float or array
      Scale density of halo.
      
    A: float or array
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
    
    G: float
      Gravitational constant. If not specified, we return Phi/G, which has
      dimensions of mass/length.
    
    zero_at_inf: bool
      If True, the zero point of energy is set so that Phi=0 at r=inf.
      Default is zero_at_inf=False, in which case Phi=0 at r=0.
    
  Returns:
    
    Phi: float or array
      Gravitational potential at radius r. If G was not specified, this is
      Phi/G instead.
  '''
  return G*rho_s*r_s**2 * __potential(r/r_s,A/(rho_s*r_s**1.5),zero_at_inf=zero_at_inf)

# distribution function

# we evaluate f(E) from an analytic fit rather than by Eddington inversion; with
# e the energy measured from the centre, Ecal = Phi_inf - e, and
#   t = e/Phi_inf, Et = Ecal/Phi_inf, L = -ln(Et), v = 1-t/L, eps = e/y**2,
#   w = Phi_inf/(4 pi) - 1
# the fit is
#   f = KNFW Phi_inf**-2.5 Et**1.5 t**0.5 L**-3 Xi(eps) exp[S(v,w)+R(eps,w)]
#   Xi(eps) = 1 + [epsstar**2 + 5 pi eps ln(1+k eps)]/eps**2
# where S and R are Chebyshev series in 2v-1 and in tanh[(ln eps-M0)/W0], each
# carrying its w dependence as a further series in 2w-1
# S vanishes at v=0 and R as eps -> inf, so the fit is exact in the limits
#   f -> KNFW e**-2.5 (e->0, y=0), KIN y**4 e**-4.5 (e->0, y>0), Et**1.5 L**-3 (Et->0)
# KNFW and KIN are the DFs of an isolated rho ~ r**-1 and rho = A r**-1.5 cusp,
# and epsstar**2 = KIN/KNFW is where the two cross over; the 21 series
# coefficients and k, M0, W0 are fitted, to 0.87% over 0 <= y <= 1 and
# 1e-16 <= Et < 1
__df_KNFW = 3./(4.*np.sqrt(2.))
__df_KIN = 1120.*np.sqrt(2.)/9.*np.pi**2
__df_epsstar2 = __df_KIN/__df_KNFW
__df_k,__df_M0,__df_W0 = 0.674147,3.57522,2.09877
__df_A = np.array([[1.37488753,0.824326109,0.125582409,-0.00863708221],
                   [-0.492795773,-0.452733042,-0.0137832097,0],
                   [0.300251105,0.0940864561,-0.00640926101,0],
                   [0,-0.00465109009,0,0],
                   [0.0247550971,0,0,0],
                   [0.00714355977,0,0,0]])
__df_B = np.array([[-1.24519817,-1.54926827,-0.326200169],
                   [-0.413672782,-0.435261035,0],
                   [0,-0.0323075178,0],
                   [-0.101499378,-0.0653558866,0]])
def __df_monomial(C,n):
  out = np.zeros((n,C.shape[1]))
  for m in range(C.shape[1]):
    c = np.polynomial.chebyshev.cheb2poly(C[:,m])
    out[:len(c),m] = c
  return out
# monomial copies of the tables, so that evaluating a series is one Horner pass
# (these degrees are low enough that the change of basis costs no precision)
__df_Ap = __df_monomial(np.insert(__df_A,0,0.,axis=0),__df_A.shape[0]+1)
__df_Bp = __df_monomial(__df_B,__df_B.shape[0])
__df_alt = (-1.)**np.arange(__df_Ap.shape[0]) # to subtract S at v=0, i.e. z=-1
def __df_horner(c,z):
  out = np.full(z.shape,c[-1])
  for ck in c[-2::-1]:
    out *= z
    out += ck
  return out
def __df(e,y,zero_at_inf=False):
  e,y = np.broadcast_arrays(np.array(e).astype(float),np.array(y).astype(float))
  shape = e.shape
  e,y = np.atleast_1d(e),np.atleast_1d(y)
  pot_inf = __potential_inf(y)
  if zero_at_inf:
    ecal,e = -e,e + pot_inf
  else:
    ecal = pot_inf - e
  bound = (e>0.)&(ecal>0.)
  if not np.any(bound):
    return np.zeros(shape)[()]
  e,ecal,y,pot_inf = e[bound],ecal[bound],y[bound],pot_inf[bound]
  t,et = e/pot_inf,ecal/pot_inf
  # take L = -ln(Et) from whichever of e, Ecal was given exactly
  # (with the zero point at r=0, Et below machine precision is unresolvable)
  L = -np.log(et) if zero_at_inf else -np.log1p(-np.minimum(t,1.-1e-16))
  # contract the w direction first, leaving one series in v and one in ln(eps);
  # with w varying from point to point that leaves a coefficient per point
  zw = 2.*(pot_inf/(4*np.pi)-1.)-1.
  Tw = np.array([np.ones(zw.shape),zw,2.*zw*zw-1.,zw*(4.*zw*zw-3.)])
  c = __df_Ap.dot(Tw[:__df_Ap.shape[1]])
  S = __df_horner(c,1.-2.*t/L) - __df_alt.dot(c)
  # transition factor and its finite-y correction, both trivial for y=0
  xi = np.ones(e.shape)
  cusp = y > 0.
  if np.any(cusp):
    eps = e[cusp]/y[cusp]**2
    xi[cusp] = 1. + (__df_epsstar2 + 5*np.pi*eps*np.log(1.+__df_k*eps))/eps**2
    ze = np.tanh((np.log(eps)-__df_M0)/__df_W0)
    S[cusp] += (1.-ze*ze)*__df_horner(__df_Bp.dot(Tw[:__df_Bp.shape[1]][:,cusp]),ze)
  f = __df_KNFW*pot_inf**-2.5 * et*np.sqrt(et*t)/(L*L*L) * xi*np.exp(S)
  out = np.zeros(bound.shape)
  out[bound] = f
  return out.reshape(shape)[()]

def df(E,r_s,rho_s,A,G=1.,zero_at_inf=False):
  '''
  Evaluate the distribution function f(E) for a cusp-NFW density profile wit
  h scale radius r_s, scale density rho_s, and cusp coefficient A. We assume an
  isotropic velocity distribution.
  
  Parameters:
    
    E: float or array
      Energy E. If G is not specified, this is assumed to be E/G instead.
      
    r_s: float or array
      Scale radius of halo.
    
    rho_s: float or array
      Scale density of halo.
      
    A: float or array
      Cusp coefficient. This profile only makes sense if rho_s * r_s**1.5 >= A.
    
    G: float
      Gravitational constant. If not specified, then we assume the first
      argument is E/G, which has dimensions of mass/length, and we return
      G^1.5 f(E).
    
    zero_at_inf: bool
      If True, the zero point of energy is set so that the potential is zero at
      r=inf. Default is zero_at_inf=False, in which case the potential is zero
      at r=0.
    
  Returns:
    
    f: float or array
      Distribution function f(E). If G was not specified, this is G^1.5 f(E)
      instead.
  '''
  return __df(E/(G*rho_s*r_s**2),A/(rho_s*r_s**1.5),zero_at_inf=zero_at_inf) / (G**1.5*r_s**3*rho_s**0.5)

# conversions from halo concentration

def r2_from_rs(r_s,rho_s,A):
  '''
  Find the radius r_{-2} at which dlog(rho)/dlog(r) = -2 for a cusp-NFW density
  profile with scale radius r_s, scale density rho_s, and cusp coefficient A.
  
  r_{-2} ranges from r_s/2 (if A=rho_s*r_s**1.5) to r_s (if A=0).
  '''
  y = A / (rho_s * r_s**1.5) # <= 1
  return 0.5 * (1. - 1.5*y**2 + np.sqrt(1.-y**2+2.25*y**4)) * r_s

def rs_from_r2(r_2,rho_s,A):
  '''
  Find the cusp-NFW scale radius r_s, given r_2 (the radius r_{-2} at which
  dlog(rho)/dlog(r) = -2), the scale density rho_s, and the cusp coefficient A.
  '''
  z = A / (rho_s * r_2**1.5) # <= 2**1.5
  f = (1. + 18.*z**2 + 0.75*z*np.sqrt(72 + 564*z**2 + 6*z**4))**(1./3)
  return ((1/3.-z**2/2.)/f + (1+f)/3.) * r_2
  
def R_from_M(M,rho_vir):
  '''Virial radius R given halo mass M and virial density rho_vir.'''
  return (3*M/(4*np.pi*rho_vir))**(1./3)
  
def __rhos_from_c_NFW(c,rho_vir):
  return c**3/(np.log(1+c)-c/(1+c))*rho_vir/3.

# for a halo of mass M and concentration c whose cusp parameter is
# y = A/(rho_s * r_s**1.5), the combination M*rho_vir/A**2 is
#   P(c,y) = 3 mass(u,1,1,y)**2/(4 pi y**2 u**3),  u = c r2_from_rs(1,1,y)
# a function of c and y alone, so the largest A allowed is at the y minimising
# it. y = 1 is a stationary point of P for every c, but is the minimum only
# below c = 10.146; above that the minimum moves to y* < 1 and the limit on A
# is larger than the y=1 value, by 0.02% at c = 16, 0.1% at c = 25 and 1.2% at
# c = 1000. __min_params_y1 is P(c,1), while q = P(c,y*)/P(c,1) and y* are
# fitted in v = log(c/__opt_ccrit) as 1-q = v**3 num/den and 1-y* = v num/den,
# to 6e-9 and 2e-6 out to c = 1e6, staying monotonic and bounded beyond it
__opt_ccrit = 10.1463872020 # above this c the largest A is at y < 1
__opt_qa,__opt_qb = (5.796430704e-05,0.0003571185986,0.002474268327,0.007381727488),(0.001143892779,0.01075713898,0.0813062605,0.339477536,0.9102772651,1.426472619)
__opt_ya,__opt_yb = (0.01960683819,0.09598029503,0.363946101,0.3136607138),(0.05872743104,0.3340454417,1.286501543,2.029260634)
def __opt_ratio(v,a,b,p):
  # v**p a(v)/(1 + v b(v)), the coefficients running from the highest order down
  num,den = 0.,0.
  for ak in a:
    num = num*v + ak
  for bk in b:
    den = den*v + bk
  return v**p*num/(1. + v*den)
def __min_params_largec(c):
  return 384*np.pi*(np.sqrt(c/(2 + c)) - np.arcsinh(np.sqrt(c/2)))**2/c**3
def __min_params_smallc(c):
  return 16*np.pi/3. - 24*np.pi/5.*c + 564*np.pi/175.*c**2
def __min_params_y1(c):
  # P(c,1); the closed form cancels as c -> 0, hence the series there
  if np.ndim(c) == 0:
    return __min_params_smallc(c) if c < 0.001 else __min_params_largec(c)
  c = np.array(c).astype(float)
  return np.where(c < 0.001,__min_params_smallc(np.minimum(c,0.001)),
                  __min_params_largec(np.maximum(c,0.001)))
def __opt_y(c):
  # the y at which A is largest, which is 1 up to __opt_ccrit
  return 1. - __opt_ratio(np.log(np.maximum(c/__opt_ccrit,1.)),__opt_ya,__opt_yb,1)
def __min_params(c):
  q = 1. - __opt_ratio(np.log(np.maximum(c/__opt_ccrit,1.)),__opt_qa,__opt_qb,3)
  return q*__min_params_y1(c)

def scale_from_c(c,M,A,rho_vir,cmin_error=True,branch='low'):
  '''
  Find the cusp-NFW profile scale parameters, given halo concentration c and
  mass M.
  
  Parameters:
    
    c, M: floats
      Halo parameters.
    
    A: float
      Cusp coefficient parameter.
    
    rho_vir: float
      The virial density (e.g., 200 times the cosmological mean value).
    
    cmin_error: boolean
      If True, we raise an exception if the concentration is too low for a halo
      of this mass and cusp coefficient to exist. If False, we increase the
      concentration to
      its minimum value and continue. Default is True.
    
    branch: 'low' or 'high'
      Which halo to return when (c,M,A) admits more than one. The two differ in
      y = A/(rho_s * r_s**1.5), and this selects the smaller or the larger of
      them; 'low', the default, is the branch this function has always returned.
      A second halo exists only for A between the y = 1 value and A_max, a range
      that is empty below c = 10.146; where there is only one, both settings
      return it. See A_max.
    
  Returns:
    
    r_s, rho_s: floats
      The scale radius and scale density, respectively.
  '''
  R = R_from_M(M,rho_vir)
  r_2 = R / c # r_{-2}
  if A == 0.:
    return r_2,__rhos_from_c_NFW(c,rho_vir)
  if M*rho_vir/A**2 < __min_params(c):
    if cmin_error:
      raise Exception('M*rho_vir/A**2=%.3e must be >%.3e for c=%.2f.'%(M*rho_vir/A**2,__min_params(c),c))
    c = c_min(M,A,rho_vir)
    y_A = __opt_y(c) # at c_min the halo sits exactly at the turning point
    r_s = R/(c*r2_from_rs(1.,1.,y_A))
    rho_s = A/(y_A*r_s**1.5)
  else:
    lo,hi = A/(2*r_2)**1.5,__rhos_from_c_NFW(c,rho_vir) # rho_s at y=1 and at y=0
    g = lambda rho_s: np.log(mass(R,rs_from_r2(r_2,rho_s,A),rho_s,A)/M)
    if M*rho_vir/A**2 > __min_params_y1(c): # A below the y=1 value: one halo
      rho_s = root_scalar(g,bracket=[lo,hi]).root
    else: # two halos, one on each side of the turning point of the curve
      y_A = __opt_y(c) # the y where A is largest
      rho_A = A/(y_A*(r_2/r2_from_rs(1.,1.,y_A))**1.5) # rho_s there
      if branch == 'high': # the two merge as A comes back down to the y=1 value
        rho_s = root_scalar(g,bracket=[lo,rho_A]).root if g(lo) > 0. else lo
      else:
        rho_s = root_scalar(g,bracket=[rho_A,hi]).root
    r_s = rs_from_r2(r_2,rho_s,A)
  return r_s, rho_s

def A_max(c,M,rho_vir):
  '''
  Maximum cusp coefficient A in a halo of mass M and concentration c. Here
  rho_vir is the virial density.
  
  This is the largest A anywhere on the curve of halos with this mass and
  concentration. Below c = 10.146 it is the A of the y = 1 halo, the one with
  rho_s * r_s**1.5 = A; above that the curve turns over before y reaches 1 and
  the maximum, slightly larger, sits at y < 1. In between the two, (c,M,A)
  admits two halos; see scale_from_c.
  '''
  return np.sqrt(M*rho_vir/__min_params(c))

def M_min(c,A,rho_vir):
  '''
  Minimum sensible halo mass M given concentration c and cusp coefficient A.
  Here rho_vir is the virial density. This inverts A_max.
  '''
  return __min_params(c) * A**2 / rho_vir

def c_min(M,A,rho_vir):
  '''
  Minimum sensible concentration c given halo mass M and cusp coefficient A.
  Here rho_vir is the virial density. This inverts A_max.
  '''
  params = M*rho_vir/A**2
  if params > 16*np.pi/3.:
    return 0.
  lnc = root_scalar(lambda lnc: np.log(__min_params(np.exp(lnc))/params),x0=0.,x1=1.).root
  return np.exp(lnc)

# phase-space core

def rc_from_fmax(fmax,r_s,rho_s,A,G=1.,prefactor=3e-5):
  '''
  Evaluate the core radius from phase-space conservation.
  
  Parameters:
    
    fmax: float
      Maximum phase-space density f_max. If G is not specified, this is assumed
      to be G^1.5 f_max.
      
    r_s: float
      Scale radius of halo.
    
    rho_s: float
      Scale density of halo.
      
    A: float
      Cusp coefficient. The profile only makes sense if rho_s * r_s**1.5 >= A.
    
    G: float
      Gravitational constant. If not specified, then we assume the first
      argument is G^1.5 f_max, which has dimensions of mass/length, and we return
      G^1.5 f(E).
    
    prefactor: float
      Prefactor in the "phase-space barrier" from arXiv:2207.05082. Default is
      prefactor=3e-5 (see appendix C of arXiv:2207.05082).
    
  Returns:
    
    r_c: float
      Core radius.
  '''
  target = prefactor * G**-3 * fmax**-2 / (rho_s * r_s**6)
  y = A / (rho_s * r_s**1.5)
  fun = lambda lnx: np.log(__density(np.exp(lnx), y) * np.exp(6*lnx) / target)
  try:
    lnx = root_scalar(fun, bracket=[-100., 100.]).root
  except ValueError:
    return 0. if fun(0.) > 0. else np.inf
  return np.exp(lnx) * r_s
