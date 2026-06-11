'''

Extended Press-Schechter (EPS) main-progenitor mass accretion history.

This module provides an alternative to the crude, universal closed-form growth
history mass_growth_fun used by CuspHalo. It computes the mean main-progenitor
mass accretion history (MAH) as the first moment of the Lacey & Cole (1993,
sec. 2.3, Fig. 4) merger rate -- the mean fractional mass-accretion rate, with
q = DeltaM/M weighting the splitting kernel -- integrated backward in time. This
is the deterministic mean of a Cole et al. (2000) binary-split merger tree.

Key simplification
------------------
With the base EPS normalization (G = 1), the linearized progenitor rate kernel

  rate(M) = integral_0^{1/2} q (dN/dq)/domega dq

depends ONLY on the halo mass M -- through the linear-theory (a=1) top-hat
variances sigma(M), sigma(qM), and |dln sigma / dln M| -- and NOT on time. The
mean-MAH ODE  dln M / domega = -rate(M)  is therefore autonomous, and its
solution is a single one-dimensional "mass flow"

  Omega(M) = integral dln M / rate(M)            (monotonic in M)
  Omega(M_prog) = Omega(M0) - [omega(a') - omega(a0)]

with the EPS "time" omega(a) = delta_c / D(a). The whole MAH is then a cheap,
fully vectorized table lookup, with no per-halo ODE integration.

References
----------
Bond, Cole, Efstathiou & Kaiser (1991); Lacey & Cole (1993); Cole et al. (2000);
van den Bosch (2002); Neistein, van den Bosch & Dekel (2006); Neistein & Dekel
(2008); Parkinson, Cole & Helly (2008).

'''

import numpy as np
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erfc

from .mpk_helpers.perturbations import W

DELTA_C = 1.686 # critical linear overdensity for spherical collapse


_GEOM_CACHE = {}      # (k.tobytes(), rho, nM) -> (log10M, W2, w_lnk)
_GEOM_CACHE_MAX = 4   # bound memory: each W2 is ~nk*nM floats


def _tophat_geometry(k, rho, nM):
  '''Mass grid, squared top-hat window W(kR)^2, and Simpson weights for the
  sigma(M) integral.

  These depend only on the k-grid, mean density rho, and nM, so they can be
  reused across power spectra (e.g. every B_bump in a bisection at fixed
  cosmology). The result is cached because W(kR)^2 dominates the cost of
  building EPSGrowth.

  w_lnk is the Simpson weight vector for integration over ln k: because Simpson
  is linear in the integrand, simpson(y, x=lnk) == w_lnk @ y for any y, so
  sigma^2(M) = int P W^2 dlnk reduces to the matvec (w_lnk*P) @ W2.
  '''
  key = (k.tobytes(), float(rho), int(nM))
  hit = _GEOM_CACHE.get(key)
  if hit is None:
    # R = 1/k -> M(k) = (4 pi / 3) rho / k^3; the widest masses the spectrum
    # can constrain span k in [k_min, k_max].
    M_of_k = lambda kk: 4.*np.pi/3. * rho / kk**3
    log10M = np.linspace(np.log10(M_of_k(k.max())),
                         np.log10(M_of_k(k.min())), nM)
    R = (3.*10.**log10M / (4.*np.pi*rho))**(1./3.)
    W2 = W(k[:,None] * R[None,:])**2 # (nk, nM)
    # extract the exact Simpson weights from the identity basis (one-time)
    lnk = np.log(k)
    w_lnk = simpson(np.eye(k.size), x=lnk, axis=0)
    if len(_GEOM_CACHE) >= _GEOM_CACHE_MAX:
      _GEOM_CACHE.clear()
    _GEOM_CACHE[key] = hit = (log10M, W2, w_lnk)
  log10M, W2, w_lnk = hit
  return log10M.copy(), W2, w_lnk # copy the small grid; share the large arrays read-only


class EPSGrowth(object):
  '''

  EPS main-progenitor mass accretion history for a fixed (a=1) power spectrum.

  Parameters:

    k, P: arrays
      Tabulated dimensionless matter power spectrum P(k), evaluated in linear
      theory at a=1 (same convention as CuspHalo: sigma_j^2 = integral
      P k^(2j) dln k, so sigma^2(M) = integral P W(kR)^2 dln k).

    rho: float
      Mean matter density at a=1, used to map mass to top-hat radius via
      R = (3 M / (4 pi rho))^(1/3). Same units as the halo masses M.

    delta_c: float
      Critical linear overdensity for collapse. Default 1.686.

    nM: int
      Number of mass-grid points for the sigma(M)/Omega(M) tables. Default 600.

    nq: int
      Number of mass-ratio-grid points for the rate integral. Default 600.

  The MAH "time" variable is omega(a) = delta_c / D(a); the caller supplies
  omega (or differences of omega) directly, so this class is cosmology-agnostic.

  '''

  def __init__(self, k, P, rho, delta_c=DELTA_C, nM=600, nq=600):
    self.k = np.asarray(k, dtype=float)
    self.P = np.asarray(P, dtype=float)
    self.rho = float(rho)
    self.delta_c = float(delta_c)
    self.lnk = np.log(self.k)

    # ---- mass grid + squared top-hat window (cached; B_bump-independent) -----
    self._log10M, W2, w_lnk = _tophat_geometry(self.k, self.rho, nM)
    M = 10.**self._log10M

    # ---- sigma(M): real-space top-hat variance, sigma^2 = int P W(kR)^2 dlnk --
    # equivalent to simpson(P[:,None]*W2, x=lnk, axis=0) but as a single matvec.
    sigma2 = (w_lnk * self.P) @ W2
    self._lnsig = 0.5*np.log(sigma2)
    # lnsigma(log10 M) cubic spline; its derivative gives dln sigma / dln M.
    self._lnsig_spl = InterpolatedUnivariateSpline(self._log10M, self._lnsig, k=3)

    # ---- rate(M) = int_{q_lo}^{1/2} (1/q) sqrt(2/pi) s1^2 a1 / (s1^2-s2^2)^1.5 dq
    self._nq = int(nq)
    rate = self._rate_of_M(M)

    # ---- autonomous flow Omega(M) = int dln M / rate(M) ---------------------
    # Restrict to the contiguous high-mass region where rate is well-resolved;
    # below it the progenitor "freezes" (rate -> 0 near the cutoff).
    good = np.isfinite(rate) & (rate > 0)
    if good.any():
      rmax = rate[good].max()
      good &= rate > rmax*1e-10
    if good.sum() < 4:
      raise RuntimeError('EPSGrowth: rate(M) is degenerate; check the spectrum '
                         'and mass range.')
    # keep the contiguous block ending at the most massive grid point
    idx = np.where(good)[0]
    lo = idx[0]
    hi = len(M)-1
    j = hi
    while j-1 >= lo and good[j-1]:
      j -= 1
    lo = j
    sl = slice(lo, hi+1)
    self._M_lo = M[lo]
    self._M_hi = M[hi]

    lnM = np.log(M[sl])
    Omega = cumulative_trapezoid(1./rate[sl], lnM, initial=0.)
    self._Omega = Omega
    # Omega is monotonic increasing in M; build both directions.
    self._Omega_spl = InterpolatedUnivariateSpline(lnM, Omega, k=3)
    self._lnM_of_Omega = InterpolatedUnivariateSpline(Omega, lnM, k=3)
    self._Omega_lo = Omega[0]
    self._Omega_hi = Omega[-1]

    # characteristic cutoff mass: where sigma(M) falls to half its plateau peak
    # (sigma is largest at the smallest resolved mass). A representative
    # "cutoff-scale halo" mass for concentration estimates when no external
    # half-mode mass is supplied.
    sig_half = self._lnsig[0] + np.log(0.5)
    self.M_cut = float(np.clip(
        10.**np.interp(sig_half, self._lnsig[::-1], self._log10M[::-1]),
        self._M_lo, self._M_hi))

  # ----------------------------------------------------------------------------
  #  sigma(M) and its logarithmic derivative
  # ----------------------------------------------------------------------------
  def sigma(self, M):
    '''rms linear (a=1) density contrast in a top-hat sphere of mass M.'''
    return np.exp(self._lnsig_spl(np.log10(M)))

  def dlnsigma_dlnM(self, M):
    '''dln sigma / dln M (negative; sigma decreases with mass).'''
    return self._lnsig_spl(np.log10(M), nu=1) / np.log(10.)

  def _rate_of_M(self, M):
    '''rate(M) = -dln M / domega for the linearized EPS kernel (G=1).

    Vectorized over M: accepts a scalar or array and returns the matching shape.
    Each mass uses its own q-grid spanning [q_lo(M), 1/2], with the lower limit
    q_lo = clip(M_min/M, 1e-30, 1/4) tracking the smallest resolved mass so the
    integration stays inside the sigma(M) table.
    '''
    M = np.asarray(M, dtype=float)
    scalar = (M.ndim == 0)
    Mf = np.atleast_1d(M)                                   # (nM,)
    M_min = 10.**self._log10M[0]
    q_lo = np.clip(M_min / Mf, 1e-30, 0.25)                 # (nM,)
    # geometric q-grid per mass, shape (nq, nM)
    t = np.linspace(0., 1., self._nq)[:, None]
    q = np.exp(np.log(q_lo)[None, :] * (1. - t) + np.log(0.5) * t)
    Mq = q * Mf[None, :]
    log10Mq = np.log10(Mq)
    s1 = np.exp(self._lnsig_spl(log10Mq.ravel()).reshape(Mq.shape))
    a1 = np.abs(self._lnsig_spl(log10Mq.ravel(), nu=1).reshape(Mq.shape) / np.log(10.))
    s2 = np.exp(self._lnsig_spl(np.log10(Mf)))              # (nM,)
    d = s1**2 - s2[None, :]**2
    good = d > 0
    d_safe = np.where(good, d, 1.0)
    integrand = np.where(good,
                         (1./q) * np.sqrt(2./np.pi) * s1**2 * a1 / d_safe**1.5,
                         0.0)
    rate = np.trapezoid(integrand, q, axis=0)               # (nM,)
    return float(rate[0]) if scalar else rate

  # ----------------------------------------------------------------------------
  #  the MAH flow
  # ----------------------------------------------------------------------------
  def main_progenitor(self, M0, domega):
    '''

    Main-progenitor mass at EPS-time offset domega earlier than the epoch at
    which the halo mass is M0:

      M_prog = Omega^{-1}( Omega(M0) - domega )

    domega = omega(a') - omega(a0) >= 0 for an earlier scale factor a' < a0.
    The result is clamped to the resolved mass range, so the progenitor freezes
    near the spectral cutoff (where the EPS rate vanishes).

    Both M0 and domega may be scalars or broadcastable arrays.

    '''
    M0 = np.asarray(M0, dtype=float)
    domega = np.asarray(domega, dtype=float)
    lnM0 = np.log(np.clip(M0, self._M_lo, self._M_hi))
    Omega0 = self._Omega_spl(lnM0)
    Omega_t = np.clip(Omega0 - domega, self._Omega_lo, self._Omega_hi)
    out = np.exp(self._lnM_of_Omega(Omega_t))
    return out if out.ndim else float(out)

  def collapsed_fraction(self, M0, domega, f=0.02):
    '''

    EPS collapsed-mass fraction (Lacey & Cole 1993; Ludlow et al. 2016, eqs 3
    and 7):

      Mcoll/M0 = erfc( domega / sqrt(2 (sigma^2(f M0) - sigma^2(M0))) ),

    the fraction of the final mass M0 that, at an EPS-time offset domega before
    the epoch at which the halo mass is M0, was already assembled into collapsed
    progenitors more massive than f*M0. The variances are the a=1 top-hat
    sigma(M), with the time dependence carried entirely by
    domega = omega(a') - omega(a0) = delta_sc(a') - delta_sc(a0) >= 0.

    The result is 1 at domega=0 and decreases toward earlier times. f*M0 is
    clamped to the resolved mass range so the variance saturates at the spectral
    cutoff. domega may be a scalar or array.

    '''
    domega = np.asarray(domega, dtype=float)
    s2 = self.sigma(max(f*M0, self._M_lo))**2 - self.sigma(M0)**2
    return erfc(domega / np.sqrt(2.*s2))

  def M_nonlinear(self, omega):
    '''

    Characteristic collapsing ("nonlinear") mass M* defined by sigma(M*) = omega,
    with omega = delta_c / D(a). Clamped to the resolved mass range.

    '''
    omega = np.asarray(omega, dtype=float)
    # lnsigma is monotonically decreasing in log10 M -> invert with reversed grid
    lns = np.log(np.clip(omega, np.exp(self._lnsig[-1]), np.exp(self._lnsig[0])))
    log10M = np.interp(lns, self._lnsig[::-1], self._log10M[::-1])
    out = np.clip(10.**log10M, self._M_lo, self._M_hi)
    return out if out.ndim else float(out)
