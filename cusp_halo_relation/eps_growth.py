'''

Extended Press-Schechter (EPS) main-progenitor mass accretion history.

This module provides an alternative to the crude, universal closed-form growth
history mass_growth_fun used by CuspHalo. It computes the mean main-progenitor
mass accretion history (MAH) by integrating the linearized Lacey & Cole (1993)
progenitor rate backward in time, following Neistein, van den Bosch & Dekel
(2006).

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

    # ---- mass grid, derived from the k-grid via the top-hat M-R relation ----
    # R = 1/k  ->  M(k) = (4 pi / 3) rho / k^3.  The widest masses the spectrum
    # can constrain span k in [k_min, k_max].
    def M_of_k(kk):
      return 4.*np.pi/3. * self.rho / kk**3
    log10M_hi = np.log10(M_of_k(self.k.min()))
    log10M_lo = np.log10(M_of_k(self.k.max()))
    self._log10M = np.linspace(log10M_lo, log10M_hi, nM)
    M = 10.**self._log10M

    # ---- sigma(M): real-space top-hat variance, sigma^2 = int P W(kR)^2 dlnk --
    R = (3.*M / (4.*np.pi*self.rho))**(1./3.)
    x = self.k[:,None] * R[None,:] # (nk, nM)
    sigma2 = simpson(self.P[:,None] * W(x)**2, x=self.lnk, axis=0)
    self._lnsig = 0.5*np.log(sigma2)
    # lnsigma(log10 M) cubic spline; its derivative gives dln sigma / dln M.
    self._lnsig_spl = InterpolatedUnivariateSpline(self._log10M, self._lnsig, k=3)

    # ---- rate(M) = int_{q_lo}^{1/2} (1/q) sqrt(2/pi) s1^2 a1 / (s1^2-s2^2)^1.5 dq
    self._nq = int(nq)
    rate = np.array([self._rate_of_M(MM) for MM in M])

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
    '''rate(M) = -dln M / domega for the linearized EPS kernel (G=1).'''
    log10M = np.log10(M)
    # q floor: stay inside the sigma(M) table, capped at 1/4.
    q_lo = max(10.**(self._log10M[0]) / M, 1e-30)
    q_lo = min(q_lo, 0.25)
    q = np.geomspace(q_lo, 0.5, self._nq)
    Mq = q*M
    s1 = np.exp(self._lnsig_spl(np.log10(Mq)))
    s2 = np.exp(self._lnsig_spl(log10M))
    a1 = np.abs(self._lnsig_spl(np.log10(Mq), nu=1) / np.log(10.))
    d = s1**2 - s2**2
    integrand = np.zeros_like(q)
    good = d > 0
    integrand[good] = (1./q[good]) * np.sqrt(2./np.pi) * s1[good]**2 \
                      * a1[good] / d[good]**1.5
    return float(np.trapezoid(integrand, q))

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
