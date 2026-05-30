import numpy as np
import pytest
from scipy.special import erfc

from cusp_halo_relation import CuspHalo, CuspHaloStandard, default_params
from conftest import make_spectrum


def build(growth_model='exp', concentration_model='L13'):
    k, P = make_spectrum()
    mp = {**default_params, 'growth_model': growth_model,
          'concentration_model': concentration_model}
    return CuspHalo(k, P, growth=1.0, rho=1.0, model_params=mp, verbose=False)


@pytest.fixture(scope="module")
def standard():
    # a concordance model with a declining cutoff (so the half-mode scale exists)
    kc = np.geomspace(1e-3, 1e3, 500)
    cutoff = (kc, np.exp(-(kc / 50.) ** 2))
    return CuspHaloStandard(cutoff, include_baryons=True, verbose=False)


# ---------------------------------------------------------------------------
#  EPS collapsed-mass fraction (L16 eq 7)
# ---------------------------------------------------------------------------

def test_collapsed_fraction_identity_and_monotonic(eps):
    M0 = 1e3
    assert eps.collapsed_fraction(M0, 0.0) == pytest.approx(1.0)
    domega = np.linspace(0., 3., 30)
    cf = eps.collapsed_fraction(M0, domega)
    assert np.all((cf > 0.) & (cf <= 1.0 + 1e-12))
    assert np.all(np.diff(cf) < 0.)   # decreasing toward earlier times


def test_collapsed_fraction_matches_erfc(eps):
    M0, f = 1e3, 0.02
    domega = np.array([0.3, 0.8, 1.5])
    s2 = eps.sigma(f * M0) ** 2 - eps.sigma(M0) ** 2
    expected = erfc(domega / np.sqrt(2. * s2))
    np.testing.assert_allclose(eps.collapsed_fraction(M0, domega, f=f), expected)


# ---------------------------------------------------------------------------
#  c(M, a) wiring and mass dependence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("growth_model", ['exp', 'eps'])
def test_L16_runs(growth_model):
    model = build(growth_model, 'L16')
    for M in [1e2, 1e3, 1e4]:
        c = model.c(M, 1.0)
        assert np.isfinite(c) and c > 1.0


def test_mass_dependence_by_model():
    masses = [1e2, 1e3, 1e4]
    # L13 + exp: mass-independent
    m = build('exp', 'L13')
    c = [m.c(M, 1.0) for M in masses]
    assert np.allclose(c, c[0])
    assert c[0] == pytest.approx(m.characteristic_c(1.0))
    # L13 + eps: mass-dependent
    m = build('eps', 'L13')
    c = [m.c(M, 1.0) for M in masses]
    assert np.ptp(c) > 1e-3 * c[0]
    # L16: mass-dependent
    m = build('exp', 'L16')
    c = [m.c(M, 1.0) for M in masses]
    assert np.ptp(c) > 1e-3 * c[0]


def test_L16_rises_to_late_times():
    model = build('eps', 'L16')
    a = np.array([0.2, 0.5, 1.0])
    c = np.array([model.c(1e3, ai) for ai in a])
    assert np.all(np.isfinite(c)) and np.all(c > 1.0)
    assert np.all(np.diff(c) > 0.)


def test_L16_vs_L13_ballpark():
    l13 = build('eps', 'L13')
    l16 = build('eps', 'L16')
    for a in [0.5, 1.0]:
        for M in [1e2, 1e3, 1e4]:
            ratio = l16.c(M, a) / l13.c(M, a)
            assert 0.3 < ratio < 3.0


def test_c_array_arguments():
    model = build('exp', 'L16')
    a = np.array([0.5, 1.0])
    c = model.c(1e3, a)
    assert c.shape == a.shape and np.all(np.isfinite(c))


# ---------------------------------------------------------------------------
#  lazy EPS, warnings, defaults
# ---------------------------------------------------------------------------

def test_lazy_eps_under_exp_growth():
    model = build('exp', 'L16')
    assert model._eps is None          # not built for 'exp' growth at construction
    model.c(1e3, 1.0)
    assert model._eps is not None       # built lazily by the L16 evaluation


def test_characteristic_c_warns_outside_exp_L13(capsys):
    k, P = make_spectrum()
    # L16 model, verbose: characteristic_c should warn
    mp = {**default_params, 'concentration_model': 'L16'}
    m = CuspHalo(k, P, growth=1.0, rho=1.0, model_params=mp, verbose=True)
    capsys.readouterr()                 # discard construction output
    m.characteristic_c(1.0)
    assert 'mass-independent' in capsys.readouterr().out
    # default exp+L13: no such warning
    m = CuspHalo(k, P, growth=1.0, rho=1.0, verbose=True)
    capsys.readouterr()
    m.characteristic_c(1.0)
    assert 'mass-independent' not in capsys.readouterr().out


def test_default_concentration_model_is_L13():
    assert default_params['concentration_model'] == 'L13'
    k, P = make_spectrum()
    model = CuspHalo(k, P, growth=1.0, rho=1.0, verbose=False)
    assert model.concentration_model == 'L13'


def test_invalid_concentration_model_raises():
    k, P = make_spectrum()
    mp = {**default_params, 'concentration_model': 'bogus'}
    with pytest.raises(Exception):
        CuspHalo(k, P, growth=1.0, rho=1.0, model_params=mp, verbose=False)


# ---------------------------------------------------------------------------
#  concordance c_at_z(M, z) and its single-argument shim
# ---------------------------------------------------------------------------

def test_c_at_z_single_arg_shim(standard):
    expected = standard.characteristic_c(1.0)
    # mass-independent characteristic concentration via several calling forms
    assert standard.c_at_z(0.0) == pytest.approx(expected)      # single positional
    assert standard.c_at_z(z=0.0) == pytest.approx(expected)    # keyword (README form)
    # default model is L13+exp (mass-independent), so the two-arg form agrees
    assert standard.c_at_z(1e10, 0.0) == pytest.approx(expected)
    assert standard.c_at_z(M=1e10, z=0.0) == pytest.approx(expected)


def test_c_at_z_mass_dependent_L16():
    kc = np.geomspace(1e-3, 1e3, 500)
    cutoff = (kc, np.exp(-(kc / 50.) ** 2))
    h = CuspHaloStandard(cutoff, include_baryons=True, verbose=False,
                         model_params={**default_params, 'concentration_model': 'L16'})
    c = [h.c_at_z(M, 0.0) for M in [1e8, 1e10, 1e12]]
    assert np.all(np.isfinite(c)) and np.ptp(c) > 1e-3 * c[0]
