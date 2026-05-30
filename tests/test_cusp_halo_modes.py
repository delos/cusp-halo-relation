import numpy as np
import pytest

from cusp_halo_relation import CuspHalo, default_params
from conftest import make_spectrum


def build(mode):
    k, P = make_spectrum()
    mp = {**default_params, 'growth_model': mode}
    return CuspHalo(k, P, growth=1.0, rho=1.0, model_params=mp, verbose=False)


@pytest.fixture(scope="module")
def exp_model():
    return build('exp')


@pytest.fixture(scope="module")
def eps_model():
    return build('eps')


@pytest.mark.parametrize("mode", ['exp', 'eps'])
def test_mode_runs(mode):
    model = build(mode)
    a = 1.0
    for M in [1e2, 1e3, 1e4]:
        a_coll = model.collapse_a(M, a)
        assert np.isfinite(a_coll) and a_coll <= a
        assert np.isfinite(model.m(M, a)) and model.m(M, a) > 0
        assert np.isfinite(model.A(M, a)) and model.A(M, a) > 0


def test_invalid_mode_raises():
    k, P = make_spectrum()
    mp = {**default_params, 'growth_model': 'bogus'}
    with pytest.raises(Exception):
        CuspHalo(k, P, growth=1.0, rho=1.0, model_params=mp, verbose=False)


def test_modes_differ_but_same_order(exp_model, eps_model):
    a = 1.0
    for M in [1e2, 1e3, 1e4]:
        m_exp, m_eps = exp_model.m(M, a), eps_model.m(M, a)
        assert not np.isclose(m_exp, m_eps, rtol=1e-3)   # genuinely different
        assert 0.1 < m_eps / m_exp < 10.0                # same ballpark


@pytest.mark.parametrize("mode_fixture", ['exp_model', 'eps_model'])
def test_concentration_rises_to_late_times(mode_fixture, request):
    model = request.getfixturevalue(mode_fixture)
    a = np.array([0.2, 0.5, 1.0])
    c = np.array([model.characteristic_c(ai) for ai in a])
    assert np.all(np.isfinite(c)) and np.all(c > 1.0)
    assert np.all(np.diff(c) > 0)   # more concentrated at later times


def test_eps_default_is_exp():
    # the package default leaves the growth model as the closed-form 'exp'
    assert default_params['growth_model'] == 'exp'
    k, P = make_spectrum()
    model = CuspHalo(k, P, growth=1.0, rho=1.0, verbose=False)
    assert model.growth_model == 'exp'
