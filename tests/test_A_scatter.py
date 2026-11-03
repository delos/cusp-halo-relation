import numpy as np

from cusp_halo_relation import CuspHalo, default_params
from conftest import make_spectrum


def build(mp=default_params):
    k, P = make_spectrum()
    return CuspHalo(k, P, growth=1.0, rho=1.0, model_params=mp, verbose=False)


def test_sigma_lnA_closed_form():
    model = build()
    a = np.array([0.1, 0.3, 1.0])
    s0 = model.sigma0(a)
    np.testing.assert_allclose(model.sigma_lnA(a), 0.45 * np.exp(-1.0 / s0), rtol=1e-12)
    # log10 form of eq. A2: 0.195 exp(-1/sigma0); 0.45/ln10 = 0.1954
    np.testing.assert_allclose(model.sigma_lnA(a) / np.log(10), 0.195 * np.exp(-1.0 / s0), rtol=3e-3)


def test_sigma_lnA_params_without_key():
    mp = {key: v for key, v in default_params.items() if key != 'A_scatter_fun'}
    model = build(mp)
    assert np.isclose(model.sigma_lnA(1.0), 0.45 * np.exp(-1.0 / model.sigma0(1.0)))


def test_sample_A_lognormal():
    model = build()
    a = 1.0
    M = 1e3
    A_med = model.A(M, a)
    assert np.isfinite(A_med)
    x = np.log(model.sample_A(M, a, rng=np.random.default_rng(1), size=200000) / A_med)
    assert abs(np.median(x)) < 0.01
    assert np.isclose(np.std(x), model.sigma_lnA(a), rtol=0.01)
