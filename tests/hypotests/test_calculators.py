#!/usr/bin/python
import pytest
import numpy as np

import zfit
from zfit.core.testing import setup_function  # allows redefinition of zfit.Parameter, needed for tests
from zfit.core.loss import UnbinnedNLL
from zfit.minimize import MinuitMinimizer

from skstats.hypotests.calculators.basecalculator import BaseCalculator
from skstats.hypotests.calculators.asymptotic_calculator import AsymptoticCalculator
from skstats.hypotests.parameters import POI
from skstats.hypotests.fitutils.api_check import is_valid_loss, is_valid_data


true_mu = 1.2
true_sigma = 0.1


def create_loss():

    obs = zfit.Space('x', limits=(0.1, 2.0))
    data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
    mean = zfit.Parameter("mu", true_mu)
    sigma = zfit.Parameter("sigma", true_sigma)
    model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
    loss = UnbinnedNLL(model=[model], data=[data], fit_range=[obs])

    return loss, (mean, sigma)


@pytest.mark.parametrize("calculator", [BaseCalculator, AsymptoticCalculator])
def test_base_calculator(calculator):
    with pytest.raises(TypeError):
        calculator()

    loss, (mean, sigma) = create_loss()

    calc_loss = calculator(loss, MinuitMinimizer())
    bestfit = calc_loss.bestfit
    calc_fitresult = calculator(bestfit, calc_loss.minimizer)

    assert calc_loss.bestfit == calc_fitresult.bestfit
    assert calc_loss.loss == calc_fitresult.loss

    mean_poi = POI(mean, [1.15, 1.2, 1.25])
    mean_nll = calc_loss.obs_nll(pois=[mean_poi])
    calc_loss.obs_nll(pois=[mean_poi])  # get from cache

    assert mean_nll[0] >= mean_nll[1]
    assert mean_nll[2] >= mean_nll[1]

    assert calc_loss.obs_nll([mean_poi[0]]) == mean_nll[0]
    assert calc_loss.obs_nll([mean_poi[1]]) == mean_nll[1]
    assert calc_loss.obs_nll([mean_poi[2]]) == mean_nll[2]

    mean_poialt = POI(mean, 1.2)

    if calculator == BaseCalculator:
        with pytest.raises(NotImplementedError):
            calc_loss.pvalue(poinull=[mean_poi], poialt=[mean_poialt])
        with pytest.raises(NotImplementedError):
            calc_loss.expected_pvalue(poinull=[mean_poi], poialt=[mean_poialt], nsigma=np.arange(-2, 3, 1))
        with pytest.raises(NotImplementedError):
            calc_loss.expected_poi(poinull=[mean_poi], poialt=[mean_poialt], nsigma=np.arange(-2, 3, 1))
    else:
        calc_loss.pvalue(poinull=[mean_poi], poialt=[mean_poialt])
        calc_loss.expected_pvalue(poinull=[mean_poi], poialt=[mean_poialt], nsigma=np.arange(-2, 3, 1))
        calc_loss.expected_poi(poinull=[mean_poi], poialt=[mean_poialt], nsigma=np.arange(-2, 3, 1))

    model = calc_loss.model[0]
    sampler = model.create_sampler(n=10000)
    assert is_valid_data(sampler)

    loss = calc_loss.lossbuilder(model=[model], data=[sampler], weights=None)
    assert is_valid_loss(loss)

    with pytest.raises(ValueError):
        calc_loss.lossbuilder(model=[model, model], data=[sampler])
    with pytest.raises(ValueError):
        calc_loss.lossbuilder(model=[model], data=[sampler, calc_loss.data[0]])
    with pytest.raises(ValueError):
        calc_loss.lossbuilder(model=[model], data=[sampler], weights=[])
    with pytest.raises(ValueError):
        calc_loss.lossbuilder(model=[model], data=[sampler], weights=[np.ones(10000), np.ones(10000)])


def test_asymptotic_calculator_one_poi():
    with pytest.raises(TypeError):
        AsymptoticCalculator()

    loss, (mean, sigma) = create_loss()
    calc = AsymptoticCalculator(loss, MinuitMinimizer())

    poi_null = POI(mean, [1.15, 1.2, 1.25])
    poi_alt = POI(mean, 1.2)

    pnull, palt = calc.pvalue([poi_null], [poi_alt])

    dataset = calc.asimov_dataset(poi_alt)
    assert all(is_valid_data(d) for d in dataset)
    loss = calc.asimov_loss(poi_alt)
    assert is_valid_loss(loss)

    null_nll = calc.asimov_nll(pois=[poi_null], poialt=[poi_alt])

    assert null_nll[0] >= null_nll[1]
    assert null_nll[2] >= null_nll[1]


def test_asymptotic_calculator_two_pois():

    loss, (mean, sigma) = create_loss()
    calc = AsymptoticCalculator(loss, MinuitMinimizer())

    poi_null = [POI(mean, [1.15, 1.2, 1.25]), POI(sigma, [0.05, 0.1])]
    poi_alt = [POI(mean, 1.2), POI(sigma, 0.1)]

    with pytest.raises(NotImplementedError):
        calc.check_pois(poi_null)
    with pytest.raises(NotImplementedError):
        calc.pvalue(poi_null, poi_alt)
    with pytest.raises(NotImplementedError):
        calc.expected_pvalue(poinull=poi_null, poialt=poi_alt, nsigma=np.arange(-2, 3, 1))
    with pytest.raises(NotImplementedError):
        calc.expected_poi(poinull=poi_null, poialt=poi_alt, nsigma=np.arange(-2, 3, 1))
