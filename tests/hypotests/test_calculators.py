#!/usr/bin/python
import pytest

from statutils.hypotests.basecalculator import BaseCalculator
from statutils.hypotests.parameters import POI
import numpy as np

import zfit
from zfit.core.testing import setup_function #allows redefinition of zfit.Parameter
from zfit.core.loss import UnbinnedNLL

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


def test_base_calculator():
    with pytest.raises(TypeError):
        BaseCalculator()

    loss, (mean, sigma) = create_loss()

    calc_loss = BaseCalculator(loss)
    bestfit = calc_loss.bestfit
    calc_fitresult = BaseCalculator(bestfit)

    assert calc_loss.bestfit == calc_fitresult.bestfit
    assert calc_loss.loss == calc_fitresult.loss

    mean_poi = POI(mean, [1.15, 1.2, 1.25])
    mean_nll = calc_loss.obs_nll(pois=[mean_poi])
    calc_loss.obs_nll(pois=[mean_poi])  # get from cache

    assert mean_nll[0] >= mean_nll[1]
    assert mean_nll[2] >= mean_nll[1]

    assert calc_loss.obs_nll(mean_poi[0]) == mean_nll[0]
    assert calc_loss.obs_nll(mean_poi[1]) == mean_nll[1]
    assert calc_loss.obs_nll(mean_poi[2]) == mean_nll[2]

    mean_poialt = POI(mean, 1.2)

    with pytest.raises(NotImplementedError):
        calc_loss.pvalue(poinull=mean_poi, poialt=mean_poialt)
    with pytest.raises(NotImplementedError):
        calc_loss.expected_pvalue(poinull=mean_poi, poialt=mean_poialt, nsigma=np.arange(-2, 3, 1))
    with pytest.raises(NotImplementedError):
        calc_loss.expected_poi(poinull=mean_poi, poialt=mean_poialt, nsigma=np.arange(-2, 3, 1))
