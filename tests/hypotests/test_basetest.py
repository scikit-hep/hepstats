# -*- coding: utf-8 -*-
import pytest
import numpy as np
import zfit
from zfit.loss import UnbinnedNLL
from zfit.minimize import Minuit

from hepstats.hypotests.calculators.basecalculator import BaseCalculator
from hepstats.hypotests.core.basetest import BaseTest
from hepstats.hypotests.parameters import POI, POIarray


def create_loss():

    obs = zfit.Space("x", limits=(0.1, 2.0))
    data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
    mean = zfit.Parameter("mu", 1.2)
    sigma = zfit.Parameter("sigma", 0.1)
    model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
    loss = UnbinnedNLL(model=model, data=data)

    return loss, (mean, sigma)


def test_constructor():
    with pytest.raises(TypeError):
        BaseTest()

    loss, (mean, sigma) = create_loss()
    calculator = BaseCalculator(loss, Minuit())

    poimean = POIarray(mean, [1.0, 1.1, 1.2, 1.3])
    poisigma = POI(sigma, 0.1)

    with pytest.raises(TypeError):
        BaseTest(calculator)

    with pytest.raises(TypeError):
        BaseTest(calculator, poimean, [poisigma])

    with pytest.raises(TypeError):
        BaseTest("calculator", poimean, poisigma)


def test_attributes():

    loss, (mean, sigma) = create_loss()
    calculator = BaseCalculator(loss, Minuit())

    poimean_1 = POIarray(mean, [1.0, 1.1, 1.2, 1.3])
    poimean_2 = POI(mean, 1.2)

    test = BaseTest(calculator, poimean_1, poimean_2)

    assert test.poinull == poimean_1
    assert test.poialt == poimean_2
    assert test.calculator == calculator
