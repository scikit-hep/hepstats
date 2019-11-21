import pytest
import numpy as np
import zfit
from zfit.core.testing import setup_function  # allows redefinition of zfit.Parameter, needed for tests
from zfit.loss import ExtendedUnbinnedNLL
from zfit.minimize import Minuit

from hepstats.hypotests.calculators.basecalculator import BaseCalculator
from hepstats.hypotests.calculators import AsymptoticCalculator
from hepstats.hypotests import ConfidenceInterval
from hepstats.hypotests.parameters import POI, POIarray
from hepstats.hypotests.exceptions import POIRangeError


def create_loss():

    bounds = (0.1, 3.0)
    obs = zfit.Space('x', limits=bounds)

    # Data and signal
    np.random.seed(0)
    tau = -2.0
    beta = -1/tau
    bkg = np.random.exponential(beta, 300)
    peak = np.random.normal(1.2, 0.1, 80)
    data = np.concatenate((bkg, peak))
    data = data[(data > bounds[0]) & (data < bounds[1])]
    N = len(data)
    data = zfit.data.Data.from_numpy(obs=obs, array=data)

    mean = zfit.Parameter("mean", 1.2, 0.5, 2.0)
    sigma = zfit.Parameter("sigma", 0.1, 0.02, 0.2)
    lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
    Nsig = zfit.Parameter("Ns", 20., -20., N)
    Nbkg = zfit.Parameter("Nbkg", N, 0., N*1.1)

    signal = Nsig * zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
    background = Nbkg * zfit.pdf.Exponential(obs=obs, lambda_=lambda_)
    tot_model = signal + background

    loss = ExtendedUnbinnedNLL(model=[tot_model], data=[data])

    return loss, mean


def test_constructor():
    with pytest.raises(TypeError):
        ConfidenceInterval()

    loss, mean = create_loss()
    calculator = BaseCalculator(loss, Minuit())

    poi_1 = POI(mean, 1.5)
    poi_2 = POI(mean, 1.2)

    with pytest.raises(TypeError):
        ConfidenceInterval(calculator)

    with pytest.raises(TypeError):
        ConfidenceInterval(calculator, [poi_1], poi_2, qtilde=True)

    with pytest.raises(TypeError):
        ConfidenceInterval(calculator, [poi_1], [poi_2], qtilde=False)


def test_with_asymptotic_calculator():

    loss, mean = create_loss()
    calculator = AsymptoticCalculator(loss, Minuit())

    poinull = POIarray(mean, np.linspace(1.15, 1.26, 100))
    ci = ConfidenceInterval(calculator, poinull)
    interval = ci.interval()

    assert interval["lower"] == pytest.approx(1.1810371356602791, rel=0.1)
    assert interval["upper"] == pytest.approx(1.2156701172321935, rel=0.1)

    with pytest.raises(POIRangeError):
        poinull = POIarray(mean, np.linspace(1.2, 1.205, 50))
        ci = ConfidenceInterval(calculator, poinull)
        ci.interval()

    with pytest.raises(POIRangeError):
        poinull = POIarray(mean, np.linspace(1.2, 1.26, 50))
        ci = ConfidenceInterval(calculator, poinull)
        ci.interval()

    with pytest.raises(POIRangeError):
        poinull = POIarray(mean, np.linspace(1.17, 1.205, 50))
        ci = ConfidenceInterval(calculator, poinull)
        ci.interval()
