import os
import pytest
import numpy as np
import zfit
from zfit.core.testing import setup_function  # allows redefinition of zfit.Parameter, needed for tests
from zfit.core.loss import ExtendedUnbinnedNLL
from zfit.minimize import Minuit

from hepstats.hypotests.calculators.basecalculator import BaseCalculator
from hepstats.hypotests.calculators import AsymptoticCalculator, FrequentistCalculator
from hepstats.hypotests import Discovery
from hepstats.hypotests.parameters import POI

pwd = os.path.dirname(__file__)


def create_loss():

    bounds = (0.1, 3.0)
    obs = zfit.Space('x', limits=bounds)

    # Data and signal
    np.random.seed(0)
    tau = -2.0
    beta = -1/tau
    bkg = np.random.exponential(beta, 300)
    peak = np.random.normal(1.2, 0.1, 25)
    data = np.concatenate((bkg, peak))
    data = data[(data > bounds[0]) & (data < bounds[1])]
    N = len(data)
    data = zfit.data.Data.from_numpy(obs=obs, array=data)

    lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
    Nsig = zfit.Parameter("Nsig", 20., -20., N)
    Nbkg = zfit.Parameter("Nbkg", N, 0., N*1.1)

    signal = Nsig * zfit.pdf.Gauss(obs=obs, mu=1.2, sigma=0.1)
    background = Nbkg * zfit.pdf.Exponential(obs=obs, lambda_=lambda_)
    tot_model = signal + background

    loss = ExtendedUnbinnedNLL(model=[tot_model], data=[data], fit_range=[obs])

    return loss, (Nsig, Nbkg)


def test_constructor():
    with pytest.raises(TypeError):
        Discovery()

    loss, (Nsig, Nbkg) = create_loss()
    calculator = BaseCalculator(loss, Minuit())

    poi_1 = POI(Nsig, 0.0)
    poi_2 = POI(Nsig, 2.0)

    with pytest.raises(TypeError):
        Discovery(calculator)

    with pytest.raises(TypeError):
        Discovery(calculator, [poi_1], poi_2)

    with pytest.raises(TypeError):
        Discovery(calculator, [poi_1], [poi_2])


def test_with_asymptotic_calculator():

    loss, (Nsig, Nbkg) = create_loss()
    calculator = AsymptoticCalculator(loss, Minuit())

    poinull = POI(Nsig, 0)

    discovery_test = Discovery(calculator, poinull)
    pnull, significance = discovery_test.result()

    assert pnull == pytest.approx(0.0007571045089567185, abs=0.05)
    assert significance == pytest.approx(3.1719464953752565, abs=0.05)
    assert significance >= 3


def test_with_frequentist_calculator():

    loss, (Nsig, Nbkg) = create_loss()
    calculator = FrequentistCalculator(loss, Minuit())
    calculator.load_toys_from_yaml(f"{pwd}/discovery_freq_zfit_toys.yaml")

    poinull = POI(Nsig, 0)

    discovery_test = Discovery(calculator, poinull)
    pnull, significance = discovery_test.result()

    assert pnull == pytest.approx(0.0006, abs=0.05)
    assert significance == pytest.approx(3.2388801183529563, abs=0.05)
    assert significance >= 3
