import pytest
import numpy as np
import zfit
import os
from zfit.loss import ExtendedUnbinnedNLL
from zfit.minimize import Minuit

from hepstats.hypotests.calculators.basecalculator import BaseCalculator
from hepstats.hypotests.calculators import AsymptoticCalculator, FrequentistCalculator
from hepstats.hypotests import UpperLimit
from hepstats.hypotests.parameters import POI, POIarray
from hepstats.hypotests.exceptions import POIRangeError

pwd = os.path.dirname(__file__)


def create_loss():

    bounds = (0.1, 3.0)
    obs = zfit.Space('x', limits=bounds)

    # Data and signal
    np.random.seed(0)
    tau = -2.0
    beta = -1/tau
    bkg = np.random.exponential(beta, 300)
    peak = np.random.normal(1.2, 0.1, 10)
    data = np.concatenate((bkg, peak))
    data = data[(data > bounds[0]) & (data < bounds[1])]
    N = len(data)
    data = zfit.data.Data.from_numpy(obs=obs, array=data)

    lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
    Nsig = zfit.Parameter("Nsig", 20., -20., N)
    Nbkg = zfit.Parameter("Nbkg", N, 0., N*1.1)

    signal = zfit.pdf.Gauss(obs=obs, mu=1.2, sigma=0.1).create_extended(Nsig)
    background = zfit.pdf.Exponential(obs=obs, lambda_=lambda_).create_extended(Nbkg)
    tot_model = zfit.pdf.SumPDF([signal, background])

    loss = ExtendedUnbinnedNLL(model=tot_model, data=data)

    return loss, (Nsig, Nbkg)


def test_constructor():
    with pytest.raises(TypeError):
        UpperLimit()

    loss, (Nsig, Nbkg) = create_loss()
    calculator = BaseCalculator(loss, Minuit())

    poi_1 = POI(Nsig, 0.0)
    poi_2 = POI(Nsig, 2.0)

    with pytest.raises(TypeError):
        UpperLimit(calculator)

    with pytest.raises(TypeError):
        UpperLimit(calculator, poi_1)

    with pytest.raises(TypeError):
        UpperLimit(calculator, [poi_1], poi_2)


def asy_calc():
    loss, (Nsig, Nbkg) = create_loss()
    return Nsig, AsymptoticCalculator(loss, Minuit())


def freq_calc():
    loss, (Nsig, Nbkg) = create_loss()
    calculator = FrequentistCalculator.from_yaml(f"{pwd}/upperlimit_freq_zfit_toys.yml", loss, Minuit())
    return Nsig, calculator


@pytest.mark.parametrize("calculator", [asy_calc, freq_calc])
def test_with_gauss_exp_example(calculator):

    Nsig, calculator = calculator()

    poinull = POIarray(Nsig, np.linspace(0.0, 25, 15))
    poialt = POI(Nsig, 0)

    ul = UpperLimit(calculator, poinull, poialt)
    ul_qtilde = UpperLimit(calculator, poinull, poialt, qtilde=True)
    limits = ul.upperlimit(alpha=0.05, CLs=True)

    assert limits["observed"] == pytest.approx(15.725784747406346, rel=0.1)
    assert limits["expected"] == pytest.approx(11.927442041887158, rel=0.1)
    assert limits["expected_p1"] == pytest.approx(16.596396280677116, rel=0.1)
    assert limits["expected_p2"] == pytest.approx(22.24864429383046, rel=0.1)
    assert limits["expected_m1"] == pytest.approx(8.592750403611896, rel=0.1)
    assert limits["expected_m2"] == pytest.approx(6.400549971360598, rel=0.1)

    ul.upperlimit(alpha=0.05, CLs=False)
    ul_qtilde.upperlimit(alpha=0.05, CLs=True)

    # test error when scan range is too small

    with pytest.raises(POIRangeError):
        poinull = POIarray(Nsig, poinull.values[:5])
        ul = UpperLimit(calculator, poinull, poialt)
        ul.upperlimit(alpha=0.05, CLs=True)
