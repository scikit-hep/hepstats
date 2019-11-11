import pytest
import numpy as np
import zfit
from zfit.core.testing import setup_function  # allows redefinition of zfit.Parameter, needed for tests
from zfit.loss import ExtendedUnbinnedNLL
from zfit.minimize import Minuit

from skstats.hypotests.calculators.basecalculator import BaseCalculator
from skstats.hypotests.calculators import AsymptoticCalculator
from skstats.hypotests import UpperLimit
from skstats.hypotests.parameters import POI
from skstats.hypotests.exceptions import POIRangeError


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
    Nsig = zfit.Parameter("Ns", 20., -20., N)
    Nbkg = zfit.Parameter("Nbkg", N, 0., N*1.1)

    signal = Nsig * zfit.pdf.Gauss(obs=obs, mu=1.2, sigma=0.1)
    background = Nbkg * zfit.pdf.Exponential(obs=obs, lambda_=lambda_)
    tot_model = signal + background

    loss = ExtendedUnbinnedNLL(model=[tot_model], data=[data])

    return loss, (Nsig, Nbkg)


def test_constructor():
    with pytest.raises(TypeError):
        UpperLimit()

    loss, (Nsig, Nbkg) = create_loss()
    calculator = BaseCalculator(loss, Minuit())

    poi_1 = POI(Nsig, [0.0])
    poi_2 = POI(Nsig, [2.0])

    with pytest.raises(TypeError):
        UpperLimit(calculator)

    with pytest.raises(TypeError):
        UpperLimit(calculator, poi_1)

    with pytest.raises(ValueError):
        UpperLimit(calculator, [poi_1], poi_2)

    with pytest.raises(ValueError):
        UpperLimit(calculator, poi_1, [poi_2])


def test_with_asymptotic_calculator():

    loss, (Nsig, Nbkg) = create_loss()
    calculator = AsymptoticCalculator(loss, Minuit())

    poinull = POI(Nsig, np.linspace(0.0, 25, 20))
    poialt = POI(Nsig, 0)

    ul = UpperLimit(calculator, [poinull], [poialt])
    ul_qtilde = UpperLimit(calculator, [poinull], [poialt], qtilde=True)
    limits = ul.upperlimit(alpha=0.05, CLs=True)

    # np.savez("cls_pvalues.npz", poivalues=poinull.value, **ul.pvalues(True))
    # np.savez("clsb_pvalues.npz", poivalues=poinull.value, **ul.pvalues(False))

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
        poinull = POI(Nsig, np.linspace(0.0, 12, 20))
        ul = UpperLimit(calculator, [poinull], [poialt])
        ul.upperlimit(alpha=0.05, CLs=True)
