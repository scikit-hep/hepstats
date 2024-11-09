import pytest
import numpy as np
zfit = pytest.importorskip("zfit")
import os
from zfit.loss import UnbinnedNLL
from zfit.minimize import Minuit

import hepstats
from hepstats.hypotests.calculators.basecalculator import BaseCalculator
from hepstats.hypotests.calculators import AsymptoticCalculator, FrequentistCalculator
from hepstats.hypotests import ConfidenceInterval
from hepstats.hypotests.parameters import POI, POIarray
from hepstats.hypotests.exceptions import POIRangeError

notebooks_dir = os.path.dirname(hepstats.__file__) + "/../../notebooks/hypotests"


def test_constructor(create_loss):
    with pytest.raises(TypeError):
        ConfidenceInterval()

    loss, (_, __, mean, _) = create_loss(npeak=80)
    calculator = BaseCalculator(loss, Minuit())

    poi_1 = POI(mean, 1.5)
    poi_2 = POI(mean, 1.2)

    with pytest.raises(TypeError):
        ConfidenceInterval(calculator)

    with pytest.raises(TypeError):
        ConfidenceInterval(calculator, [poi_1], poi_2, qtilde=True)

    with pytest.raises(TypeError):
        ConfidenceInterval(calculator, [poi_1], [poi_2], qtilde=False)


def asy_calc(create_loss, nbins=None):
    loss, (_, __, mean, ___) = create_loss(npeak=80, nbins=nbins)
    return mean, AsymptoticCalculator(loss, Minuit())


def asy_calc_old(create_loss, nbins=None):
    loss, (_, __, mean, ___) = create_loss(npeak=80, nbins=nbins)

    class calculator(AsymptoticCalculator):
        UNBINNED_TO_BINNED_LOSS = {}

    assert calculator is not AsymptoticCalculator, "Must not be the same"
    assert AsymptoticCalculator.UNBINNED_TO_BINNED_LOSS, "Has to be filled"
    return mean, calculator(loss, Minuit())


def freq_calc(create_loss, nbins=None):
    loss, (_, __, mean, ___) = create_loss(npeak=80, nbins=nbins)
    calculator = FrequentistCalculator.from_yaml(
        f"{notebooks_dir}/toys/ci_freq_zfit_toys.yml", loss, Minuit()
    )
    return mean, calculator


@pytest.mark.parametrize("calculator", [asy_calc, freq_calc, asy_calc_old])
@pytest.mark.parametrize("nbins", [None, 47, 300], ids=lambda x: f"nbins={x}")
def test_with_gauss_exp_example(create_loss, calculator, nbins):
    if calculator is asy_calc_old and nbins is not None:
        pytest.skip("Not implemented for old calculator")
    mean, calculator = calculator(create_loss, nbins=nbins)
    scan_values = np.linspace(1.15, 1.26, 50)
    poinull = POIarray(mean, scan_values)
    ci = ConfidenceInterval(calculator, poinull)
    interval = ci.interval()
    assert interval["lower"] == pytest.approx(1.1810371356602791, rel=0.1)
    assert interval["upper"] == pytest.approx(1.2156701172321935, rel=0.1)
    with pytest.raises(POIRangeError):
        poinull = POIarray(
            mean, scan_values[(scan_values >= 1.2) & (scan_values <= 1.205)]
        )

        ci = ConfidenceInterval(calculator, poinull)
        ci.interval()
    with pytest.raises(POIRangeError):
        poinull = POIarray(mean, scan_values[scan_values >= 1.2])
        ci = ConfidenceInterval(calculator, poinull)
        ci.interval()
    with pytest.raises(POIRangeError):
        poinull = POIarray(mean, scan_values[scan_values <= 1.205])
        ci = ConfidenceInterval(calculator, poinull)
        ci.interval()


def test_with_gauss_fluctuations():
    x_true = -2.0

    minimizer = Minuit()
    bounds = (-10, 10)
    obs = zfit.Space("x", limits=bounds)

    mean = zfit.Parameter("mean", 0)
    sigma = zfit.Parameter("sigma", 1.0)
    model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)

    npzfile = f"{notebooks_dir}/toys/FC_toys_{x_true}.npz"
    data = zfit.data.Data.from_numpy(obs=obs, array=np.load(npzfile)["x"])

    nll = UnbinnedNLL(model=model, data=data)

    minimum = minimizer.minimize(loss=nll)
    minimum.hesse()

    toys_fname = f"{notebooks_dir}/toys/FC_toys_{x_true}.yml"
    calculator = FrequentistCalculator.from_yaml(toys_fname, minimum, minimizer)
    keys = np.unique([k[0].value for k in calculator.keys()])
    keys.sort()
    poinull = POIarray(mean, keys)

    ci = ConfidenceInterval(calculator, poinull, qtilde=False)
    with pytest.warns(UserWarning):
        ci.interval(alpha=0.05, printlevel=0)

    ci = ConfidenceInterval(calculator, poinull, qtilde=True)
    ci.interval(alpha=0.05, printlevel=0)


@pytest.mark.parametrize("n", [0.5])
@pytest.mark.parametrize("min_x", [0, -10])
def test_with_gauss_qtilde(n, min_x):
    sigma_x = 0.032

    minimizer = Minuit()
    bounds = (-10, 10)
    obs = zfit.Space("x", limits=bounds)

    mean = zfit.Parameter("mean", n * sigma_x)
    sigma = zfit.Parameter("sigma", 1.0)
    model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)

    data = model.sample(n=1000)

    nll = UnbinnedNLL(model=model, data=data)

    minimum = minimizer.minimize(loss=nll)
    minimum.hesse()

    x = minimum.params[mean]["value"]
    x_err = minimum.params[mean]["hesse"]["error"]

    x_min = x - x_err * 3
    x_max = x + x_err * 3

    x_min = max([x_min, min_x])

    poinull = POIarray(mean, np.linspace(x_min, x_max, 50))
    calculator = AsymptoticCalculator(nll, minimizer)

    ci = ConfidenceInterval(calculator, poinull, qtilde=True)
    ci.interval(alpha=0.05, printlevel=1)
