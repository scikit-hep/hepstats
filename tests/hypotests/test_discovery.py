import os

import numpy as np
import pytest

from tests.conftest import create_loss_func, create_sim_loss_func

zfit = pytest.importorskip("zfit")
from zfit.loss import UnbinnedNLL
from zfit.minimize import Minuit

import hepstats
from hepstats.hypotests import Discovery
from hepstats.hypotests.calculators import AsymptoticCalculator, FrequentistCalculator
from hepstats.hypotests.calculators.basecalculator import BaseCalculator
from hepstats.hypotests.parameters import POI

notebooks_dir = f"{os.path.dirname(hepstats.__file__)}/../../notebooks/hypotests"


@pytest.mark.parametrize("nbins", [None, 30], ids=["unbinned", "binned"])
def test_constructor(create_loss, nbins):
    with pytest.raises(TypeError):
        Discovery()

    loss, (Nsig, Nbkg, _, _) = create_loss(nbins=nbins, npeak=25)
    calculator = BaseCalculator(loss, Minuit())

    poi_1 = POI(Nsig, 0.0)
    poi_2 = POI(Nsig, 2.0)

    with pytest.raises(TypeError):
        Discovery(calculator)

    with pytest.raises(TypeError):
        Discovery(calculator, [poi_1], poi_2)

    with pytest.raises(TypeError):
        Discovery(calculator, [poi_1], [poi_2])


class AsymptoticCalculatorOld(AsymptoticCalculator):
    UNBINNED_TO_BINNED_LOSS = {}


@pytest.mark.parametrize(
    "nbins", [None, 76, 253], ids=lambda x: "unbinned" if x is None else f"nbin={x}"
)
@pytest.mark.parametrize("Calculator", [AsymptoticCalculator, AsymptoticCalculatorOld])
def test_with_asymptotic_calculator(create_loss, nbins, Calculator):
    if Calculator is AsymptoticCalculatorOld and nbins is not None:
        pytest.skip("Old AsymptoticCalculator does not support binned loss")

    loss, (Nsig, Nbkg, mean, sigma) = create_loss(npeak=25, nbins=nbins)
    mean.floating = False
    sigma.floating = False
    calculator = Calculator(loss, Minuit())

    poinull = POI(Nsig, 0)

    discovery_test = Discovery(calculator, poinull)
    pnull, significance = discovery_test.result()

    uncertainty = 0.05
    if nbins is not None and nbins < 80:
        uncertainty *= 4

    # check absolute significance
    assert pnull == pytest.approx(0.000757, abs=uncertainty)
    assert significance == pytest.approx(3.17, abs=uncertainty)
    assert significance >= 3


@pytest.mark.parametrize(
    "nbins", [None, 95, 153], ids=lambda x: "unbinned" if x is None else f"nbin={x}"
)
@pytest.mark.parametrize("losscreator", [create_loss_func,
                                         # create_sim_loss_func
                                         ], ids=["simple",
                                                 # "sim"
                                                 ])
def test_with_frequentist_calculator(losscreator, nbins):
    loss, (Nsig, Nbkg, mean, sigma) = losscreator(npeak=25, nbins=nbins)
    mean.floating = False
    sigma.floating = False
    calculator = FrequentistCalculator.from_yaml(
        f"{notebooks_dir}/toys/discovery_freq_zfit_toys.yml", loss, Minuit()
    )
    # calculator = FrequentistCalculator(loss, Minuit(), ntoysnull=500, ntoysalt=500)

    poinull = POI(Nsig, 0)

    discovery_test = Discovery(calculator, poinull)
    pnull, significance = discovery_test.result()

    abserr = 0.1
    if nbins is not None and nbins < 120:
        abserr *= 4
    abserr_pnull = 0.0005
    if nbins is not None and nbins < 120:
        abserr_pnull *= 4
    assert pnull == pytest.approx(0.0004, rel=0.05, abs=abserr_pnull)
    assert significance == pytest.approx(3.3427947805048592, rel=0.05, abs=abserr)
    assert significance >= 3


def create_loss_counting():
    n = 370
    nbkg = 340

    Nsig = zfit.Parameter("Nsig", 0, -100.0, 100)
    Nbkg = zfit.Parameter("Nbkg", nbkg, floating=False)
    Nobs = zfit.ComposedParameter("Nobs", lambda a, b: a + b, params=[Nsig, Nbkg])

    obs = zfit.Space("N", limits=(0, 800))
    model = zfit.pdf.Poisson(obs=obs, lamb=Nobs)

    data = zfit.data.Data.from_numpy(obs=obs, array=np.array([n]))

    loss = UnbinnedNLL(model=model, data=data)

    return loss, Nsig


def test_counting_with_asymptotic_calculator():
    (
        loss,
        Nsig,
    ) = create_loss_counting()
    calculator = AsymptoticCalculator(loss, Minuit())

    poinull = POI(Nsig, 0)

    discovery_test = Discovery(calculator, poinull)
    pnull, significance = discovery_test.result()

    assert significance < 2


def test_counting_with_frequentist_calculator():
    (
        loss,
        Nsig,
    ) = create_loss_counting()
    calculator = FrequentistCalculator(loss, Minuit(), ntoysnull=1000)

    poinull = POI(Nsig, 0)

    discovery_test = Discovery(calculator, poinull)
    pnull, significance = discovery_test.result()

    assert significance < 2


def test_likelihood_ratio_fmin():
    import numpy as np
    import zfit
    from zfit.loss import UnbinnedNLL
    from zfit.minimize import Minuit
    from hepstats.hypotests import Discovery, UpperLimit
    from hepstats.hypotests.calculators import (AsymptoticCalculator,
                                                FrequentistCalculator)
    from hepstats.hypotests.parameters import POI, POIarray

    Nsig = zfit.Parameter("Nsig", 40, -100., 100)
    Nbkg = zfit.Parameter("Nbkg", 340, 0, 500)
    Nobs = zfit.ComposedParameter("Nobs", lambda a, b: a + b, params=[Nsig, Nbkg])

    from collections import OrderedDict
    import tensorflow_probability as tfp
    from zfit.models.dist_tfp import WrapDistribution
    from zfit.util import ztyping


    obs = zfit.Space('N', limits=(0, 800))
    model = zfit.pdf.Poisson(obs=obs, lamb=Nobs)

    n = 370
    nbkg = 340

    data = zfit.data.Data.from_numpy(obs=obs, array=np.array([n]))
    Nbkg.set_value(nbkg)
    Nbkg.floating = False

    nll = UnbinnedNLL(model=model, data=data)
    minimizer = Minuit(verbosity=0)
    minimum = minimizer.minimize(loss=nll)

    calculator = AsymptoticCalculator(nll, minimizer)
    calculator.bestfit = minimum

    discovery_test = Discovery(calculator, POI(Nsig, 0))
    pnull, significance = discovery_test.result()
    assert pytest.approx(pnull, abs=0.01) == 0.05
    assert pytest.approx(significance, abs=0.1) == 1.6
