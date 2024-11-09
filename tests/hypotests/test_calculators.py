#!/usr/bin/python

import numpy as np
import pytest
zfit = pytest.importorskip("zfit")
from zfit.loss import UnbinnedNLL
from zfit.minimize import Minuit

from hepstats.hypotests.calculators import AsymptoticCalculator, FrequentistCalculator
from hepstats.hypotests.calculators.basecalculator import BaseCalculator
from hepstats.hypotests.parameters import POI, POIarray
from hepstats.utils.fit.api_check import is_valid_loss, is_valid_data

true_mu = 1.2
true_sigma = 0.1


def create_loss(constraint=False, nbins=None, make2d=False):
    if not isinstance(nbins, list):
        nbins = [nbins] * 2 if make2d else [nbins]
    obs1 = zfit.Space("x", limits=(0.1, 2.0), binning=nbins[0])
    obs = obs1
    if make2d:
        obs2 = zfit.Space("y", limits=(-0.1, 3.0), binning=nbins[1])
        obs = obs1 * obs2

    array1 = np.random.normal(1.2, 0.1, (10000, 2 if make2d else 1))
    data = zfit.data.Data.from_numpy(obs=obs.with_binning(None), array=array1)
    if nbins[0] is not None:
        data = data.to_binned(obs)
    mean = zfit.Parameter("mu", true_mu, true_mu - 2, true_mu + 2)
    sigma = zfit.Parameter("sigma", true_sigma, 0.01, 1.0)
    model = zfit.pdf.Gauss(obs=obs1.with_binning(None), mu=mean, sigma=sigma)
    if make2d:
        model2 = zfit.pdf.Gauss(obs=obs2.with_binning(None), mu=mean, sigma=sigma)
        model = model * model2
    if nbins[0] is not None:
        model = zfit.pdf.BinnedFromUnbinnedPDF(model, space=obs)
    if constraint:
        constraint = zfit.constraint.GaussianConstraint(
            params=mean, observation=true_mu, uncertainty=0.01
        )
    else:
        constraint = None
    if nbins[0] is None:
        loss = UnbinnedNLL(model=model, data=data, constraints=constraint)
    else:
        loss = zfit.loss.BinnedNLL(model=model, data=data, constraints=constraint)

    return loss, (mean, sigma)


@pytest.mark.parametrize(
    "calculator",
    [BaseCalculator, AsymptoticCalculator, FrequentistCalculator, "AsymptoticOld"],
)
@pytest.mark.parametrize("make2d", [False, True], ids=["1d", "2d"])
@pytest.mark.parametrize(
    "nbins",
    [None, [10, 13], [9, 50]],
    ids=lambda x: f"Binning {x}" if x is not None else "Unbinned",
)
@pytest.mark.parametrize(
    "constraint", [False, True], ids=["No constraint", "With constraint"]
)
def test_base_calculator(calculator, make2d, nbins, constraint):
    if calculator == "AsymptoticOld":
        if make2d:
            pytest.skip("AsymptoticOld does not support 2D")
        if nbins is not None:
            pytest.skip("AsymptoticOld does not support binned")

        class calculator(AsymptoticCalculator):  # we disable the converter
            UNBINNED_TO_BINNED_LOSS = {}

        assert calculator is not AsymptoticCalculator, "Must not be the same"
        assert AsymptoticCalculator.UNBINNED_TO_BINNED_LOSS, "Has to be filled"
    with pytest.raises(TypeError):
        calculator()

    loss, (mean, sigma) = create_loss(constraint=constraint, make2d=make2d, nbins=nbins)

    with pytest.raises(ValueError):
        calculator("loss", Minuit())

    with pytest.raises(ValueError):
        calculator(loss, "Minuit()")

    calc_loss = calculator(loss, Minuit())

    with pytest.raises(ValueError):
        calc_loss.bestfit = "bestfit"

    bestfit = calc_loss.bestfit
    calc_fitresult = calculator(bestfit, calc_loss.minimizer)

    assert calc_loss.bestfit == calc_fitresult.bestfit
    assert calc_loss.loss == calc_fitresult.loss

    mean_poi = POIarray(mean, [1.15, 1.2, 1.25])
    mean_nll = calc_loss.obs_nll(pois=mean_poi)
    calc_loss.obs_nll(pois=mean_poi)  # get from cache

    assert mean_nll[0] >= mean_nll[1]
    assert mean_nll[2] >= mean_nll[1]

    assert calc_loss.obs_nll(mean_poi[0]) == mean_nll[0]
    assert calc_loss.obs_nll(mean_poi[1]) == mean_nll[1]
    assert calc_loss.obs_nll(mean_poi[2]) == mean_nll[2]

    mean_poialt = POI(mean, 1.2)

    def pvalue():
        return calc_loss.pvalue(poinull=mean_poi, poialt=mean_poialt)

    def exp_pvalue():
        return calc_loss.expected_pvalue(
            poinull=mean_poi, poialt=mean_poialt, nsigma=np.arange(-2, 3, 1)
        )

    def exp_poi():
        return calc_loss.expected_poi(
            poinull=mean_poi, poialt=mean_poialt, nsigma=np.arange(-2, 3, 1)
        )

    if calculator == BaseCalculator:
        with pytest.raises(NotImplementedError):
            pvalue()
        with pytest.raises(NotImplementedError):
            exp_pvalue()
    else:
        pvalue()
        exp_pvalue()

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
        calc_loss.lossbuilder(
            model=[model], data=[sampler], weights=[np.ones(10000), np.ones(10000)]
        )

    assert calc_loss.get_parameter(mean_poi.name) == mean
    with pytest.raises(KeyError):
        calc_loss.get_parameter("dummy_parameter")


def test_asymptotic_calculator_one_poi():
    with pytest.raises(TypeError):
        AsymptoticCalculator()

    loss, (mean, sigma) = create_loss()
    calc = AsymptoticCalculator(loss, Minuit())

    poi_null = POIarray(mean, [1.15, 1.2, 1.25])
    poi_alt = POI(mean, 1.2)

    dataset = calc.asimov_dataset(poi_alt)
    assert all(is_valid_data(d) for d in dataset)
    loss = calc.asimov_loss(poi_alt)
    assert is_valid_loss(loss)

    null_nll = calc.asimov_nll(pois=poi_null, poialt=poi_alt)

    assert null_nll[0] >= null_nll[1]
    assert null_nll[2] >= null_nll[1]


@pytest.mark.parametrize("constraint", [False, True])
def test_frequentist_calculator_one_poi(constraint):
    with pytest.raises(TypeError):
        FrequentistCalculator()

    loss, (mean, sigma) = create_loss(constraint=constraint)
    calc = FrequentistCalculator(loss, Minuit(), ntoysnull=100, ntoysalt=100)

    assert calc.ntoysnull == 100
    assert calc.ntoysalt == 100

    samplers = calc.sampler()
    assert all(is_valid_data(s) for s in samplers)
    loss = calc.toys_loss(mean.name)
    assert is_valid_loss(loss)
