# -*- coding: utf-8 -*-
import pytest
import numpy as np
import zfit
import os
from zfit.loss import ExtendedUnbinnedNLL, UnbinnedNLL
from zfit.minimize import Minuit

import hepstats
from hepstats.hypotests.parameters import POI, POIarray
from hepstats.hypotests.exceptions import ParameterNotFound
from hepstats.hypotests.toyutils import ToyResult, ToysManager
from hepstats.utils.fit.api_check import is_valid_loss, is_valid_data

pwd = os.path.dirname(__file__)
notebooks_dir = os.path.dirname(hepstats.__file__) + "/../../notebooks/hypotests"


def create_loss():
    bounds = (0.1, 3.0)
    obs = zfit.Space("x", limits=bounds)

    # Data and signal
    np.random.seed(0)
    tau = -2.0
    beta = -1 / tau
    bkg = np.random.exponential(beta, 300)
    peak = np.random.normal(1.2, 0.1, 25)
    data = np.concatenate((bkg, peak))
    data = data[(data > bounds[0]) & (data < bounds[1])]
    N = len(data)
    data = zfit.data.Data.from_numpy(obs=obs, array=data)

    lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
    Nsig = zfit.Parameter("Nsig", 20.0, -20.0, N)
    Nbkg = zfit.Parameter("Nbkg", N, 0.0, N * 1.1)

    signal = zfit.pdf.Gauss(obs=obs, mu=1.2, sigma=0.1).create_extended(Nsig)
    background = zfit.pdf.Exponential(obs=obs, lambda_=lambda_).create_extended(Nbkg)
    tot_model = zfit.pdf.SumPDF([signal, background])

    loss = ExtendedUnbinnedNLL(model=tot_model, data=data)

    poigen = POI(Nsig, 0.0)
    poieval = POIarray(Nsig, [0.0])

    return loss, (Nsig, poigen, poieval)


def create_loss_1():
    obs = zfit.Space("x", limits=(0.1, 2.0))
    data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
    mean = zfit.Parameter("mu", 1.2)
    sigma = zfit.Parameter("sigma", 0.1)
    model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
    loss = UnbinnedNLL(model=model, data=data)

    return loss


def test_constructors():
    loss, (Nsig, poigen, poieval) = create_loss()
    ToyResult(poigen, poieval)

    with pytest.raises(TypeError):
        ToyResult(poigen, "poieval")
    with pytest.raises(TypeError):
        ToyResult(poieval, poieval)

    ToysManager(loss, Minuit())


def test_toyresult_attributes():

    _, (_, poigen, poieval) = create_loss()
    tr = ToyResult(poigen, poieval)

    assert tr.ntoys == 0
    assert tr.poigen == poigen
    assert tr.poieval == poieval

    bf = np.array([0.5, 0.1, 0.2])
    nll_bf = np.array([-1000, -1001, -1002])
    nlls = {poieval[0]: np.array([-1001, -1002, -1003])}

    tr.add_entries(bestfit=bf, nll_bestfit=nll_bf, nlls=nlls)
    assert tr.ntoys == 3

    with pytest.raises(ValueError):
        tr.add_entries(bestfit=bf, nll_bestfit=nll_bf, nlls={})

    tr.add_entries(bestfit=bf, nll_bestfit=nll_bf, nlls=nlls)
    assert tr.ntoys == 6

    tr.to_dict()


def test_toymanager_attributes():

    loss, (Nsig, poigen, poieval) = create_loss()

    tm = ToysManager.from_yaml(
        f"{notebooks_dir}/toys/discovery_freq_zfit_toys.yml", loss, Minuit()
    )

    with pytest.raises(ParameterNotFound):
        ToysManager.from_yaml(
            f"{notebooks_dir}/toys/discovery_freq_zfit_toys.yml",
            create_loss_1(),
            Minuit(),
        )

    tr = list(tm.values())[0]
    assert isinstance(tr, ToyResult)
    assert list(tm.keys())[0] == (poigen, poigen)
    assert (poigen, poieval) in tm.keys()

    assert tm.get_toyresult(poigen, poieval) == tr
    tr1 = ToyResult(poigen, poieval.append(1))
    tm.add_toyresult(tr1)
    with pytest.raises(TypeError):
        tm.add_toyresult("tr1")
    assert (tr1.poigen, tr1.poieval) in tm.keys()

    tm.to_yaml(f"{pwd}/test_toyutils.yml")
    tm.to_yaml(f"{pwd}/test_toyutils.yml")
    tmc = ToysManager.from_yaml(f"{pwd}/test_toyutils.yml", loss, Minuit())
    assert (
        tm.get_toyresult(poigen, poieval).ntoys
        == tmc.get_toyresult(poigen, poieval).ntoys
    )

    samplers = tm.sampler(floating_params=[poigen.parameter])
    assert all(is_valid_data(s) for s in samplers)
    loss = tm.toys_loss(poigen.name)
    assert is_valid_loss(loss)

    os.remove(f"{pwd}/test_toyutils.yml")
