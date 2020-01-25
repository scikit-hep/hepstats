import pytest
import numpy as np
import zfit
import os
from zfit.core.testing import teardown_function # allows redefinition of zfit.Parameter, needed for tests

from hepstats.hypotests.parameters import POI, POIarray
from hepstats.hypotests.exceptions import ParameterNotFound
from hepstats.hypotests.toyutils import ToyResult, ToysManager

pwd = os.path.dirname(__file__)


def get_pois():
    Nsig = zfit.Parameter("Nsig", 0)
    poigen = POI(Nsig, 0.)
    poieval = POIarray(Nsig, [0.])
    return poigen, poieval


def test_constructors():
    poigen, poieval = get_pois()
    ToyResult(poigen, poieval)

    with pytest.raises(TypeError):
        ToyResult(poigen, "poieval")
    with pytest.raises(TypeError):
        ToyResult(poieval, poieval)

    ToysManager()


def test_toyresult_attributes():

    poigen, poieval = get_pois()
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

    tr2 = tr + tr.copy()
    assert tr2.ntoys == 6

    tr.to_dict()


def test_toymanager_attributes():

    poigen, poieval = get_pois()

    tm = ToysManager.from_yaml(f"{pwd}/discovery_freq_zfit_toys.yaml", [poigen.parameter])

    with pytest.raises(ParameterNotFound):
        ToysManager.from_yaml(f"{pwd}/discovery_freq_zfit_toys.yaml", [])

    tr = list(tm.values())[0]
    assert isinstance(tr, ToyResult)
    assert list(tm.keys())[0] == (poigen, poigen)
    assert (poigen, poieval) in tm
    tm.items()

    assert tm[poigen, poieval] == tr
    trc = tr.copy()
    tm[poigen, poieval.append(1)[-1]] = trc
    assert tm[poigen, poieval.append(1)[-1]] == trc

    tm.to_yaml(f"{pwd}/test_toyutils.yml")
    tmc = ToysManager.from_yaml(f"{pwd}/test_toyutils.yml", [poigen.parameter])
    assert tm[poigen, poieval].ntoys == tmc[poigen, poieval].ntoys
