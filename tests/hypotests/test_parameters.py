#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import zfit

from hepstats.hypotests.parameters import POI, POIarray


def test_pois():

    mean = zfit.Parameter("mu", 1.2, 0.1, 2)

    p0 = POI(mean, 0)
    p1 = POI(mean, 1.0)
    values = np.linspace(0.0, 1.0, 10)
    pn = POIarray(mean, values)
    pnc = POIarray(mean, values)

    for cls in [POI, POIarray]:
        with pytest.raises(ValueError):
            cls("mean", 0)
        with pytest.raises(TypeError):
            cls(mean)

    with pytest.raises(TypeError):
        POI(mean, values)
    with pytest.raises(TypeError):
        POIarray(mean, 0)

    repr(p0)
    repr(pn)

    assert p0.value == 0
    assert p0.name == mean.name
    assert p0 != p1

    assert all(pn.values == values)
    assert pn.name == mean.name
    assert len(pn) == len(values)
    iter(pn)
    assert pn == pnc
    assert hash(pn) == hash(pnc)

    assert pn != p0
    assert pn != p1

    assert pn[0] == p0
    assert pn[1] != p0
    assert pn[-1] == p1

    pn1 = pn.append(12)
    assert pn1.values[-1] == 12
    assert all(pn.values == values)
    assert pn1 != pn
    pn2 = pn.append([15, 20, 30])
    assert pn2.values[-1] == 30
    assert pn2.values[-2] == 20
    assert pn2.values[-3] == 15
    assert pn2 != pn

    {p0: "p0", p1: "p1", pn: "pn"}
