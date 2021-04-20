#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--cmdopt",
        action="store",
        default="test",
        help="option: 'test' or \
    'generate'. Only use 'generate' if you've changed the tests and need to update the expected \
    output!",
    )


@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--cmdopt")


@pytest.fixture(scope="session")
def data_gen():
    np.random.seed(111)
    data1 = np.random.normal(size=1000)
    data2 = np.random.normal(2, 1, size=1000)
    weights = np.random.uniform(1, 2, size=1000)
    return data1, data2, weights


# TODO: manually ported, use pre-made: https://github.com/zfit/zfit-development/issues/73
@pytest.fixture(autouse=True)
def setup_teardown():
    import zfit

    old_chunksize = zfit.run.chunking.max_n_points
    old_active = zfit.run.chunking.active

    yield

    from zfit.core.parameter import ZfitParameterMixin

    ZfitParameterMixin._existing_params.clear()

    from zfit.util.cache import clear_graph_cache

    clear_graph_cache()
    zfit.run.chunking.active = old_active
    zfit.run.chunking.max_n_points = old_chunksize
    zfit.run.set_graph_mode()
    zfit.run.set_autograd_mode()
