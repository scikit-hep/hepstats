#!/usr/bin/python
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


def create_loss_func(npeak, nbins=None):
    import zfit

    bounds = (0.1, 3.0)
    obs = zfit.Space("x", limits=bounds)

    # Data and signal
    np.random.seed(0)
    tau = -2.0
    beta = -1 / tau
    bkg = np.random.exponential(beta, 300)
    peak = np.random.normal(1.2, 0.1, npeak)
    data = np.concatenate((bkg, peak))
    data = data[(data > bounds[0]) & (data < bounds[1])]
    N = len(data)
    data = zfit.data.Data.from_numpy(obs=obs, array=data)

    mean = zfit.Parameter("mean", 1.2, 0.5, 2.0)
    sigma = zfit.Parameter("sigma", 0.1, 0.02, 0.2)
    lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
    Nsig = zfit.Parameter("Nsig", 20.0, -20.0, N)
    Nbkg = zfit.Parameter("Nbkg", N, 0.0, N * 1.1)

    signal = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma).create_extended(Nsig)
    background = zfit.pdf.Exponential(obs=obs, lambda_=lambda_).create_extended(Nbkg)

    tot_model = zfit.pdf.SumPDF([signal, background])

    if nbins is not None:
        binned_space = obs.with_binning(nbins)
        data = data.to_binned(binned_space)
        tot_model = tot_model.to_binned(binned_space)
        loss = zfit.loss.ExtendedBinnedNLL(tot_model, data)
    else:
        loss = zfit.loss.ExtendedUnbinnedNLL(model=tot_model, data=data)

    return loss, (Nsig, Nbkg, mean, sigma)


@pytest.fixture()
def create_loss():
    return create_loss_func
