from __future__ import annotations

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
def _setup_teardown():
    try:
        import zfit
    except ImportError:
        yield
        return

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


def create_loss_func(npeak, nbins=None, nbkg=None, nameadd="", obs=None):
    import zfit

    bounds = (0.1, 3.0)
    obs = "x" if obs is None else obs
    obs = zfit.Space(obs, limits=bounds)

    # Data and signal
    np.random.seed(0)
    tau = -2.0
    beta = -1 / tau
    nbkg = 300 if nbkg is None else nbkg
    bkg = np.random.exponential(beta, nbkg)
    peak = np.random.normal(1.2, 0.1, npeak)
    data = np.concatenate((bkg, peak))
    data = data[(data > bounds[0]) & (data < bounds[1])]
    N = len(data)
    data = zfit.data.Data.from_numpy(obs=obs, array=data)

    mean = zfit.Parameter("mean" + nameadd, 1.2, 0.5, 2.0)
    sigma = zfit.Parameter("sigma" + nameadd, 0.1, 0.02, 0.2)
    lambda_ = zfit.Parameter("lambda" + nameadd, -2.0, -4.0, -1.0)
    Nsig = zfit.Parameter("Nsig" + nameadd, 20.0, -20.0, N * 3)
    Nbkg = zfit.Parameter("Nbkg" + nameadd, N, 0.0, N * 3)

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


def create_sim_loss_func(npeak, nbins=None):
    loss1, params1 = create_loss_func(npeak, nbins=nbins, nameadd="_1", obs="x1")
    loss2, params2 = create_loss_func(npeak * 10, nbins=nbins, nameadd="_2", obs="x2", nbkg=500)
    loss = loss1 + loss2

    return loss, params1


@pytest.fixture
def create_loss():
    return create_loss_func


@pytest.fixture
def create_sim_loss():
    return create_sim_loss_func
