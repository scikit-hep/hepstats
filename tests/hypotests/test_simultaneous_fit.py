"""
Test simultaneous fits with different parameter sharing scenarios.
"""

from __future__ import annotations

import os

os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import pytest

zfit = pytest.importorskip("zfit")
from zfit.loss import UnbinnedNLL  # noqa: E402
from zfit.minimize import Minuit  # noqa: E402

from hepstats.hypotests.calculators import FrequentistCalculator  # noqa: E402
from hepstats.hypotests.parameters import POI, POIarray  # noqa: E402


def create_no_shared_params_loss():
    """Create simultaneous loss where PDFs share NO parameters (2 losses)."""
    obs1 = zfit.Space("x", limits=(0.0, 3.0))
    mu1 = zfit.Parameter("mu1", 1.0, 0.0, 3.0)
    sigma1 = zfit.Parameter("sigma1", 0.3)
    sigma1.floating = False
    model1 = zfit.pdf.Gauss(obs=obs1, mu=mu1, sigma=sigma1)

    obs2 = zfit.Space("y", limits=(0.0, 3.0))
    mu2 = zfit.Parameter("mu2", 2.0, 0.0, 3.0)
    sigma2 = zfit.Parameter("sigma2", 0.3)
    sigma2.floating = False
    model2 = zfit.pdf.Gauss(obs=obs2, mu=mu2, sigma=sigma2)

    data1 = zfit.data.Data.from_numpy(obs=obs1, array=np.random.normal(1.0, 0.3, 500))
    data2 = zfit.data.Data.from_numpy(obs=obs2, array=np.random.normal(2.0, 0.3, 500))

    loss = UnbinnedNLL(model=model1, data=data1) + UnbinnedNLL(model=model2, data=data2)
    return loss, mu1, 1.5, 0.0


def create_all_shared_params_loss():
    """Create simultaneous loss where PDFs share ALL parameters (2 losses)."""
    mu = zfit.Parameter("mu_shared", 1.5, 0.0, 3.0)
    sigma = zfit.Parameter("sigma_shared", 0.3)
    sigma.floating = False

    obs1 = zfit.Space("x", limits=(0.0, 3.0))
    obs2 = zfit.Space("y", limits=(0.0, 3.0))

    model1 = zfit.pdf.Gauss(obs=obs1, mu=mu, sigma=sigma)
    model2 = zfit.pdf.Gauss(obs=obs2, mu=mu, sigma=sigma)

    data1 = zfit.data.Data.from_numpy(obs=obs1, array=np.random.normal(1.5, 0.3, 500))
    data2 = zfit.data.Data.from_numpy(obs=obs2, array=np.random.normal(1.5, 0.3, 500))

    loss = UnbinnedNLL(model=model1, data=data1) + UnbinnedNLL(model=model2, data=data2)
    return loss, mu, 2.0, 0.0


def create_some_shared_params_loss():
    """Create simultaneous loss where PDFs share SOME parameters (2 losses)."""
    mu = zfit.Parameter("mu_common", 1.2, 0.0, 3.0)
    sigma1 = zfit.Parameter("sigma1", 0.3)
    sigma2 = zfit.Parameter("sigma2", 0.4)
    sigma1.floating = False
    sigma2.floating = False

    obs1 = zfit.Space("x", limits=(0.0, 3.0))
    obs2 = zfit.Space("y", limits=(0.0, 3.0))

    model1 = zfit.pdf.Gauss(obs=obs1, mu=mu, sigma=sigma1)
    model2 = zfit.pdf.Gauss(obs=obs2, mu=mu, sigma=sigma2)

    data1 = zfit.data.Data.from_numpy(obs=obs1, array=np.random.normal(1.2, 0.3, 500))
    data2 = zfit.data.Data.from_numpy(obs=obs2, array=np.random.normal(1.2, 0.4, 500))

    loss = UnbinnedNLL(model=model1, data=data1) + UnbinnedNLL(model=model2, data=data2)
    return loss, mu, 1.8, 0.5


def create_mixed_loss():
    """Create simultaneous loss mixing all sharing patterns (5 losses).

    - Loss 1-2: Share mu_shared (POI) and sigma_shared
    - Loss 3: Independent mu3, sigma3
    - Loss 4-5: Share mu_shared (POI) but have independent sigmas
    """
    mu_shared = zfit.Parameter("mu_shared", 1.5, 0.0, 3.0)
    sigma_shared = zfit.Parameter("sigma_shared", 0.3)
    sigma_shared.floating = False

    # Losses 1-2: All shared params
    obs1 = zfit.Space("x1", limits=(0.0, 3.0))
    obs2 = zfit.Space("x2", limits=(0.0, 3.0))
    model1 = zfit.pdf.Gauss(obs=obs1, mu=mu_shared, sigma=sigma_shared)
    model2 = zfit.pdf.Gauss(obs=obs2, mu=mu_shared, sigma=sigma_shared)
    data1 = zfit.data.Data.from_numpy(obs=obs1, array=np.random.normal(1.5, 0.3, 500))
    data2 = zfit.data.Data.from_numpy(obs=obs2, array=np.random.normal(1.5, 0.3, 500))

    # Loss 3: No shared params (independent)
    obs3 = zfit.Space("x3", limits=(0.0, 3.0))
    mu3 = zfit.Parameter("mu3", 2.0, 0.0, 3.0)
    sigma3 = zfit.Parameter("sigma3", 0.35)
    sigma3.floating = False
    model3 = zfit.pdf.Gauss(obs=obs3, mu=mu3, sigma=sigma3)
    data3 = zfit.data.Data.from_numpy(obs=obs3, array=np.random.normal(2.0, 0.35, 500))

    # Losses 4-5: Share mu_shared but independent sigmas
    obs4 = zfit.Space("x4", limits=(0.0, 3.0))
    obs5 = zfit.Space("x5", limits=(0.0, 3.0))
    sigma4 = zfit.Parameter("sigma4", 0.25)
    sigma5 = zfit.Parameter("sigma5", 0.4)
    sigma4.floating = False
    sigma5.floating = False
    model4 = zfit.pdf.Gauss(obs=obs4, mu=mu_shared, sigma=sigma4)
    model5 = zfit.pdf.Gauss(obs=obs5, mu=mu_shared, sigma=sigma5)
    data4 = zfit.data.Data.from_numpy(obs=obs4, array=np.random.normal(1.5, 0.25, 500))
    data5 = zfit.data.Data.from_numpy(obs=obs5, array=np.random.normal(1.5, 0.4, 500))

    loss = (
        UnbinnedNLL(model=model1, data=data1)
        + UnbinnedNLL(model=model2, data=data2)
        + UnbinnedNLL(model=model3, data=data3)
        + UnbinnedNLL(model=model4, data=data4)
        + UnbinnedNLL(model=model5, data=data5)
    )
    return loss, mu_shared, 2.0, 0.5


@pytest.mark.parametrize(
    "loss_factory",
    [
        create_no_shared_params_loss,
        create_all_shared_params_loss,
        create_some_shared_params_loss,
        create_mixed_loss,
    ],
    ids=["no_shared", "all_shared", "some_shared", "mixed_5losses"],
)
def test_simultaneous_fit_null_toys(loss_factory):
    """Test that null toys cluster around the null hypothesis value."""
    np.random.seed(42)

    loss, poi_param, null_value, alt_value = loss_factory()

    calc = FrequentistCalculator(loss, Minuit(), ntoysnull=20, ntoysalt=20)

    poi_null = POIarray(poi_param, [null_value])
    poi_alt = POI(poi_param, alt_value)

    toysresults = calc.get_toys_null(poi_null, poi_alt, qtilde=False)
    toys = toysresults[POI(poi_param, null_value)]

    assert np.abs(np.mean(toys.bestfit) - null_value) < 0.15, (
        f"Expected null toys around {null_value}, got {np.mean(toys.bestfit):.3f}"
    )


@pytest.mark.parametrize(
    "loss_factory",
    [
        create_some_shared_params_loss,
        create_mixed_loss,
    ],
    ids=["some_shared", "mixed_5losses"],
)
def test_simultaneous_fit_alt_toys(loss_factory):
    """Test that alt toys cluster around the alt hypothesis value."""
    np.random.seed(42)

    loss, poi_param, null_value, alt_value = loss_factory()

    calc = FrequentistCalculator(loss, Minuit(), ntoysnull=20, ntoysalt=20)

    poi_null = POIarray(poi_param, [null_value])
    poi_alt = POI(poi_param, alt_value)

    toysresults_alt = calc.get_toys_alt(poi_alt, poi_null, qtilde=False)
    toys_alt = toysresults_alt[poi_alt]

    assert np.abs(np.mean(toys_alt.bestfit) - alt_value) < 0.15, (
        f"Expected alt toys around {alt_value}, got {np.mean(toys_alt.bestfit):.3f}"
    )
