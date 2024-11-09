import numpy as np
import pytest
from scipy.stats import ks_2samp

zfit = pytest.importorskip("zfit")
from zfit.loss import ExtendedUnbinnedNLL
from zfit.minimize import Minuit

from hepstats.splot import compute_sweights
from hepstats.splot.sweights import is_sum_of_extended_pdfs
from hepstats.utils.fit import get_value
from hepstats.splot.exceptions import ModelNotFittedToData
from hepstats.splot.warnings import AboveToleranceWarning


def get_data_and_loss():
    bounds = (0.0, 3.0)
    obs = zfit.Space("x", limits=bounds)
    nbkg = 10000
    nsig = 5000

    # Data and signal
    def get_sel(arr):
        return (arr > bounds[0]) & (arr < bounds[1])

    np.random.seed(0)

    N = nsig + nbkg
    mean = zfit.Parameter("mean", 1.2, 0.5, 2.0)
    sigma = zfit.Parameter("sigma", 0.1, 0.02, 0.3)
    lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
    Nsig = zfit.Parameter("Nsig", nsig, 0.0, N)
    Nbkg = zfit.Parameter("Nbkg", nbkg, 0.0, N)

    signal = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma).create_extended(Nsig)
    background = zfit.pdf.Exponential(obs=obs, lambda_=lambda_).create_extended(Nbkg)
    tot_model = zfit.pdf.SumPDF([signal, background])

    bkg = background.sample(nbkg, params={lambda_: -2.1})

    peak = signal.sample(nsig, params={sigma: 0.2})

    mass = np.concatenate((bkg["x"], peak["x"]))

    bck_p = np.random.normal(3, 1, size=nbkg)
    sig_p = np.random.normal(5, 1, size=nsig)
    p = np.concatenate([bck_p, sig_p])

    data = zfit.data.concat([bkg, peak], axis="index")

    loss = ExtendedUnbinnedNLL(model=tot_model, data=data)

    return mass, p, loss, Nsig, Nbkg, sig_p, bck_p


def test_sweights_constructor():
    mass, p, loss, Nsig, Nbkg, sig_p, bkg_p = get_data_and_loss()

    with pytest.raises(ValueError):
        compute_sweights("model", mass)

    with pytest.raises(ValueError):
        compute_sweights(loss.model[0].get_models()[0], mass)


def test_sweights():
    minimizer = Minuit()
    mass, p, loss, Nsig, Nbkg, sig_p, bkg_p = get_data_and_loss()

    with pytest.raises(ModelNotFittedToData):
        compute_sweights(loss.model[0], mass)

    result = minimizer.minimize(loss)
    assert result.valid

    model = loss.model[0]
    assert is_sum_of_extended_pdfs(model)

    yields = [Nsig, Nbkg]

    sweights = compute_sweights(loss.model[0], mass)

    assert np.allclose(
        [np.sum(sweights[y]) / get_value(y.value()) for y in yields], 1.0
    )

    nbins = 30
    hist_conf = dict(bins=nbins, range=[0, 10])

    hist_sig_true_p, _ = np.histogram(sig_p, **hist_conf)
    sel = hist_sig_true_p != 0
    hist_sig_true_p = hist_sig_true_p[sel]
    hist_sig_sweights_p = np.histogram(p, weights=sweights[Nsig], **hist_conf)[0][sel]

    assert ks_2samp(hist_sig_sweights_p, hist_sig_true_p)[-1] > 0.001

    hist_bkg_true_p, _ = np.histogram(bkg_p, **hist_conf)
    sel = hist_bkg_true_p != 0
    hist_bkg_true_p = hist_bkg_true_p[sel]
    hist_bkg_sweights_p = np.histogram(p, weights=sweights[Nbkg], **hist_conf)[0][sel]

    assert ks_2samp(hist_bkg_sweights_p, hist_bkg_true_p)[-1] > 0.001

    with pytest.warns(AboveToleranceWarning):
        compute_sweights(
            loss.model[0], np.concatenate([mass, np.random.normal(0.8, 0.1, 100)])
        )

    with pytest.raises(ModelNotFittedToData):
        compute_sweights(
            loss.model[0], np.concatenate([mass, np.random.normal(0.8, 0.1, 1000)])
        )
