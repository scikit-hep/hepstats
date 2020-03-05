import numpy as np
from scipy.stats import chisquare

import zfit
from zfit.core.testing import teardown_function  # allows redefinition of zfit.Parameter, needed for tests
from zfit.loss import ExtendedUnbinnedNLL
from zfit.minimize import Minuit

from hepstats.splot import compute_sweights
from hepstats.splot.sweights import is_sum_of_extended_pdfs
from hepstats.utils.fit import get_value


def get_data_and_loss():

    bounds = (0.0, 3.0)
    obs = zfit.Space('x', limits=bounds)
    nbkg = 10000
    nsig = 5000

    # Data and signal

    np.random.seed(0)
    tau = -2.0
    beta = -1/tau
    bkg = np.random.exponential(beta, nbkg)
    peak = np.random.normal(1.2, 0.2, nsig)
    mass = np.concatenate((bkg, peak))

    sig_p = np.random.normal(5, 1, size=nsig)
    bck_p = np.random.normal(3, 1, size=nbkg)
    p = np.concatenate([bck_p, sig_p])

    sel = (mass > bounds[0]) & (mass < bounds[1])
    mass = mass[sel]
    p = p[sel]

    N = len(mass)
    data = zfit.data.Data.from_numpy(obs=obs, array=mass)

    mean = zfit.Parameter("mean", 1.2, 0.5, 2.0)
    sigma = zfit.Parameter("sigma", 0.1, 0.02, 0.2)
    lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
    Nsig = zfit.Parameter("Nsig", nsig, 0., N)
    Nbkg = zfit.Parameter("Nbkg", nbkg, 0., N)

    signal = Nsig * zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
    background = Nbkg * zfit.pdf.Exponential(obs=obs, lambda_=lambda_)
    tot_model = signal + background

    loss = ExtendedUnbinnedNLL(model=tot_model, data=data)

    return mass, p, loss, Nsig, Nbkg, sig_p, bck_p


def test_sweights():

    minimizer = Minuit()
    mass, p, loss, Nsig, Nbkg, sig_p, bkg_p = get_data_and_loss()
    minimizer.minimize(loss)

    model = loss.model[0]
    assert is_sum_of_extended_pdfs(model)

    yields = [Nsig, Nbkg]

    sweights = compute_sweights(loss.model[0], mass)

    assert np.allclose([np.sum(sweights[y])/get_value(y) for y in yields], 1.0)

    nbins = 30
    hist_conf = dict(bins=nbins, range=[0, 10])

    hist_sig_true_p, _ = np.histogram(sig_p, **hist_conf)
    sel = hist_sig_true_p != 0
    hist_sig_true_p = hist_sig_true_p[sel]
    hist_sig_sweights_p = np.histogram(p, weights=sweights[Nsig], **hist_conf)[0][sel]

    assert chisquare(hist_sig_sweights_p, hist_sig_true_p)[-1] < 0.01

    hist_bkg_true_p, _ = np.histogram(bkg_p, **hist_conf)
    sel = hist_bkg_true_p != 0
    hist_bkg_true_p = hist_bkg_true_p[sel]
    hist_bkg_sweights_p = np.histogram(p, weights=sweights[Nbkg], **hist_conf)[0][sel]

    assert chisquare(hist_bkg_sweights_p, hist_bkg_true_p)[-1] < 0.01
