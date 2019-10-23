# scikit-stats: statistics tools and utilities

[![Build Status](https://dev.azure.com/matthieumarinangeli/matthieumarinangeli/_apis/build/status/scikit-hep.scikit-stats?branchName=master)](https://dev.azure.com/matthieumarinangeli/matthieumarinangeli/_build/latest?definitionId=3&branchName=master)

## Getting Started

The scikit-stats module includes modeling and hypotests submodules. This a quick user  guide to each submodule.

### modeling

To be done.

### hypotests

This submodule provides tools to do hypothesis tests such as discovery test and computations of upper limits or confidence intervals. scikit-stats needs a fitting backend to perform computations such as [zfit](https://github.com/zfit/zfit). Any fitting library can be used if their API is compatible  with scikit-stats (see [api checks](https://github.com/scikit-hep/scikit-stats/blob/master/skstats/hypotests/fitutils/api_check.py)).

We give here a simple example of a discovery test, using zfit as backend, of gaussian signal with known mean and sigma over an exponential background.

```python
>>> import zfit
>>> from zfit.core.loss import ExtendedUnbinnedNLL
>>> from zfit.minimize import Minuit

>>> bounds = (0.1, 3.0)
>>> zfit.Space('x', limits=bounds)

>>> bkg = np.random.exponential(0.5, 300)
>>> peak = np.random.normal(1.2, 0.1, 25)
>>> data = np.concatenate((bkg, peak))
>>> data = data[(data > bounds[0]) & (data < bounds[1])]
>>> N = data.size
>>> data = zfit.data.Data.from_numpy(obs=obs, array=data)

>>> lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
>>> Nsig = zfit.Parameter("Ns", 20., -20., N)
>>> Nbkg = zfit.Parameter("Nbkg", N, 0., N*1.1)
>>> signal = Nsig * zfit.pdf.Gauss(obs=obs, mu=1.2, sigma=0.1)
>>> background = Nbkg * zfit.pdf.Exponential(obs=obs, lambda_=lambda_)
>>> loss = ExtendedUnbinnedNLL(model=[signal + background], data=[data], fit_range=[obs])

>>> calculator = AsymptoticCalculator(loss, Minuit())
>>> poinull = POI(Nsig, 0)
>>> discovery_test = Discovery(calculator, [poinull])
>>> discovery_test.result()

p_value for the Null hypothesis = 0.0007571045424956679
Significance = 3.1719464825102244
```

The discovery test prints out the pvalue and the significance of the null hypothesis to be rejected.
