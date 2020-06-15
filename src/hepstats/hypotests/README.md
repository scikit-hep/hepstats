# Hypotests

This submodule provides tools to do hypothesis tests such as discovery test and computations of upper limits or confidence intervals. hepstats needs a fitting backend to perform computations such as [zfit](https://github.com/zfit/zfit). Any fitting library can be used if their API is compatible  with hepstats (see [api checks](https://github.com/scikit-hep/hepstats/blob/master/hepstats/hypotests/fitutils/api_check.py)).

We give here a simple example of a discovery test, using the [zfit](https://github.com/zfit/zfit)
fitting package as backend, of a Gaussian signal with known mean and sigma over an exponential background.

```python
>>> import zfit
>>> from zfit.loss import ExtendedUnbinnedNLL
>>> from zfit.minimize import Minuit

>>> bounds = (0.1, 3.0)
>>> obs = zfit.Space('x', limits=bounds)

>>> bkg = np.random.exponential(0.5, 300)
>>> peak = np.random.normal(1.2, 0.1, 25)
>>> data = np.concatenate((bkg, peak))
>>> data = data[(data > bounds[0]) & (data < bounds[1])]
>>> N = data.size
>>> data = zfit.Data.from_numpy(obs=obs, array=data)

>>> lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
>>> Nsig = zfit.Parameter("Ns", 20., -20., N)
>>> Nbkg = zfit.Parameter("Nbkg", N, 0., N*1.1)
>>> signal = zfit.pdf.Gauss(obs=obs, mu=1.2, sigma=0.1).create_extended(Nsig)
>>> background = zfit.pdf.Exponential(obs=obs, lambda_=lambda_).create_extended(Nbkg)
>>> total = zfit.pdf.SumPDF([signal, background])
>>> loss = ExtendedUnbinnedNLL(model=total, data=data)

>>> from hepstats.hypotests.calculators import AsymptoticCalculator
>>> from hepstats.hypotests import Discovery
>>> from hepstats.hypotests.parameters import POI

>>> calculator = AsymptoticCalculator(input=loss, minimizer=Minuit())
>>> poinull = POI(Nsig, 0)
>>> discovery_test = Discovery(calculator, poinull)
>>> discovery_test.result()

p_value for the Null hypothesis = 0.0007571045424956679
Significance (in units of sigma) = 3.1719464825102244
```

The discovery test prints out the p-value and the significance of the null hypothesis to be rejected.
