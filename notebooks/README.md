# Notebooks

In this directory are stored all the notebooks demo that you can either run with [binder](https://mybinder.org/v2/gh/scikit-hep/hepstats/master) or by downloading the jupyter notebooks `ipynb` files.

The notebooks are divided for each `submodule`:
- `hypotests`:
    * discovery_asy_zfit.ipynb: computes the significance of a gaussian signal over an exponential background, fitted with `zfit`, using the `AsymptoticCalculator`.
    * discovery_freq_zfit.ipynb: computes the significance of a gaussian signal over an exponential background, fitted with `zfit`, using the `FrequentistCalculator`.
    * upperlimit_asy_zfit.ipynb: computes the upper limit on the number signal of a gaussian signal over an exponential background, fitted with `zfit`, using the `AsymptoticCalculator`.
    * upperlimit_freq_zfit.ipynb: computes the upper limit on the number signal of a gaussian signal over an exponential background, fitted with `zfit`, using the `FrequentistCalculator`.
    * confidenceinterval_asy_zfit.ipynb: computes the 68% confidence level interval on the mean of a gaussian signal over an exponential background, fitted with `zfit`, using the `AsymptoticCalculator`.
    * confidenceinterval_freq_zfit.ipynb: computes the 68% confidence level interval on the mean of a gaussian signal over an exponential background, fitted with `zfit`, using the `FrequentistCalculator`.
    * FC_interval_asy.ipynb: computes the 90% confidence level Feldman and Cousins interval on the measured mean 𝑥 of a gaussian for several values of the true mean μ, using the `AsymptoticCalculator`.
    * FC_interval_asy.ipynb: computes the 90% confidence level Feldman and Cousins interval on the measured mean 𝑥 of a gaussian for several values of the true mean μ, using the `FrequentistCalculator`.
    * counting.ipynb: shows examples of inferences with `hepstats` using a counting analysis instead of a shape analysis.

- `modeling`
    * bayesian_blocks.ipynb: presentation of the Bayesian Blocks algorithm and comparison with other binning methods.

- `splots`
    * splot_example.ipynb: example of `sPlot` on fake mass and momentum distributions for some signal and some background. The `sWeights` are derived using mass fit of a gaussian signal over an exponential background with `zfit`. The `sWeights` are applied on the momentum distribution to retrieve the signal distribution. This example is a reproduction of the example in [hep_ml](https://github.com/arogozhnikov/hep_ml/blob/master/notebooks/sPlot.ipynb) using `hepstats`.
    * splot_example_2.ipynb: example of `sPlot` on fake mass and lifetime distributions for some signal and some background. The `sWeights` are derived using mass fit of a gaussian signal over an exponential background with `zfit`. The `sWeights` are applied on the lifetime distribution to retrieve the signal distribution. This example is a reproduction of the example of the [LHCb statistics guidelines](https://gitlab.cern.ch/lhcb/statistics-guidelines/-/blob/add_sweights_item/resources/appendix_f4.ipynb) using `hepstats`.
