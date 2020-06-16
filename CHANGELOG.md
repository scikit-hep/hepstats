Changelog
=========

master
------
- Redesigned packaging system, GHA deployment.
- `expected_poi` removed from `BaseCalculator` and `AsymptoticCalculator`

Version 0.2.3
--------------
- `hepstats` is now compatible with zfit > 0.5 api
- expected intervals in upper limit are now calculated from the pvalues and not from the `expected_poi` function anymore.

Version 0.2.2
--------------
- Addition of the **Splot** algorithm

Version 0.2.1
--------------
- Addition of the `FrequentistCalculator` to performs hypothesis test, upper limit and interval calculations with toys. Toys can be saved and loaded in / from yaml files using the methods:
    * `to_yaml`
    * `from_yaml`

Version 0.2.0
--------------
- New version for the new `hepstats` name of the package

Version 0.1.3
--------------
- Package name changed from `scikit-stats` to `hepstats`

Version 0.1.2
--------------
- Additions of classes to compute upper limits and confidence intervals.

Version 0.1.1
--------------
- Release for Zenodo DOI

Version 0.1.0
--------------
- First release of `scikit-stats`
- Addition of the `modeling` submodule with the  Bayesian Blocks algorithm
- Addition of the `hypotests` submodule
