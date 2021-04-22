Changelog
=========

master
******


Version 0.4.0
*************

* loss: upgrade API to use `create_new` to make sure that the losses are comparable. Compatible with zfit 0.6.4+

Version 0.3.1
*************
* sPlot: Increase the tolerance of the sanity check from 1e-3 to 5e-2, if above the tolerance a ModelNotFittedToData
  exception is raised. In addition if the the check is above the 5e-3 tolerance a warning message is printed.


Version 0.3.0
*************
* New documentation style
* `hepstats` can now do hypothesis tests, and compute upper limits and confidence intervals for counting analysis
* Progess bars are used to see the progression of the generation of the toys

Version 0.2.5
*************
* ConfidenceInterval can compute Feldman and Cousin intervals with boundaries (i.e ``qtilde=True``)
* **AsymptoticCalculator** asymov weights are now scaled to the number of entries in dataset from loss
  function if the loss is not extended
* **hepstats.hypotests** can now be used even if there is no nuisances. The **pll** function in **utils/fit/diverse.py**
  had to be modified such that if there are no nuisances, the **pll** function returns the value of the loss function.
* add notebooks demos for FC intervals with the `FrequentistCalculator` and `AsymptoticCalculator`.
* add warnings when multiple roots are found in `ConfidenceInterval`
* move toys .yml files from notebook to notebook/toys

Version 0.2.4
*************
* Redesigned packaging system, GHA deployment.
* **expected_poi** removed from **BaseCalculator** and **AsymptoticCalculator**
* add type checks in the **hypotests** submodule

Version 0.2.3
**************
* **hepstats** is now compatible with zfit > 0.5 api
* expected intervals in upper limit are now calculated from the pvalues and not from the **expected_poi**
  function anymore.

Version 0.2.2
**************
* Addition of the **sPlot** algorithm

Version 0.2.1
**************
* Addition of the **FrequentistCalculator** to performs hypothesis test, upper limit and interval calculations
  with toys. Toys can be saved and loaded in / from yaml files using the methods:
   * `to_yaml`
   * `from_yaml`

Version 0.2.0
**************
* New version for the new **hepstats** name of the package

Version 0.1.3
**************
* Package name changed from **scikit*stats** to **hepstats**

Version 0.1.2
**************
* Additions of classes to compute upper limits and confidence intervals.

Version 0.1.1
**************
* Release for Zenodo DOI

Version 0.1.0
**************
* First release of **scikit*stats**
* Addition of the **modeling** submodule with the ``Bayesian Blocks algorithm``
* Addition of the **hypotests** submodule
