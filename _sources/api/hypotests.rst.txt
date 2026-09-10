hepstats.hypotests
------------------

.. currentmodule:: hepstats.hypotests.core.discovery

.. autosummary::

    Discovery

.. currentmodule:: hepstats.hypotests.core.upperlimit

.. autosummary::

    UpperLimit

.. currentmodule:: hepstats.hypotests.core.confidence_interval

.. autosummary::

    ConfidenceInterval

Parameters
""""""""""

.. currentmodule:: hepstats.hypotests.parameters

.. autosummary::

    POIarray
    POI

Calculators
"""""""""""

Module defining the base class for the calculators for statistical tests based on the likelyhood ratio.

Acronyms used in the code:
    * nll = negative log-likehood, which is the value of the `loss` attribute of a calculator;
    * obs = observed, i.e. measured on provided data.

.. currentmodule:: hepstats.hypotests.calculators.asymptotic_calculator

.. autosummary::

    AsymptoticCalculator

.. currentmodule:: hepstats.hypotests.calculators.frequentist_calculator

.. autosummary::

    FrequentistCalculator

Toys utils
""""""""""

.. currentmodule:: hepstats.hypotests.toyutils

.. autosummary::

    ToyResult
    ToysManager
