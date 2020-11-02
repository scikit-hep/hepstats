hepstats.utils
==============

hepstats.utils.fit
------------------

hepstats.utils.fit.api_check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Module for testing a fitting library validity with hepstats.

A fitting library should provide six basic objects:

    * model / probability density function
    * parameters of the models
    * data
    * loss / likelihood function
    * minimizer
    * fitresult (optional)

A function for each object is defined in this module, all should return `True` to work
with hepstats.

The `zfit` API is currently the standard fitting API in hepstats.


.. currentmodule:: hepstats.utils.fit.api_check

.. autosummary::

    is_valid_parameter
    is_valid_data
    is_valid_pdf
    is_valid_loss
    is_valid_fitresult
    is_valid_minimizer

hepstats.utils.fit.diverse
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: hepstats.utils.fit.diverse

.. autosummary::

    get_value
    eval_pdf
    pll
    array2dataset
    get_nevents

hepstats.utils.fit.sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: hepstats.utils.fit.sampling

.. autosummary::

    base_sampler
    base_sample
