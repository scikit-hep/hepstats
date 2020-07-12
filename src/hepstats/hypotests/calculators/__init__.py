# -*- coding: utf-8 -*-
"""
Module defining the base class for the calculators for statistical tests based on the likelyhood ratio.

Acronyms used in the code:
    * nll = negative log-likehood, which is the value of the `loss` attribute of a calculator;
    * obs = observed, i.e. measured on provided data.

"""

from .asymptotic_calculator import AsymptoticCalculator
from .frequentist_calculator import FrequentistCalculator
