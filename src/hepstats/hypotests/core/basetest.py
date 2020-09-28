# -*- coding: utf-8 -*-
from typing import Union

from ..calculators.basecalculator import BaseCalculator
from ..parameters import POI, POIarray

"""
Module defining the base class for hypothesis tests.
"""


class BaseTest(object):
    def __init__(
        self,
        calculator: BaseCalculator,
        poinull: Union[POI, POIarray],
        poialt: Union[POI, POIarray, None] = None,
    ):
        """Base class for hypothesis tests.

        Args:
            calculator: calculator to use for computing the pvalues
            poinull: parameters of interest for the null hypothesis
            poialt: parameters of interest for the alternative hypothesis

        Raises:
            TypeError: if calculator is not a BaseCalculator instance
        """

        if not isinstance(calculator, BaseCalculator):
            msg = "Invalid type, {0}, for calculator. Calculator required."
            raise TypeError(msg)
        self._calculator = calculator

        self.calculator.check_pois(poinull)
        if poialt:
            self.calculator.check_pois(poialt)
            self.calculator.check_pois_compatibility(poinull, poialt)

        self._poinull = poinull
        self._poialt = poialt

    @property
    def poinull(self):
        """
        Returns the POI for the null hypothesis.
        """
        return self._poinull

    @property
    def poialt(self):
        """
        Returns the POI for the alternative hypothesis.
        """
        return self._poialt

    @property
    def calculator(self):
        """
        Returns the calculator.
        """
        return self._calculator
