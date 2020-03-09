from ..calculators.basecalculator import BaseCalculator

"""
Module defining the base class for hypothesis tests.
"""


class BaseTest(object):
    def __init__(self, calculator, poinull, poialt=None):
        """Base class for hypothesis tests.

            Args:
                calculator (`sktats.hypotests.BaseCalculator`): calculator to use for computing the pvalues
                poinull (List[`hypotests.POI`]): parameters of interest for the null hypothesis
                poialt (List[`hypotests.POI`], optional): parameters of interest for the alternative hypothesis
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
