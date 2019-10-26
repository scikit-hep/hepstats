from scipy import interpolate

from .basetest import BaseTest
from ..parameters import POI


class ConfidenceInterval(BaseTest):
    def __init__(self, calculator, poinull, qtilde=False):

        super(ConfidenceInterval, self).__init__(calculator, poinull)

        self._qtilde = qtilde

    @property
    def qtilde(self):
        """
        Returns True if qtilde test statistic is used, else False.
        """
        return self._qtilde

    def pvalues(self):
        """
        Returns p-values scanned for the values of the parameters of interest
        in the null hypothesis.
        """

        return self.calculator.pvalue(self.poinull, qtilde=self.qtilde, onesided=False)[0]

    def interval(self, alpha=0.32, printlevel=1):
        """
        Returns the confidence level on the parameter of interest.

        Args:
            alpha (float, default=0.05): significance level
            printlevel (int, default=1): if > 0 print the result

        """

        poinull = self.poinull[0]
        observed = self.calculator.bestfit.params[poinull.parameter]["value"]

        tck = interpolate.splrep(poinull.value, self.pvalues()-alpha, s=0)
        root = interpolate.sproot(tck)

        bands = {}
        bands["observed"] = observed
        bands["band_p"] = root[0]
        bands["band_m"] = root[1]

        if printlevel > 0:

            msg = f"\nConfidence interval on {poinull.name}:\n"
            msg += f"\t{root[0]} < {poinull.name} < {root[1]} at {1-alpha:.1f}% C.L."
            print(msg)

        return bands
