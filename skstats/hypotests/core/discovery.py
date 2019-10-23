from scipy.stats import norm

from .basetest import BaseTest


class Discovery(BaseTest):
    def __init__(self, calculator, poinull):
        """Class for discovery test.

            Args:
                calculator (`sktats.hypotests.BaseCalculator`): calculator to use for computing the pvalues
                poinull (List[`hypotests.POI`]): parameters of interest for the null hypothesis
        """
        super(Discovery, self).__init__(calculator, poinull)

    def result(self, printlevel=1):
        """
        Returns the result of the discovery hypothesis test.

            Args:
                printlevel (int, default=1): if > 0 print the result
        """
        pnull, _ = self.calculator.pvalue(self.poinull, onesideddiscovery=True)
        pnull = pnull[0]

        Z = norm.ppf(1. - pnull)

        if printlevel > 0:
            print("\np_value for the Null hypothesis = {0}".format(pnull))
            print("Significance = {0}".format(Z))

        ret = {
               "pnull": pnull,
               "significance": Z,
               }

        return ret
