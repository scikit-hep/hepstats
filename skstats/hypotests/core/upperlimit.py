from scipy import interpolate

from .basetest import BaseTest
from ..calculators import AsymptoticCalculator
from ..parameters import POI


class UpperLimit(BaseTest):
    def __init__(self, calculator, poinull, poialt=None, qtilde=False):
        """Class for upper limit calculation.

            Args:
                calculator (`sktats.hypotests.BaseCalculator`): calculator to use for computing the pvalues
                poinull (List[`hypotests.POI`]): parameters of interest for the null hypothesis
                poialt (List[`hypotests.POI`], optional): parameters of interest for the alternative hypothesis
                qtilde (bool, optional): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                    the $$q$$ test statistic
        """

        super(UpperLimit, self).__init__(calculator, poinull, poialt)

        self._qtilde = qtilde

    @property
    def qtilde(self):
        """
        Returns True if qtilde statistic is used, else False.
        """
        return self._qtilde

    def pvalues(self, CLs=True):
        """
        Returns p-values scanned for the values of the parameters of interest
        in the null hypothesis.
        """
        pvalue_func = self.calculator.pvalue

        pnull, palt = pvalue_func(poinull=self.poinull, poialt=self.poialt, qtilde=self.qtilde, onesided=True)

        pvalues = {"clsb": pnull, "clb": palt}

        sigmas = [0.0, 1.0, 2.0, -1.0, -2.0]

        exppvalue_func = self.calculator.expected_pvalue

        result = exppvalue_func(poinull=self.poinull, poialt=self.poialt, nsigma=sigmas, CLs=CLs,
                                qtilde=self.qtilde, onesided=True)

        pvalues["expected"] = result[0]
        pvalues["expected_p1"] = result[1]
        pvalues["expected_p2"] = result[2]
        pvalues["expected_m1"] = result[3]
        pvalues["expected_m2"] = result[4]

        pvalues["cls"] = pnull / palt

        return pvalues

    def upperlimit(self, alpha=0.05, CLs=True, printlevel=1):
        """
        Returns the upper limit of the parameter of interest.
        """

        poinull = self.poinull[0]

        # create a filter for -1 and -2 sigma expected limits
        bestfitpoi = self.calculator.bestfit.params[poinull.parameter]["value"]
        filter = poinull.value > bestfitpoi

        if CLs:
            observed_key = "cls"
        else:
            observed_key = "clsb"

        if isinstance(self.calculator, AsymptoticCalculator):
            to_interpolate = [observed_key]
        else:
            to_interpolate = [observed_key] + [f"expected{i}" for i in ["", "_p1", "_m1", "_p2", "_m2"]]

        limits = {}
        for k in to_interpolate:
            if k not in ["expected_m1", "expected_m2"]:
                pvalues = self.pvalues(CLs)[k][filter]
                values = poinull.value[filter]
            else:
                pvalues = self.pvalues(CLs)[k]
                values = poinull.value

            tck = interpolate.splrep(values, pvalues-alpha, s=0)
            root = interpolate.sproot(tck)

            if k == observed_key:
                k = "observed"

            if len(root) > 1:
                root = root[0]

            limits[k] = float(root)

        if isinstance(self.calculator, AsymptoticCalculator):
            poiul = POI(poinull.parameter, limits["observed"])
            exppoi_func = self.calculator.expected_poi
            sigmas = [0.0, 1.0, -1.0, 2.0, -2.0]

            results = exppoi_func(poinull=[poiul], poialt=self.poialt, nsigma=sigmas, alpha=alpha, CLs=CLs)
            keys = [f"expected{i}" for i in ["", "_p1", "_m1", "_p2", "_m2"]]

            for r, k in zip(results, keys):
                limits[k] = float(r)

        if printlevel > 0:
            print(f"\nObserved upper limit: {poinull.name} = {limits['observed']}")
            print(f"Expected upper limit: {poinull.name} = {limits['expected']}")
            for sigma in ["+1", "-1", "+2", "-2"]:
                key = sigma.replace("+", "p").replace("-", "m")
                print(f"Expected upper limit {sigma} sigma: {poinull.name} = {limits[f'expected_{key}']}")

        return limits
