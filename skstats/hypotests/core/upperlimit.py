from .basetest import BaseTest


class UpperLimit(BaseTest):
    def __init__(self, calculator, poinull, poialt=None, qtilde=False, alpha=0.05):
        """Class for upper limit calculation.

            Args:
                calculator (`sktats.hypotests.BaseCalculator`): calculator to use for computing the pvalues
                poinull (List[`hypotests.POI`]): parameters of interest for the null hypothesis
                poialt (List[`hypotests.POI`], optional): parameters of interest for the alternative hypothesis
                qtilde (bool, optional): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                    the $$q$$ test statistic
        """

        super(UpperLimit, self).__init__(calculator, poinull, poialt)

        self._pvalues = {}
        self._qtilde = qtilde

    @property
    def qtilde(self):
        """
        Returns True if qtilde statistic is used, else False.
        """
        return self._qtilde

    def pvalues(self):
        """
        Returns p-values scanned for the values of the parameters of interest
        in the null hypothesis.
        """
        if not self._pvalues:

            pvalue_func = self.calculator.pvalue

            pnull, palt = pvalue_func(self.poinull, self.poialt, qtilde=self.qtilde, onesided=True)

            self._pvalues = {"clsb": pnull, "clb": palt}

            sigmas = [0.0, 1.0, 2.0, -1.0, -2.0]

            exppvalue_func = self.calculator.expected_pvalue

            result = exppvalue_func(self.poinull, self.poialt, sigmas, self.CLs)

            self._pvalues["exp"] = result[0]
            self._pvalues["exp_p1"] = result[1]
            self._pvalues["exp_p2"] = result[2]
            self._pvalues["exp_m1"] = result[3]
            self._pvalues["exp_m2"] = result[4]

            self._pvalues["cls"] = pnull / palt

        return self._pvalues

    def upperlimit(self, alpha=0.05, CLs=False, printlevel=1):
        """
        Returns the upper limit of the parameter of interest.
        """

        pvalues = self.pvalues()
        poinull = self.poinull
        poivalues = poinull.value
        poiname = poinull.name
        poiparam = poinull.parameter

        bestfitpoi = self.calculator.config.bestfit.params[poiparam]["value"]
        sel = poivalues > bestfitpoi

        if CLs:
            k = "cls"
        else:
            k = "clsb"

        values = {}
        if isinstance(self.calculator, AsymptoticCalculator):
            keys = [k]
        else:
            keys = [k, "exp", "exp_p1", "exp_m1", "exp_p2", "exp_m2"]

        for k_ in keys:
            p_ = pvalues[k_]
            pvals = poivalues
            if k_ not in ["exp_m1", "exp_m2"]:
                p_ = p_[sel]
                pvals = pvals[sel]
            p_ = p_ - self.alpha

            s = InterpolatedUnivariateSpline(pvals, p_)
            val = s.roots()

            if len(val) > 0:
                poiul = val[0]
            else:
                poiul = None
            if k_ == k:
                k_ = "observed"

            values[k_] = poiul

        if isinstance(self.calculator, AsymptoticCalculator):
            poiul = POI(poiparam, poiul)
            exp_poi = self.calculator.expected_poi
            sigmas = [0.0, 1.0, 2.0, -1.0, -2.0]
            kwargs = dict(poinull=poiul, poialt=self.poialt, nsigma=sigmas,
                          alpha=self.alpha, CLs=self.CLs)

            results = exp_poi(**kwargs)
            keys = ["exp", "exp_p1", "exp_p2", "exp_m1", "exp_m2"]

            for r, k_ in zip(results, keys):
                values[k_] = r

        if printlevel > 0:

            msg = "\nObserved upper limit: {0} = {1}"
            print(msg.format(poiname, values["observed"]))
            msg = "Expected upper limit: {0} = {1}"
            print(msg.format(poiname, values["exp"]))
            msg = "Expected upper limit +1 sigma: {0} = {1}"
            print(msg.format(poiname, values["exp_p1"]))
            msg = "Expected upper limit -1 sigma: {0} = {1}"
            print(msg.format(poiname, values["exp_m1"]))
            msg = "Expected upper limit +2 sigma: {0} = {1}"
            print(msg.format(poiname, values["exp_p2"]))
            msg = "Expected upper limit -2 sigma: {0} = {1}"
            print(msg.format(poiname, values["exp_m2"]))

        return values
