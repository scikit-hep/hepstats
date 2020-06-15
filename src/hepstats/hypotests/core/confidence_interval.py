from scipy import interpolate

from .basetest import BaseTest
from ..exceptions import POIRangeError


class ConfidenceInterval(BaseTest):
    """Class for confidence interval calculation.

        Args:
            * **calculator** (`sktats.hypotests.BaseCalculator`): calculator to use for computing the pvalues
            * **poinull** (`POIarray`): parameters of interest for the null hypothesis
            * **qtilde** (bool, optional): if `True` use the :math:`\widetilde{q}` test statistics else (default) use the :math:`q` test statistic

        Example with `zfit`:
            >>> import numpy as np
            >>> import zfit
            >>> from zfit.loss import ExtendedUnbinnedNLL
            >>> from zfit.minimize import Minuit

            >>> bounds = (0.1, 3.0)
            >>> zfit.Space('x', limits=bounds)

            >>> bkg = np.random.exponential(0.5, 300)
            >>> peak = np.random.normal(1.2, 0.1, 80)
            >>> data = np.concatenate((bkg, peak))
            >>> data = data[(data > bounds[0]) & (data < bounds[1])]
            >>> N = data.size
            >>> data = zfit.data.Data.from_numpy(obs=obs, array=data)

            >>> mean = zfit.Parameter("mean", 1.2, 0.5, 2.0)
            >>> sigma = zfit.Parameter("sigma", 0.1, 0.02, 0.2)
            >>> lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
            >>> Nsig = zfit.Parameter("Ns", 20., -20., N)
            >>> Nbkg = zfit.Parameter("Nbkg", N, 0., N*1.1)
            >>> signal = Nsig * zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
            >>> background = Nbkg * zfit.pdf.Exponential(obs=obs, lambda_=lambda_)
            >>> loss = ExtendedUnbinnedNLL(model=signal + background, data=data)

            >>> from hepstats.hypotests.calculators import AsymptoticCalculator
            >>> from hepstats.hypotests import ConfidenceInterval
            >>> from hepstats.hypotests.parameters import POI, POIarray

            >>> calculator = AsymptoticCalculator(loss, Minuit())
            >>> poinull = POIarray(mean, np.linspace(1.15, 1.26, 100))
            >>> ci = ConfidenceInterval(calculator, poinull)
            >>> ci.interval()
            Confidence interval on mean:
                1.1810371356602791 < mean < 1.2156701172321935 at 68.0% C.L.
    """

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

        Returns:
            pvalues (`np.array`): CLsb, CLs, expected (+/- sigma bands) p-values
        """

        return self.calculator.pvalue(self.poinull, qtilde=self.qtilde, onesided=False)[0]

    def interval(self, alpha=0.32, printlevel=1):
        """
        Returns the confidence level on the parameter of interest.

        Args:
            * **alpha** (float, default=0.32/1 sigma): significance level,
            * **printlevel** (int, default=1): if > 0 print the result

        Returns:
            limits (Dict): central, upper and lower bounds on the parameter of interest

        """

        poinull = self.poinull
        observed = self.calculator.bestfit.params[poinull.parameter]["value"]

        if min(self.pvalues()) > alpha:
            msg = f"The minimum of the scanned p-values is {min(self.pvalues())} which is larger than the"
            msg += f" confidence level alpha = {alpha}. Try to increase the range of POI values."
            raise POIRangeError(msg)

        tck = interpolate.splrep(poinull.values, self.pvalues() - alpha, s=0)
        root = interpolate.sproot(tck)

        bands = {}
        bands["observed"] = observed

        if len(root) < 2:
            msg = f" bound on the POI `{poinull.name}` cannot not be interpolated."
            if root[0] < observed:
                msg = "Upper" + msg + " Try to increase the maximum POI value."
            else:
                msg = "Low" + msg + " Try to decrease the minimum POI value."
            raise POIRangeError(msg)

        else:
            bands["lower"] = root[0]
            bands["upper"] = root[1]

        if printlevel > 0:

            msg = f"\nConfidence interval on {poinull.name}:\n"
            msg += f"\t{root[0]} < {poinull.name} < {root[1]} at {(1-alpha)*100:.1f}% C.L."
            print(msg)

        return bands
