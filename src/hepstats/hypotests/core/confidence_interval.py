# -*- coding: utf-8 -*-
from scipy import interpolate
import numpy as np
import warnings
from typing import Dict

from .basetest import BaseTest
from ..exceptions import POIRangeError
from ..parameters import POIarray
from ..calculators.basecalculator import BaseCalculator
from ..calculators import FrequentistCalculator


class ConfidenceInterval(BaseTest):
    """Class for confidence interval calculation."""

    def __init__(
        self, calculator: BaseCalculator, poinull: POIarray, qtilde: bool = False
    ):
        """
        Args:
            calculator: calculator to use for computing the pvalues.
            poinull: parameters of interest for the null hypothesis.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else (default)
               use the :math:`q` test statistic.

        Example with **zfit**:
            >>> import numpy as np
            >>> import zfit
            >>> from zfit.loss import ExtendedUnbinnedNLL
            >>> from zfit.minimize import Minuit
            >>>
            >>> bounds = (0.1, 3.0)
            >>> zfit.Space('x', limits=bounds)
            >>>
            >>> bkg = np.random.exponential(0.5, 300)
            >>> peak = np.random.normal(1.2, 0.1, 80)
            >>> data = np.concatenate((bkg, peak))
            >>> data = data[(data > bounds[0]) & (data < bounds[1])]
            >>> N = data.size
            >>> data = zfit.data.Data.from_numpy(obs=obs, array=data)
            >>>
            >>> mean = zfit.Parameter("mean", 1.2, 0.5, 2.0)
            >>> sigma = zfit.Parameter("sigma", 0.1, 0.02, 0.2)
            >>> lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
            >>> Nsig = zfit.Parameter("Ns", 20., -20., N)
            >>> Nbkg = zfit.Parameter("Nbkg", N, 0., N*1.1)
            >>> signal = Nsig * zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
            >>> background = Nbkg * zfit.pdf.Exponential(obs=obs, lambda_=lambda_)
            >>> loss = ExtendedUnbinnedNLL(model=signal + background, data=data)
            >>>
            >>> from hepstats.hypotests.calculators import AsymptoticCalculator
            >>> from hepstats.hypotests import ConfidenceInterval
            >>> from hepstats.hypotests.parameters import POI, POIarray
            >>>
            >>> calculator = AsymptoticCalculator(loss, Minuit())
            >>> poinull = POIarray(mean, np.linspace(1.15, 1.26, 100))
            >>> ci = ConfidenceInterval(calculator, poinull)
            >>> ci.interval()
            Confidence interval on mean:
                1.1810371356602791 < mean < 1.2156701172321935 at 68.0% C.L.
        """
        super(ConfidenceInterval, self).__init__(calculator, poinull)

        self._qtilde = qtilde

    @property
    def qtilde(self) -> bool:
        """
        Returns True if qtilde test statistic is used, else False.
        """
        return self._qtilde

    def pvalues(self) -> np.ndarray:
        """
        Returns p-values scanned for the values of the parameters of interest
        in the null hypothesis.

        Returns:
            Array of p-values for CLsb, CLs, expected (+/- sigma bands).
        """

        poialt = None
        return self.calculator.pvalue(
            poinull=self.poinull, poialt=poialt, qtilde=self.qtilde, onesided=False
        )[0]

    def interval(self, alpha: float = 0.32, printlevel: int = 1) -> Dict[str, float]:
        """
        Returns the confidence level on the parameter of interest.

        Args:
            alpha: significance level.
            printlevel: if > 0 print the result.

        Returns:
            Dict of the values for the central, upper and lower bounds on the parameter of interest.

        """

        bands = {}
        poinull = self.poinull
        observed = self.calculator.bestfit.params[poinull.parameter]["value"]
        bands["observed"] = observed

        if min(self.pvalues()) > alpha:
            msg = f"The minimum of the scanned p-values is {min(self.pvalues())} which is larger than the"
            msg += f" confidence level alpha = {alpha}. Try to increase the range of POI values."
            raise POIRangeError(msg)

        tck = interpolate.splrep(poinull.values, self.pvalues() - alpha, s=0)
        roots = np.array(interpolate.sproot(tck))

        msg = f" bound on the POI `{poinull.name}` cannot not be interpolated."

        if roots.size > 2:
            msg_warn = "Multiple roots have been founds."
            if isinstance(self.calculator, FrequentistCalculator):
                msg_warn += " Try to increase the number of toys, 'ntoysnull', to reduce fluctuations."
            warnings.warn(msg_warn)

        lower_roots = roots[roots < observed]
        upper_roots = roots[roots > observed]

        if upper_roots.size == 0:
            msg = "Upper" + msg + " Try to increase the maximum POI value."
            raise POIRangeError(msg)
        else:
            bands["upper"] = max(upper_roots)

        if lower_roots.size == 0:
            if self.qtilde:
                bands["lower"] = 0.0
            else:
                msg = "Low" + msg + " Try to decrease the minimum POI value."
                raise POIRangeError(msg)
        else:
            bands["lower"] = min(lower_roots)

            if self.qtilde and bands["lower"] < 0.0:
                bands["lower"] = 0.0

        if printlevel > 0:

            msg = f"\nConfidence interval on {poinull.name}:\n"
            msg += f"\t{bands['lower']} < {poinull.name} < {bands['upper']} at {(1-alpha)*100:.1f}% C.L."
            print(msg)

        return bands
