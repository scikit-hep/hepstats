# -*- coding: utf-8 -*-
from scipy import interpolate
import numpy as np
from typing import Union, Dict

from .basetest import BaseTest
from ..calculators.basecalculator import BaseCalculator
from ..parameters import POI, POIarray
from ..exceptions import POIRangeError


class UpperLimit(BaseTest):
    """Class for upper limit calculation."""

    def __init__(
        self,
        calculator: BaseCalculator,
        poinull: Union[POI, POIarray],
        poialt: POI,
        qtilde: bool = False,
    ):
        """
        Args:
            calculator: calculator to use for computing the pvalues.
            poinull: parameters of interest for the null hypothesis.
            poialt: parameters of interest for the alternative hypothesis.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else (default) use the :math:`q`
               test statistic.

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
            >>> peak = np.random.normal(1.2, 0.1, 10)
            >>> data = np.concatenate((bkg, peak))
            >>> data = data[(data > bounds[0]) & (data < bounds[1])]
            >>> N = data.size
            >>> data = zfit.data.Data.from_numpy(obs=obs, array=data)
            >>>
            >>> lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
            >>> Nsig = zfit.Parameter("Ns", 20., -20., N)
            >>> Nbkg = zfit.Parameter("Nbkg", N, 0., N*1.1)
            >>> signal = Nsig * zfit.pdf.Gauss(obs=obs, mu=1.2, sigma=0.1)
            >>> background = Nbkg * zfit.pdf.Exponential(obs=obs, lambda_=lambda_)
            >>> loss = ExtendedUnbinnedNLL(model=signal + background, data=data)
            >>>
            >>> from hepstats.hypotests.calculators import AsymptoticCalculator
            >>> from hepstats.hypotests import UpperLimit
            >>> from hepstats.hypotests.parameters import POI, POIarray
            >>>
            >>> calculator = AsymptoticCalculator(loss, Minuit())
            >>> poinull = POIarray(Nsig, np.linspace(0.0, 25, 20))
            >>> poialt = POI(Nsig, 0)
            >>> ul = UpperLimit(calculator, poinull, poialt)
            >>> ul.upperlimit(alpha=0.05, CLs=True)
            Observed upper limit: Nsig = 15.725784747406346
            Expected upper limit: Nsig = 11.927442041887158
            Expected upper limit +1 sigma: Nsig = 16.596396280677116
            Expected upper limit -1 sigma: Nsig = 8.592750403611896
            Expected upper limit +2 sigma: Nsig = 22.24864429383046
            Expected upper limit -2 sigma: Nsig = 6.400549971360598
        """

        super(UpperLimit, self).__init__(calculator, poinull, poialt)

        self._qtilde = qtilde

    @property
    def qtilde(self) -> bool:
        """
        Returns True if qtilde test statistic is used, else False.
        """
        return self._qtilde

    def pvalues(self, CLs: int = True) -> Dict[str, np.ndarray]:
        """
        Returns p-values scanned for the values of the parameters of interest
        in the null hypothesis.

        Args:
            CLs: if `True` uses pvalues as :math:`p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}`
               else as :math:`p_{clsb} = p_{null}`.

        Returns:
            Dictionnary of p-values for CLsb, CLs, expected (+/- sigma bands).
        """
        pvalue_func = self.calculator.pvalue

        pnull, palt = pvalue_func(
            poinull=self.poinull, poialt=self.poialt, qtilde=self.qtilde, onesided=True
        )

        pvalues = {"clsb": pnull, "clb": palt}

        sigmas = [0.0, 1.0, 2.0, -1.0, -2.0]

        exppvalue_func = self.calculator.expected_pvalue

        result = exppvalue_func(
            poinull=self.poinull,
            poialt=self.poialt,
            nsigma=sigmas,
            CLs=CLs,
            qtilde=self.qtilde,
            onesided=True,
        )

        pvalues["expected"] = result[0]
        pvalues["expected_p1"] = result[1]
        pvalues["expected_p2"] = result[2]
        pvalues["expected_m1"] = result[3]
        pvalues["expected_m2"] = result[4]

        pvalues["cls"] = pnull / palt

        return pvalues

    def upperlimit(
        self, alpha: float = 0.05, CLs: bool = True, printlevel: int = 1
    ) -> Dict[str, float]:
        """
        Returns the upper limit of the parameter of interest.

        Args:
            alpha: significance level.
            CLs: if `True` uses pvalues as :math:`p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}` else as
               :math:`p_{clsb} = p_{null}`.
            printlevel: if > 0 print the result.

        Returns:
            Dictionnary of upper limits for observed, expected (+/- sigma bands).

        """

        poinull = self.poinull

        # create a filter for -1 and -2 sigma expected limits
        bestfit = self.calculator.bestfit.params[poinull.parameter]["value"]
        filter = poinull.values >= bestfit

        if CLs:
            observed_key = "cls"
        else:
            observed_key = "clsb"

        to_interpolate = [observed_key] + [
            f"expected{i}" for i in ["", "_p1", "_m1", "_p2", "_m2"]
        ]

        limits: Dict = {}
        for k in to_interpolate:

            pvalues = self.pvalues(CLs)[k]
            values = poinull.values

            if k == observed_key:
                k = "observed"
                pvalues = pvalues[filter]
                values = values[filter]

            if min(pvalues) > alpha:
                if k in ["expected", "observed"]:
                    msg = f"The minimum of the scanned p-values is {min(pvalues)} which is larger than the"
                    msg += f" confidence level alpha = {alpha}. Try to increase the maximum POI value."
                    raise POIRangeError(msg)
                else:
                    limits[k] = None
                    continue

            tck = interpolate.splrep(values, pvalues - alpha, s=0)
            root = interpolate.sproot(tck)

            if len(root) > 1:
                root = root[0]

            try:
                limits[k] = float(root)
            except TypeError:
                limits[k] = None

        if printlevel > 0:
            print(f"\nObserved upper limit: {poinull.name} = {limits['observed']}")
            print(f"Expected upper limit: {poinull.name} = {limits['expected']}")
            for sigma in ["+1", "-1", "+2", "-2"]:
                key = sigma.replace("+", "p").replace("-", "m")
                print(
                    f"Expected upper limit {sigma} sigma: {poinull.name} = {limits[f'expected_{key}']}"
                )

        return limits
