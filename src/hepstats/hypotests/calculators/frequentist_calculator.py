# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
from typing import Callable, Union, Optional

from .basecalculator import ToysCalculator
from ...utils import base_sampler, base_sample
from ..parameters import POI, POIarray


class FrequentistCalculator(ToysCalculator):
    """Frequentist calculator class."""

    def __init__(
        self,
        input,
        minimizer,
        ntoysnull: int = 100,
        ntoysalt: int = 100,
        sampler: Callable = base_sampler,
        sample: Callable = base_sample,
    ):
        """
        Args:
            input: loss or fit result
            minimizer: minimizer to use to find the minimum of the loss function.
            ntoysnull: minimum number of toys to generate for the null hypothesis.
            ntoysalt: minimum number of toys to generate for the alternative hypothesis.
            sampler: function used to create sampler with models, number of events and floating parameters
              in the sample. Default is :func:`hepstats.utils.fit.sampling.base_sampler`.
            sample: function used to get samples from the sampler. Default is
              :func:`hepstats.utils.fit.sampling.base_sample`.

        Example with **zfit**:
            >>> import zfit
            >>> from zfit.core.loss import UnbinnedNLL
            >>> from zfit.minimize import MinuitMinimizer
            >>>
            >>> obs = zfit.Space('x', limits=(0.1, 2.0))
            >>> data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> sigma = zfit.Parameter("sigma", 0.1)
            >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
            >>> loss = UnbinnedNLL(model=[model], data=[data], fit_range=[obs])
            >>>
            >>> calc = FrequentistCalculator(input=loss, minimizer=MinuitMinimizer(), ntoysnull=1000, ntoysalt=1000)
        """
        super(FrequentistCalculator, self).__init__(
            input=input,
            minimizer=minimizer,
            ntoysnull=ntoysnull,
            ntoysalt=ntoysalt,
            sampler=sampler,
            sample=sample,
        )

    def qnull(
        self,
        poinull: Union[POI, POIarray],
        poialt: Optional[POI] = None,
        onesided: bool = True,
        onesideddiscovery: bool = False,
        qtilde: bool = False,
    ):
        """Computes null hypothesis values of the :math:`\\Delta` log-likelihood test statistic.

        Args:
            poinull: parameters of interest for the null hypothesis.
            poialt: parameters of interest for the alternative hypothesis.
            onesided: if `True` computes onesided pvalues.
            onesideddiscovery: if `True` computes onesided pvalues for a discovery test.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else use the :math:`q`
              test statistic.

        Returns:
            Q distribution for the null hypothesis.

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POIarray(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> q = calc.qnull(poinull, poialt)
        """
        toysresults = self.get_toys_null(poinull, poialt, qtilde)
        ret = {}

        for p in poinull:
            toysresult = toysresults[p]
            nll1 = toysresult.nlls[p]
            nll2 = toysresult.nll_bestfit
            bestfit = toysresult.bestfit

            if qtilde:
                nllat0 = toysresult.nlls[POI(poinull.parameter, 0.0)]
                nll2 = np.where(bestfit < 0, nllat0, nll2)
                bestfit = np.where(bestfit < 0, 0, bestfit)

            poi1 = POIarray(poinull.parameter, np.full(nll1.size, p.value))
            poi2 = POIarray(poinull.parameter, bestfit)

            q_p = self.q(
                nll1=nll1,
                nll2=nll2,
                poi1=poi1,
                poi2=poi2,
                onesided=onesided,
                onesideddiscovery=onesideddiscovery,
            )

            ret[p] = q_p[~np.isnan(q_p)]

        return ret

    def qalt(
        self,
        poinull: Union[POI, POIarray],
        poialt: POI,
        onesided: bool = True,
        onesideddiscovery: bool = False,
        qtilde: bool = False,
    ):
        """Computes alternative hypothesis values of the :math:`\\Delta` log-likelihood test statistic.

        Args:
            poinull: parameters of interest for the null hypothesis.
            poialt: parameters of interest for the alternative hypothesis.
            onesided: if `True` computes onesided pvalues.
            onesideddiscovery: if `True` computes onesided pvalues for a discovery test.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else use the :math:`q`
              test statistic.

        Returns:
            Q distribution for the alternative hypothesis.

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POIarray(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> q = calc.qalt(poinull, poialt)
        """
        toysresult = self.get_toys_alt(poialt, poinull, qtilde)[poialt]
        ret = {}

        for p in poinull:
            nll1 = toysresult.nlls[p]
            nll2 = toysresult.nll_bestfit
            bestfit = toysresult.bestfit

            if qtilde:
                nllat0 = toysresult.nlls[POI(poinull.parameter, 0.0)]
                nll2 = np.where(bestfit < 0, nllat0, nll2)
                bestfit = np.where(bestfit < 0, 0, bestfit)

            poi1 = POIarray(poialt.parameter, np.full(nll1.size, p.value))
            poi2 = POIarray(poialt.parameter, bestfit)

            q_p = self.q(
                nll1=nll1,
                nll2=nll2,
                poi1=poi1,
                poi2=poi2,
                onesided=onesided,
                onesideddiscovery=onesideddiscovery,
            )

            ret[p] = q_p[~np.isnan(q_p)]

        return ret

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):

        qobs = self.qobs(
            poinull,
            onesided=onesided,
            qtilde=qtilde,
            onesideddiscovery=onesideddiscovery,
        )

        def compute_pvalue(qdist, qobs):
            qdist = qdist[~(np.isnan(qdist) | np.isinf(qdist))]
            p = len(qdist[qdist >= qobs]) / len(qdist)
            return p

        qnulldist = self.qnull(
            poinull=poinull,
            poialt=poialt,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
            qtilde=qtilde,
        )
        pnull = np.empty(len(poinull))
        for i, p in enumerate(poinull):
            pnull[i] = compute_pvalue(qnulldist[p], qobs[i])

        if poialt is not None:
            qaltdist = self.qalt(
                poinull=poinull,
                poialt=poialt,
                onesided=onesided,
                onesideddiscovery=onesideddiscovery,
                qtilde=qtilde,
            )
            palt = np.empty(len(poinull))
            for i, p in enumerate(poinull):
                palt[i] = compute_pvalue(qaltdist[p], qobs[i])
        else:
            palt = None

        return pnull, palt

    def _expected_pvalue_(
        self, poinull, poialt, nsigma, CLs, onesided, onesideddiscovery, qtilde
    ):

        ps = {
            ns: {"p_clsb": np.empty(len(poinull)), "p_clb": np.empty(len(poinull))}
            for ns in nsigma
        }

        qnulldist = self.qnull(
            poinull=poinull,
            poialt=poialt,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
            qtilde=qtilde,
        )
        qaltdist = self.qalt(
            poinull=poinull,
            poialt=poialt,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
            qtilde=qtilde,
        )

        filter_nan = lambda q: q[~(np.isnan(q) | np.isinf(q))]

        for i, p in enumerate(poinull):

            qaltdist_p = filter_nan(qaltdist[p])
            lqaltdist = len(qaltdist_p)

            qnulldist_p = filter_nan(qnulldist[p])
            lqnulldist = len(qnulldist_p)

            p_clsb_i = np.empty(lqaltdist)
            p_clb_i = np.empty(lqaltdist)

            for j, q in np.ndenumerate(qaltdist_p):
                p_clsb_i[j] = len(qnulldist_p[qnulldist_p >= q]) / lqnulldist
                p_clb_i[j] = len(qaltdist_p[qaltdist_p >= q]) / lqaltdist

            for ns in nsigma:
                frac = norm.cdf(ns) * 100
                ps[ns]["p_clsb"][i] = np.percentile(p_clsb_i, frac)
                ps[ns]["p_clb"][i] = np.percentile(p_clb_i, frac)

        expected_pvalues = []
        for ns in nsigma:
            if CLs:
                p_cls = ps[ns]["p_clsb"] / ps[ns]["p_clb"]
                expected_pvalues.append(np.where(p_cls < 0, 0, p_cls))
            else:
                p_clsb = ps[ns]["p_clsb"]
                expected_pvalues.append(np.where(p_clsb < 0, 0, p_clsb))

        return expected_pvalues
