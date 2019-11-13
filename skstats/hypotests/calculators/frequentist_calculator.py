import numpy as np

from .basecalculator import BaseCalculator
from ..fitutils.utils import eval_pdf, array2dataset, pll
from ..fitutils.sampling import base_sampler, base_sample
from ..parameters import POI


class FrequentistCalculator(BaseCalculator):
    """
    Class for frequentist calculators.
    """

    def __init__(self, input, minimizer, ntoysnull=1000, ntoysalt=1000):
        """Frequentist calculator class.

            Args:
                input : loss or fit result
                minimizer : minimizer to use to find the minimum of the loss function
                ntoysnull (int, optionnal): number of toys to generate for the null hypothesis
                ntoysalt (int, optionnal): number of toys to generate for the alternative hypothesis

            Example with `zfit`:
                >>> import zfit
                >>> from zfit.core.loss import UnbinnedNLL
                >>> from zfit.minimize import MinuitMinimizer

                >>> obs = zfit.Space('x', limits=(0.1, 2.0))
                >>> data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
                >>> mean = zfit.Parameter("mu", 1.2)
                >>> sigma = zfit.Parameter("sigma", 0.1)
                >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
                >>> loss = UnbinnedNLL(model=[model], data=[data], fit_range=[obs])

                >>> calc = FrequentistCalculator(input=loss, minimizer=MinuitMinimizer(), ntoysnull=1000, ntoysalt=1000)
        """

        super(FrequentistCalculator, self).__init__(input, minimizer)
        self._toysresults = {}
        self._ntoysnull = ntoysnull
        self._ntoysalt = ntoysalt

        self._samplers = {}
        self._loss_toys = {}

        self._sampler = None
        self._sample = None

    def sampler(self, floatting_params=None, *args, **kwargs):

        if self._sampler is None:
            return base_sampler(self.model, floatting_params, *args, **kwargs)
        else:
            return self._sampler(self.model, floatting_params, *args, **kwargs)

    def sample(self, sampler, ntoys, param=None, value=None):

        if self._sample is None:
            return base_sample(sampler, ntoys, param, value)
        else:
            return self._sample(sampler, ntoys, param, value)

    def dotoys(self, poigen, ntoys, poieval, printfreq=0.2):
        minimizer = self.minimizer

        poigen_param = poigen.parameter
        poigen_value = poigen.value

        try:
            sampler = self._samplers[poigen_param]
        except KeyError:
            sampler = self.sampler(floatting_params=[poigen_param])
            self._samplers[poigen_param] = sampler

        try:
            loss_toys = self._loss_toys[poigen_param]
        except KeyError:
            loss_toys = self.lossbuilder(self.model, sampler)
            self.loss_toys[poigen_param] = loss_toys

        result = {"bestfit": {"values": np.empty(ntoys), "nll": np.empty(ntoys)}}

        result["nll"] = {p: np.empty(ntoys) for p in poieval}

        printfreq = ntoys * printfreq

        toys = self.sample(sampler, int(ntoys*1.2), poigen_param, poigen_value)

        for i in range(ntoys):
            converged = False
            toprint = i % printfreq == 0
            while converged is False:
                try:
                    next(toys)
                except StopIteration:
                    to_gen = ntoys - i
                    toys = self.config.sample(sampler, int(to_gen*1.2), poigen_param, poigen_value)
                    next(toys)

                bestfit = minimizer.minimize(loss=loss_toys)
                converged = bf.converged

                if not converged:
                    config.deps_to_bestfit()
                    continue

                bestfit = bestfit.params[g_param]["value"]
                result["bestfit"]["values"][i] = bestfit
                nll = config.pll(minimizer, loss_toys, g_param, bestfit)
                result["bestfit"]["nll"][i] = nll

                for p in poieval:
                    param_ = p.parameter
                    val_ = p.value
                    nll = config.pll(minimizer, loss_toys, param_, val_)
                    result["nll"][p][i] = nll

            if toprint:
                print("{0} toys generated, fitted and scanned!".format(i))

            if i > ntoys:
                break
            i += 1

        return result

    def dotoys_null(self, poinull, poialt=None, qtilde=False, printlevel=1):

        ntoys = self._ntoysnull

        for p in poinull:
            if p in self._toysresults.keys():
                continue
            msg = "Generating null hypothesis toys for {0}."
            if printlevel >= 0:
                print(msg.format(p))

            toeval = [p]
            if poialt is not None:
                for palt in poialt:
                    toeval.append(palt)
            if qtilde:
                poi0 = POI(poinull.parameter, 0.)
                if poi0 not in toeval:
                    toeval.append(poi0)

            toyresult = self.dotoys(p, ntoys, toeval)

            self._toysresults[p] = toyresult

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):

        qobs = self.qobs(poinull, onesided=onesided, qtilde=qtilde,
                         onesideddiscovery=onesideddiscovery)

        def pvalue_i(qdist, qobs):
            qdist = qdist[~(np.isnan(qdist) | np.isinf(qdist))]
            p = len(qdist[qdist >= qobs])/len(qdist)
            return p

        self.dotoys_null(poinull, poialt, qtilde)

        needpalt = poialt is not None

        if needpalt:
            self.dotoys_alt(poialt, poinull, qtilde)

        pnull = np.empty(len(poinull))
        if needpalt:
            palt = np.empty(len(poinull))
        else:
            palt = None

        for i, p in enumerate(poinull):
            qnulldist = self.qnull(p, qtilde)
            bestfitnull = self.poi_bestfit(p, qtilde)
            qnulldist = self.qdist(qnulldist, bestfitnull, p.value,
                                   onesided=onesided,
                                   onesideddiscovery=onesideddiscovery)
            pnull[i] = pvalue_i(qnulldist, qobs[i])
            if needpalt:
                qaltdist = self.qalt(p, poialt, qtilde)
                bestfitalt = self.poi_bestfit(poialt, qtilde)
                qaltdist = self.qdist(qaltdist, bestfitalt, p.value,
                                      onesided=onesided,
                                      onesideddiscovery=onesideddiscovery)
                palt[i] = pvalue_i(qaltdist, qobs[i])

        return pnull, palt
