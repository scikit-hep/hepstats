import numpy as np

from .basecalculator import BaseCalculator
from ..fitutils.utils import pll
from ..fitutils.sampling import base_sampler, base_sample
from ..parameters import POI

# TODO Think of other cases with more than one POI, only one can be ran now


class FrequentistCalculator(BaseCalculator):
    """
    Class for frequentist calculators.
    """

    def __init__(self, input, minimizer, ntoysnull=1000, ntoysalt=1000, sampler=base_sampler, sample=base_sample):
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
        self._ntoysnull = int(ntoysnull)
        self._ntoysalt = int(ntoysalt)
        self._sampler = sampler
        self._sample = sample
        self._loss_toys = {}
        self._printlevel = 1

    @property
    def ntoysnull(self):
        return self._ntoysnull

    @property
    def ntoysalt(self):
        return self._ntoysalt

    @property
    def printlevel(self):
        return self._printlevel

    def sampler(self, floatting_params=None, *args, **kwargs):
        return self._sampler(self.model, floatting_params, *args, **kwargs)

    def sample(self, sampler, ntoys, param=None, value=None):
        return self._sample(sampler, ntoys, param, value)

    def _generate_fit_toys(self, poigen, ntoys, poieval, printfreq=0.2):
        minimizer = self.minimizer

        param = poigen.parameter
        gen_value = poigen.value

        sampler = self.sampler(floatting_params=[param])

        try:
            loss_toys = self._loss_toys[param]
        except KeyError:
            loss_toys = self.lossbuilder(self.model, sampler)
            self._loss_toys[param] = loss_toys

        result = {"bestfit": {"values": np.empty(ntoys), "nll": np.empty(ntoys)},
                  "nll": {p: np.empty(ntoys) for p in poieval}}

        printfreq = ntoys * printfreq

        toys = self.sample(sampler, int(ntoys*1.2), param, gen_value)

        for i in range(ntoys):
            converged = False
            toprint = i % printfreq == 0
            while converged is False:
                try:
                    next(toys)
                except StopIteration:
                    to_gen = ntoys - i
                    toys = self.sample(sampler, int(to_gen*1.2), param, gen_value)
                    next(toys)

                minimum = minimizer.minimize(loss=loss_toys)
                converged = minimum.converged

                if not converged:
                    self.set_dependents_to_bestfit()
                    continue

                bestfit = minimum.params[param]["value"]
                result["bestfit"]["values"][i] = bestfit
                nll = pll(minimizer, loss_toys, POI(param, bestfit))
                result["bestfit"]["nll"][i] = nll

                for p in poieval:
                    nll = pll(minimizer, loss_toys, p)
                    result["nll"][p][i] = nll

            if toprint:
                print("{0} toys generated, fitted and scanned!".format(i))

            if i > ntoys:
                break
            i += 1

        return result

    def _gettoys(self, poigen, poieval=None, qtilde=False, hypotesis="null"):

        assert hypotesis in ["null", "alternative"]

        if hypotesis == "null":
            ntoys = self.ntoysnull
        else:
            ntoys = self.ntoysalt

        for p in poigen:
            if p not in self._toysresults:
                ntogen = ntoys
            else:
                ngenerated = self._toysresults[p]["bestfit"]["values"].size
                if ngenerated < ntoys:
                    ntogen = ntoys - ngenerated
                else:
                    ntogen = 0

            if ntogen > 0:
                if self.printlevel >= 0:
                    print(f"Generating {hypotesis} hypothesis toys for {p}.")

                eval_values = [p.value]
                if poieval:
                    eval_values += poieval.values_array.tolist()
                if qtilde:
                    eval_values.append(0.)
                poieval = POI(poigen.parameter, eval_values)

                toyresult = self._generate_fit_toys(p, ntogen, poieval)

                if p in self._toysresults:
                    self._toysresults[p].update(toyresult)
                else:
                    self._toysresults[p] = toyresult

            return self._toysresults[p]

    def gettoys_null(self, poigen, poieval=None, qtilde=False):
        return self._gettoys(poigen, poieval=poieval, qtilde=qtilde, hypotesis="null")

    def gettoys_alt(self, poigen, poieval=None, qtilde=False):
        return self._gettoys(poigen, poieval=poieval, qtilde=qtilde, hypotesis="alternative")

    def qnull(self, poinull, poialt, onesided, onesideddiscovery, qtilde=False):
        toysresult = self.gettoys_null(poinull, poialt, qtilde)

        nll1 = toysresult["nll"][poinull[0]]
        nll2 = toysresult["bestfit"]["nll"]
        bestfit = toysresult["bestfit"]["values"]

        if qtilde:
            nllat0 = toysresult["nll"][0]
            nll2 = np.where(bestfit < 0, nllat0, nll2)
            bestfit = np.where(bestfit < 0, 0, bestfit)

        poi1 = POI(poinull.parameter, np.full(self.ntoysnull, poinull.value))
        poi2 = POI(poinull.parameter, bestfit)

        return self.q(nll1=nll1, nll2=nll2, poi1=[poi1], poi2=[poi2],
                      onesided=onesided, onesideddiscovery=onesideddiscovery)

    def qalt(self, poinull, poialt, onesided, onesideddiscovery, qtilde=False):
        toysresult = self.gettoys_null(poialt, poinull, qtilde)

        # 1 POI
        nll1 = toysresult["nll"][poinull[0]]
        nll2 = toysresult["bestfit"]["nll"]

        if qtilde:
            nllat0 = toysresult["nll"][0]
            bestfit = toysresult["bestfit"]["values"]
            nll2 = np.where(bestfit < 0, nllat0, nll2)

        poi1 = POI(poialt.parameter, np.full(self.ntoysnull, poialt.value))
        poi2 = POI(poialt.parameter, bestfit)

        return self.q(nll1=nll1, nll2=nll2, poi1=[poi1], poi2=[poi2],
                      onesided=onesided, onesideddiscovery=onesideddiscovery)

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):

        qobs = self.qobs(poinull, onesided=onesided, qtilde=qtilde,
                         onesideddiscovery=onesideddiscovery)

        def compute_pvalue(qdist, qobs):
            qdist = qdist[~(np.isnan(qdist) | np.isinf(qdist))]
            p = len(qdist[qdist >= qobs])/len(qdist)
            return p

        needpalt = poialt is not None

        poinull = poinull[0]
        if needpalt:
            poialt = poialt[0]

        pnull = np.empty(len(poinull))
        if needpalt:
            palt = np.empty(len(poinull))
        else:
            palt = None

        for i, p in enumerate(poinull):
            qnulldist = self.qnull(p, poialt, onesided, onesideddiscovery, qtilde)
            pnull[i] = compute_pvalue(qnulldist, qobs[i])

            if needpalt:
                qaltdist = self.qalt(p, poialt, onesided, onesideddiscovery, qtilde)
                palt[i] = compute_pvalue(qaltdist, qobs[i])

        return pnull, palt
