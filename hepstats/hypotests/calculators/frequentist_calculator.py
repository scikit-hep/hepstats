import numpy as np
from scipy.stats import norm

from .basecalculator import BaseCalculator
from ..fitutils.utils import pll, get_nevents
from ..fitutils.sampling import base_sampler, base_sample
from ..parameters import POI, POIarray
from ..toyutils import Toys, ToysCollection


class FrequentistCalculator(BaseCalculator):
    """
    Class for frequentist calculators.
    """

    def __init__(self, input, minimizer, ntoysnull=100, ntoysalt=100, sampler=base_sampler, sample=base_sample):
        """Frequentist calculator class.

            Args:
                input : loss or fit result
                minimizer : minimizer to use to find the minimum of the loss function
                ntoysnull (int, default=100): number of toys to generate for the null hypothesis
                ntoysalt (int, default=100): number of toys to generate for the alternative hypothesis
                sampler : function used to create sampler with models, number of events and
                    floating parameters in the sample Default is `hepstats.fitutils.sampling.base_sampler`.
                sample : function used to get samples from the sampler.
                    Default is `hepstats.fitutils.sampling.base_sample`.

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

        self._toysnull = {}
        self._toysalt = {}
        self._toyscollection = ToysCollection()
        self._ntoysnull = ntoysnull
        self._ntoysalt = ntoysalt
        self._sampler = sampler
        self._sample = sample
        self._toys_loss = {}

    @property
    def ntoysnull(self):
        """
        Returns the number of toys loss for the null hypothesis.
        """
        return self._ntoysnull

    @property
    def ntoysalt(self):
        """
        Returns the number of toys loss for the alternative hypothesis.
        """
        return self._ntoysalt

    def sampler(self, floating_params=None):
        """
        Create sampler with models.

        Args:
            floating_params (list): floating parameters in the sampler

        Example with `zfit`:
            >>> sampler = calc.sampler(floating_params=[zfit.Parameter("mean")])
        """
        self.set_dependents_to_bestfit()
        nevents = []
        # self.model and self.data defined in BaseCalculator
        for m, d in zip(self.model, self.data):
            if m.is_extended:
                nevents.append("extended")
            else:
                nevents.append(get_nevents(d))

        return self._sampler(self.model,  nevents, floating_params)

    def sample(self, sampler, ntoys, poi=None):
        """
        Returns the samples generated from the sampler for a given value of a parameter of interest

        Args:
            sampler (list): generator of samples
            ntoys (int): number of samples to generate
            poi (POI, optional):  in the sampler

        Example with `zfit`:
            >>> mean = zfit.Parameter("mean")
            >>> sampler = calc.sampler(floating_params=[mean])
            >>> sample = calc.sample(sampler, 1000, POI(mean, 1.2))
        """
        return self._sample(sampler, ntoys, parameter=poi.parameter, value=poi.value)

    def toys_loss(self, parameter_name):
        """
        Construct a loss function constructed with a sampler for a given floating parameter

        Args:
            parameter_name: name floating parameter in the sampler
        Returns:
             Loss function

        Example with `zfit`:
            >>> loss = calc.toys_loss(zfit.Parameter("mean"))
        """
        if parameter_name not in self._toys_loss:
            parameter = self.get_parameter(parameter_name)
            sampler = self.sampler(floating_params=[parameter])
            self._toys_loss[parameter.name] = self.lossbuilder(self.model, sampler)
        return self._toys_loss[parameter_name]

    def _generate_and_fit_toys(self, ntoys, toys, printfreq=0.2):
        """
        Generate and fit toys for at a given POI (poigen). The toys are then fitted, and the likelihood
        is profiled at the values of poigen and poieval.

        Args:
            ntoys (int): number of toys to generate
            poigen (POI): POI used to generate the toys
            poieval (POIarray, optional): POI values to evaluate the loss function
            printfreq: print frequency of the toys generation
        """

        poigen = toys.poigen
        poieval = toys.poieval

        minimizer = self.minimizer
        param = poigen.parameter

        toys_loss = self.toys_loss(poigen.name)
        sampler = toys_loss.data

        bestfit = np.empty(ntoys)
        nll_bestfit = np.empty(ntoys)
        nlls = {p: np.empty(ntoys) for p in poieval}

        printfreq = ntoys * printfreq

        samples = self.sample(sampler, int(ntoys*1.2), poigen)

        for i in range(ntoys):
            converged = False
            toprint = i % printfreq == 0
            while converged is False:
                try:
                    next(samples)
                except StopIteration:
                    to_gen = ntoys - i
                    samples = self.sample(sampler, int(to_gen*1.2), poigen)
                    next(samples)

                minimum = minimizer.minimize(loss=toys_loss)
                converged = minimum.converged

                if not converged:
                    self.set_dependents_to_bestfit()
                    continue

                bestfit[i] = minimum.params[param]["value"]
                nll_bestfit[i] = pll(minimizer, toys_loss, POI(param, bestfit[i]))

                for p in poieval:
                    nlls[p][i] = pll(minimizer, toys_loss, p)

            if toprint:
                print("{0} toys generated, fitted and scanned!".format(i))

            if i > ntoys:
                break
            i += 1

        toys.add_entries(bestfit=bestfit, nll_bestfit=nll_bestfit, nlls=nlls)

    def _get_toys(self, poigen, poieval=None, qtilde=False, hypothesis="null"):
        """
        Return the generated toys for a given POI.

        Args:
            poigen (POI): POI used to generate the toys
            poieval (POIarray): POI values to evaluate the loss function
            qtilde (bool, optional): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                the $$q$$ test statistic
            hypothesis: `null` or `alternative`
        """

        assert hypothesis in ["null", "alternative"]

        if hypothesis == "null":
            ntoys = self.ntoysnull
        else:
            ntoys = self.ntoysalt

        ret = {}

        for p in poigen:

            if poieval is None:
                poieval = POIarray(poigen.parameter, p.value)
            else:
                if p not in poieval:
                    poieval = poieval.append(p.value)

            if qtilde and 0. not in poieval:
                poieval = poieval.append(0.0)

            if (p, poieval) not in self._toyscollection:
                ntogen = ntoys
                toysresults = Toys(p, poieval)
                self._toyscollection[p, poieval] = toysresults
            else:
                ngenerated = self._toyscollection[p, poieval].ntoys
                if ngenerated < ntoys:
                    ntogen = ntoys - ngenerated
                else:
                    ntogen = 0
                toysresults = self._toyscollection[p, poieval]

            if ntogen > 0:
                print(f"Generating {hypothesis} hypothesis toys for {p}.")

                assert all(p in toysresults.poieval for p in poieval)

                self._generate_and_fit_toys(ntoys=ntogen, toys=toysresults)

                # if p in toysdict:
                #    toysdict[p].update(toysresults)
                # else:
                #    toysdict[p] = toysresults

            ret[p] = toysresults

        return ret
        #return {p: toysdict[p] for p in poigen}

    def get_toys_null(self, poigen, poieval, qtilde=False):
        """
        Return the generated toys for the null hypothesis.

        Args:
            poigen (POI): POI used to generate the toys
            ntoys (int): number of toys to generate
            poieval (POIarray): POI values to evaluate the loss function
            qtilde (bool, optional): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                the $$q$$ test statistic

        Example with `zfit`:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POIarray(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> for p in poinull:
            ...     calc.get_toys_alt(p, poieval=poialt)
        """
        return self._get_toys(poigen=poigen, poieval=poieval, qtilde=qtilde, hypothesis="null")

    def get_toys_alt(self, poigen, poieval, qtilde=False):
        """
        Return the generated toys for the alternative hypothesis.

        Args:
            poigen (POI): POI used to generate the toys
            ntoys (int): number of toys to generate
            poieval (POIarray): POI values to evaluate the loss function
            qtilde (bool, optional): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                the $$q$$ test statistic

        Example with `zfit`:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POIarray(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> calc.get_toys_alt(poialt, poieval=poinull)
        """
        return self._get_toys(poigen=poigen, poieval=poieval, qtilde=qtilde, hypothesis="alternative")

    def qnull(self, poinull, poialt, onesided, onesideddiscovery, qtilde=False):
        """ Compute null hypothesis values of the $$\\Delta$$ log-likelihood test statistic.

            Args:
                poinull (`POIarray`): parameters of interest for the null hypothesis
                poialt (`POIarray`): parameters of interest for the alternative hypothesis
                onesided (bool): if `True` computes onesided pvalues
                onesideddiscovery (bool): if `True` computes onesided pvalues for a discovery
                    test
                qtilde (bool): if `True` use the $$\tilde{q}$$ test statistics else use
                    the $$q$$ test statistic

            Returns:
                `numpy.array`: observed values of q

            Example with `zfit`:
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
                nllat0 = toysresult.nlls[0]
                nll2 = np.where(bestfit < 0, nllat0, nll2)
                bestfit = np.where(bestfit < 0, 0, bestfit)

            poi1 = POIarray(poinull.parameter, np.full(self.ntoysnull, p.value))
            poi2 = POIarray(poinull.parameter, bestfit)

            ret[p] = self.q(nll1=nll1, nll2=nll2, poi1=poi1, poi2=poi2, onesided=onesided,
                            onesideddiscovery=onesideddiscovery)

        return ret

    def qalt(self, poinull, poialt, onesided, onesideddiscovery, qtilde=False):
        """ Compute alternative hypothesis values of the $$\\Delta$$ log-likelihood test statistic.

            Args:
                poinull (`POIarray`): parameters of interest for the null hypothesis
                poialt (`POIarray`): parameters of interest for the alternative hypothesis
                onesided (bool): if `True` computes onesided pvalues
                onesideddiscovery (bool, optional): if `True` (default) computes onesided pvalues for a discovery
                    test
                qtilde (bool): if `True` use the $$\tilde{q}$$ test statistics else use
                    the $$q$$ test statistic

            Returns:
                `numpy.array`: observed values of q

            Example with `zfit`:
                >>> mean = zfit.Parameter("mu", 1.2)
                >>> poinull = POIarray(mean, [1.1, 1.2, 1.0])
                >>> poialt = POI(mean, 1.2)
                >>> q = calc.qalt(poinull, poialt)
        """
        toysresults = self.get_toys_alt(poialt, poinull, qtilde)
        ret = {}

        for p in poinull:
            toysresult = toysresults[poialt]

            nll1 = toysresult.nlls[p]
            nll2 = toysresult.nll_bestfit
            bestfit = toysresult.bestfit

            if qtilde:
                nllat0 = toysresult.nll[0]
                nll2 = np.where(bestfit < 0, nllat0, nll2)
                bestfit = np.where(bestfit < 0, 0, bestfit)

            poi1 = POIarray(poialt.parameter, np.full(self.ntoysalt, p.value))
            poi2 = POIarray(poialt.parameter, bestfit)

            ret[p] = self.q(nll1=nll1, nll2=nll2, poi1=poi1, poi2=poi2, onesided=onesided,
                            onesideddiscovery=onesideddiscovery)

        return ret

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):

        qobs = self.qobs(poinull, onesided=onesided, qtilde=qtilde,
                         onesideddiscovery=onesideddiscovery)

        def compute_pvalue(qdist, qobs):
            qdist = qdist[~(np.isnan(qdist) | np.isinf(qdist))]
            p = len(qdist[qdist >= qobs])/len(qdist)
            return p

        qnulldist = self.qnull(poinull, poialt, onesided, onesideddiscovery, qtilde)
        pnull = np.empty(len(poinull))
        for i, p in enumerate(poinull):
            pnull[i] = compute_pvalue(qnulldist[p], qobs[i])

        if poialt is not None:
            qaltdist = self.qalt(poinull, poialt, onesided, onesideddiscovery, qtilde)
            palt = np.empty(len(poinull))
            for i, p in enumerate(poinull):
                palt[i] = compute_pvalue(qaltdist[p], qobs[i])
        else:
            palt = None

        return pnull, palt

    def _expected_pvalue_(self, poinull, poialt, nsigma, CLs, onesided, onesideddiscovery, qtilde):

        ps = {ns: {"p_clsb": np.empty(len(poinull)),
                   "p_clb": np.empty(len(poinull))} for ns in nsigma}

        qnulldist = self.qnull(poinull, poialt, onesided, onesideddiscovery, qtilde)
        qaltdist = self.qalt(poinull, poialt, onesided, onesideddiscovery, qtilde)

        filter_nan = lambda q: q[~(np.isnan(q) | np.isinf(q))]

        for i, p in enumerate(poinull):

            qaltdist_p = filter_nan(qaltdist[p])
            lqaltdist = len(qaltdist_p)

            qnulldist_p = filter_nan(qnulldist[p])
            lqnulldist = len(qnulldist_p)

            p_clsb_i = np.empty(lqaltdist)
            p_clb_i = np.empty(lqaltdist)

            for j, q in np.ndenumerate(qaltdist_p):
                p_clsb_i[j] = (len(qnulldist_p[qnulldist_p >= q])/lqnulldist)
                p_clb_i[j] = (len(qaltdist_p[qaltdist_p >= q])/lqaltdist)

            for ns in nsigma:
                frac = norm.cdf(ns)*100
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
