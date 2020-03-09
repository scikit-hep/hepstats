#!/usr/bin/python
from typing import Dict, Union, Tuple, List
import numpy as np

from ..hypotests_object import HypotestsObject
from ..parameters import POI, POIarray, asarray
from ...utils.fit import pll
from ..toyutils import ToysManager
from ...utils.fit.sampling import base_sampler, base_sample


class BaseCalculator(HypotestsObject):
    """Base class for calculator.

        Args:
            * **input** : loss or fit result
            * **minimizer** : minimizer to use to find the minimum of the loss function

        Example with `zfit`:
            >>> import zfit
            >>> from zfit.core.loss import UnbinnedNLL
            >>> from zfit.minimize import Minuit

            >>> obs = zfit.Space('x', limits=(0.1, 2.0))
            >>> data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> sigma = zfit.Parameter("sigma", 0.1)
            >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
            >>> loss = UnbinnedNLL(model=model, data=data)

            >>> calc = BaseCalculator(input=loss, minimizer=Minuit())
    """

    def __init__(self, input, minimizer):

        super(BaseCalculator, self).__init__(input, minimizer)

        self._obs_nll = {}

        self._parameters = {}
        for m in self.model:
            for d in m.get_dependents():
                self._parameters[d.name] = d

    def obs_nll(self, pois) -> np.ndarray:
        """ Compute observed negative log-likelihood values for given parameters of interest.

            Args:
                * **pois** (List[`hypotests.POI`]): parameters of interest

            Returns:
                 `numpy.array`: observed nll values

            Example with `zfit`:
                >>> mean = zfit.Parameter("mu", 1.2)
                >>> poi = POI(mean, [1.1, 1.2, 1.0])
                >>> nll = calc.obs_nll(poi)

        """
        ret = np.empty(pois.shape)
        for i, p in enumerate(pois):
            if p not in self._obs_nll.keys():
                nll = pll(minimizer=self.minimizer, loss=self.loss, pois=p)
                self._obs_nll[p] = nll
            ret[i] = self._obs_nll[p]
        return ret

    def qobs(self, poinull: List[POI], onesided=True, onesideddiscovery=False, qtilde=False):
        """Computes observed values of the :math:`\Delta` log-likelihood test statistic.

            Args:
                * **poinull** (List[`hypotests.POI`]): parameters of interest for the null hypothesis
                * **qtilde** (bool, optional): if `True` use the :math:`\\tilde{q}` test statistics else (default)
                  use the :math:`q` test statistic
                * **onesided** (bool, optional): if `True` (default) computes onesided pvalues
                * **onesideddiscovery** (bool, optional): if `True` (default) computes onesided pvalues for a
                  discovery test

            Returns:
                `numpy.array`: observed values of q

            Example with `zfit`:
                >>> mean = zfit.Parameter("mu", 1.2)
                >>> poi = POI(mean, [1.1, 1.2, 1.0])
                >>> q = calc.qobs(poi)
        """

        self.check_pois(poinull)

        param = poinull.parameter
        bestfit = self.bestfit.params[param]["value"]
        if qtilde and poinull.ndim == 1:
            bestfitpoi = POI(param, 0)
        else:
            bestfitpoi = POI(param, bestfit)
            if len(poinull) == 1:
                self._obs_nll[bestfitpoi] = self.bestfit.fmin

        nll_poinull_obs = self.obs_nll(poinull)
        nll_bestfitpoi_obs = self.obs_nll(bestfitpoi)
        qobs = self.q(
            nll1=nll_poinull_obs,
            nll2=nll_bestfitpoi_obs,
            poi1=poinull,
            poi2=bestfitpoi,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
        )

        return qobs

    def pvalue(
        self,
        poinull: List[POI],
        poialt: Union[List[POI], None] = None,
        qtilde=False,
        onesided=True,
        onesideddiscovery=False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes pvalues for the null and alternative hypothesis.

        Args:
            * **poinull** (List[`hypotests.POI`]): parameters of interest for the null hypothesis
            * **poialt** (List[`hypotests.POI`], optional): parameters of interest for the alternative hypothesis
            * **qtilde** (bool, optional): if `True` use the :math:`\widetilde{q}` test statistics else (default)
              use the :math:`q` test statistic
            * **onesided** (bool, optional): if `True` (default) computes onesided pvalues
            * **onesideddiscovery** (bool, optional): if `True` (default) computes onesided pvalues for a discovery test

        Returns:
            Tuple(`numpy.array`, `numpy.array`): pnull, palt

        Example with `zfit`:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POI(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> pvalues = calc.pavalue(poinull, poialt)
        """
        self.check_pois(poinull)
        if poialt:
            self.check_pois(poialt)
            self.check_pois_compatibility(poinull, poialt)

        return self._pvalue_(
            poinull=poinull,
            poialt=poialt,
            qtilde=qtilde,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
        )

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):
        """
        To be overwritten in `BaseCalculator` subclasses.
        """
        raise NotImplementedError

    def expected_pvalue(
        self,
        poinull: List[POI],
        poialt: List[POI],
        nsigma,
        CLs=False,
        qtilde=False,
        onesided=True,
        onesideddiscovery=False,
    ) -> Dict[int, np.array]:
        """Computes the expected pvalues and error bands for different values of :math:`\sigma` (0=expected/median)

        Args:
            * **poinull** (List[`hypotests.POI`]): parameters of interest for the null hypothesis
            * **poialt** (List[`hypotests.POI`], optional): parameters of interest for the alternative hypothesis
            * **nsigma** (`numpy.array`): array of values of :math:`\sigma` to compute the expected pvalue
            * **CLs** (bool, optional): if `True` computes pvalues as :math:`p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}`
              else as :math:`p_{clsb} = p_{null}`
            * **qtilde** (bool, optional): if `True` use the :math:`\widetilde{q}` test statistics else (default)
              use the :math:`q` test statistic
            * **onesided** (bool, optional): if `True` (default) computes onesided pvalues
            * **onesideddiscovery** (bool, optional): if `True` (default) computes onesided pvalues for a discovery

        Returns:
            `numpy.array`: array of expected pvalues for each :math:`\sigma` value

        Example with `zfit`:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POI(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> nll = calc.expected_pvalue(poinull, poialt)
        """
        self.check_pois(poinull)
        if poialt:
            self.check_pois(poialt)
            self.check_pois_compatibility(poinull, poialt)

        return self._expected_pvalue_(
            poinull=poinull,
            poialt=poialt,
            nsigma=nsigma,
            CLs=CLs,
            qtilde=qtilde,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
        )

    def _expected_pvalue_(self, poinull, poialt, nsigma, CLs, qtilde, onesided, onesideddiscovery):
        """
        To be overwritten in `BaseCalculator` subclasses.
        """
        raise NotImplementedError

    def expected_poi(
        self,
        poinull: List[POI],
        poialt: List[POI],
        nsigma,
        alpha=0.05,
        CLs=False,
        onesided=True,
        onesideddiscovery=False,
    ):
        """Computes the expected parameter of interest values such that the expected p_values = :math:`\alpha` for
        different values of :math:`\sigma` (0=expected/median)

        Args:
            * **poinull** (List[`hypotests.POI`]): parameters of interest for the null hypothesis
            * **poialt** (List[`hypotests.POI`], optional): parameters of interest for the alternative hypothesis
            * **nsigma** (`numpy.array`): array of values of :math:`\sigma` to compute the expected pvalue
            * **alpha** (float, default=0.05): significance level
            * **CLs** (bool, optional): if `True` uses pvalues as :math:`p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}`
              else as :math:`p_{clsb} = p_{null}`
            * **onesided** (bool, optional): if `True` (default) computes onesided pvalues
            * **onesideddiscovery** (bool, optional): if `True` (default) computes onesided pvalues for a discovery

        Returns:
            `numpy.array`: array of expected POI values for each :math:`\sigma  value

        Example with `zfit`:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POI(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> nll = calc.expected_poi(poinull, poialt)
        """
        self.check_pois(poinull)
        if poialt:
            self.check_pois(poialt)
            self.check_pois_compatibility(poinull, poialt)

        return self._expected_poi_(
            poinull=poinull,
            poialt=poialt,
            nsigma=nsigma,
            alpha=alpha,
            CLs=CLs,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
        )

    def _expected_poi_(self, poinull, poialt, nsigma, alpha, CLs, onesided, onesideddiscovery):
        """
        To be overwritten in `BaseCalculator` subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def check_pois(pois):
        """
        Checks if the parameters of interest are all `hepstats.parameters.POI/POIarray` instances.
        """

        msg = "POI/POIarray is required."
        if not isinstance(pois, POIarray):
            raise TypeError(msg)
        if pois.ndim > 1:
            msg = "Tests with more that one parameter of interest are not yet implemented."
            raise NotImplementedError(msg)

    @staticmethod
    def check_pois_compatibility(poi1, poi2):
        """
        Checks compatibility between two lists of `hepstats.parameters.POIarray` instances.
        """

        if poi1.ndim != poi2.ndim:
            msg = f"POIs should have the same dimensions, poi1={poi1.ndim}, poi2={poi2.ndim}"
            raise ValueError(msg)

        if poi1.ndim == 1:

            if poi1.name != poi2.name:
                msg = "The variables used in the parameters of interest should have the same names,"
                msg += f" poi1={poi1.name}, poi2={poi2.name}"
                raise ValueError(msg)

    def q(
        self, nll1: np.array, nll2: np.array, poi1, poi2, onesided=True, onesideddiscovery=False
    ) -> np.ndarray:
        """Compute values of the test statistic q defined as the difference between negative log-likelihood
            values :math:`q = nll1 - nll2`.

            Args:
                * **nll1** (`numpy.array`): array of nll values #1, evaluated with poi1
                * **nll2** (`numpy.array`): array of nll values #2, evaluated with poi2
                * **poi1** ((List[`hypotests.POI`])): list of POI's #1
                * **poi2** ((List[`hypotests.POI`])): list of POI's #2
                * **onesided** (bool, optional, default=True)
                * **onesideddiscovery** (bool, optional, default=True)

            Returns:
                `numpy.array`: array of :math:`q` values
        """

        self.check_pois(poi1)
        self.check_pois(poi2)
        self.check_pois_compatibility(poi1, poi2)

        assert len(nll1) == len(poi1)
        assert len(nll2) == len(poi2)

        poi1 = poi1.values
        poi2 = poi2.values

        q = 2 * (nll1 - nll2)
        # filter_non_nan = ~(np.isnan(q) | np.isinf(q))
        # q = q[filter_non_nan]
        #
        # if isinstance(poi2, np.ndarray):
        #     poi2 = poi2[filter_non_nan]
        zeros = np.zeros(q.shape)

        if onesideddiscovery:
            condition = (poi2 < poi1) | (q < 0)
            q = np.where(condition, zeros, q)
        elif onesided:
            condition = (poi2 > poi1) | (q < 0)
            q = np.where(condition, zeros, q)
        else:
            q = q

        return q


class BaseToysCalculator(BaseCalculator):
    def __init__(self, input, minimizer, sampler, sample):
        """Basis for toys calculator class.

            Args:
                * **input** : loss or fit result
                * **minimizer** : minimizer to use to find the minimum of the loss function
                * **sampler** : function used to create sampler with models, number of events and floating parameters in the sample Default is `hepstats.fitutils.sampling.base_sampler`.
                * **sample** : function used to get samples from the sampler. Default is `hepstats.fitutils.sampling.base_sample`.
        """
        super(BaseToysCalculator, self).__init__(input, minimizer)


class ToysCalculator(BaseToysCalculator, ToysManager):
    """
    Class for calculators using toys.
    """

    def __init__(
        self, input, minimizer, ntoysnull=100, ntoysalt=100, sampler=base_sampler, sample=base_sample
    ):
        """Toys calculator class.

            Args:
                * **input** : loss or fit result
                * **minimizer** : minimizer to use to find the minimum of the loss function
                * **ntoysnull** (int, default=100): minimum number of toys to generate for the null hypothesis
                * **ntoysalt** (int, default=100): minimum number of toys to generate for the alternative hypothesis
                * **sampler** : function used to create sampler with models, number of events and floating** parameters in the sample Default is `hepstats.fitutils.sampling.base_sampler`.
                * **sample : function used to get samples from the sampler. Default is `hepstats.fitutils.sampling.base_sample`.
        """
        super(ToysCalculator, self).__init__(input, minimizer, sampler, sample)

        self._ntoysnull = ntoysnull
        self._ntoysalt = ntoysalt

    @classmethod
    def from_yaml(
        cls, filename, loss, minimizer, ntoysnull=100, ntoysalt=100, sampler=base_sampler, sample=base_sample
    ):
        """
        ToysCalculator constructor with the toys loaded from a yaml file.

        Args:
            * **filename** (str)
            * **input** : loss or fit result
            * **minimizer** : minimizer to use to find the minimum of the loss function
            * **ntoysnull** (int, default=100): minimum number of toys to generate for the null hypothesis
            * **ntoysalt** (int, default=100): minimum number of toys to generate for the alternative hypothesis
            * **sampler** : function used to create sampler with models, number of events and floating parameters in the sample Default is `hepstats.fitutils.sampling.base_sampler`.
            * **sample** : function used to get samples from the sampler. Default is `hepstats.fitutils.sampling.base_sample`.

        Returns
            ToysCalculator
        """

        calculator = cls(loss, minimizer, ntoysnull, ntoysalt, sampler, sample)
        toysresults = calculator.toysresults_from_yaml(filename)

        for t in toysresults:
            calculator.add_toyresult(t)

        return calculator

    @property
    def ntoysnull(self):
        """
        Returns the number of toys generated for the null hypothesis.
        """
        return self._ntoysnull

    @property
    def ntoysalt(self):
        """
        Returns the number of toys generated for the alternative hypothesis.
        """
        return self._ntoysalt

    def _get_toys(self, poigen, poieval=None, qtilde=False, hypothesis="null"):
        """
        Return the generated toys for a given POI.

        Args:
            * **poigen** (POI): POI used to generate the toys
            * **poieval** (POIarray): POI values to evaluate the loss function
            * **qtilde** (bool, optional): if `True` use the :math:`\tilde{q}` test statistics else (default) use the :math:`q` test statistic
            * **hypothesis** : `null` or `alternative`
        """

        assert hypothesis in ["null", "alternative"]

        if hypothesis == "null":
            ntoys = self.ntoysnull
        else:
            ntoys = self.ntoysalt

        ret = {}

        for p in poigen:
            poieval_p = poieval

            if poieval_p is None:
                poieval_p = POIarray(poigen.parameter, [p.value])
            else:
                if p not in poieval_p:
                    poieval_p = poieval_p.append(p.value)

            if qtilde and 0.0 not in poieval_p.values:
                poieval_p = poieval_p.append(0.0)

            poieval_p = asarray(poieval_p)

            ngenerated = self.ntoys(p, poieval_p)
            if ngenerated < ntoys:
                ntogen = ntoys - ngenerated
            else:
                ntogen = 0

            if ntogen > 0:
                print(f"Generating {hypothesis} hypothesis toys for {p}.")

                self.generate_and_fit_toys(ntoys=ntogen, poigen=p, poieval=poieval_p)

            ret[p] = self.get_toyresult(p, poieval_p)

        return ret

    def get_toys_null(self, poigen, poieval, qtilde=False):
        """
        Return the generated toys for the null hypothesis.

        Args:
            * **poigen** (POI): POI used to generate the toys
            * **ntoys** (int): number of toys to generate
            * **poieval** (POIarray): POI values to evaluate the loss function
            * **qtilde** (bool, optional): if `True` use the :math:`\tilde{q}` test statistics else (default) use the :math:`q` test statistic

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
            * **poigen** (POI): POI used to generate the toys
            * **ntoys** (int): number of toys to generate
            * **poieval** (POIarray): POI values to evaluate the loss function
            * **qtilde** (bool, optional): if `True` use the :math:`\tilde{q}` test statistics else (default) use the :math:`q` test statistic

        Example with `zfit`:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POIarray(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> calc.get_toys_alt(poialt, poieval=poinull)
        """
        return self._get_toys(poigen=poigen, poieval=poieval, qtilde=qtilde, hypothesis="alternative")
