#!/usr/bin/python
from typing import Dict, Union, Tuple, List
import numpy as np

from ..parameters import POI, POIarray
from ..fitutils.api_check import is_valid_loss, is_valid_fitresult, is_valid_minimizer
from ..fitutils.api_check import is_valid_data, is_valid_pdf
from ..fitutils.utils import pll

"""
Module defining the base class for the calculators for statistical tests based on the likelyhood ratio.

Any calculator can be a subclass of `BaseCalculator`. Currently implemented:

    * `AsymptoticCalculator`: calculator using the asymptotic formulaed of the likehood ratio.

Acronyms used in the code:
    * nll = negative log-likehood, the likehood being the `loss` attribute of a calculator;
    * obs = observed, i.e. measured on provided data.

"""


class BaseCalculator(object):

    def __init__(self, input, minimizer):
        """Base class for calculator.

            Args:
                input : loss or fit result
                minimizer : minimizer to use to find the minimum of the loss function

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

                >>> calc = BaseCalculator(input=loss, minimizer=MinuitMinimizer())
        """

        if is_valid_fitresult(input):
            self._loss = input.loss
            self._bestfit = input
        elif is_valid_loss(input):
            self._loss = input
            self._bestfit = None
        else:
            raise ValueError("{} is not a valid loss funtion or fit result!".format(input))

        if not is_valid_minimizer(minimizer):
            raise ValueError("{} is not a valid minimizer !".format(minimizer))

        self._minimizer = minimizer
        self.minimizer.verbosity = 0
        # cache of the observed nll values
        self._obs_nll = {}

    @property
    def loss(self):
        """
        Returns the loss / likelihood function used in the calculator.
        """
        return self._loss

    @property
    def minimizer(self):
        """
        Returns the minimzer used in the calculator.
        """
        return self._minimizer

    @property
    def bestfit(self):
        """
        Returns the best fit values of the model parameters.
        """
        if getattr(self, "_bestfit", None):
            return self._bestfit
        else:
            print("Get fit best values!")
            self.minimizer.verbosity = 5
            mininum = self.minimizer.minimize(loss=self.loss)
            self.minimizer.verbosity = 0
            self._bestfit = mininum
            return self._bestfit

    @bestfit.setter
    def bestfit(self, value):
        """
        Set the best fit values  of the model parameters.

            Args:
                value: fit result
        """
        if not is_valid_fitresult(value):
            raise ValueError()
        self._bestfit = value

    @property
    def model(self):
        """
        Returns the model used in the calculator.
        """
        return self.loss.model

    @property
    def data(self):
        """
        Returns the data used in the calculator.
        """
        return self.loss.data

    @property
    def constraints(self):
        """
        Returns the constraints on the loss / likehood function used in the calculator.
        """
        return self.loss.constraints

    def lossbuilder(self, model, data, weights=None):
        """ Method to build a new loss function.

            Args:
                model (List): The model or models to evaluate the data on
                data (List): Data to use
                weights (optional, List): the data weights

            Example with `zfit`:
                >>> data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
                >>> mean = zfit.Parameter("mu", 1.2)
                >>> sigma = zfit.Parameter("sigma", 0.1)
                >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
                >>> loss = calc.lossbuilder(model, data)

            Returns:
                Loss function

        """

        assert all(is_valid_pdf(m) for m in model)
        assert all(is_valid_data(d) for d in data)

        msg = "{0} must have the same number of components as {1}"
        if len(data) != len(self.data):
            raise ValueError(msg.format("data", "`self.data"))
        if len(model) != len(self.model):
            raise ValueError(msg.format("model", "`self.model"))
        if weights is not None and len(weights) != len(self.data):
            raise ValueError(msg.format("weights", "`self.data`"))

        fit_range = self.loss.fit_range

        if weights is not None:
            for d, w in zip(data, weights):
                d.set_weights(w)

        loss = type(self.loss)(model=model, data=data, fit_range=fit_range)
        loss.add_constraints(self.constraints)

        return loss

    def obs_nll(self, pois) -> np.array:
        """ Compute observed negative log-likelihood values for given parameters of interest.

            Args:
                pois (List[`hypotests.POI`]): parameters of interest

            Returns:
                 numpy.array`: observed nll values

            Example with `zfit`:
                >>> mean = zfit.Parameter("mu", 1.2)
                >>> poi = POI(mean, [1.1, 1.2, 1.0])
                >>> nll = calc.obs_nll([poi])

        """
        ret = np.empty(pois.shape)
        for i, p in enumerate(pois):
            if p not in self._obs_nll.keys():
                nll = pll(minimizer=self.minimizer, loss=self.loss, pois=p)
                self._obs_nll[p] = nll
            ret[i] = self._obs_nll[p]
        return ret

    def qobs(self, poinull: List[POI], onesided=True, onesideddiscovery=False, qtilde=False):
        """ Compute observed values of the $$\\Delta$$ log-likelihood test statistic.

            Args:
                poinull (List[`hypotests.POI`]): parameters of interest for the null hypothesis
                qtilde (bool, optional): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                    the $$q$$ test statistic
                onesided (bool, optional): if `True` (default) computes onesided pvalues
                onesideddiscovery (bool, optional): if `True` (default) computes onesided pvalues for a discovery
                    test

            Returns:
                `numpy.array`: observed values of q

            Example with `zfit`:
                >>> mean = zfit.Parameter("mu", 1.2)
                >>> poi = POI(mean, [1.1, 1.2, 1.0])
                >>> q = calc.qobs([poi])
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
        qobs = self.q(nll1=nll_poinull_obs, nll2=nll_bestfitpoi_obs, poi1=poinull, poi2=bestfitpoi,
                      onesided=onesided, onesideddiscovery=onesideddiscovery)

        return qobs

    def pvalue(self, poinull: List[POI], poialt: Union[List[POI], None] = None, qtilde=False, onesided=True,
               onesideddiscovery=False) -> Tuple[np.array, np.array]:
        """Computes pvalues for the null and alternative hypothesis.

        Args:
            poinull (List[`hypotests.POI`]): parameters of interest for the null hypothesis
            poialt (List[`hypotests.POI`], optional): parameters of interest for the alternative hypothesis
            qtilde (bool, optional): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                the $$q$$ test statistic
            onesided (bool, optional): if `True` (default) computes onesided pvalues
            onesideddiscovery (bool, optional): if `True` (default) computes onesided pvalues for a discovery
                test

        Returns:
            Tuple(`numpy.array`, `numpy.array`): pnull, palt

        Example with `zfit`:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POI(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> pvalues = calc.pavalue([poinull], [poialt])
        """
        self.check_pois(poinull)
        if poialt:
            self.check_pois(poialt)
            self.check_pois_compatibility(poinull, poialt)

        return self._pvalue_(poinull=poinull, poialt=poialt, qtilde=qtilde, onesided=onesided,
                             onesideddiscovery=onesideddiscovery)

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):
        """
        To be overwritten in `BaseCalculator` subclasses.
        """
        raise NotImplementedError

    def expected_pvalue(self, poinull: List[POI], poialt: List[POI], nsigma, CLs=False, qtilde=False,
                        onesided=True, onesideddiscovery=False) -> Dict[int, np.array]:
        """Computes the expected pvalues and error bands for different values of $$\\sigma$$ (0=expected/median)

        Args:
            poinull (List[`hypotests.POI`]): parameters of interest for the null hypothesis
            poialt (List[`hypotests.POI`], optional): parameters of interest for the alternative hypothesis
            nsigma (`numpy.array`): array of values of $$\\sigma$$ to compute the expected pvalue
            CLs (bool, optional): if `True` computes pvalues as $$p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}$$
                else as $$p_{clsb} = p_{null}$
            qtilde (bool, optional): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                the $$q$$ test statistic
            onesided (bool, optional): if `True` (default) computes onesided pvalues
            onesideddiscovery (bool, optional): if `True` (default) computes onesided pvalues for a discovery

        Returns:
            `numpy.array`: array of expected pvalues for each $$\\sigma$$ value

        Example with `zfit`:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POI(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> nll = calc.expected_pvalue([poinull], [poialt])
        """
        self.check_pois(poinull)
        if poialt:
            self.check_pois(poialt)
            self.check_pois_compatibility(poinull, poialt)

        return self._expected_pvalue_(poinull=poinull, poialt=poialt, nsigma=nsigma, CLs=CLs, qtilde=qtilde,
                                      onesided=onesided, onesideddiscovery=onesideddiscovery)

    def _expected_pvalue_(self, poinull, poialt, nsigma, CLs, qtilde, onesided, onesideddiscovery):
        """
        To be overwritten in `BaseCalculator` subclasses.
        """
        raise NotImplementedError

    def expected_poi(self, poinull: List[POI], poialt: List[POI], nsigma, alpha=0.05, CLs=False,
                     onesided=True, onesideddiscovery=False):
        """Computes the expected parameter of interest values such that the expected p_values == $$\alpha$$
        for different values of $$\\sigma$$ (0=expected/median)

        Args:
            poinull (List[`hypotests.POI`]): parameters of interest for the null hypothesis
            poialt (List[`hypotests.POI`], optional): parameters of interest for the alternative hypothesis
            nsigma (`numpy.array`): array of values of $$\\sigma$$ to compute the expected pvalue
            alpha (float, default=0.05): significance level
            CLs (bool, optional): if `True` uses pvalues as $$p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}$$
                else as $$p_{clsb} = p_{null}$
            onesided (bool, optional): if `True` (default) computes onesided pvalues
            onesideddiscovery (bool, optional): if `True` (default) computes onesided pvalues for a discovery

        Returns:
            `numpy.array`: array of expected POI values for each $$\\sigma$$ value

        Example with `zfit`:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POI(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> nll = calc.expected_poi([poinull], [poialt])
        """
        self.check_pois(poinull)
        if poialt:
            self.check_pois(poialt)
            self.check_pois_compatibility(poinull, poialt)

        return self._expected_poi_(poinull=poinull, poialt=poialt, nsigma=nsigma, alpha=alpha, CLs=CLs,
                                   onesided=onesided, onesideddiscovery=onesideddiscovery)

    def _expected_poi_(self, poinull, poialt, nsigma, alpha, CLs, onesided, onesideddiscovery):
        """
        To be overwritten in `BaseCalculator` subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def check_pois(pois):
        """
        Check if the parameters of interest are all `hepstats.parameters.POI/POIarray` instances.
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
        Check compatibility between two lists of `hepstats.parameters.POIarray` instances.
        """

        if poi1.ndim != poi2.ndim:
            msg = f"POIs should have the same dimensions, poi1={poi1.ndim}, poi2={poi2.ndim}"
            raise ValueError(msg)

        if poi1.ndim == 1:

            if poi1.name != poi2.name:
                msg = "The variables used in the parameters of interest should have the same names,"
                msg += f" poi1={poi1.name}, poi2={poi2.name}"
                raise ValueError(msg)

    def q(self, nll1: np.array, nll2: np.array, poi1, poi2,
          onesided=True, onesideddiscovery=False) -> np.array:
        """ Compute value of the test statistic q defined as the difference between negative log-likelihood
            values $$q = nll1 - nll2$$

            Args:
                nll1 (`numpy.array`): array of nll values #1, evaluated with poi1
                nll2 (`numpy.array`): array of nll values #2, evaluated with poi2
                poi1 ((List[`hypotests.POI`])): list of POI's #1
                poi2 ((List[`hypotests.POI`])): list of POI's #2
                onesided (bool, optional, default=True)
                onesideddiscovery (bool, optional, default=True)

            Returns:
                `np.array`: array of q values
        """

        self.check_pois(poi1)
        self.check_pois(poi2)
        self.check_pois_compatibility(poi1, poi2)

        assert len(nll1) == len(poi1)
        assert len(nll2) == len(poi2)

        poi1 = poi1.values
        poi2 = poi2.values

        q = 2*(nll1 - nll2)
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
