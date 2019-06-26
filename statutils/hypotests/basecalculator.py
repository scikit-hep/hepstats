#!/usr/bin/python
from typing import Dict, Union, Optional, Tuple, List

import numpy as np
from .parameters import POI
from zfit.core.loss import BaseLoss
from zfit.data import Data
from zfit.minimizers.minimizer_minuit import MinuitMinimizer
from zfit.minimizers.fitresult import FitResult
from zfit.minimizers.baseminimizer import BaseMinimizer
from contextlib import ExitStack


def q(cls, nll1: np.array, nll2: np.array, bestfit, poival, onesided=True,
      onesideddiscovery=False) -> np.array:
    """ Compute difference between log-likelihood values."""
    q = nll1 - nll2
    sel = ~(np.isnan(q) | np.isinf(q))
    q = q[sel]
    if isinstance(bestfit, np.ndarray):
        bestfit = bestfit[sel]
    zeros = np.zeros(q.shape)

    if onesideddiscovery:
        condition = (bestfit < poival) | (q < 0)
        q = np.where(condition, zeros, q)
    elif onesided:
        condition = (bestfit > poival) | (q < 0)
        q = np.where(condition, zeros, q)
    else:
        q = q

    return q


def pll(minimizer, loss, pois) -> float:
    """ Compute minimum profile likelihood for given parameters values. """
    with ExitStack() as stack:
        for p in pois:
            param = p.parameter
            stack.enter_context(param.set_value(p.value))
            param.floating = False
        minimum = minimizer.minimize(loss=loss)
        for p in pois:
            p.parameter.floating = True
    return minimum.fmin


class BaseCalculator(object):

    def __init__(self, input: Union[BaseLoss, FitResult], minimizer: Optional[BaseMinimizer] = MinuitMinimizer):
        """Base class for calculator.

            Args:
                input (`zfit.core.BaseLoss`/`zfit.minimizers.fitresult import FitResult`): loss
                minimizer (`zfit.minimizers.baseminimizer.BaseMinimizer`, optionnal): minimizer to use to find
                    loss function minimun
        """

        if isinstance(input, FitResult):
            self._loss = input.loss
            self._bestfit = input
        else:
            self._loss = input
            self._bestfit = None
        self._minimizer = minimizer()
        self.minimizer.verbosity = 0

        # cache of the observed nll values
        self._obs_nll = {}

    @property
    def loss(self):
        return self._loss

    @property
    def minimizer(self):
        return self._minimizer

    @property
    def bestfit(self):
        """
        Returns the best fit values.
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
        Set the best fit values.

            Args:
                value (`zfit.minimizers.fitresult import FitResult`)
        """
        if not isinstance(value, FitResult):
            raise ValueError()
        self._bestfit = value

    @property
    def model(self):
        return self.loss.model

    @property
    def data(self):
        return self.loss.data

    @property
    def constraints(self):
        return self.loss.constraints

    def loss_builder(self, model, data, weights=None):

        msg = "{0} must have the same number of components as {1}"
        if len(data) != len(self.data):
            raise ValueError(msg.format("data", "`self.data"))
        if len(model) != len(self.model):
            raise ValueError(msg.format("model", "`self.model"))
        if weights is not None and len(weights) != len(self.data):
            raise ValueError(msg.format("weights", "`self.data`"))

        fit_range = self.loss.fit_range

        if all(isinstance(d, (Data)) for d in data):
            if weights is not None:
                for d, w in zip(data, weights):
                    d.set_weights(w)
        elif all(isinstance(d, (np.ndarray)) for d in data):
            data_converted = []
            if weights is None:
                weights = [None]*len(data)
            for d, w in zip(data, weights):
                data_zfit = Data.from_numpy(obs=fit_range, array=d, weights=w)
                data_converted.append(data_zfit)
            data = data_converted
        else:
            raise ValueError("data must be `zfit.data.Data` or a numpy array")

        loss = type(self.loss)(model=model, data=data, fit_range=fit_range)
        loss.add_constraints(self.constraints)

        return loss

    def obs_nll(self, pois: List[POI]) -> np.array:
        """ Compute observed negative log-likelihood values."""
        grid = np.array([g.ravel() for g in np.meshgrid(*pois)]).T
        ret = np.empty(len(grid))
        for i, g in enumerate(grid):
            k = tuple(g)
            if k not in self._obs_nll.keys():
                nll = pll(minimizer=self.minimizer, loss=self.loss, pois=g)
                self._obs_nll[k] = nll
            ret[i] = self._obs_nll[k]
        return ret

    def qobs(self, poinull: List[POI], onesided=True, onesideddiscovery=False, qtilde=False):
        """ Compute observed $$\\Delta$$ log-likelihood values."""
        print("Compute qobs for the null hypothesis!")

        params = [p.parameter for p in poinull]
        bestfit = [self.bestfit.params[p]["value"] for p in params]
        bestfitpoi = []
        for param, bf in zip(params, bestfit):
            if qtilde and len(poinull) == 1:
                bestfitpoi.append(POI(param, 0))
            else:
                bestfitpoi.append(POI(param, bf))
                if len(poinull) == 1:
                    self._obs_nll[bestfitpoi] = self.bestfit.fmin

        nll_poinull_obs = self.obs_nll(poinull)
        nll_bestfitpoi_obs = self.obs_nll(bestfitpoi)
        qobs = q(nll1=nll_poinull_obs, nll2=nll_bestfitpoi_obs, bestfit=bestfitpoi.value,
                 poival=poinull.value, onesided=onesided, onesideddiscovery=onesideddiscovery)

        return qobs

    def pvalue(self, poinull: List[POI], poialt: Union[List[POI], None] = None, qtilde=False, onesided=True,
               onesideddiscovery=False) -> Tuple[np.array, np.array]:
        """Computes pvalues for the null and alternative hypothesis.

        Args:
            poinull (Iterable[`hypotests.POI`]): parameters of interest for the null hypothesis
            poialt (Iterable[`hypotests.POI`], optionnal): parameters of interest for the alternative hypothesis
            qtilde (bool, optionnal): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                the $$q$$ test statistic
            onesided (bool, optionnal): if `True` (default) computes onesided pvalues
            onesideddiscovery (bool, optionnal): if `True` (default) computes onesided pvalues for a discovery
                test
        Returns:
            Tuple(`numpy.array`, `numpy.array`): pnull, palt
        """
        return self._pvalue(poinull=poinull, poialt=poialt, qtilde=qtilde, onesided=onesided,
                            onesideddiscovery=onesideddiscovery)

    def _pvalue(self, poinull, poialt, qtilde, onesided, onesideddiscovery):

        raise NotImplementedError

    def expected_pvalue(self, poinull: List[POI], poialt: List[POI], nsigma, CLs=False) -> Dict[int, np.array]:
        """Computes the expected pvalues and error bands for different values of $$\\sigma$$ (0=expected/median)

        Args:
            poinull (Iterable[`hypotests.POI`]): parameters of interest for the null hypothesis
            poialt (Iterable[`hypotests.POI`], optionnal): parameters of interest for the alternative hypothesis
            nsigma (`numpy.array`): array of values of $$\\sigma$$ to compute the expected pvalue
            CLs (bool, optionnal): if `True` computes pvalues as $$p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}$$
                else as $$p_{clsb} = p_{null}$

        Returns:
            Dict($$\\sigma$$, `numpy.array`): dictionnary of pvalue arrays for each $$\\sigma$$ value
        """
        return self._expected_pvalue(poinull=poinull, poialt=poialt, nsigma=nsigma, CLs=CLs)

    def _expected_pvalue(self, poinull, poialt, nsigma, CLs):

        raise NotImplementedError

    def expected_poi(self, poinull: List[POI], poialt: List[POI], nsigma, alpha=0.05, CLs=False):
        """Computes the expected parameter of interest values such that the expected p_values == $$\alpha$$
        for different values of $$\\sigma$$ (0=expected/median)

        Args:
            poinull (Iterable[`hypotests.POI`]): parameters of interest for the null hypothesis
            poialt (Iterable[`hypotests.POI`], optionnal): parameters of interest for the alternative hypothesis
            nsigma (`numpy.array`): array of values of $$\\sigma$$ to compute the expected pvalue
            CLs (bool, optionnal): if `True` uses pvalues as $$p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}$$
                else as $$p_{clsb} = p_{null}$

        """
        return self._expected_poi(poinull=poinull, poialt=poialt, nsigma=nsigma, alpha=alpha, CLs=CLs)

    def _expected_poi(self, poinull, poialt, nsigma, alpha, CLs):

        raise NotImplementedError
