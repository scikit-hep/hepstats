#!/usr/bin/python
from typing import Dict, Union, Tuple, List

import numpy as np
from .parameters import POI
from contextlib import ExitStack

# nll = negative log-likelihood
# obs = observed

# API checks
def is_valid_pdf(object):

    has_params = hasattr(object, "params")
    has_pdf = hasattr(object, "pdf")
    has_integrate = hasattr(object, "integrate")
    has_sample = hasattr(object, "sample")

    return has_params and has_pdf and has_integrate and has_sample


def is_valid_loss(object):
    has_model = hasattr(object, "model")

    if not has_model:
        return False
    else:
        model = object.model

    has_data = hasattr(object, "data")
    has_constraints = hasattr(object, "constraints")
    has_fit_range = hasattr(object, "fit_range")
    all_valid_pdfs = all(is_valid_pdf(m) for m in model)

    return all_valid_pdfs and has_data and has_constraints and has_fit_range


def is_valid_fitresult(object):
    has_loss = hasattr(object, "loss")

    if not has_loss:
        return False
    else:
        loss = object.loss
        has_params = hasattr(object, "params")

        return is_valid_loss(loss) and has_params


def is_valid_minimizer(object):
    has_minimize = hasattr(object, "minimize")

    return has_minimize


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

    def __init__(self, input, minimizer):
        """Base class for calculator.

            Args:
                input : loss or fit result
                minimizer : minimizer to use to find the minimum of the loss function
        """

        if is_valid_fitresult(input):
            self._loss = input.loss
            self._bestfit = input
        elif is_valid_loss(input):
            self._loss = input
            self._bestfit = None
        else:
            raise ValueError(input + " is not a valid loss funtion or fit result!")

        if not is_valid_minimizer(minimizer):
            raise ValueError(minimizer + " is not a valid minimizer !")

        self._minimizer = minimizer
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
                value: fit result
        """
        if not is_valid_fitresult(value):
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

        if weights is not None:
            for d, w in zip(data, weights):
                d.set_weights(w)

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
