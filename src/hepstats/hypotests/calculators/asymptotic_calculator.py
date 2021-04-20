# -*- coding: utf-8 -*-
from typing import Tuple, Union, Dict, Any
import numpy as np
from scipy.stats import norm
import warnings

from .basecalculator import BaseCalculator
from ...utils import eval_pdf, array2dataset, pll, get_value
from ..parameters import POI, POIarray


def generate_asimov_hist(
    model, params: Dict[Any, Dict[str, Any]], nbins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the Asimov histogram using a model and dictionary of parameters.

    Args:
        model: model used to generate the dataset.
        params: values of the parameters of the models.
        nbins: number of bins.

    Returns:
        Tuple of hist and bin_edges.

    Example with **zfit**:
        >>> obs = zfit.Space('x', limits=(0.1, 2.0))
        >>> mean = zfit.Parameter("mu", 1.2)
        >>> sigma = zfit.Parameter("sigma", 0.1)
        >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
        >>> hist, bin_edges = generate_asimov_hist(model, {"mean": 1.2, "sigma": 0.1})
    """

    space = model.space
    bounds = space.limit1d
    bin_edges = np.linspace(*bounds, nbins + 1)
    bin_centers = bin_edges[0:-1] + np.diff(bin_edges) / 2

    hist = eval_pdf(model, bin_centers, params, allow_extended=True)
    hist *= space.area() / nbins

    return hist, bin_edges


class AsymptoticCalculator(BaseCalculator):
    """
    Class for asymptotic calculators, using asymptotic formulae of the likelihood ratio described in
    :cite:`Cowan:2010js`. Can be used only with one parameter of interest.
    """

    def __init__(self, input, minimizer, asimov_bins: int = 100):
        """Asymptotic calculator class.

        Args:
            input: loss or fit result.
            minimizer: minimizer to use to find the minimum of the loss function.
            asimov_bins: number of bins of the Asimov dataset.

        Example with **zfit**:
            >>> import zfit
            >>> from zfit.core.loss import UnbinnedNLL
            >>> from zfit.minimize import Minuit
            >>>
            >>> obs = zfit.Space('x', limits=(0.1, 2.0))
            >>> data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> sigma = zfit.Parameter("sigma", 0.1)
            >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
            >>> loss = UnbinnedNLL(model=model, data=data)
            >>>
            >>> calc = AsymptoticCalculator(input=loss, minimizer=Minuit(), asimov_bins=100)
        """

        super(AsymptoticCalculator, self).__init__(input, minimizer)
        self._asimov_bins = asimov_bins
        self._asimov_dataset: Dict = {}
        self._asimov_loss: Dict = {}
        # cache of nll values computed with the asimov dataset
        self._asimov_nll: Dict[POI, np.ndarray] = {}

    @staticmethod
    def check_pois(pois: Union[POI, POIarray]):
        """
        Checks if the parameter of interest is a :class:`hepstats.parameters.POIarray` instance.

        Args:
            pois: the parameter of interest to check.

        Raises:
            TypeError: if pois is not an instance of :class:`hepstats.parameters.POIarray`.
        """

        msg = "POI/POIarray is required."
        if not isinstance(pois, POIarray):
            raise TypeError(msg)
        if pois.ndim > 1:
            msg = "Tests using the asymptotic calculator can only be used with one parameter of interest."
            raise NotImplementedError(msg)

    def asimov_dataset(self, poi: POI, ntrials_fit: int = 5):
        """Gets the Asimov dataset for a given alternative hypothesis.

        Args:
            poi: parameter of interest of the alternative hypothesis.
            ntrials_fit: maximum number of fits to perform

        Returns:
             The asymov dataset.

        Example with **zfit**:
            >>> poialt = POI(mean, 1.2)
            >>> dataset = calc.asimov_dataset(poialt)

        """
        if poi not in self._asimov_dataset.keys():
            model = self.model
            data = self.data
            minimizer = self.minimizer
            oldverbose = minimizer.verbosity
            minimizer.verbosity = 0

            poiparam = poi.parameter
            poivalue = poi.value

            msg = "\nGet fitted values of the nuisance parameters for the"
            msg += " alternative hypothesis!"
            print(msg)

            self.set_params_to_bestfit()
            poiparam.floating = False

            if not self.loss.get_params():
                values = {poiparam: {"value": poivalue}}

            else:
                with poiparam.set_value(poivalue):
                    for trial in range(5):
                        minimum = minimizer.minimize(loss=self.loss)
                        if minimum.valid:
                            break
                        else:
                            # shift other parameter values to change starting point of minimization
                            for p in self.parameters:
                                if p != poiparam:
                                    p.set_value(
                                        get_value(p) * np.random.normal(1, 0.02, 1)[0]
                                    )
                    else:
                        msg = "No valid minimum was found when fitting the loss function for the alternative"
                        msg += f"hypothesis ({poi}), after {ntrials_fit} trials."
                        warnings.warn(msg)

                print(minimum)

                values = dict(minimum.params)
                values[poiparam] = {"value": poivalue}

            poiparam.floating = True
            minimizer.verbosity = oldverbose

            asimov_data = []

            if not isinstance(self._asimov_bins, list):
                asimov_bins = [self._asimov_bins] * len(data)
            else:
                asimov_bins = self._asimov_bins
                assert len(asimov_bins) == len(data)

            for i, (m, nbins) in enumerate(zip(model, asimov_bins)):

                weights, bin_edges = generate_asimov_hist(m, values, nbins)
                bin_centers = bin_edges[0:-1] + np.diff(bin_edges) / 2

                if not model[i].is_extended:
                    weights *= get_value(data[i].n_events)

                asimov_data.append(
                    array2dataset(type(data[i]), data[i].space, bin_centers, weights)
                )

            self._asimov_dataset[poi] = asimov_data

        return self._asimov_dataset[poi]

    def asimov_loss(self, poi: POI):
        """Constructs a loss function using the Asimov dataset for a given alternative hypothesis.

        Args:
            poi: parameter of interest of the alternative hypothesis.

        Returns:
             Loss function.

        Example with **zfit**:
            >>> poialt = POI(mean, 1.2)
            >>> loss = calc.asimov_loss(poialt)
        """
        if poi not in self._asimov_loss.keys():
            loss = self.lossbuilder(self.model, self.asimov_dataset(poi))
            self._asimov_loss[poi] = loss

        return self._asimov_loss[poi]

    def asimov_nll(self, pois: POIarray, poialt: POI) -> np.ndarray:
        """Computes negative log-likelihood values for given parameters of interest using the Asimov dataset
        generated with a given alternative hypothesis.

        Args:
            pois: parameters of interest.
            poialt: parameter of interest of the alternative hypothesis.

        Returns:
            Array of nll values for the alternative hypothesis.

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POIarray(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, 1.2)
            >>> nll = calc.asimov_nll(poinull, poialt)

        """
        self.check_pois(pois)
        self.check_pois(poialt)

        minimizer = self.minimizer
        ret = np.empty(pois.shape)
        for i, p in enumerate(pois):
            if p not in self._asimov_nll.keys():
                loss = self.asimov_loss(poialt)
                nll = pll(minimizer, loss, p)
                self._asimov_nll[p] = nll
            ret[i] = self._asimov_nll[p]
        return ret

    def pnull(
        self,
        qobs: np.ndarray,
        qalt: Union[np.ndarray, None] = None,
        onesided: bool = True,
        onesideddiscovery: bool = False,
        qtilde: bool = False,
        nsigma: int = 0,
    ) -> np.ndarray:
        """Computes the pvalue for the null hypothesis.

        Args:
            qobs: observed values of the test-statistic q.
            qalt: alternative values of the test-statistic q using the asimov dataset.
            onesided: if `True` (default) computes onesided pvalues.
            onesideddiscovery: if `True` (default) computes onesided pvalues for a discovery.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else (default)
              use the :math:`q` test statistic.
            nsigma: significance shift.

        Returns:
             Array of the pvalues for the null hypothesis.
        """
        sqrtqobs = np.sqrt(qobs)

        # 1 - norm.cdf(x) == norm.cdf(-x)
        if onesided or onesideddiscovery:
            pnull = 1.0 - norm.cdf(sqrtqobs - nsigma)
        else:
            pnull = (1.0 - norm.cdf(sqrtqobs - nsigma)) * 2.0

        if qalt is not None and qtilde:
            cond = (qobs > qalt) & (qalt > 0)
            sqrtqalt = np.sqrt(qalt)
            pnull_2 = 1.0 - norm.cdf((qobs + qalt) / (2.0 * sqrtqalt) - nsigma)

            if not (onesided or onesideddiscovery):
                pnull_2 += 1.0 - norm.cdf(sqrtqobs - nsigma)

            pnull = np.where(cond, pnull_2, pnull)

        return pnull

    def qalt(
        self,
        poinull: POIarray,
        poialt: POI,
        onesided: bool,
        onesideddiscovery: bool,
        qtilde: bool = False,
    ) -> np.ndarray:
        """Computes alternative hypothesis values of the :math:`\\Delta` log-likelihood test statistic using the asimov
        dataset.

        Args:
            poinull: parameters of interest for the null hypothesis.
            poialt: parameters of interest for the alternative hypothesis.
            onesided: if `True` (default) computes onesided pvalues.
            onesideddiscovery: if `True` (default) computes onesided pvalues for a
              discovery test.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else (default)
              use the :math:`q` test statistic.

        Returns:
            Q values for the alternative hypothesis.

        Example with **zfit**:
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> poinull = POI(mean, [1.1, 1.2, 1.0])
            >>> poialt = POI(mean, [1.2])
            >>> q = calc.qalt(poinull, poialt)
        """
        param = poialt.parameter

        if qtilde and poialt.value < 0:
            poialt_bf = POI(param, 0)
        else:
            poialt_bf = poialt

        nll_poialt_asy = self.asimov_nll(poialt_bf, poialt)
        nll_poinull_asy = self.asimov_nll(poinull, poialt)

        return self.q(
            nll1=nll_poinull_asy,
            nll2=nll_poialt_asy,
            poi1=poinull,
            poi2=poialt,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
        )

    def palt(
        self,
        qobs: np.ndarray,
        qalt: np.ndarray,
        onesided: int = True,
        onesideddiscovery: int = False,
        qtilde: int = False,
    ) -> np.ndarray:
        """Computes the pvalue for the alternative hypothesis.

        Args:
            qobs: observed values of the test-statistic q.
            qalt: alternative values of the test-statistic q using the Asimov dataset.
            onesided: if `True` (default) computes onesided pvalues.
            onesideddiscovery: if `True` (default) computes onesided pvalues for a discovery.
            qtilde: if `True` use the :math:`\\widetilde{q}` test statistics else (default)
              use the :math:`q` test statistic.

        Returns:
             Array of the pvalues for the alternative hypothesis.
        """
        sqrtqobs = np.sqrt(qobs)
        sqrtqalt = np.sqrt(qalt)

        # 1 - norm.cdf(x) == norm.cdf(-x)
        if onesided or onesideddiscovery:
            palt = 1.0 - norm.cdf(sqrtqobs - sqrtqalt)
        else:
            palt = 1.0 - norm.cdf(sqrtqobs + sqrtqalt)
            palt += 1.0 - norm.cdf(sqrtqobs - sqrtqalt)

        if qtilde:
            cond = qobs > qalt
            palt_2 = 1.0 - norm.cdf((qobs - qalt) / (2.0 * sqrtqalt))

            if not (onesided or onesideddiscovery):
                palt_2 += 1.0 - norm.cdf(sqrtqobs + sqrtqalt)

            palt = np.where(cond, palt_2, palt)

        return palt

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):

        qobs = self.qobs(
            poinull,
            onesided=onesided,
            qtilde=qtilde,
            onesideddiscovery=onesideddiscovery,
        )

        needpalt = poialt is not None

        if needpalt:
            qalt = self.qalt(
                poinull=poinull,
                poialt=poialt,
                onesided=onesided,
                onesideddiscovery=onesideddiscovery,
                qtilde=qtilde,
            )
            palt = self.palt(
                qobs=qobs,
                qalt=qalt,
                onesided=onesided,
                qtilde=qtilde,
                onesideddiscovery=onesideddiscovery,
            )
        else:
            qalt = None
            palt = None

        pnull = self.pnull(
            qobs=qobs,
            qalt=qalt,
            onesided=onesided,
            qtilde=qtilde,
            onesideddiscovery=onesideddiscovery,
        )

        return pnull, palt

    def _expected_pvalue_(
        self, poinull, poialt, nsigma, CLs, onesided, onesideddiscovery, qtilde
    ):

        qalt = self.qalt(
            poinull, poialt, onesided=onesided, onesideddiscovery=onesideddiscovery
        )
        qalt = np.where(qalt < 0, 0, qalt)

        expected_pvalues = []
        for ns in nsigma:
            p_clsb = self.pnull(
                qobs=qalt,
                qalt=None,
                onesided=onesided,
                qtilde=qtilde,
                onesideddiscovery=onesideddiscovery,
                nsigma=ns,
            )
            if CLs:
                p_clb = norm.cdf(ns)
                p_cls = p_clsb / p_clb
                expected_pvalues.append(np.where(p_cls < 0, 0, p_cls))
            else:
                expected_pvalues.append(np.where(p_clsb < 0, 0, p_clsb))

        return expected_pvalues
