from __future__ import annotations

import math
import typing
import warnings
from typing import Any

import numpy as np
from scipy.stats import norm

from ...utils import array2dataset, eval_pdf, get_value, pll, set_values
from ...utils.fit.api_check import is_valid_fitresult, is_valid_loss
from ...utils.fit.diverse import get_ndims
from ..parameters import POI, POIarray
from .basecalculator import BaseCalculator


def generate_asimov_hist(
    model, params: dict[Any, dict[str, Any]], nbins: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
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
    if nbins is None:
        nbins = 100
    space = model.space
    bounds = space.limit1d
    bin_edges = np.linspace(*bounds, nbins + 1)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    hist = eval_pdf(model, bin_centers, params, allow_extended=True)
    hist *= space.area() / nbins

    return hist, bin_edges


def generate_asimov_dataset(data, model, is_binned, nbins, values):
    """Generate the Asimov dataset using a model and dictionary of parameters.

    Args:
        data: Data, the same class should be used for the generated dataset.
        model: Model to use for the generation. Can be binned or unbinned.
        is_binned: If the model is binned.
        nbins: Number of bins for the asimov dataset.
        values: Dictionary of parameters values.

    Returns:
        Dataset with the asimov dataset.
    """
    nsample = None
    if not model.is_extended:
        nsample = get_value(data.n_events)
    if is_binned:
        with set_values(list(values), [v["value"] for v in values.values()]):
            dataset = model.to_binneddata()
            if nsample is not None:
                dataset = type(dataset).from_hist(dataset.to_hist() * nsample)
    else:
        if len(nbins) > 1:  # meaning we have multiple dimensions
            msg = (
                "Currently, only one dimension is supported for models that do not follow"
                " the new binned loss convention. New losses can be registered with the"
                " asymtpotic calculator."
            )
            raise ValueError(msg)
        weights, bin_edges = generate_asimov_hist(model, values, nbins[0])
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

        if nsample is not None:  # It's not extended
            weights *= nsample

        dataset = array2dataset(type(data), data.space, bin_centers, weights)
    return dataset


class AsymptoticCalculator(BaseCalculator):
    """
    Class for asymptotic calculators, using asymptotic formulae of the likelihood ratio described in
    :cite:`Cowan:2010js`. Can be used only with one parameter of interest.
    """

    UNBINNED_TO_BINNED_LOSS: typing.ClassVar = {}
    try:
        from zfit.loss import (
            BinnedNLL,
            ExtendedBinnedNLL,
            ExtendedUnbinnedNLL,
            UnbinnedNLL,
        )
    except ImportError:
        pass
    else:
        UNBINNED_TO_BINNED_LOSS[UnbinnedNLL] = BinnedNLL
        UNBINNED_TO_BINNED_LOSS[ExtendedUnbinnedNLL] = ExtendedBinnedNLL

    def __init__(
        self,
        input,
        minimizer,
        asimov_bins: int | list[int] | None = None,
    ):
        """Asymptotic calculator class using Wilk's and Wald's asymptotic formulae.

        The asympotic formula is significantly faster than the Frequentist calculator, as it does not
        require the calculation of the frequentist p-value, which involves the calculation of toys (sample-and-fit).


        Args:
            input: loss or fit result.
            minimizer: minimizer to use to find the minimum of the loss function.
            asimov_bins: number of bins of the Asimov dataset.

        Example with **zfit**:
            >>> import zfit
            >>> from zfit.loss import UnbinnedNLL
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
        if is_valid_fitresult(input):
            loss = input.loss
        elif is_valid_loss(input):
            loss = input
        else:
            msg = "input must be a fitresult or a loss"
            raise ValueError(msg)

        asimov_bins_converted = self._check_convert_asimov_bins(asimov_bins, loss.data)

        super().__init__(input, minimizer)
        self._asimov_bins = asimov_bins_converted
        self._asimov_dataset: dict = {}
        self._asimov_loss: dict = {}
        self._binned_loss = None
        # cache of nll values computed with the asimov dataset
        self._asimov_nll: dict[POI, np.ndarray] = {}

    def _convert_to_binned(self, loss, asimov_bins):
        """Converts the loss to binned if necessary."""

        for unbinned_loss, binned_loss in self.UNBINNED_TO_BINNED_LOSS.items():
            if type(loss) is unbinned_loss:
                datasets = []
                models = []
                for d, m, nbins in zip(loss.data, loss.model, asimov_bins):
                    binnings = m.space.with_binning(nbins)
                    model_binned = m.to_binned(binnings)
                    data_binned = d.to_binned(binnings)
                    datasets.append(data_binned)
                    models.append(model_binned)
                loss = binned_loss(model=models, data=datasets, constraints=loss.constraints)
                break
            if type(loss) is binned_loss:
                break
        else:
            loss = False

        return loss

    def _get_binned_loss(self):
        """Returns the binned loss."""
        binned_loss = self._binned_loss
        if binned_loss is None:
            binned_loss = self._convert_to_binned(self.loss, self._asimov_bins)
            self._binned_loss = binned_loss
        return binned_loss

    @staticmethod
    def _check_convert_asimov_bins(asimov_bins, datasets) -> list[list[int]]:  # TODO: we want to allow axes from UHI
        nsimultaneous = len(datasets)
        ndims = [get_ndims(dataset) for dataset in datasets]
        if asimov_bins is None:
            asimov_bins = [[math.ceil(100 / ndim**0.5)] * ndim for ndim in ndims]
        if isinstance(asimov_bins, int):
            if nsimultaneous == 1:
                asimov_bins = [[asimov_bins] * ndim for ndim in ndims]
            else:
                msg = (
                    "asimov_bins is an int but there are multiple datasets. "
                    "Please provide a list of int for each dataset."
                )
                raise ValueError(msg)
        elif isinstance(asimov_bins, list):
            if len(asimov_bins) != nsimultaneous:
                msg = "asimov_bins is a list but the number of elements is different from the number of datasets."
                raise ValueError(msg)
        else:
            msg = f"asimov_bins must be an int or a list of int (or list of list of int), not {type(asimov_bins)}"
            raise TypeError(msg)

        for i, (asimov_bin, ndim) in enumerate(zip(asimov_bins, ndims)):
            if isinstance(asimov_bin, int):
                if ndim == 1:
                    asimov_bins[i] = [asimov_bin]
                else:
                    msg = f"asimov_bins[{i}] is not a list but the dataset has {ndim} dimensions."
                    raise ValueError(msg)
            elif isinstance(asimov_bin, list):
                if len(asimov_bin) != ndim:
                    msg = (
                        f"asimov_bins[{i}] is a list with {len(asimov_bin)} elements but the"
                        f" dataset has {ndim} dimensions."
                    )
                    raise ValueError(msg)
                if not all(isinstance(x, int) for x in asimov_bin):
                    msg = f"asimov_bins[{i}] is a list with non-int elements."
                    raise ValueError(msg)
            else:
                msg = f"asimov_bins[{i}] is not an int or a list but a {type(asimov_bin)}."
                raise TypeError(msg)
        assert isinstance(asimov_bins, list), "INTERNAL ERROR: Could not correctly convert asimov_bins"
        assert all(
            isinstance(asimov_bin, list) and len(asimov_bin) == ndim for ndim, asimov_bin in zip(ndims, asimov_bins)
        ), "INTERNAL ERROR: Could not correctly convert asimov_bins, dimensions wrong"
        return asimov_bins

    @staticmethod
    def check_pois(pois: POI | POIarray):
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

    def asimov_dataset(self, poi: POI, ntrials_fit: int | None = None):
        """Gets the Asimov dataset for a given alternative hypothesis.

        Args:
            poi: parameter of interest of the alternative hypothesis.
            ntrials_fit: (default: 5) maximum number of fits to perform

        Returns:
             The asymov dataset.

        Example with **zfit**:
            >>> poialt = POI(mean, 1.2)
            >>> dataset = calc.asimov_dataset(poialt)

        """
        if ntrials_fit is None:
            ntrials_fit = 5
        if poi not in self._asimov_dataset:
            binned_loss = self._get_binned_loss()
            if binned_loss is False:  # LEGACY
                model = self.model
                data = self.data
                loss = self.loss
            else:
                model = binned_loss.model
                data = binned_loss.data
                loss = binned_loss
            minimizer = self.minimizer
            oldverbose = minimizer.verbosity
            minimizer.verbosity = 0

            poiparam = poi.parameter
            poivalue = poi.value

            msg = "\nGet fitted values of the nuisance parameters for the"
            msg += " alternative hypothesis!"

            self.set_params_to_bestfit()
            poiparam.floating = False

            if not self.loss.get_params():
                values = {poiparam: {"value": poivalue}}

            else:
                with poiparam.set_value(poivalue):
                    for _trial in range(ntrials_fit):
                        minimum = minimizer.minimize(loss=loss)
                        if minimum.valid:
                            break

                        # shift other parameter values to change starting point of minimization
                        for p in self.parameters:
                            if p != poiparam:
                                p.set_value(get_value(p) * np.random.normal(1, 0.02, 1)[0])
                    else:
                        msg = "No valid minimum was found when fitting the loss function for the alternative"
                        msg += f"hypothesis ({poi}), after {ntrials_fit} trials."
                        warnings.warn(msg, stacklevel=2)

                values = dict(minimum.params)
                values[poiparam] = {"value": poivalue}

            poiparam.floating = True
            minimizer.verbosity = oldverbose

            asimov_data = []
            asimov_bins = self._asimov_bins
            assert len(asimov_bins) == len(data)
            is_binned_loss = isinstance(loss, tuple(self.UNBINNED_TO_BINNED_LOSS.values()))
            for _i, (m, d, nbins) in enumerate(zip(model, data, asimov_bins)):
                dataset = generate_asimov_dataset(d, m, is_binned_loss, nbins, values)
                asimov_data.append(dataset)

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
        oldloss = self._get_binned_loss()
        if oldloss is False:  # LEGACY
            oldloss = self.loss
        if oldloss is None:
            msg = "No loss function was provided."
            raise ValueError(msg)
        if poi not in self._asimov_loss:
            loss = self.lossbuilder(oldloss.model, self.asimov_dataset(poi), oldloss=oldloss)
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
            if p not in self._asimov_nll:
                loss = self.asimov_loss(poialt)
                nll = pll(minimizer, loss, p)
                self._asimov_nll[p] = nll
            ret[i] = self._asimov_nll[p]
        return ret

    def pnull(
        self,
        qobs: np.ndarray,
        qalt: np.ndarray | None = None,
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

        poialt_bf = POI(param, 0) if qtilde and poialt.value < 0 else poialt

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

    def _expected_pvalue_(self, poinull, poialt, nsigma, CLs, onesided, onesideddiscovery, qtilde):
        qalt = self.qalt(poinull, poialt, onesided=onesided, onesideddiscovery=onesideddiscovery)
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
