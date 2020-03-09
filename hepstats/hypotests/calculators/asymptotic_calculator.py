from typing import Tuple
import numpy as np
from scipy.stats import norm

from .basecalculator import BaseCalculator
from ...utils.fit import eval_pdf, array2dataset, pll
from ..parameters import POI, POIarray


def generate_asimov_hist(model, params, nbins=100):
    """ Generate the Asimov histogram using a model and dictionary of parameters.

        Args:
            * **model** : model used to generate the dataset
            * **params** (Dict) : values of the parameters of the models
            * **nbins** (int, optional) : number of bins

        Returns:
             (`numpy.array`, `numpy.array`) : hist, bin_edges

        Example with `zfit`:
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
    Class for asymptotic calculators. Can be used only with one parameter
    of interest.

    See G. Cowan, K. Cranmer, E. Gross and O. Vitells: Asymptotic formulae for
    likelihood- based tests of new physics. Eur. Phys. J., C71:1–19, 2011
    """

    def __init__(self, input, minimizer, asimov_bins=100):
        """Asymptotic calculator class.

            Args:
                * **input** : loss or fit result
                * **minimizer** : minimizer to use to find the minimum of the loss function
                * **asimov_bins** (Optional, int) : number of bins of the Asimov dataset

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

                >>> calc = AsymptoticCalculator(input=loss, minimizer=Minuit(), asimov_bins=100)
        """

        super(AsymptoticCalculator, self).__init__(input, minimizer)
        self._asimov_bins = asimov_bins
        self._asimov_dataset = {}
        self._asimov_loss = {}
        # cache of nll values computed with the asimov dataset
        self._asimov_nll = {}

    @staticmethod
    def check_pois(pois):
        """
        Checks if the parameters of interest are all `hepstats.parameters.POI/POIarray` instances.
        """

        msg = "POI/POIarray is required."
        if not isinstance(pois, POIarray):
            raise TypeError(msg)
        if pois.ndim > 1:
            msg = "Tests using the asymptotic calculator can only be used with one parameter of interest."
            raise NotImplementedError(msg)

    def asimov_dataset(self, poi) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the Asimov dataset for a given alternative hypothesis.

            Args:
                * **poi** (`POI`): parameter of interest of the alternative hypothesis

            Returns:
                 Dataset

            Example with `zfit`:
                >>> poialt = POI(mean, 1.2)
                >>> dataset = calc.asimov_dataset(poialt)

        """

        if poi not in self._asimov_dataset.keys():
            model = self.model
            data = self.data
            minimizer = self.minimizer
            oldverbose = minimizer.verbosity
            minimizer.verbosity = 5

            poiparam = poi.parameter
            poivalue = poi.value

            msg = "\nGet fit best values for nuisance parameters for the"
            msg += " alternative hypothesis!"
            print(msg)

            with poiparam.set_value(poivalue):
                poiparam.floating = False
                minimum = minimizer.minimize(loss=self.loss)
                poiparam.floating = True

            minimizer.verbosity = oldverbose

            values = minimum.params
            values[poiparam] = {"value": poivalue}

            asimov_data = []
            for i, ad in enumerate([generate_asimov_hist(m, values, self._asimov_bins) for m in model]):
                weights, bin_edges = ad
                bin_centers = bin_edges[0:-1] + np.diff(bin_edges) / 2
                asimov_data.append(array2dataset(type(data[i]), data[i].space, bin_centers, weights))

            self._asimov_dataset[poi] = asimov_data

        return self._asimov_dataset[poi]

    def asimov_loss(self, poi):
        """Constructs a loss function using the Asimov dataset for a given alternative hypothesis.

            Args:
                * **poi** (`POI`): parameter of interest of the alternative hypothesis

            Returns:
                 Loss function

            Example with `zfit`:
                >>> poialt = POI(mean, 1.2)
                >>> loss = calc.asimov_loss(poialt)

        """
        if poi not in self._asimov_loss.keys():
            loss = self.lossbuilder(self.model, self.asimov_dataset(poi))
            self._asimov_loss[poi] = loss

        return self._asimov_loss[poi]

    def asimov_nll(self, pois, poialt) -> np.ndarray:
        """Computes negative log-likelihood values for given parameters of interest using the Asimov dataset
            generated with a given alternative hypothesis.

            Args:
                * **pois** (`POIarray`): parameters of interest
                * **poialt** (`POI`): parameter of interest of the alternative hypothesis

            Returns:
                 `numpy.array`: alternative nll values

            Example with `zfit`:
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
        self, qobs, qalt=None, onesided=True, onesideddiscovery=False, qtilde=False, nsigma=0
    ) -> np.ndarray:
        """Computes the pvalue for the null hypothesis.

            Args:
                * **qobs** (`numpy.array`): observed values of the test-statistic q
                * **qalt** (`numpy.array`): alternative values of the test-statistic q using the asimov dataset
                * **onesided** (bool, optional): if `True` (default) computes onesided pvalues
                * **onesideddiscovery** (bool, optional): if `True` (default) computes onesided pvalues for a discovery
                * **qtilde** (bool, optional): if `True` use the :math:`\widetilde{q}` test statistics else (default)
                  use the :math:`q` test statistic
                * **nsigma** (float, optional): significance shift

            Returns:
                 `np.array` : array of the pvalues for the null hypothesis
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

    def qalt(self, poinull, poialt, onesided, onesideddiscovery) -> np.ndarray:
        """Computes alternative hypothesis values of the :math:`\Delta` log-likelihood test statistic using the asimov
            dataset.

            Args:
                * **poinull** (`POIarray`): parameters of interest for the null hypothesis
                * **poialt** (`POIarray`): parameters of interest for the alternative hypothesis
                * **onesided** (bool, optional): if `True` (default) computes onesided pvalues
                * **onesideddiscovery** (bool, optional): if `True` (default) computes onesided pvalues for a
                  discovery test

            Returns:
                `numpy.array`: observed values of q

            Example with `zfit`:
                >>> mean = zfit.Parameter("mu", 1.2)
                >>> poinull = POI(mean, [1.1, 1.2, 1.0])
                >>> poialt = POI(mean, [1.2])
                >>> q = calc.qalt([poinull], [poialt])
        """
        nll_poinull_asy = self.asimov_nll(poinull, poialt)
        nll_poialt_asy = self.asimov_nll(poialt, poialt)
        return self.q(
            nll1=nll_poinull_asy,
            nll2=nll_poialt_asy,
            poi1=poinull,
            poi2=poialt,
            onesided=onesided,
            onesideddiscovery=onesideddiscovery,
        )

    def palt(self, qobs, qalt, onesided=True, onesideddiscovery=False, qtilde=False) -> np.ndarray:
        """Computes the pvalue for the alternative hypothesis.

            Args:
                * **qobs** (`np.array`): observed values of the test-statistic q
                * **qalt** (`np.array`): alternative values of the test-statistic q using the Asimov dataset
                * **onesided** (bool, optional): if `True` (default) computes onesided pvalues
                * **onesideddiscovery** (bool, optional): if `True` (default) computes onesided pvalues for a discovery
                * **qtilde** (bool, optional): if `True` use the :math:`\widetilde{q}` test statistics else (default)
                  use the :math:`q` test statistic

            Returns:
                 `numpy.array` : array of the pvalues for the alternative hypothesis
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
            cond = (qobs > qalt) & (qalt > 0)
            palt_2 = 1.0 - norm.cdf((qobs - qalt) / (2.0 * sqrtqalt))

            if not (onesided or onesideddiscovery):
                palt_2 += 1.0 - norm.cdf(sqrtqobs + sqrtqalt)

            palt = np.where(cond, palt_2, palt)

        return palt

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):

        qobs = self.qobs(poinull, onesided=onesided, qtilde=qtilde, onesideddiscovery=onesideddiscovery)

        needpalt = poialt is not None

        if needpalt:
            qalt = self.qalt(poinull, poialt, onesided, onesideddiscovery)
            palt = self.palt(
                qobs=qobs, qalt=qalt, onesided=onesided, qtilde=qtilde, onesideddiscovery=onesideddiscovery
            )
        else:
            qalt = None
            palt = None

        pnull = self.pnull(
            qobs=qobs, qalt=qalt, onesided=onesided, qtilde=qtilde, onesideddiscovery=onesideddiscovery
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

    def _expected_poi_(self, poinull, poialt, nsigma, alpha, CLs, onesided, onesideddiscovery):

        qalt = self.qalt(poinull, poialt, onesided=onesided, onesideddiscovery=onesideddiscovery)
        qalt = np.where(qalt < 0, 0, qalt)

        sigma = np.sqrt((poinull[0].value - poialt[0].value) ** 2 / qalt)

        expected_values = []
        for ns in nsigma:
            if CLs:
                exp = sigma * (norm.ppf(1 - alpha * norm.cdf(ns)) + ns)
            else:
                exp = sigma * (norm.ppf(1 - alpha) + ns)

            expected_values.append(poialt[0].value + exp)

        return expected_values
