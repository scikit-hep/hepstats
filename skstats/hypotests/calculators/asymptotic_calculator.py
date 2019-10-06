from .basecalculator import BaseCalculator
from ..fit_utils.utils import eval_pdf, array2dataset, pll
from ..parameters import POI
import numpy as np
from scipy.stats import norm


def generate_asymov_hist(model, params, nbins=100):

    space = model.space
    bounds = space.limit1d
    bin_edges = np.linspace(*bounds, nbins+1)
    bin_centers = bin_edges[0: -1] + np.diff(bin_edges)/2

    weights = eval_pdf(model, bin_centers, params)
    weights *= (space.area() / nbins)

    return bin_centers, weights


class AsymptoticCalculator(BaseCalculator):
    """
    Class for for asymptotic calculators. Can be used only with one parameter
    of interest.

    See G. Cowan, K. Cranmer, E. Gross and O. Vitells: Asymptotic formulae for
    likelihood- based tests of new physics. Eur. Phys. J., C71:1–19, 2011
    """

    def __init__(self, input, minimizer, asymov_bins=100):
        """Base class for calculator.

            Args:
                input : loss or fit result
                minimizer : minimizer to use to find the minimum of the loss function
                asymov_bins (Optional, int) : number of bins of the asymov dataset

            Example:
                import zfit
                from zfit.core.loss import UnbinnedNLL
                from zfit.minimize import MinuitMinimizer

                obs = zfit.Space('x', limits=(0.1, 2.0))
                data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
                mean = zfit.Parameter("mu", 1.2)
                sigma = zfit.Parameter("sigma", 0.1)
                model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
                loss = UnbinnedNLL(model=[model], data=[data], fit_range=[obs])

                calc = AsymptoticCalculator(input=loss, minimizer=MinuitMinimizer(), nbins=100)
        """

        super(AsymptoticCalculator, self).__init__(input, minimizer)
        self._asymov_bins = asymov_bins
        self._asymov_hist = {}
        self._asymov_loss = {}
        # cache of nll values computed with the asymov dataset
        self._asymov_nll = {}

    @staticmethod
    def checkpois(pois):
        msg = "A list of POIs is required."
        if not isinstance(pois, (list, tuple)):
            raise ValueError(msg)
        if not all(isinstance(p, POI) for p in pois):
            raise ValueError(msg)
        if len(pois) > 1:
            msg = "Tests using the asymptotic calcultor can only be used with one parameter of interest."
            raise NotImplementedError(msg)

    def asymov_hist(self, poi) -> (np.array, np.array):
        """ Generate the asymov histogram a given alternative hypothesis.

            Args:
                poi (List[`hypotests.POI`]): parameter of interest of the alternative hypothesis

            Returns:
                 Tuple('np.array', 'np.array'): hist, bin_edges

            Example:
                poialt = POI(mean, [1.2])
                hist, bin_edges = calc.asymov_hist([poialt])

        """
        self.checkpois(poi)
        poi = poi[0]

        if poi not in self._asymov_hist.keys():
            model = self.model
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

            datasets = [generate_asymov_hist(m, values, self._asymov_bins) for m in model]

            self._asymov_hist[poi] = datasets

        return self._asymov_hist[poi]

    def asymov_loss(self, poi):
        """ Construct a loss function using the asymov dataset for a given alternative hypothesis.

            Args:
                poi (List[`hypotests.POI`]): parameter of interest of the alternative hypothesis

            Returns:
                 Loss function

            Example:
                poialt = POI(mean, [1.2])
                loss = calc.asymov_loss([poialt])

        """
        self.checkpois(poi)

        if poi[0] not in self._asymov_loss.keys():
            model = self.model
            data = self.data
            asymov_data = []

            for i, ad in enumerate(self.asymov_hist(poi)):
                bin_centers, weights = ad
                asymov_data.append(array2dataset(type(data[i]), data[i].obs, bin_centers, weights))

            loss = self.lossbuilder(model, asymov_data)

            self._asymov_loss[poi[0]] = loss

        return self._asymov_loss[poi[0]]

    def asymov_nll(self, poi, poialt) -> np.array:
        """ Compute negative log-likelihood values for given parameters of interest using the asymov dataset
            generated with a given alternative hypothesis.

            Args:
                pois (List[`hypotests.POI`]): parameters of interest
                poialt (List[`hypotests.POI`]): parameter of interest of the alternative hypothesis

            Returns:
                 `numpy.array`: alternative nll values

            Example:
                mean = zfit.Parameter("mu", 1.2)
                poinull = POI(mean, [1.1, 1.2, 1.0])
                poialt = POI(mean, [1.2])
                nll = calc.asymov_nll([poinull], [poialt])

        """
        self.checkpois(poi)
        self.checkpois(poialt)

        minimizer = self.minimizer
        ret = np.empty(len(poi[0]))
        for i, p in enumerate(poi[0]):
            if p not in self._asymov_nll.keys():
                loss = self.asymov_loss(poialt)
                nll = pll(minimizer, loss, p)
                self._asymov_nll[p] = nll
            ret[i] = self._asymov_nll[p]
        return ret

    def pnull(self, qobs, qalt=None, onesided=True, onesideddiscovery=False, qtilde=False, nsigma=0) -> np.array:
        """ Compute the pvalue for the null hypotesis.

            Args:
                qobs (`np.array`): observed values of the test-statistic q
                qalt (`np.array`): alternative values of the test-statistic q using the asymov dataset
                onesided (bool, optionnal): if `True` (default) computes onesided pvalues
                onesideddiscovery (bool, optionnal): if `True` (default) computes onesided pvalues for a discovery
                qtilde (bool, optionnal): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                    the $$q$$ test statistic
                nsigma (float, optionnal): significance shift

            Returns:
                 `np.array` : array of the pvalues for the null hypothesis
        """
        sqrtqobs = np.sqrt(qobs)

        # 1 - norm.cdf(x) == norm.cdf(-x)
        if onesided or onesideddiscovery:
            pnull = 1. - norm.cdf(sqrtqobs - nsigma)
        else:
            pnull = (1. - norm.cdf(sqrtqobs - nsigma))*2.

        if qalt is not None and qtilde:
            cond = (qobs > qalt) & (qalt > 0)
            sqrtqalt = np.sqrt(qalt)
            pnull_2 = 1. - norm.cdf((qobs + qalt) / (2. * sqrtqalt) - nsigma)

            if not (onesided or onesideddiscovery):
                pnull_2 += 1. - norm.cdf(sqrtqobs - nsigma)

            pnull = np.where(cond, pnull_2, pnull)

        return pnull

    def qalt(self, poinull, poialt, onesided, onesideddiscovery) -> np.array:
        """ Compute alternative values of the $$\\Delta$$ log-likelihood test statistic using the asymov
            datset.

            Args:
                poinull (List[`hypotests.POI`]): parameters of interest for the null hypothesis
                poialt (List[`hypotests.POI`]): parameters of interest for the alternative hypothesis
                onesided (bool, optionnal): if `True` (default) computes onesided pvalues
                onesideddiscovery (bool, optionnal): if `True` (default) computes onesided pvalues for a discovery
                    test

            Returns:
                `numpy.array`: observed values of q

            Example:
                mean = zfit.Parameter("mu", 1.2)
                poinull = POI(mean, [1.1, 1.2, 1.0])
                poialt = POI(mean, [1.2])
                q = calc.qalt([poinull], [poialt])
        """
        nll_poinull_asy = self.asymov_nll(poinull, poialt)
        nll_poialt_asy = self.asymov_nll(poialt, poialt)
        return self.q(nll1=nll_poinull_asy, nll2=nll_poialt_asy, poi1=poinull, poi2=poialt,
                      onesided=onesided, onesideddiscovery=onesideddiscovery)

    def palt(self, qobs, qalt, onesided=True, onesideddiscovery=False, qtilde=False) -> np.array:
        """ Compute the pvalue for the alternative hypotesis.

            Args:
                qobs (`np.array`): observed values of the test-statistic q
                qalt (`np.array`): alternative values of the test-statistic q using the asymov dataset
                onesided (bool, optionnal): if `True` (default) computes onesided pvalues
                onesideddiscovery (bool, optionnal): if `True` (default) computes onesided pvalues for a discovery
                qtilde (bool, optionnal): if `True` use the $$\tilde{q}$$ test statistics else (default) use
                    the $$q$$ test statistic

            Returns:
                 `np.array` : array of the pvalues for the alternative hypothesis
        """
        sqrtqobs = np.sqrt(qobs)
        sqrtqalt = np.sqrt(qalt)

        # 1 - norm.cdf(x) == norm.cdf(-x)
        if onesided or onesideddiscovery:
            palt = 1. - norm.cdf(sqrtqobs - sqrtqalt)
        else:
            palt = 1. - norm.cdf(sqrtqobs + sqrtqalt)
            palt += 1. - norm.cdf(sqrtqobs - sqrtqalt)

        if qtilde:
            cond = (qobs > qalt) & (qalt > 0)
            palt_2 = 1. - norm.cdf((qobs - qalt) / (2. * sqrtqalt))

            if not (onesided or onesideddiscovery):
                palt_2 += 1. - norm.cdf(sqrtqobs + sqrtqalt)

            palt = np.where(cond, palt_2, palt)

        return palt

    def _pvalue_(self, poinull, poialt, qtilde, onesided, onesideddiscovery):

        qobs = self.qobs(poinull, onesided=onesided, qtilde=qtilde,
                         onesideddiscovery=onesideddiscovery)

        needpalt = poialt is not None

        if needpalt:
            qalt = self.qalt(poinull, poialt, onesided, onesideddiscovery)
            palt = self.palt(qobs=qobs, qalt=qalt, onesided=onesided, qtilde=qtilde,
                             onesideddiscovery=onesideddiscovery)
        else:
            qalt = None
            palt = None

        pnull = self.pnull(qobs=qobs, qalt=qalt, onesided=onesided, qtilde=qtilde,
                           onesideddiscovery=onesideddiscovery)

        return pnull, palt

    def _expected_pvalue_(self, poinull, poialt, nsigma, CLs, onesided, onesideddiscovery, qtilde):

        qalt = self.qalt(poinull, poialt, onesided=onesided, onesideddiscovery=onesideddiscovery)
        qalt = np.where(qalt < 0, 0, qalt)

        expected_pvalues = []
        for ns in nsigma:
            p_clsb = self.pnull(qobs=qalt, qalt=None, onesided=onesided, qtilde=qtilde,
                                onesideddiscovery=onesideddiscovery, nsigma=ns)
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

        sigma = np.sqrt((poinull[0].value - poialt[0].value)**2 / qalt)

        expected_values = []
        for ns in nsigma:
            if CLs:
                exp = sigma * (norm.ppf(1 - alpha*norm.cdf(ns)) + ns)
            else:
                exp = sigma * (norm.ppf(1 - alpha) + ns)

            expected_values.append(poialt[0].value + exp)

        return expected_values
