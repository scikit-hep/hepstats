from .basecalculator import BaseCalculator
from ..fit_utils.utils import eval_pdf, array2dataset, pll
from ..parameters import POI
import numpy as np
from scipy.stats import norm


def generate_asymov_dataset(model, params, nbins=100):

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
        """

        super(AsymptoticCalculator, self).__init__(input, minimizer)
        self._asymov_bins = asymov_bins
        self._asymov_dataset = {}
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

    def asymov_dataset(self, poi):
        assert len(poi) == 1

        if poi not in self._asymov_dataset.keys():
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

            datasets = [generate_asymov_dataset(m, values, self._asymov_bins) for m in model]

            self._asymov_dataset[poi] = datasets

        return self._asymov_dataset[poi]

    def asymov_loss(self, poi):
        if poi not in self._asymov_loss.keys():
            model = self.model
            data = self.data
            asymov_data = []

            for i, ad in enumerate(self.asymov_dataset(poi)):
                bin_centers, weights = ad
                asymov_data.append(array2dataset(type(data[i]), data[i].obs, bin_centers, weights))

            loss = self.lossbuilder(model, asymov_data)

            self._asymov_loss[poi] = loss

        return self._asymov_loss[poi]

    def asymov_nll(self, poi, poialt):
        minimizer = self.minimizer
        ret = np.empty(len(poi))
        for i, p in enumerate(poi):
            if p not in self._asymov_nll.keys():
                loss = self.asymov_loss(poialt)
                nll = pll(minimizer, loss, p)
                self._asymov_nll[p] = nll
            ret[i] = self._asymov_nll[p]
        return ret

    def pnull(self, qobs, qalt=None, onesided=True, onesideddiscovery=False, qtilde=False, nsigma=0):
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

    def qalt(self, poinull, poialt, onesided, onesideddiscovery):
        """ Compute $$\\Delta$$ log-likelihood values for alternative hypothesis
            using the asymov dataset.
        """
        nll_poinull_asy = self.asymov_nll(poinull, poialt)
        nll_poialt_asy = self.asymov_nll(poialt, poialt)
        return self.q(nll1=nll_poinull_asy, nll2=nll_poialt_asy, bestfit=[poialt], poival=[poinull],
                      onesided=onesided, onesideddiscovery=onesideddiscovery)

    def palt(self, qobs, qalt, onesided=True, onesideddiscovery=False, qtilde=False):
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

        poinull = poinull[0]
        needpalt = poialt is not None

        if needpalt:
            poialt = poialt[0]
            qalt = self.qalt(poinull, poialt, onesided, onesideddiscovery)
            palt = self.palt(qobs=qobs, qalt=qalt, onesided=onesided, qtilde=qtilde,
                             onesideddiscovery=onesideddiscovery)
        else:
            qalt = None
            palt = None

        pnull = self.pnull(qobs=qobs, qalt=qalt, onesided=onesided, qtilde=qtilde,
                           onesideddiscovery=onesideddiscovery)

        return pnull, palt

    def _expected_pvalue_(self, poinull, poialt, nsigma, CLs, onesided=True, onesideddiscovery=False,
                          qtilde=False):

        poinull = poinull[0]
        if poialt is not None:
            poialt = poialt[0]

        qalt = self.qalt(poinull, poialt, onesided=onesided, onesideddiscovery=onesideddiscovery)  # TO CHECK
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

    def _expected_poi_(self, poinull, poialt, nsigma, alpha, CLs, onesided=True, onesideddiscovery=False):

        poinull = poinull[0]
        if poialt is not None:
            poialt = poialt[0]

        qalt = self.qalt(poinull, poialt, onesided=True, onesideddiscovery=False)  # TO CHECK
        qalt = np.where(qalt < 0, 0, qalt)

        sigma = np.sqrt((poinull.value - poialt.value)**2 / qalt)

        expected_values = []
        for ns in nsigma:
            if CLs:
                exp = sigma * (norm.ppf(1 - alpha*norm.cdf(ns)) + ns)
            else:
                exp = sigma * (norm.ppf(1 - alpha) + ns)

            expected_values.append(poialt.value + exp)

        return expected_values
