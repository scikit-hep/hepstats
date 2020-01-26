import asdf
import os
import numpy as np

from .parameters import POI, POIarray
from .exceptions import ParameterNotFound
from .fitutils.utils import pll
from .fitutils.sampling import base_sampler, base_sample
from .hypotests_object import ToysObject


class ToyResult(object):

    def __init__(self, poigen, poieval):
        """
        Class to store the results of toys generated for a given value of a POI.
        The best fit value of the POI, the NLL evaluate at the best fit, and the NLL evaluated
        at several values of the POI are stored.

        Args:
            poigen (POI): POI used to generate the toys
            poieval (POIarray): POI values to evaluate the loss function
        """

        if not isinstance(poigen, POI):
            raise TypeError("A `hypotests.parameters.POI` is required for poigen.")
        if not isinstance(poieval, POIarray):
            raise TypeError("A `hypotests.parameters.POIarray` is required for poieval.")

        self._poigen = poigen
        self._bestfit = np.array([])
        self._nll_bestfit = np.array([])
        self._poieval = poieval
        self._nlls = {p: np.array([]) for p in poieval}

    @property
    def poigen(self):
        return self._poigen

    @property
    def bestfit(self):
        return self._bestfit

    @property
    def poieval(self):
        return self._poieval

    @property
    def nll_bestfit(self):
        return self._nll_bestfit

    @property
    def nlls(self):
        return self._nlls

    @property
    def ntoys(self):
        return len(self.bestfit)

    def add_entries(self, bestfit, nll_bestfit, nlls):
        if not all(k in nlls.keys() for k in self.poieval):
            missing_keys = [k for k in self.poieval if k not in nlls.keys()]
            raise ValueError(f"NLLs values for {missing_keys} are missing.")

        nentries = bestfit.size
        assert nll_bestfit.size == nentries
        assert all(a.size == nentries for a in nlls.values())

        self._bestfit = np.concatenate([self.bestfit, bestfit])
        self._nll_bestfit = np.concatenate([self.nll_bestfit, nll_bestfit])

        self._nlls = {p: np.concatenate([v, nlls[p]]) for p, v in self.nlls.items()}

    def __add__(self, toys):
        if not isinstance(toys, ToyResult):
            raise TypeError("A `Toys` is required.")

        assert self.poigen == toys.poigen
        assert self.poieval == toys.poieval

        newtoys = self.copy()
        print(toys.bestfit)
        newtoys.add_entries(bestfit=toys.bestfit, nll_bestfit=toys.nll_bestfit,
                            nlls=toys.nlls)

        return newtoys

    def to_dict(self):
        ret = {"poi": self.poigen.name, "bestfit": self.bestfit}
        ret["nlls"] = {n.value: nll for n, nll in self.nlls.items()}
        ret["genvalue"] = self.poigen.value
        ret["evalvalues"] = self.poieval.values
        ret["nlls"]["bestfit"] = self.nll_bestfit
        return ret

    def copy(self):
        newtoys = ToyResult(self.poigen, self.poieval)
        newtoys.add_entries(bestfit=self.bestfit, nll_bestfit=self.nll_bestfit,
                            nlls=self.nlls)
        return newtoys


class ToysManager(ToysObject):
    """
    Class handling the toy generation and fit.
    """

    def __init__(self, loss, minimizer, sampler=base_sampler, sample=base_sample):

        super(ToysManager, self).__init__(loss, minimizer, sampler, sample)
        self._toys = {}

    def get_toyresult(self, poigen, poieval):
        """
        Getter function.

        Args:
            index (`POI`, `POIarray`): POI used to generate the toys and
                POI values to evaluate the loss function

        Returns:
            `Toys`
        """

        index = (poigen, poieval)

        if index not in self.keys():
            for k in self.keys():
                if np.isin(poieval.values, k[-1].values).all():
                    index = k
                    break

        return self._toys[index]

    def set_toyresult(self, poigen, poieval, toy):
        """
        Setter function.

        Args:
            index (`POI`, `POIarray`): POI used to generate the toys and
                POI values to evaluate the loss function
            toy: (`Toys`)
        """
        index = (poigen, poieval)

        if not isinstance(poigen, POI):
            raise TypeError("A `hypotests.parameters.POI` is required for poigen.")
        if not isinstance(poieval, POIarray):
            raise TypeError("A `hypotests.parameters.POIarray` is required for poieval.")
        if not isinstance(toy, ToyResult):
            raise TypeError("A `hypotests.toyutils.Toys` is required for toy.")

        self._toys[index] = toy

    def ntoys(self, poigen, poieval):
        if (poigen, poieval) not in self.keys():
            return 0
        else:
            return self.get_toyresult(poigen, poieval).ntoys

    def generate_and_fit_toys(self, ntoys, poigen, poieval, printfreq=0.2):
        """
        Generate and fit toys for at a given POI (poigen). The toys are then fitted, and the likelihood
        is profiled at the values of poigen and poieval.

        Args:
            ntoys (int): number of toys to generate
            poigen (POI): POI used to generate the toys
            poieval (POIarray, optional): POI values to evaluate the loss function
            printfreq: print frequency of the toys generation
        """

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

        if (poigen, poieval) not in self.keys():
            toysresult = ToyResult(p, poieval)
            self.set_toyresult(poigen, poieval, toysresult)
        else:
            toysresult = self.get_toyresult(poigen, poieval)

        toysresult.add_entries(bestfit=bestfit, nll_bestfit=nll_bestfit, nlls=nlls)

    def keys(self):
        return self._toys.keys()

    def values(self):
        return self._toys.values()

    def items(self):
        return self._toys.items()

    def toyresults_to_dict(self):
        return [v.to_dict() for v in self.values()]

    def to_yaml(self, filename):
        """
        Save the toys into a yaml file.

        Args:
            filename (str)
        """
        if os.path.isfile(filename):
            tree = asdf.open(filename).tree
        else:
            tree = {}

        tree["toys"] = self.toyresults_to_dict()
        af = asdf.AsdfFile(tree)
        af.write_to(filename)

    def toysresults_from_yaml(self, filename):
        ret = {}

        toys = asdf.open(filename).tree["toys"]

        for t in toys:
            poiparam = None
            for p in self.loss.get_dependents():
                if t["poi"] == p.name:
                    poiparam = p

            if poiparam is None:
                raise ParameterNotFound(f"Parameter with name {t['poi']} is not found.")

            poigen = POI(poiparam, t["genvalue"])
            poieval = POIarray(poiparam, np.asarray(t["evalvalues"]))

            bestfit = t["bestfit"]
            nll_bestfit = t["nlls"]["bestfit"]
            nlls = {p: t["nlls"][p.value] for p in poieval}

            t = ToyResult(poigen, poieval)
            t.add_entries(bestfit=bestfit, nll_bestfit=nll_bestfit, nlls=nlls)

            ret[poigen, poieval] = t

        return ret

    @classmethod
    def from_yaml(cls, filename, loss, minimizer, sampler=base_sampler, sample=base_sample):
        """
        Read the toys from a yaml file.

        Args:
            filename (str)

        Returns
            `Toys`
        """

        toyscollection = cls(loss, minimizer, sampler, sample)
        toysresults = toyscollection.toysresults_from_yaml(filename)

        for (poigen, poieval), t in toysresults.items():
            toyscollection.set_toyresult(poigen, poieval, t)

        return toyscollection
