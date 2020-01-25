import asdf
import os
import numpy as np

from .parameters import POI, POIarray
from .exceptions import ParameterNotFound


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

        self._poigen = poigen
        self._bestfit = []
        self._nll_bestfit = []
        self._poieval = poieval
        self._nlls = {p: [] for p in poieval}

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
        assert all(k in self.poieval for k in nlls.keys())

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

        toys = self.copy()
        toys.add_entries(bestfit=toys.bestfit, nll_bestfit=toys.nll_bestfit,
                         nlls=toys.nlls)

        return toys

    def to_dict(self):
        ret = {"poi": self.poigen.name, "bestfit": self.bestfit}
        ret["nlls"] = {n.value: nll for n, nll in self.nlls.items()}
        ret["genvalue"] = self.poigen.value
        ret["evalvalues"] = self.poieval.values
        ret["nlls"]["bestfit"] = self.nll_bestfit
        return ret

    def copy(self):
        toys = ToyResult(self.poigen, self.poieval)
        toys.add_entries(bestfit=toys.bestfit, nll_bestfit=toys.nll_bestfit,
                         nlls=toys.nlls)
        return toys


class ToysManager(object):
    """
    Class containing instances of `Toys` in a dictionnary
    """

    def __init__(self):
        self._toys = {}

    def __getitem__(self, index):
        """
        Getter function.

        Args:
            index (`POI`, `POIarray`): POI used to generate the toys and
                POI values to evaluate the loss function

        Returns:
            `Toys`
        """
        return self._toys[index]

    def __setitem__(self, index, toy):
        """
        Setter function.

        Args:
            index (`POI`, `POIarray`): POI used to generate the toys and
                POI values to evaluate the loss function
            toy: (`Toys`)
        """
        poigen, poieval = index

        if not isinstance(poigen, POI):
            raise TypeError("A `hypotests.parameters.POI` is required for poigen.")
        if not isinstance(poieval, POIarray):
            raise TypeError("A `hypotests.parameters.POIarray` is required for poieval.")
        if not isinstance(toy, ToyResult):
            raise TypeError("A `hypotests.toyutils.Toys` is required for toy.")

        self._toys[index] = toy

    def keys(self):
        return self._toys.keys()

    def values(self):
        return self._toys.values()

    def items(self):
        return self._toys.items()

    def __contains__(self, index):
        return index in self._toys

    def __add__(self, toyscollection):
        if not isinstance(toyscollection, ToysManager):
            raise TypeError("A `ToysCollection` is required.")

        ntoyscollection = self.copy()

        for index, toy in toyscollection.items():
            if index not in ntoyscollection:
                ntoyscollection[index] = toy
            else:
                ntoyscollection[index] = ntoyscollection[index] + toy

        return ntoyscollection

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

        tree["toys"]: [v.to_dict() for v in self._toys.values()]
        af = asdf.AsdfFile(tree)
        af.write_to(filename)

    @classmethod
    def from_yaml(cls, filename, parameters):
        """
        Read the toys from a yaml file. At least one parameter in the argument `parameters` must have
        the name of the parameter of interest save in the yaml file.

        Args:
            filename (str)
            parameters (list): list of parameters

        Returns
            `Toys`
        """

        toys = asdf.open(filename).tree["toys"]
        toyscollection = cls()

        for t in toys:
            poiparam = None
            for p in parameters:
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

            toyscollection._toys[poigen, poieval] = t

        return toyscollection

    def copy(self):
        toyscollection = ToysManager()
        toyscollection._toys = dict(self._toys)
        return toyscollection
