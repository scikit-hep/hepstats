import asdf
import numpy as np

from .parameters import POI, POIarray
from .exceptions import ParameterNotFound


class Toys(object):

    def __init__(self, poigen, poieval):
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

    def to_dict(self):
        ret = {"poi": self.poigen.name, "bestfit": self.bestfit}
        ret["nlls"] = {n: nll for n, nll in self.nlls.items()}
        ret["genvalue"] = self.poigen.value
        ret["evalvalues"] = self.poieval.values
        ret["nlls"]["bestfit"] = self.nll_bestfit
        return ret


class ToysCollection(object):

    def __init__(self):
        self._toys = {}

    def __getitem__(self, index):
        return self._toys[index]

    def __setitem__(self, index, toy):
        poigen, poieval = index

        if not isinstance(poigen, POI):
            raise TypeError("A `hepstats.parameters.POI` is required for poigen.")
        if not isinstance(poieval, POIarray):
            raise TypeError("A `hepstats.parameters.POIarray` is required for poieval.")
        if not isinstance(toy, Toys):
            raise TypeError("A `hepstats.toyutils.Toys` is required for toy.")

        self._toys[index] = toy

    def __contains__(self, index):
        return index in self._toys

    def to_yaml(self, filename):
        tree = {"toys": [v.to_dict() for v in self._toys.values()]}
        af = asdf.AsdfFile(tree)
        af.write_to(filename)

    @classmethod
    def from_yaml(cls, filename, parameters):
        toys = asdf.open(filename).tree["toys"]
        tc = cls.__init__()

        for t in toys:
            poiparam = None
            for p in parameters:
                if t["poi"] == p.name:
                    poiparam = p

            if poiparam is None:
                raise ParameterNotFound(f"Parameter with name {t['poi']} is not found.")

            poigen = POI(poiparam, t["genvalue"])
            poieval = POIarray(poiparam, t["evalvalues"])

            bestfit = t["bestfit"]
            nll_bestfit = t["nll_bestfit"]
            nlls = {p: t["nlls"][p.value] for p in poieval}

            t = Toys.from_dict(poigen, t)
            t.add_entries(besfit=bestfit, nll_bestfit=nll_bestfit, nlls=nlls)

            tc._toys[poigen, poieval] = t

        return tc
