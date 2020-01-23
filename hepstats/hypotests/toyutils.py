import asdf
import numpy as np

from .parameters import POI, POIarray


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

    @classmethod
    def from_dict(cls, poigen, dict):
        poieval = POIarray(poigen.parameter, dict["evalvalues"])
        bestfit = dict["bestfit"]
        nll_bestfit = dict["nll_bestfit"]
        nlls = {p: dict["nlls"][p.value] for p in poieval}

        t = cls.__init__(poigen, poieval)
        t.add_entries(besfit=bestfit, nll_bestfit=nll_bestfit, nlls=nlls)
        return t


class ToysCollection(object):

    def __init__(self):
        self._toys = {}

    def __getitem__(self, index):
        return self._toys[index]

    def __setitem__(self, index, toy):
        poigen, poieval = index

        if not isinstance(poigen, POI):
            raise NotImplementedError
        if not isinstance(poieval, POIarray):
            raise NotImplementedError
        if not isinstance(toy, Toys):
            raise NotImplementedError

        self._toys[index] = toy

    def __contains__(self, index):
        return index in self._toys

    def to_dict(self):
        pass
        return {(p, p.value): t.dict() for (pg, pe), t in self._toys.items()}

    def to_yaml(self, filename):
        tree = {"toys": self.to_dict()}
        af = asdf.AsdfFile(tree)
        af.write_to(filename)

    @classmethod
    def from_yaml(cls, filename, parameters):
        dict_from_yaml = asdf.open(filename).tree["toys"]
        tc = cls.__init__()

        for (n, v), t in dict_from_yaml.items():
            poiparam = None
            for p in parameters:
                if n == p.name:
                    poiparam = p

            if poiparam is None:
                raise NotImplementedError

            poigen = POI(poiparam, v)

            tc._toys[poigen] = Toys.from_dict(poigen, t)

        return tc
