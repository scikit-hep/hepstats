"""
Module defining the classes to perform and store the results of toy experiments.

Acronyms used in the code:
    * nll = negative log-likehood, which is the value of the `loss` attribute of a calculator;
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import asdf
import numpy as np
from tqdm.auto import tqdm

from ..utils import base_sample, base_sampler, pll, set_values
from .exceptions import FormatError, ParameterNotFound
from .hypotests_object import ToysObject
from .parameters import POI, POIarray


class ToyResult:
    """
    Class to store the results of toys generated for a given value of a POI.
    The best fit value of the POI, the NLL evaluate at the best fit, and the NLL evaluated
    at several values of the POI are stored. The results can serialized using the **to_dict** method.
    """

    def __init__(self, poigen: POI, poieval: POIarray):
        """
        Args:
            poigen: POI used to generate the toys.
            poieval: POI values to evaluate the loss function.

        Raises:
            TypeError: if poigen is not a POI instance.
            TypeError: if poieval is not a POIarray instance.
        """

        if not isinstance(poigen, POI):
            msg = "A `hypotests.parameters.POI` is required for poigen."
            raise TypeError(msg)
        if not isinstance(poieval, POIarray):
            msg = "A `hypotests.parameters.POIarray` is required for poieval."
            raise TypeError(msg)

        self._poigen = poigen
        self._bestfit = np.array([])
        self._nll_bestfit = np.array([])
        self._poieval = poieval
        self._nlls = {p: np.array([]) for p in poieval}

    @property
    def poigen(self):
        """
        Returns the POI used to generate the toys.
        """
        return self._poigen

    @property
    def bestfit(self):
        """
        Returns the best fitted values of the POI for each toys.
        """
        return self._bestfit

    @property
    def poieval(self):
        """
        Returns the scanned POIarray.
        """
        return self._poieval

    @property
    def nll_bestfit(self):
        """
        Returns the NLL evaluated at the best fitted values of the POI for each toys.
        """
        return self._nll_bestfit

    @property
    def nlls(self):
        """
        Returns the NLL evaluated at the poigeval values of the POI for each toys.
        """
        return self._nlls

    @property
    def ntoys(self):
        """
        Returns the number of toys.
        """
        return len(self.bestfit)

    def add_entries(self, bestfit: np.ndarray, nll_bestfit: np.ndarray, nlls: dict[POI, np.ndarray]):
        """
        Add new result entries.

        Args:
            bestfit: best fitted values of the POI
            nll_bestfit: NLL evaluated at the best fitted values of the POI
            nlls: NLL evaluated at the best fitted values of the POI
        """
        if any(k not in nlls for k in self.poieval):
            missing_keys = [k for k in self.poieval if k not in nlls]
            msg = f"NLLs values for {missing_keys} are missing."
            raise ValueError(msg)

        nentries = bestfit.size
        assert nll_bestfit.size == nentries
        assert all(a.size == nentries for a in nlls.values())

        self._bestfit = np.concatenate([self.bestfit, bestfit])
        self._nll_bestfit = np.concatenate([self.nll_bestfit, nll_bestfit])

        self._nlls = {p: np.concatenate([v, nlls[p]]) for p, v in self.nlls.items()}

    def to_dict(self) -> dict:
        """
        Returns dictionary of the toy results.

        Keys:
            poi: name of the parameter of interest
            genvalues: fixed vale of the poi used to generate the toys
            evalvalues: values to evaluate the NLL
            bestfit: array of best fitted values of the poi for each toy
            nlls: dictionary of NLL values for each value in `evalvalues` and best fit
        """
        ret = {"poi": self.poigen.name, "bestfit": self.bestfit}
        ret["nlls"] = {n.value: nll for n, nll in self.nlls.items()}
        ret["genvalue"] = self.poigen.value
        ret["evalvalues"] = self.poieval.values
        ret["nlls"]["bestfit"] = self.nll_bestfit
        return ret


class FitFailuresWarning(UserWarning):
    pass


class ToysManager(ToysObject):
    """Class handling the toy generation and fit, results are stored in **ToyResult** instances stored
    themselves in a dictionary.
    """

    def __init__(
        self,
        input,
        minimizer,
        sampler: Callable = base_sampler,
        sample: Callable = base_sample,
    ):
        """
        Args:
            input: loss or fit result
            minimizer: minimizer to use to find the minimum of the loss function
            sampler: function used to create sampler with models, number of events and floating parameters in the
               sample. Default is :func:`hepstats.utils.fit.sampling.base_sampler`.
            sample: function used to get samples from the sampler. Default is
               :func:`hepstats.utils.fit.sampling.base_sample`.
        """

        super().__init__(input=input, minimizer=minimizer, sampler=sampler, sample=sample)
        self._toys = {}

    def get_toyresult(self, poigen: POI, poieval: POIarray) -> ToyResult:
        """
        Getter function.

        Args:
            poigen: POI used to generate the toys
            poieval: POI values to evaluate the loss function
        """

        index = (poigen, poieval)

        if index not in self.keys():
            for k in self.keys():
                poigen_k, poieval_k = k
                if poigen_k != poigen:
                    continue
                if np.isin(poieval.values, k[-1].values).all():
                    index = k
                    break

        return self._toys[index]

    def add_toyresult(self, toy: ToyResult):
        """
        Add ToyResult to the manager.

        Args:
            toy: the toy result to add
        """
        if not isinstance(toy, ToyResult):
            msg = "A `hypotests.toyutils.ToyResult` is required for toy."
            raise TypeError(msg)

        index = (toy.poigen, toy.poieval)

        self._toys[index] = toy

    def ntoys(self, poigen: POI, poieval: POIarray) -> int:
        """
        Return the number of toys generated from given value of a POI, and scanned/evaluated for given values_equal
        of the same POI.

        Args:
            poigen: POI used to generate the toys
            poieval: POI values to evaluate the loss function
        """
        try:
            return self.get_toyresult(poigen, poieval).ntoys
        except KeyError:
            return 0

    def generate_and_fit_toys(  # TODO PROFILE THIS
        self,
        ntoys: int,
        poigen: POI,
        poieval: POIarray,
    ):
        """
        Generate and fit toys for at a given POI (poigen). The toys are then fitted, and the likelihood
        is profiled at the values of poigen and poieval.

        Args:
            ntoys: number of toys to generate
            poigen: POI used to generate the toys
            poieval: POI values to evaluate the loss function
        """

        minimizer = self.minimizer
        param = poigen.parameter

        # generate toys from best fit at profile point
        param.set_value(poigen.value)
        is_poi_floating = param.floating
        param.floating = False
        profile_minimum = minimizer.minimize(loss=self.loss)
        if not profile_minimum.valid:
            print(poigen, "PROBLEM: profile fit not valid, toys won't be either")
        param.floating = is_poi_floating

        toys_loss = self.toys_loss(poigen.name)
        sampler = toys_loss.data

        bestfit = np.empty(ntoys)
        nll_bestfit = np.empty(ntoys)
        nlls = {p: np.empty(ntoys) for p in poieval}

        samples = self.sample(
            sampler=sampler,
            ntoys=int(ntoys * 1.2),
            poi=poigen,
            constraints=toys_loss.constraints,
        )

        try:
            toysresult = self.get_toyresult(poigen, poieval)
            poieval = toysresult.poieval
        except KeyError:
            toysresult = ToyResult(poigen, poieval)
            self.add_toyresult(toysresult)

        nfailures = 0
        ntrials = 0

        progressbar = tqdm(total=ntoys)
        minimum = None

        for i in range(ntoys):
            ntrials += 1
            converged = False
            while not converged:
                try:
                    param_dict = next(samples)
                except StopIteration:
                    to_gen = ntoys - i
                    samples = self.sample(
                        sampler=sampler,
                        ntoys=int(to_gen * 1.2),
                        poi=poigen,
                        constraints=toys_loss.constraints,
                    )
                    param_dict = next(samples)

                with set_values(param_dict):
                    for _ in range(2):
                        try:
                            minimum = minimizer.minimize(
                                loss=toys_loss
                            )  # TODO: , init=minimum use previous minimum as starting point for parameter uncertainties
                            converged = minimum.converged
                            if converged:
                                break
                        except RuntimeError:
                            converged = False
                            break

                if not converged:
                    self.set_params_to_bestfit()
                    nfailures += 1
                    if nfailures > 0.15 * ntrials and ntrials > 10:
                        msg = f"{nfailures} out of {ntrials} fits failed or didn't converge."
                        warnings.warn(msg, FitFailuresWarning, stacklevel=2)
                    continue
                if minimum is None:
                    msg = "No minimum found."
                    raise RuntimeError(msg)
                bestfit[i] = minimum.params[param]["value"]
                nll_bestfit[i] = pll(minimizer, toys_loss, POI(param, bestfit[i]))

                for p in poieval:
                    nlls[p][i] = pll(minimizer, toys_loss, p)

            progressbar.update(1)

            if i > ntoys:
                break
            i += 1

        progressbar.close()

        if (poigen, poieval) not in self.keys():
            toysresult = ToyResult(p, poieval)
            self.add_toyresult(toysresult)
        else:
            toysresult = self.get_toyresult(poigen, poieval)

        toysresult.add_entries(bestfit=bestfit, nll_bestfit=nll_bestfit, nlls=nlls)

    def keys(self):
        """
        Returns keys of the **ToysManager** instance defined as **key = (toy.poigen, toy.poieval)** for a
        given **ToyResult** instance `toy`.
        """
        return self._toys.keys()

    def values(self):
        """
        Returns values of **ToysManager** instance that are **ToyResult** instances.
        """
        return self._toys.values()

    def toyresults_to_dict(self) -> list[dict]:
        """
        Returns a list of all the toy results converted into dictionaries.
        """
        return [v.to_dict() for v in self.values()]

    def to_yaml(self, filename: str):
        """
        Save the toys into a yaml file under the key **toys**.

        Args:
            filename: the yaml file name.
        """
        tree = asdf.open(filename).tree if Path(filename).is_file() else {}

        tree["toys"] = self.toyresults_to_dict()
        af = asdf.AsdfFile(tree)
        af.write_to(
            filename, version="1.5.0"
        )  # TODO: change the dict layout not to have floats as keys. Fails with new ASDF version
        af.close()

    def toysresults_from_yaml(self, filename: str) -> list[ToyResult]:
        """
        Extract toy results from a yaml file.

        Args:
            filename: the yaml file name.
        """
        ret = []
        with asdf.open(filename) as asdf_file:
            try:
                toys = asdf_file.tree["toys"]
            except KeyError as error:
                msg = f"The key `toys` is not found in {filename}, not a valid toy file."
                raise FormatError(msg) from error

            for t in toys:
                poiparam = None
                for p in self.loss.get_params():
                    if t["poi"] == p.name:
                        poiparam = p

                if poiparam is None:
                    msg = f"Parameter with name {t['poi']} is not found."
                    raise ParameterNotFound(msg)

                poigen = POI(poiparam, t["genvalue"])
                poieval = POIarray(poiparam, np.asarray(t["evalvalues"]))

                bestfit = t["bestfit"]
                nll_bestfit = t["nlls"]["bestfit"]
                nlls = {p: t["nlls"][p.value] for p in poieval}

                t = ToyResult(poigen, poieval)
                t.add_entries(bestfit=bestfit, nll_bestfit=nll_bestfit, nlls=nlls)
                ret.append(t)

        return ret

    @classmethod
    def from_yaml(
        cls,
        filename: str,
        input,
        minimizer,
        sampler: Callable = base_sampler,
        sample: Callable = base_sample,
    ):
        """
        Read the toys from a yaml file.

        Args:
            filename: the yaml file name.
            input: loss or fit result
            minimizer: minimizer to use to find the minimum of the loss function
            sampler: function used to create sampler with models, number of events and floating parameters in the
               sample. Default is :func:`hepstats.utils.fit.sampling.base_sampler`.
            sample: function used to get samples from the sampler. Default is
               :func:`hepstats.utils.fit.sampling.base_sample`.
        """

        toyscollection = cls(input, minimizer, sampler, sample)
        toysresults = toyscollection.toysresults_from_yaml(filename)

        for t in toysresults:
            toyscollection.add_toyresult(t)

        return toyscollection
