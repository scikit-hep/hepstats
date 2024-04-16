from __future__ import annotations

import warnings

import numpy as np

from ..utils.fit import get_nevents, set_values_once
from ..utils.fit.api_check import is_valid_data, is_valid_fitresult, is_valid_loss, is_valid_minimizer, is_valid_pdf
from .parameters import POI


class HypotestsObject:
    """Base object in `hepstats.hypotests` to manipulate a loss function and a minimizer.

    Args:
        input: loss or fit result
        minimizer: minimizer to use to find the minimum of the loss function
    """

    def __init__(self, input, minimizer, **kwargs):
        super().__init__(**kwargs)
        if is_valid_fitresult(input):
            self._loss = input.loss
            self._bestfit = input
        elif is_valid_loss(input):
            self._loss = input
            self._bestfit = None
        else:
            msg = f"{input} is not a valid loss function or fit result!"
            raise ValueError(msg)

        if not is_valid_minimizer(minimizer):
            msg = f"{minimizer} is not a valid minimizer !"
            raise ValueError(msg)

        self._minimizer = minimizer
        self.minimizer.verbosity = 0

        self._parameters = {param.name: param for param in self.loss.get_params()}

    @property
    def loss(self):
        """
        Returns the loss / likelihood function.
        """
        return self._loss

    @property
    def minimizer(self):
        """
        Returns the minimizer.
        """
        return self._minimizer

    @property
    def bestfit(self):
        """
        Returns the best fit values of the model parameters.
        """
        if getattr(self, "_bestfit", None):
            return self._bestfit
        else:
            old_verbosity = self.minimizer.verbosity
            self.minimizer.verbosity = 5
            minimum = self.minimizer.minimize(loss=self.loss)
            self.minimizer.verbosity = old_verbosity
            self._bestfit = minimum
            return self._bestfit

    @bestfit.setter
    def bestfit(self, value):
        """
        Set the best fit values  of the model parameters.

        Args:
            value: fit result
        """
        if not is_valid_fitresult(value):
            msg = f"{input} is not a valid fit result!"
            raise ValueError(msg)
        self._bestfit = value

    @property
    def model(self):
        """
        Returns the model.
        """
        return self.loss.model

    @property
    def data(self):
        """
        Returns the data.
        """
        return self.loss.data

    @property
    def constraints(self):
        """
        Returns the constraints on the loss / likehood function.
        """
        return self.loss.constraints

    def get_parameter(self, name: str):
        """
        Returns the parameter in loss function with given input name.

        Args:
            name: name of the parameter to return
        """
        return self._parameters[name]

    @property
    def parameters(self):
        """
        Returns the list of free parameters in loss / likelihood function.
        """
        return list(self._parameters.values())

    def set_params_to_bestfit(self):
        """
        Set the values of the parameters in the models to the best fit values
        """
        set_values_once(self.parameters, self.bestfit)

    def lossbuilder(self, model, data, weights=None, oldloss=None):
        """Method to build a new loss function.

        Args:
            * **model** (List): The model or models to evaluate the data on
            * **data** (List): Data to use
            * **weights** (optional, List): the data weights
            * **oldloss**: Previous loss that has data, models, type

        Example with `zfit`:
            >>> data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
            >>> mean = zfit.Parameter("mu", 1.2)
            >>> sigma = zfit.Parameter("sigma", 0.1)
            >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
            >>> loss = calc.lossbuilder(model, data)

        Returns:
            Loss function

        """

        if oldloss is None:
            oldloss = self.loss
        assert all(is_valid_pdf(m) for m in model)
        assert all(is_valid_data(d) for d in data)

        msg = "{0} must have the same number of components as {1}"
        if len(data) != len(self.data):
            raise ValueError(msg.format("data", "`self.data"))
        if len(model) != len(self.model):
            raise ValueError(msg.format("model", "`self.model"))
        if weights is not None and len(weights) != len(self.data):
            raise ValueError(msg.format("weights", "`self.data`"))

        if weights is not None:
            for d, w in zip(data, weights):
                d = d.with_weights(w)

        if hasattr(oldloss, "create_new"):
            loss = oldloss.create_new(model=model, data=data, constraints=self.constraints)
        else:
            warnings.warn(
                "A loss should have a `create_new` method. If you are using zfit, please make sure to"
                "upgrade to >= 0.6.4",
                FutureWarning,
                stacklevel=2,
            )
            loss = type(oldloss)(model=model, data=data)
            loss.add_constraints(self.constraints)

        return loss


class ToysObject(HypotestsObject):
    """Base object in `hepstats.hypotests` to manipulate a loss function, a minimizer and sample a
    model (within the loss function) to do toy experiments.

        Args:
            input: loss or fit result
            minimizer: minimizer to use to find the minimum of the loss function
            sampler: function used to create sampler with models, number of events and floating parameters
            in the sample.
            sample: function used to get samples from the sampler.
    """

    def __init__(self, input, minimizer, sampler, sample):
        super().__init__(input, minimizer)
        self._toys = {}
        self._sampler = sampler
        self._sample = sample
        self._toys_loss = {}

    def sampler(self):
        """
        Create sampler with models.

        >>> sampler = calc.sampler()
        """
        self.set_params_to_bestfit()
        nevents = []
        for m, d in zip(self.loss.model, self.loss.data):
            nevents_data = get_nevents(d)
            if m.is_extended:
                nevents.append(np.random.poisson(lam=nevents_data))  # TODO: handle constraint yields correctly?
            else:
                nevents.append(nevents_data)

        return self._sampler(self.loss.model, nevents)

    def sample(self, sampler, ntoys, poi: POI, constraints=None):
        """
        Generator function of samples from the sampler for a given value of a parameter of interest. Returns a
        dictionnary of samples constraints in any.

        Args:
            sampler (list): generator of samples
            ntoys (int): number of samples to generate
            poi (POI):  in the sampler
            constraints (list, optional): list of constraints to sample

        Example with `zfit`:
            >>> mean = zfit.Parameter("mean")
            >>> sampler = calc.sampler()
            >>> sample = calc.sample(sampler, 1000, POI(mean, 1.2))

        Returns:
            dictionnary of sampled values of the constraints at each iteration
        """
        return self._sample(
            sampler,
            ntoys,
            parameter=poi.parameter,
            value=poi.value,
            constraints=constraints,
        )

    def toys_loss(self, parameter_name: str):
        """
        Construct a loss function constructed with a sampler for a given floating parameter

        Args:
            parameter_name: name floating parameter in the sampler
        Returns:
             Loss function

        Example with `zfit`:
            >>> loss = calc.toys_loss(zfit.Parameter("mean"))
        """
        if parameter_name not in self._toys_loss:
            parameter = self.get_parameter(parameter_name)
            sampler = self.sampler()
            self._toys_loss[parameter.name] = self.lossbuilder(self.model, sampler)
        return self._toys_loss[parameter_name]
