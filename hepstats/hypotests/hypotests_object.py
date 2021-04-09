import warnings

from ..utils.fit import get_nevents
from ..utils.fit.api_check import is_valid_loss, is_valid_fitresult, is_valid_minimizer, is_valid_data, is_valid_pdf


class HypotestsObject(object):
    """Base object in `hepstats.hypotests` to manipulate a loss function and a minimizer.

        Args:
            * **input** : loss or fit result
            * **minimizer** : minimizer to use to find the minimum of the loss function
    """

    def __init__(self, input, minimizer):

        if is_valid_fitresult(input):
            self._loss = input.loss
            self._bestfit = input
        elif is_valid_loss(input):
            self._loss = input
            self._bestfit = None
        else:
            raise ValueError("{} is not a valid loss funtion or fit result!".format(input))

        if not is_valid_minimizer(minimizer):
            raise ValueError("{} is not a valid minimizer !".format(minimizer))

        self._minimizer = minimizer
        self.minimizer.verbosity = 0

        self._parameters = {}
        for m in self.model:
            for d in m.get_params():
                self._parameters[d.name] = d

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
            print("Get fit best values!")
            self.minimizer.verbosity = 5
            mininum = self.minimizer.minimize(loss=self.loss)
            self.minimizer.verbosity = 0
            self._bestfit = mininum
            return self._bestfit

    @bestfit.setter
    def bestfit(self, value):
        """
        Set the best fit values  of the model parameters.

            Args:
                * **value**: fit result
        """
        if not is_valid_fitresult(value):
            raise ValueError("{} is not a valid fit result!".format(input))
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

    def get_parameter(self, name):
        """
        Returns the parameter in loss function with given input name.

        Args:
            name (str): name of the parameter to return
        """
        return self._parameters[name]

    def set_dependents_to_bestfit(self):
        """
        Set the values of the parameters in the models to the best fit values
        """
        for m in self.model:
            for d in m.get_params():
                d.set_value(self.bestfit.params[d]["value"])

    def lossbuilder(self, model, data, weights=None):
        """ Method to build a new loss function.

            Args:
                * **model** (List): The model or models to evaluate the data on
                * **data** (List): Data to use
                * **weights** (optional, List): the data weights

            Example with `zfit`:
                >>> data = zfit.data.Data.from_numpy(obs=obs, array=np.random.normal(1.2, 0.1, 10000))
                >>> mean = zfit.Parameter("mu", 1.2)
                >>> sigma = zfit.Parameter("sigma", 0.1)
                >>> model = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
                >>> loss = calc.lossbuilder(model, data)

            Returns:
                Loss function

        """

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
                d.set_weights(w)

        # fix for newly introduce https://github.com/zfit/zfit-development/issues/68
        if hasattr(self.loss, 'create_new'):
            loss = self.loss.create_new(model=model, data=data, constraints=self.constraints)
        else:
            warnings.warn("A loss should have a `create_new` method. If you are using zfit, please make sure to"
                          "upgrade to >= 0.6.3", FutureWarning)
            loss = type(self.loss)(model=model, data=data)
            loss.add_constraints(self.constraints)

        return loss


class ToysObject(HypotestsObject):
    """Base object in `hepstats.hypotests` to manipulate a loss function, a minimizer and sample a
    model (within the loss function) to do toy experiments.

        Args:
            * **input** : loss or fit result
            * **minimizer** : minimizer to use to find the minimum of the loss function
            * **sampler** : function used to create sampler with models, number of events and floating parameters in the sample.
            * **sample** : function used to get samples from the sampler.
    """

    def __init__(self, input, minimizer, sampler, sample):

        super(ToysObject, self).__init__(input, minimizer)
        self._toys = {}
        self._sampler = sampler
        self._sample = sample
        self._toys_loss = {}

    def sampler(self, floating_params=None):
        """
        Create sampler with models.

        Args:
            * **floating_params** (list): floating parameters in the sampler

        Example with `zfit`:
            >>> sampler = calc.sampler(floating_params=[zfit.Parameter("mean")])
        """
        self.set_dependents_to_bestfit()
        nevents = []
        for m, d in zip(self.loss.model, self.loss.data):
            if m.is_extended:
                nevents.append("extended")
            else:
                nevents.append(get_nevents(d))

        return self._sampler(self.loss.model, nevents, floating_params)

    def sample(self, sampler, ntoys, poi=None):
        """
        Returns the samples generated from the sampler for a given value of a parameter of interest

        Args:
            * **sampler** (list): generator of samples
            * **ntoys** (int): number of samples to generate
            * **poi** (POI, optional):  in the sampler

        Example with `zfit`:
            >>> mean = zfit.Parameter("mean")
            >>> sampler = calc.sampler(floating_params=[mean])
            >>> sample = calc.sample(sampler, 1000, POI(mean, 1.2))
        """
        return self._sample(sampler, ntoys, parameter=poi.parameter, value=poi.value)

    def toys_loss(self, parameter_name):
        """
        Construct a loss function constructed with a sampler for a given floating parameter

        Args:
            * **parameter_name**: name floating parameter in the sampler
        Returns:
             Loss function

        Example with `zfit`:
            >>> loss = calc.toys_loss(zfit.Parameter("mean"))
        """
        if parameter_name not in self._toys_loss:
            parameter = self.get_parameter(parameter_name)
            sampler = self.sampler(floating_params=[parameter])
            self._toys_loss[parameter.name] = self.lossbuilder(self.model, sampler)
        return self._toys_loss[parameter_name]
