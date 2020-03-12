from contextlib import ExitStack
import numpy as np

from .sampling import base_sampler, base_sample
from .api_check import is_valid_parameter


def get_value(value):
    return np.array(value)


def eval_pdf(model, x, params={}, allow_extended=False):
    """ Compute pdf of model at a given point x and for given parameters values """

    def pdf(model, x):
        if model.is_extended and allow_extended:
            ret = model.pdf(x) * model.get_yield()
        else:
            ret = model.pdf(x)

        return get_value(ret)

    with ExitStack() as stack:
        for param in model.get_dependents():
            if param in params:
                value = params[param]["value"]
                stack.enter_context(param.set_value(value))
        return pdf(model, x)


def convert_to_list(object):
    if not isinstance(object, list):
        object = [object]
    return object


def pll(minimizer, loss, param, value) -> float:
    """ Compute minimum profile likelihood for given parameters values. """
    verbosity = minimizer.verbosity

    param = convert_to_list(param)
    value = convert_to_list(value)

    assert all(is_valid_parameter(p) for p in param)
    if not len(param) == len(value):
        raise ValueError(
            f"Incompatible length of parameters and values: {param}, {value}"
        )

    with ExitStack() as stack:
        for p, v in zip(param, value):
            stack.enter_context(p.set_value(v))
            p.floating = False
        minimizer.verbosity = 0
        minimum = minimizer.minimize(loss=loss)

    for p in param:
        p.floating = True
    minimizer.verbosity = verbosity

    return minimum.fmin


def array2dataset(dataset_cls, obs, array, weights=None):
    """
    dataset_cls: only used to get the class in which array/weights will be
    converted.
    """

    if hasattr(dataset_cls, "from_numpy"):
        return dataset_cls.from_numpy(obs=obs, array=array, weights=weights)
    else:
        return dataset_cls(obs=obs, array=array, weights=weights)


def get_nevents(dataset):
    """ Returns the number of events in the dataset """

    return get_value(dataset.nevents)
