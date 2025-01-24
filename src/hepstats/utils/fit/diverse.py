from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, suppress

import numpy as np


def get_ndims(dataset):
    """Return the number of dimensions in the dataset"""
    return len(dataset.obs)


def get_value(value):
    return np.array(value)


def set_values_once(params, values):
    with suppress(ImportError):
        import zfit

        return zfit.param.set_values(params, values)  # more efficient

    for p, v in zip(params, values):
        p.set_value(v)
    return None


def eval_pdf(model, x, params=None, allow_extended=False):
    """Compute pdf of model at a given point x and for given parameters values"""

    if params is None:
        params = {}

    def pdf(model, x):
        ret = model.ext_pdf(x) if model.is_extended and allow_extended else model.pdf(x)

        return get_value(ret)

    with ExitStack() as stack:
        for param in model.get_params():
            if param in params:
                value = params[param]["value"]
                stack.enter_context(param.set_value(value))
        return pdf(model, x)


def pll(minimizer, loss, pois, init=None) -> float:
    """Compute minimum profile likelihood for fixed given parameters values."""
    del init  # unused currently

    with ExitStack() as stack:
        for p in pois:
            param = p.parameter
            stack.enter_context(param.set_value(p.value))
            param.floating = False

        if any(param_loss.floating for param_loss in loss.get_params()):
            minimum = minimizer.minimize(loss=loss)  # TODO: add init?
            value = minimum.fmin
        else:
            value = get_value(loss.value())

        for p in pois:
            p.parameter.floating = True

    return value


@contextmanager
def set_values(params, values=None):
    if values is None:
        if isinstance(params, Mapping):
            values = tuple(params.values())
            params = tuple(params.keys())
        else:
            msg = "values must be provided if params is not a Mapping (dict-like)"
            raise ValueError(msg)
    old_values = [p.value() for p in params]
    for p, v in zip(params, values):
        p.set_value(v)
    yield
    for p, v in zip(params, old_values):
        p.set_value(v)


def array2dataset(dataset_cls, obs, array, weights=None):
    """
    dataset_cls: only used to get the class in which array/weights will be
    converted.
    """

    if hasattr(dataset_cls, "from_numpy"):
        return dataset_cls.from_numpy(obs, array=array, weights=weights)
    else:
        return dataset_cls(obs, array=array, weights=weights)


def get_nevents(dataset):
    """Returns the number of events in the dataset"""

    return get_value(dataset.nevents)
