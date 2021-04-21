# -*- coding: utf-8 -*-
"""
Module for testing a fitting library validity with hepstats.

A fitting library should provide six basic objects:

    * model / probability density function
    * parameters of the models
    * data
    * loss / likelihood function
    * minimizer
    * fitresult (optional)

A function for each object is defined in this module, all should return `True` to work
with hepstats.

The `zfit` API is currently the standard fitting API in hepstats.

"""


def is_valid_parameter(object):
    """
    Checks if a parameter has the following attributes/methods:
        * value
        * set_value
        * floating
    """
    has_value = hasattr(object, "value")
    has_set_value = hasattr(object, "set_value")
    has_floating = hasattr(object, "floating")

    return has_value and has_set_value and has_floating


def is_valid_data(object):
    """
    Checks if the data object has the following attributes/methods:
        * nevents
        * weights
        * set_weights
        * space
    """
    is_sampled_data = hasattr(object, "resample")

    try:
        has_nevents = hasattr(object, "nevents")
    except RuntimeError:
        if is_sampled_data:
            object.resample()
            has_nevents = hasattr(object, "nevents")
        else:
            has_nevents = False

    has_weights = hasattr(object, "weights")
    has_set_weights = hasattr(object, "set_weights")
    has_space = hasattr(object, "space")
    return has_nevents and has_weights and has_set_weights and has_space


def is_valid_pdf(object):
    """
    Checks if the pdf object has the following attributes/methods:
        * get_params
        * pdf
        * integrate
        * sample
        * get_yield

    Also the function **is_valid_parameter** is called with each of the parameters returned by get_params
    as argument.
    """
    has_get_params = hasattr(object, "get_params")
    if not has_get_params:
        return False
    else:
        params = object.get_params()

    all_valid_params = all(is_valid_parameter(p) for p in params)
    has_pdf = hasattr(object, "pdf")
    has_integrate = hasattr(object, "integrate")
    has_sample = hasattr(object, "sample")
    has_space = hasattr(object, "space")
    has_get_yield = hasattr(object, "get_yield")

    return (
        all_valid_params
        and has_pdf
        and has_integrate
        and has_sample
        and has_space
        and has_get_yield
    )


def is_valid_loss(object):
    """
    Checks if the loss object has the following attributes/methods:
        * model
        * data
        * get_params
        * constraints
        * fit_range

    Also the function **is_valid_pdf** is called with each of the models returned by model
    as argument. Additionnally the function **is_valid_data** is called with each of the data objects
    return by data as argument.
    """
    if not hasattr(object, "model"):
        return False
    else:
        model = object.model

    if not hasattr(object, "data"):
        return False
    else:
        data = object.data

    has_get_params = hasattr(object, "get_params")
    has_constraints = hasattr(object, "constraints")
    has_create_new = hasattr(object, "create_new")
    if not has_create_new:
        warnings.warn(
            "Loss should have a `create_new` method.", FutureWarning, stacklevel=3
        )
        has_create_new = True  # TODO: allowed now, will be dropped in the future
    all_valid_pdfs = all(is_valid_pdf(m) for m in model)
    all_valid_datasets = all(is_valid_data(d) for d in data)

    return (
        all_valid_pdfs
        and all_valid_datasets
        and has_constraints
        and has_create_new
        and has_get_params
    )


def is_valid_fitresult(object):
    """
    Checks if the fit result object has the following attributes/methods:
        * loss
        * params
        * covariance

    Also the function **is_valid_loss** is called with the loss as argument.
    """
    has_loss = hasattr(object, "loss")

    if not has_loss:
        return False
    else:
        loss = object.loss
        has_params = hasattr(object, "params")
        has_covariance = hasattr(object, "covariance")
        return is_valid_loss(loss) and has_params and has_covariance


def is_valid_minimizer(object):
    """
    Checks if the minimzer object has the following attributes/methods:
        * minimize
    """
    has_minimize = hasattr(object, "minimize")
    return has_minimize
