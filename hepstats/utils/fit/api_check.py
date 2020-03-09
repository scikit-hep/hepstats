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
    has_value = hasattr(object, "value")
    has_set_value = hasattr(object, "set_value")
    has_floating = hasattr(object, "floating")

    return has_value and has_set_value and has_floating


def is_valid_data(object):
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
    has_get_dependents = hasattr(object, "get_dependents")
    if not has_get_dependents:
        return False
    else:
        params = object.get_dependents()

    all_valid_params = all(is_valid_parameter(p) for p in params)
    has_pdf = hasattr(object, "pdf")
    has_integrate = hasattr(object, "integrate")
    has_sample = hasattr(object, "sample")
    has_space = hasattr(object, "space")
    has_get_yield = hasattr(object, "get_yield")

    return all_valid_params and has_pdf and has_integrate and has_sample and has_space and has_get_yield


def is_valid_loss(object):
    if not hasattr(object, "model"):
        return False
    else:
        model = object.model

    if not hasattr(object, "data"):
        return False
    else:
        data = object.data

    has_constraints = hasattr(object, "constraints")
    has_fit_range = hasattr(object, "fit_range")
    all_valid_pdfs = all(is_valid_pdf(m) for m in model)
    all_valid_datasets = all(is_valid_data(d) for d in data)

    return all_valid_pdfs and all_valid_datasets and has_constraints and has_fit_range


def is_valid_fitresult(object):
    has_loss = hasattr(object, "loss")

    if not has_loss:
        return False
    else:
        loss = object.loss
        has_params = hasattr(object, "params")
        return is_valid_loss(loss) and has_params


def is_valid_minimizer(object):
    has_minimize = hasattr(object, "minimize")
    return has_minimize
