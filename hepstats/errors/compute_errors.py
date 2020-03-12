import numpy as np
from scipy import optimize

from ..utils.fit.api_check import is_valid_fitresult
from ..utils import pll


def set_params_to_fitresult(params, fit_result):
    for param in params:
        param.set_value(fit_result.params[param]["value"])


def get_crossing_value(fit_result, parameters, direction, sigma, covariance):

    assert is_valid_fitresult(fit_result)
    all_parameters = list(fit_result.params.keys())
    loss = fit_result.loss
    up = loss.errordef
    fmin = fit_result.fmin
    minimizer = fit_result.minimizer

    sigma = sigma * direction

    result = {}

    for param in parameters:
        set_params_to_fitresult(all_parameters, fit_result)
        hesse_error = covariance[(param, param)] ** 0.5
        fitted_value = fit_result.params[param]["value"]

        for ap in all_parameters:
            if ap == param:
                continue

            ap_value = fit_result.params[ap]["value"]
            ap_error = covariance[(ap, ap)] ** 0.5

            ap_value += sigma * covariance[(param, ap)] * np.sqrt(up * 2) / ap_error
            ap.set_value(ap_value)

        if direction == -1:
            lower_bound = fitted_value + 2 * hesse_error * sigma
            upper_bound = fitted_value
        else:
            lower_bound = fitted_value
            upper_bound = fitted_value + 2 * hesse_error * sigma

        root, results = optimize.toms748(
            lambda v: pll(minimizer, loss, param, v) - fmin - up,
            lower_bound,
            upper_bound,
            full_output=True,
            k=2,
        )

        result[param] = root

    return result


def compute_errors(fit_result, parameters=None, sigma=1):

    assert is_valid_fitresult(fit_result)
    all_parameters = list(fit_result.params.keys())
    covariance = fit_result.covariance(as_dict=True)

    if parameters is None:
        parameters = all_parameters
    elif not isinstance(parameters, (list, tuple)):
        parameters = [parameters]

    upper_values = get_crossing_value(
        fit_result=fit_result,
        parameters=parameters,
        direction=1,
        sigma=sigma,
        covariance=covariance,
    )

    lower_values = get_crossing_value(
        fit_result=fit_result,
        parameters=parameters,
        direction=-1,
        sigma=sigma,
        covariance=covariance,
    )

    result = {}

    for param in parameters:
        fitted_value = fit_result.params[param]["value"]
        result[param] = {
            "lower": lower_values[param] - fitted_value,
            "upper": upper_values[param] - fitted_value,
        }

    return result
