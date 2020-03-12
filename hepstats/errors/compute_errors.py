import numpy as np
from scipy import optimize

from ..utils.fit.api_check import is_valid_fitresult
from ..utils import pll, convert_to_list


def set_params_to_result(params, result):
    for param in params:
        param.set_value(result.params[param]["value"])


def get_crossing_value(result, params, direction, sigma):

    all_params = list(result.params.keys())
    loss = result.loss
    up = loss.errordef
    fmin = result.fmin
    minimizer = result.minimizer.copy()
    covariance = result.covariance(as_dict=True)
    sigma = sigma * direction

    to_return = {}
    for param in params:
        set_params_to_result(all_params, result)
        hesse_error = covariance[(param, param)] ** 0.5
        fitted_value = result.params[param]["value"]

        for ap in all_params:
            if ap == param:
                continue

            ap_value = result.params[ap]["value"]
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

        to_return[param] = root

    return to_return


def compute_errors(result, params=None, sigma=1):

    if not is_valid_fitresult(result):
        raise ValueError(f"{result} is not a valid fit result!")
    all_params = list(result.params.keys())

    if params is None:
        params = all_params
    params = convert_to_list(params)

    upper_values = get_crossing_value(
        result=result, params=params, direction=1, sigma=sigma
    )

    lower_values = get_crossing_value(
        result=result, params=params, direction=-1, sigma=sigma
    )

    to_return = {}
    for param in params:
        fitted_value = result.params[param]["value"]
        to_return[param] = {
            "lower": lower_values[param] - fitted_value,
            "upper": upper_values[param] - fitted_value,
        }

    return to_return
