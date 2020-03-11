import numpy as np

from ..utils.fit.api_check import is_valid_fitresult
from ..utils import pll


def set_params_to_fitresult(params, fit_result):
    for p in params:
        p.set_value(fit_result.params[p]["value"])


def compute_errors(fit_result, parameters=None, sigma=1):

    assert is_valid_fitresult(fit_result)
    all_parameters = list(fit_result.params.keys())
    loss = fit_result.loss

    if parameters is None:
        parameters = all_parameters

    covariance = fit_result.covariance(as_dict=True)
    up = loss.errordef
    fmin = fit_result.fmin

    for p in parameters:
        set_params_to_fitresult(all_parameters, fit_result)
        hesse_error = covariance[(p, p)] ** 0.5
        fitted_value = fit_result.params[p]["value"]

        p.set_value(fitted_value + sigma * hesse_error)

        for ap in all_parameters:
            if ap == p:
                continue

            ap_value = fit_result.params[ap]["value"]
            ap_error = covariance[(ap, ap)] ** 0.5

            ap_value += sigma * covariance[(p, ap)] * np.sqrt(up * 2) / ap_error

            ap.set_value(ap_value)

            delta_loss = fmin - pll()
