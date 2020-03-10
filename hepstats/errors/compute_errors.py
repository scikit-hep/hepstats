from ..utils.fit.api_check import is_valid_fitresult


def compute_errors(fit_result, parameters, sigma=1):

    assert is_valid_fitresult(fit_result)
