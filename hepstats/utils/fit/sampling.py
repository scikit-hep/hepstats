from .api_check import is_valid_pdf
from .diverse import get_value

"""
Module providing basic sampling methods.
"""

class Sampler

    def __init__(loss, nevents, floating_params=None):


def base_sampler(models, nevents, floating_params=None):
    """
    Creates samplers from models.

    Args:
        models (list(model)): models to sample
        nevents (list(int)): number of in each sampler
        floating_params (list(parameter), optionnal): floating parameter in the samplers
    """

    assert all(is_valid_pdf(m) for m in models)
    assert len(nevents) == len(models)

    if floating_params:
        floating_params_names = [f.name for f in floating_params]

    samplers = []
    fixed_params = []
    for m in models:

        def to_fix(p):
            if floating_params:
                return p.name in floating_params_names
            else:
                return False

        fixed = [p for p in m.get_params() if not to_fix(p)]
        fixed_params.append(fixed)

    for i, (m, p) in enumerate(zip(models, fixed_params)):
        sampler = m.create_sampler(n=nevents[i], fixed_params=p)
        samplers.append(sampler)

    return samplers


def base_sample(samplers, ntoys, parameter=None, value=None):
    """
    Sample from samplers. The parameters that are floating in the samplers can be set to a specific value
    using the `parameter` and `value` argument.

    Args:
        samplers (list): generators of samples
        ntoys (int): number of samples to generate
        parameter (optional): floating parameter in the sampler
        value (optional): value of the parameter
        constraints (optional): constraints to sample
    """

    sampled_constraints = {}
    if constraints is not None:
        for constr in constraints:
            sampled_constraints.update({k: get_value(v) for k, v in constr.sample(n=ntoys).items()})

    for i in range(ntoys):
        if not (parameter is None or value is None):
            with parameter.set_value(value):
                for s in samplers:
                    s.resample()
        else:
            for s in samplers:
                s.resample()

        for param, value in sampled_constraints:
            param.set_value(value[i])

        yield i


def base_minimize_sample(minimizer, loss, n_trials=2):

    for minimize_trial in range(n_trials):
        try:
            minimum = minimizer.minimize(loss=toys_loss)
            converged = minimum.converged
            if converged:
                break
        except RuntimeError:
            converged = False
            break
