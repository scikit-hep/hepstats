from .api_check import is_valid_pdf

"""
Module providing basic sampling methods.
"""


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

        fixed = [p for p in m.get_dependents() if not to_fix(p)]
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
    """

    for i in range(ntoys):
        if not (parameter is None or value is None):
            with parameter.set_value(value):
                for s in samplers:
                    s.resample()
        else:
            for s in samplers:
                s.resample()

        yield i
