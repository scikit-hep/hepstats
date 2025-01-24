"""
Module providing basic sampling methods.
"""

from __future__ import annotations

from .api_check import is_valid_pdf
from .diverse import get_value, set_values


def base_sampler(models, nevents):
    """
    Creates samplers from models.

    Args:
        models (list(model)): models to sample
        nevents (list(int)): number of in each sampler

    Returns:
        Samplers
    """

    assert all(is_valid_pdf(m) for m in models)
    assert len(nevents) == len(models)

    samplers = []

    for i, m in enumerate(models):
        sampler = m.create_sampler(n=nevents[i])
        samplers.append(sampler)

    return samplers


def base_sample(samplers, ntoys, parameter=None, value=None, constraints=None):
    """
    Samples from samplers. The parameters that are floating in the samplers can be set to a specific value
    using the `parameter` and `value` argument.

    Args:
        samplers (list): generators of samples
        ntoys (int): number of samples to generate
        parameter (optional): floating parameter in the sampler
        value (optional): value of the parameter
        constraints (optional): constraints to sample

    Returns:
        dict: sampled values for each constraint
    """

    sampled_constraints = {}
    if constraints is not None:
        for constr in constraints:
            try:
                sampled_constraints.update({k: get_value(v) for k, v in constr.sample(n=ntoys).items()})
            except AttributeError:
                continue

    params = {} if parameter is None or value is None else {parameter: value}
    for i in range(ntoys):
        with set_values(params):
            for s in samplers:
                s.resample()  # do not pass parameters as arguments as it will fail in simultaneous fits

        if constraints is not None:
            yield {param: value[i] for param, value in sampled_constraints.items()}
        else:
            yield {}
