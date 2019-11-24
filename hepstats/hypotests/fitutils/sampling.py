from .api_check import is_valid_pdf


def base_sampler(models, nevents, floatting_params=None, *args, **kwargs):

    assert all(is_valid_pdf(m) for m in models)
    assert len(nevents) == len(models)

    if floatting_params:
        floatting_params = [f.name for f in floatting_params]

    samplers = []
    fixed_params = []
    for m in models:
        def to_fix(p):
            if floatting_params:
                return p.name in floatting_params
            else:
                return False
        fixed = [p for p in m.get_dependents() if not to_fix(p)]
        fixed_params.append(fixed)

    for i, (m, p) in enumerate(zip(models, fixed_params)):
        sampler = m.create_sampler(n=nevents[i], fixed_params=p)
        samplers.append(sampler)

    return samplers


def base_sample(sampler, ntoys, parameter=None, value=None, *args, **kwargs):
    for i in range(ntoys):
        if not (parameter is None or value is None):
            with parameter.set_value(value):
                for s in sampler:
                    s.resample()
        else:
            for s in sampler:
                s.resample()

        yield i
