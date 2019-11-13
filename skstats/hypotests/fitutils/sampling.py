from .api_check import is_valid_pdf


def base_sampler(models, floatting_params=None, *args, **kwargs):

    assert all(is_valid_pdf(m) for m in models)

    if floatting_params:
        floatting_params = [f.name for f in floatting_params]

    samplers = []
    fixed_params = []
    for m in models:
        def cond(p):
            if floatting_params:
                return p.name in floatting_params
            else:
                return False
        fixed = [p for p in m.get_dependents() if not cond(p)]
        fixed_params.append(fixed)

    for m, p in zip(models, fixed_params):
        n_ = kwargs.get("n", None)
        if n_ is None:
            if m.is_extended:
                n_ = "extended"
        sampler = m.create_sampler(n=n_, fixed_params=p)
        samplers.append(sampler)

    return samplers


def base_sample(sampler, ntoys, param=None, value=None, *args, **kwargs):
    for i in range(ntoys):
        if not (param is None or value is None):
            with param.set_value(value):
                for s in sampler:
                    s.resample()
        else:
            for s in sampler:
                s.resample()

        yield i
