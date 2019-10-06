from contextlib import ExitStack


def eval_pdf(model, x, params):
    """ Compute pdf of model at a given point x and for given parameters values """
    if "zfit" in str(model.__class__):
        import zfit

        def pdf(model, x):
            ret = zfit.run(model.pdf(x) * model.get_yield())
            return ret
    else:
        def pdf(model, x):
            return model.pdf(x) * model.get_yield()

    dependents = list(model.get_dependents())

    with ExitStack() as stack:
        for param in dependents:
            value = params[param]["value"]
            stack.enter_context(param.set_value(value))
        return pdf(model, x)


def pll(minimizer, loss, pois) -> float:
    """ Compute minimum profile likelihood for given parameters values. """
    with ExitStack() as stack:
        for p in pois:
            param = p.parameter
            stack.enter_context(param.set_value(p.value))
            param.floating = False
        minimum = minimizer.minimize(loss=loss)
        for p in pois:
            p.parameter.floating = True
    return minimum.fmin


def array2dataset(dataset_cls, obs, array, weights=None):
    """
    dataset: only used to get the class in which array/weights will be
    converted.
    """

    if hasattr(dataset_cls, "from_numpy"):
        return dataset_cls.from_numpy(obs=obs, array=array, weights=weights)
    else:
        return dataset_cls(obs=obs, array=array, weights=weights)
