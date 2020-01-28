from contextlib import ExitStack
import numbers


def eval_pdf(model, x, params):
    """ Compute pdf of model at a given point x and for given parameters values """

    def pdf(model, x):
        if model.is_extended:
            ret = model.pdf(x) * model.get_yield()
        else:
            ret = model.pdf(x)

        if hasattr(ret, "numpy"):
            return ret.numpy()
        else:
            return ret

    with ExitStack() as stack:
        for param in model.get_dependents():
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
    dataset_cls: only used to get the class in which array/weights will be
    converted.
    """

    if hasattr(dataset_cls, "from_numpy"):
        return dataset_cls.from_numpy(obs=obs, array=array, weights=weights)
    else:
        return dataset_cls(obs=obs, array=array, weights=weights)


def get_nevents(dataset):
    """ Returns the number of events in the dataset """

    nevents = dataset.nevents

    if hasattr(nevents, "numpy"):
        return nevents.numpy()
    else:
        return nevents
