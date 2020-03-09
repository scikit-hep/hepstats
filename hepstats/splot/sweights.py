import numpy as np

from ..utils.fit import eval_pdf, get_value
from ..utils.fit.api_check import is_valid_pdf
from .exceptions import ModelNotFittedToData


def is_sum_of_extended_pdfs(model):
    if not hasattr(model, "get_models"):
        return False

    return all(m.is_extended for m in model.get_models())


def compute_sweights(model, x):
    """Computes sWeights from probability density functions for different components/species in a fit model
    (for instance signal and background) fitted on some data `x`.

        i.e. model = Nsig * pdf_signal + Nbkg * pdf_bkg

        Args:
            * **model**: sum of extended pdfs.
            * **x** (`np.array`): data on which `model` is fitted

        Returns:
            * `dict(`yield`, `np.array`)`: dictionary with yield parameters as keys, and sWeights for
              correspoind species as values.

        Example with `zfit`:
            >>> import numpy as np
            >>> import zfit
            >>> from zfit.loss import ExtendedUnbinnedNLL
            >>> from zfit.minimize import Minuit

            >>> bounds = (0.0, 3.0)
            >>> nbkg = 10000
            >>> nsig = 5000
            >>> zfit.Space('x', limits=bounds)

            >>> bkg = np.random.exponential(0.5, nbkg)
            >>> peak = np.random.normal(1.2, 0.1, nsig)
            >>> data = np.concatenate((bkg, peak))
            >>> data = data[(data > bounds[0]) & (data < bounds[1])]
            >>> N = data.size
            >>> data = zfit.data.Data.from_numpy(obs=obs, array=data)

            >>> mean = zfit.Parameter("mean", 1.2, 0.5, 2.0)
            >>> sigma = zfit.Parameter("sigma", 0.1, 0.02, 0.2)
            >>> lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
            >>> Nsig = zfit.Parameter("Nsig", nsig, 0., N)
            >>> Nbkg = zfit.Parameter("Nbkg", nbkg, 0., N)
            >>> signal = Nsig * zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma)
            >>> background = Nbkg * zfit.pdf.Exponential(obs=obs, lambda_=lambda_)
            >>> loss = ExtendedUnbinnedNLL(model=signal + background, data=data)
            >>> minimizer = Minuit()
            >>> minimum = minimizer.minimize(loss)

            >>> from hepstats.splot import compute_sweights

            >>> sweights = compute_sweights(tot_model, data)
    """

    if not is_valid_pdf(model):
        raise ValueError("{} is not a valid pdf!".format(model))
    if not is_sum_of_extended_pdfs(model):
        raise ValueError("Input model, {}, should be a sum of extended pdfs!".format(model))

    models = model.get_models()
    yields = [m.get_yield() for m in models]

    p = np.vstack([eval_pdf(m, x) for m in models]).T
    Nx = eval_pdf(model, x, allow_extended=True)
    pN = p / Nx[:, None]

    if not np.allclose(pN.sum(axis=0), 1, atol=1e-3):
        raise ModelNotFittedToData("The model needs to fitted to input data in order to comput the sWeights.")

    Vinv = (pN).T.dot(pN)
    V = np.linalg.inv(Vinv)

    sweights = p.dot(V) / Nx[:, None]

    return {y: sweights[:, i] for i, y in enumerate(yields)}
