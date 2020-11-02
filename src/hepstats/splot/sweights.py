# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, Any
import warnings

from ..utils import eval_pdf
from ..utils.fit.api_check import is_valid_pdf
from .exceptions import ModelNotFittedToData
from .warnings import AboveToleranceWarning


def is_sum_of_extended_pdfs(model) -> bool:
    """Checks if the input model is a sum of extended models.

    Args:
         model: the input model/pdf

    Returns:
         True if the model is a sum of extended models, False if not.
    """
    if not hasattr(model, "get_models"):
        return False

    return all(m.is_extended for m in model.get_models())


def compute_sweights(model, x: np.ndarray) -> Dict[Any, np.ndarray]:
    """Computes sWeights from probability density functions for different components/species in a fit model
    (for instance signal and background) fitted on some data `x`.

    i.e. model = Nsig * pdf_signal + Nbkg * pdf_bkg

    Args:
        model: sum of extended pdfs.
        x: data on which `model` is fitted

    Returns:
        dictionary with yield parameters as keys, and sWeights for correspoind species as values.

    Example with **zfit**:

        Imports:

        >>> import numpy as np
        >>> import zfit
        >>> from zfit.loss import ExtendedUnbinnedNLL
        >>> from zfit.minimize import Minuit

        Definition of the bounds and yield of background and signal species:

        >>> bounds = (0.0, 3.0)
        >>> nbkg = 10000
        >>> nsig = 5000
        >>> obs = zfit.Space('x', limits=bounds)

        Generation of data:

        >>> bkg = np.random.exponential(0.5, nbkg)
        >>> peak = np.random.normal(1.2, 0.1, nsig)
        >>> data = np.concatenate((bkg, peak))
        >>> data = data[(data > bounds[0]) & (data < bounds[1])]
        >>> N = data.size
        >>> data = zfit.data.Data.from_numpy(obs=obs, array=data)

        Model definition:

        >>> mean = zfit.Parameter("mean", 1.2, 0.5, 2.0)
        >>> sigma = zfit.Parameter("sigma", 0.1, 0.02, 0.2)
        >>> lambda_ = zfit.Parameter("lambda", -2.0, -4.0, -1.0)
        >>> Nsig = zfit.Parameter("Nsig", nsig, 0., N)
        >>> Nbkg = zfit.Parameter("Nbkg", nbkg, 0., N)
        >>> signal = zfit.pdf.Gauss(obs=obs, mu=mean, sigma=sigma).create_extended(Nsig)
        >>> background = zfit.pdf.Exponential(obs=obs, lambda_=lambda_).create_extended(Nbkg)
        >>> tot_model = zfit.pdf.SumPDF([signal, background])

        Loss construction and minimization:

        >>> loss = ExtendedUnbinnedNLL(model=signal + background, data=data)
        >>> minimizer = Minuit()
        >>> minimum = minimizer.minimize(loss)

        sWeights computation:

        >>> from hepstats.splot import compute_sweights
        >>> sweights = compute_sweights(tot_model, data)
        >>> print(sweights)
        {<zfit.Parameter 'Nsig' floating=True value=4985>: array([-0.09953299, -0.09953299, -0.09953299, ...,
        0.78689884, 1.08823111,  1.05948873]),
        <zfit.Parameter 'Nbkg' floating=True value=9989>: array([ 1.09953348,  1.09953348,  1.09953348, ...,
        0.21310097, -0.08823153, -0.05948912])}
    """

    if not is_valid_pdf(model):
        raise ValueError("{} is not a valid pdf!".format(model))
    if not is_sum_of_extended_pdfs(model):
        raise ValueError(
            "Input model, {}, should be a sum of extended pdfs!".format(model)
        )

    models = model.get_models()
    yields = [m.get_yield() for m in models]

    p = np.vstack([eval_pdf(m, x) for m in models]).T
    Nx = eval_pdf(model, x, allow_extended=True)
    pN = p / Nx[:, None]

    MLSR = pN.sum(axis=0)
    atol_warning = 5e-3
    atol_exceptions = 5e-2

    def msg_fn(tolerance):
        msg = (
            "The Maximum Likelihood Sum Rule sanity check, described in equation 17 of"
            + " arXiv:physics/0402083, failed. According to this check the following quantities\n"
        )
        for y, mlsr in zip(yields, MLSR):
            msg += f"\t* {y.name}: {mlsr},\n"
        msg += f"should be equal to 1.0 with an absolute tolerance of {tolerance}."
        return msg

    if not np.allclose(MLSR, 1, atol=atol_exceptions):
        msg = msg_fn(atol_exceptions)
        msg += " The numbers suggest that the model is not fitted to the data. Please check your fit."
        raise ModelNotFittedToData(msg)

    if not np.allclose(MLSR, 1, atol=atol_warning):
        msg = msg_fn(atol_warning)
        msg += " If the fit to the data is good please ignore this warning."
        warnings.warn(msg, AboveToleranceWarning)

    Vinv = (pN).T.dot(pN)
    V = np.linalg.inv(Vinv)

    sweights = p.dot(V) / Nx[:, None]

    return {y: sweights[:, i] for i, y in enumerate(yields)}
