import os
import numpy as np
import matplotlib.pyplot as plt

import skstats
from skstats.hypotests.plotutils import plotlimit

pvalues_dir = os.path.dirname(skstats.__file__)+'/../tests/hypotests/data'


def test_plotlimit():

    cls_pvalues = dict(np.load(pvalues_dir+'/cls_pvalues.npz'))
    clsb_pvalues = dict(np.load(pvalues_dir+'/clsb_pvalues.npz'))

    poivalues = cls_pvalues.pop("poivalues")
    clsb_pvalues.pop("poivalues")

    ax = plotlimit(poivalues, cls_pvalues, CLs=True)

    f, ax = plt.subplots()
    plotlimit(poivalues, clsb_pvalues, CLs=False, ax=ax)
