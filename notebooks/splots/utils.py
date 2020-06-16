# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import zfit


def pltdist(data, bins, bounds, weights=None, label=None):
    y, bin_edges = np.histogram(data, bins=bins, range=bounds, weights=weights)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    yerr = np.sqrt(y)
    plt.errorbar(bin_centers, y, yerr=yerr, fmt=".", color="royalblue", label=label)


def plotfitresult(model, bounds, nbins, **kwargs):
    x = np.linspace(*bounds, num=1000)
    pdf = zfit.run(model.pdf(x, norm_range=bounds) * model.get_yield())
    plt.plot(x, ((bounds[1] - bounds[0]) / nbins) * (pdf), **kwargs)
