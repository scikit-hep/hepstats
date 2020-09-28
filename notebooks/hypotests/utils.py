# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import zfit


def pltdist(data, bins, bounds):
    y, bin_edges = np.histogram(data, bins=bins, range=bounds)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    yerr = np.sqrt(y)
    plt.errorbar(bin_centers, y, yerr=yerr, fmt=".", color="royalblue")


def plotfitresult(model, bounds, nbins):
    x = np.linspace(*bounds, num=1000)
    if model.is_extended:
        pdf = model.ext_pdf(x, norm_range=bounds) * ((bounds[1] - bounds[0]) / nbins)
    else:
        pdf = model.pdf(x, norm_range=bounds)
    plt.plot(x, pdf, "-r", label="fit result")


def plotlimit(ul, alpha=0.05, CLs=True, ax=None):
    """
    plot pvalue scan for different values of a parameter of interest (observed, expected and +/- sigma bands)

    Args:
        ul: UpperLimit instance
        alpha (float, default=0.05): significance level
        CLs (bool, optional): if `True` uses pvalues as $$p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}$$
            else as $$p_{clsb} = p_{null}$
        ax (matplotlib axis, optionnal)

    """
    if ax is None:
        ax = plt.gca()

    poivalues = ul.poinull.values
    pvalues = ul.pvalues(CLs=CLs)

    if CLs:
        cls_clr = "r"
        clsb_clr = "b"
    else:
        cls_clr = "b"
        clsb_clr = "r"

    color_1sigma = "mediumseagreen"
    color_2sigma = "gold"

    ax.plot(
        poivalues,
        pvalues["cls"],
        label="Observed CL$_{s}$",
        marker=".",
        color="k",
        markerfacecolor=cls_clr,
        markeredgecolor=cls_clr,
        linewidth=2.0,
        ms=11,
    )

    ax.plot(
        poivalues,
        pvalues["clsb"],
        label="Observed CL$_{s+b}$",
        marker=".",
        color="k",
        markerfacecolor=clsb_clr,
        markeredgecolor=clsb_clr,
        linewidth=2.0,
        ms=11,
        linestyle=":",
    )

    ax.plot(
        poivalues,
        pvalues["clb"],
        label="Observed CL$_{b}$",
        marker=".",
        color="k",
        markerfacecolor="k",
        markeredgecolor="k",
        linewidth=2.0,
        ms=11,
    )

    ax.plot(
        poivalues,
        pvalues["expected"],
        label="Expected CL$_{s}-$Median",
        color="k",
        linestyle="--",
        linewidth=1.5,
        ms=10,
    )

    ax.plot(
        [poivalues[0], poivalues[-1]],
        [alpha, alpha],
        color="r",
        linestyle="-",
        linewidth=1.5,
    )

    ax.fill_between(
        poivalues,
        pvalues["expected"],
        pvalues["expected_p1"],
        facecolor=color_1sigma,
        label="Expected CL$_{s} \\pm 1 \\sigma$",
        alpha=0.8,
    )

    ax.fill_between(
        poivalues,
        pvalues["expected"],
        pvalues["expected_m1"],
        facecolor=color_1sigma,
        alpha=0.8,
    )

    ax.fill_between(
        poivalues,
        pvalues["expected_p1"],
        pvalues["expected_p2"],
        facecolor=color_2sigma,
        label="Expected CL$_{s} \\pm 2 \\sigma$",
        alpha=0.8,
    )

    ax.fill_between(
        poivalues,
        pvalues["expected_m1"],
        pvalues["expected_m2"],
        facecolor=color_2sigma,
        alpha=0.8,
    )

    ax.set_ylim(-0.01, 1.1)
    ax.set_ylabel("p-value")
    ax.set_xlabel("parameter of interest")
    ax.legend(loc="best", fontsize=14)

    return ax


def one_minus_cl_plot(x, pvalues, alpha=[0.32], ax=None):

    if ax is None:
        ax = plt.gca()

    ax.plot(x, pvalues, ".--")
    for a in alpha:
        ax.axhline(a, color="red", label="$\\alpha = " + str(a) + "$")
    ax.set_ylabel("1-CL")

    return ax
