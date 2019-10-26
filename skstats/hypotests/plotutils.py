import matplotlib.pyplot as plt


def plotlimit(poivalues, pvalues, alpha=0.05, CLs=True, ax=None):

    if ax is None:
        _, ax = plt.subplots()

    if CLs:
        cls_clr = "r"
        clsb_clr = "b"
    else:
        cls_clr = "b"
        clsb_clr = "r"

    ax.plot(poivalues, pvalues["cls"], label="Observed CL$_{s}$", marker=".", color='k',
            markerfacecolor=cls_clr, markeredgecolor=cls_clr, linewidth=2.0, ms=11)

    ax.plot(poivalues, pvalues["clsb"], label="Observed CL$_{s+b}$", marker=".", color='k',
            markerfacecolor=clsb_clr, markeredgecolor=clsb_clr, linewidth=2.0, ms=11, linestyle=":")

    ax.plot(poivalues, pvalues["clb"], label="Observed CL$_{b}$", marker=".", color='k', markerfacecolor="k",
            markeredgecolor="k", linewidth=2.0, ms=11)

    ax.plot(poivalues, pvalues["expected"], label="Expected CL$_{s}-$Median", color='k', linestyle="--",
            linewidth=1.5, ms=10)

    ax.plot([poivalues[0], poivalues[-1]], [alpha, alpha], color='r', linestyle='-', linewidth=1.5)

    ax.fill_between(poivalues, pvalues["expected"], pvalues["expected_p1"], facecolor="lime",
                    label="Expected CL$_{s} \\pm 1 \\sigma$")

    ax.fill_between(poivalues, pvalues["expected"], pvalues["expected_m1"], facecolor="lime")

    ax.fill_between(poivalues, pvalues["expected_p1"], pvalues["expected_p2"], facecolor="yellow",
                    label="Expected CL$_{s} \\pm 2 \\sigma$")

    ax.fill_between(poivalues, pvalues["expected_m1"], pvalues["expected_m2"], facecolor="yellow")

    ax.set_ylim(-0.01, 1.1)
    ax.set_ylabel("p-value")
    ax.set_xlabel("parameter of interest")
    ax.legend(loc="best", fontsize=14)

    return ax
