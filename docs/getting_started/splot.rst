splot
#####

A full example using the **sPlot** algorithm can be found `here <https://github.com/scikit-hep/hepstats/tree/master/notebooks/splots/splot_example.ipynb>`_ . **sWeights** for different components in a data sample, modeled with a sum of extended probability density functions, are derived using the ``compute_sweights`` function:

.. code-block:: python

    >>> from hepstats.splot import compute_sweights
    >>> # using same model as above for illustration
    >>> sweights = compute_sweights(zfit.pdf.SumPDF([signal, background]), data)
    >>> bkg_sweights = sweights[Nbkg]
    >>> sig_sweights = sweights[Nsig]


The model needs to be fitted to the data for the computation of the **sWeights**, if not an error is raised.
