

|hepstats_logo|

==============================
Statistics tools and utilities
==============================


The hepstats package is a python library providing statistics tools and utilities for particles physics.
In particular hepstats can work with a fitting library, such as `zfit <https://github.com/zfit/zfit>`_, to build
likelihoods function that hepstats will use to perform statistical inferences. hepstats offers a pythonic
oriented alternative to the RooStat library from the `ROOT <https://root.cern.ch/>`_ data analysis package but
also other tools.

You can install hepstats from PyPI_ with pip:

.. code-block:: console

    $ pip install hepstats


.. toctree::
   :maxdepth: 2

   getting_started/index
   whats_new
   api/index
   bibliography


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |hepstats_logo| image:: images/logo_small.png
   :target: https://github.com/scikit-hep/hepstats
   :alt: hepstats logo


.. _PyPI: https://pypi.org/project/hepstats
