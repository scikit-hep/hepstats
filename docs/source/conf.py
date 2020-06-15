# -*- coding: utf-8 -*-
import recommonmark
from recommonmark.transform import AutoStructify
from recommonmark.parser import CommonMarkParser

source_parsers = {".md": CommonMarkParser}

# import sphinx_bootstrap_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "hepstats"
copyright = "2020, Matthieu Marinangeli"
author = "Matthieu Marinangeli"

# The full version, including alpha/beta/rc tags
release = "0.2.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.bibtex",
]

extensions += [
    #'matplotlib.sphinxext.only_directives',
    "matplotlib.sphinxext.plot_directive",
]
#'matplotlib.sphinxext.ipython_directive',
#'matplotlib.sphinxext.ipython_console_highlighting']

source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {
            "enable_math": True,
            "enable_eval_rst": True,
            "enable_auto_doc_ref": True,
            "auto_code_block": True,
        },
        True,
    )
    app.add_transform(AutoStructify)
