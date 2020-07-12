# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
import shutil
from pathlib import Path
import easydev

project_dir = Path(__file__).parents[1]
sys.path.insert(0, str(project_dir))


# -- Project information -----------------------------------------------------

project = "hepstats"
copyright = "2020, Matthieu Marinangeli"
author = "Matthieu Marinangeli"

# The full version, including alpha/beta/rc tags
release = "0.2.5"


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
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
]

source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Napoleon settings ---------------------------------------------

using_numpy_style = False  # False -> google style
napoleon_google_docstring = not using_numpy_style
napoleon_numpy_docstring = using_numpy_style
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- sphinx_autodoc_typehints settings ---------------------------------------------

# if True, set typing.TYPE_CHECKING to True to enable “expensive” typing imports
set_type_checking_flag = False
# if True, class names are always fully qualified (e.g. module.for.Class). If False, just the class
# name displays (e.g. Class)
typehints_fully_qualified = False
# (default: False): If False, do not add ktype info for undocumented parameters. If True, add stub documentation for
# undocumented parameters to be able to add type info.
always_document_param_types = False
# (default: True): If False, never add an :rtype: directive. If True, add the :rtype: directive if no existing :rtype:
# is found.
typehints_document_rtype = True

# -- autodoc settings ---------------------------------------------

# also doc __init__ docstrings
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": True,
}
autodoc_inherit_docstrings = False

# -- sphinx.ext.todo settings ---------------------------------------------
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
copybutton_prompt_text = ">>> "

html_static_path = ["_static"]


html_logo = "images/logo.png"

html_theme_options = {
    "github_url": "https://github.com/scikit-hep/hepstats",
    "use_edit_page_button": True,
    "search_bar_text": "Search hepstats...",
    "navigation_with_keys": True,
    "search_bar_position": "sidebar",
}

html_context = {
    "github_user": "scikit-hep",
    "github_repo": "hepstats",
    "github_version": "master",
    "doc_path": "docs",
}
