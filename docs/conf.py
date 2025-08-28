# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

from hepstats import __version__ as version

project_dir = Path(__file__).parents[1]


# -- Project information -----------------------------------------------------

project = "hepstats"
copyright = "2019-2025, The Scikit-HEP Administrators"
author = "Matthieu Marinangeli"

# The full version, including alpha/beta/rc tags

release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.inheritance_diagram",
    # "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

bibtex_bibfiles = [
    str(project_dir / "docs" / "bib" / "references.bib")
]  # TODO: currently string, Path doesn't work: https://github.com/mcmtroffaes/sphinxcontrib-bibtex/issues/314
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

copybutton_prompt_text = ">>> "

# -- autodoc settings ---------------------------------------------

# also doc __init__ docstrings
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": True,
}
autodoc_inherit_docstrings = False

html_static_path = []  # "_static"


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
    "github_version": "main",
    "doc_path": "docs",
}
