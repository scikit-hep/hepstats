# Licensed under a 3-clause BSD style license, see LICENSES folder.
"""
Specific exceptions for the `splot` submodule.
"""

from __future__ import annotations


class ModelNotFittedToData(Exception):
    """Exception class for model not fitted to data provided to compute sweights"""
