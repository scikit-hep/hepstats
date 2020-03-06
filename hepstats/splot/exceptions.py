"""
Specific exceptions for the `splot` submodule
"""

class ModelNotFittedToData(Exception):
    """Exception class for model not fitted to data provided to compute sweights"""

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
