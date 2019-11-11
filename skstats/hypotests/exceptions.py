"""
Specific exceptions for the `hypotests` submodule
"""


class POIRangeError(Exception):
    """Exception class non adequate POI scan range"""

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
