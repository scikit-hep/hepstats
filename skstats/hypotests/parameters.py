#!/usr/bin/python
from typing import Union, List
import numpy as np

from .fitutils.api_check import is_valid_parameter


class POI(object):

    def __init__(self, parameter, values: Union[float, List[float], np.array]):
        """
        Class for parameters of interest:

            Args:
                parameter: the parameter of interest
                values (`float`,`list(float)`,`numpy.array`): values of the parameter of interest

            Example with `zfit`:
                >>> Nsig = zfit.Parameter("Nsig")
                >>> poi = POI(Nsig, value=0)
                >>> poi = POI(Nsig, value=np.linspace(0,10,10))
        """
        if not is_valid_parameter(parameter):
            return NotImplementedError

        self.parameter = parameter
        self.name = parameter.name
        self._values_array = np.atleast_1d(values)

    @property
    def value(self):
        """
        Returns the value of the `POI`.
        """
        if len(self) > 1:
            return self.values_array
        else:
            return self.values_array[0]

    @property
    def values_array(self):
        """
        Returns the array of values of the `POI`.
        """
        return self._values_array

    def __repr__(self):
        return "POI('{0}', value={1})".format(self.name, self.value)

    def __getitem__(self, i):
        """
        Get the i th element the array of values of the `POI`.
        """
        return POI(self.parameter, self.values_array[i])

    def __iter__(self):
        for v in self.values_array:
            yield POI(self.parameter, v)

    def __len__(self):
        return len(self.values_array)

    def __eq__(self, other):
        if not isinstance(other, POI):
            return NotImplemented

        value_equal = self.values_array == other.values_array
        name_equal = self.name == other.name
        return value_equal.all() and name_equal

    def __hash__(self):
        return hash((self.name, self.value.tostring()))
