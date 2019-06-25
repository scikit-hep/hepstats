#!/usr/bin/python
from typing import Union, List, Iterable, Any, Callable

from zfit import Parameter
import numpy as np


def convert_to_container(value: Any, container: Callable = list) -> Iterable:

    if not isinstance(value, container):
        try:
            if isinstance(value, (str)):
                raise TypeError
            value = container(value)
        except TypeError:
            value = container((value,))
    return value


class POI(object):

    def __init__(self, parameter: Parameter, values: Union[float, List[float], np.array]):
        """
        Class for parameters of interest:

            Args:
                parameter (`zfit.Parameter`): the parameter of interest
                values (`float`,`list(float)`,`numpy.array`): values of the parameter of interest

            Example:
                Nsig = zfit.Parameter("Nsig")
                poi = POI(Nsig, value=0)
                poi = POI(Nsig, value=np.linspace(0,10,10))
        """
        if not isinstance(parameter, Parameter):
            raise TypeError("Please provide a `zfit.Parameter`.")

        self.parameter = parameter
        self.name = parameter.name
        self._values_tuple = convert_to_container(values, tuple)

    @property
    def value(self):
        if len(self) > 1:
            return self.values_tuple
        else:
            return self.values_tuple[0]

    @property
    def values_tuple(self):
        return self._values_tuple

    def __repr__(self):
        return "POI('{0}', value={1})".format(self.name, self.value)

    def __getitem__(self, i):
        return POI(self.parameter, self.values_tuple[i])

    def __iter__(self):
        for v in self.values_tuple:
            yield POI(self.parameter, v)

    def __len__(self):
        return len(self.values_tuple)

    def __eq__(self, other):
        if not isinstance(other, POI):
            return NotImplemented

        value_equal = self.values_tuple == other.values_tuple
        name_equal = self.name == other.name
        return value_equal and name_equal

    def __hash__(self):
        return hash((self.name, self.value))
