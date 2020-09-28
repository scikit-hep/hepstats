# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license, see LICENSE
"""
Module defining the parameter of interest classes, currently includes:

* **POIarray**
* **POI**
"""
import numpy as np
from typing import Union, Collection

from ..utils.fit.api_check import is_valid_parameter


class POIarray(object):
    """
    Class for parameters of interest with multiple values:
    """

    def __init__(self, parameter, values: Union[Collection, np.array]):
        """
        Args:
            parameter: the parameter of interest
            values: values of the parameter of interest

        Raises:
            ValueError: if is_valid_parameter(parameter) returns False
            TypeError: if parameter is not an iterable

        Example with `zfit`:
            >>> Nsig = zfit.Parameter("Nsig")
            >>> poi = POIarray(Nsig, value=np.linspace(0,10,10))
        """

        if not is_valid_parameter(parameter):
            raise ValueError(f"{parameter} is not a valid parameter!")

        if not isinstance(values, Collection):
            raise TypeError("A list/array of values of the POI is required.")

        self.parameter = parameter
        self.name = parameter.name
        self._values = np.array(values, dtype=np.float64)
        self._ndim = 1
        self._shape = (len(values),)

    @property
    def values(self):
        """
        Returns the values of the **POIarray**.
        """
        return self._values

    def __repr__(self):
        return "POIarray('{0}', values={1})".format(self.name, self.values)

    def __getitem__(self, i):
        """
        Get the i-th element the array of values of the **POIarray**.
        """
        return POI(self.parameter, self.values[i])

    def __iter__(self):
        for v in self.values:
            yield POI(self.parameter, v)

    def __len__(self):
        return len(self.values)

    def __eq__(self, other):
        if not isinstance(other, POIarray):
            return NotImplemented

        if len(self) != len(other):
            return False

        values_equal = self.values == other.values
        name_equal = self.name == other.name
        return values_equal.all() and name_equal

    def __hash__(self):
        return hash((self.name, self.values.tostring()))

    @property
    def ndim(self):
        """
        Returns the number of dimension of the **POIarray**.
        """
        return self._ndim

    @property
    def shape(self):
        """
        Returns the shape of the **POIarray**.
        """
        return self._shape

    def append(self, values: Union[int, float, Collection, np.ndarray]):
        """
        Append values in the **POIarray**.

        Args:
            values: values to append
        """
        if not isinstance(values, Collection):
            values = [values]
        values = np.concatenate([self.values, values])
        return POIarray(parameter=self.parameter, values=values)


class POI(POIarray):
    """
    Class for single value parameter of interest:
    """

    def __init__(self, parameter, value: Union[int, float]):
        """
        Args:
            parameter: the parameter of interest
            values: value of the parameter of interest

        Raises:
            TypeError: if value is an iterable

        Example with `zfit`:
            >>> Nsig = zfit.Parameter("Nsig")
            >>> poi = POI(Nsig, value=0)
        """
        if isinstance(value, Collection):
            raise TypeError("A single value for the POI is required.")

        super(POI, self).__init__(parameter=parameter, values=[value])
        self._value = value

    @property
    def value(self):
        """
        Returns the value of the **POI**.
        """
        return self._value

    def __eq__(self, other):
        if not isinstance(other, POI):
            return NotImplemented

        value_equal = self.value == other.value
        name_equal = self.name == other.name
        return value_equal and name_equal

    def __repr__(self):
        return "POI('{0}', value={1})".format(self.name, self.value)

    def __hash__(self):
        return hash((self.name, self.value))


def asarray(poi: POI) -> POIarray:
    """
    Transforms a **POI** instance into a **POIarray** instance.

    Args:
        poi: the parameter of interest.
    """
    return POIarray(parameter=poi.parameter, values=poi.values)
