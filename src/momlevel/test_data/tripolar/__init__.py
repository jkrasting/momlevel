""" tripolar - module for generating test data on a tripolar grid """

from .horizontal import xy_fields
from .vertical import zlevel_fields

__all__ = [
    "xy_fields",
    "zlevel_fields",
]
