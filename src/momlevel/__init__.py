""" momlevel - sea level routines for the MOM ocean model """

import importlib.metadata as ilm

msg = ilm.metadata("momlevel")

__name__ = msg["Name"]
__version__ = msg["Version"]
__license__ = msg["License"]
__email__ = msg["Maintainer-email"]
__description__ = msg["Summary"]
__requires__ = msg["Requires-Dist"]
__requires_python__ = msg["Requires-Python"]

from . import derived
from . import eos
from . import reference
from . import spice
from . import test_data
from . import trend
from . import util

from .dynamic import inverse_barometer

from .steric import (
    halosteric,
    steric,
    thermosteric,
)

from .tidegauge import extract_tidegauge
