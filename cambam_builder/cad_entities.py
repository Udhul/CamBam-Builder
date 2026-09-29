"""Compatibility imports for :mod:`cambam_builder.native.cad`."""

from .native.cad import *
from .native import cad as _owner


def __getattr__(name):
    return getattr(_owner, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_owner)))
